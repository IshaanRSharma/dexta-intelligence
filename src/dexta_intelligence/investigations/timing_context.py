"""Timing context - a prospective, time-bucket-scoped briefing.

The retrospective twin of ``explain_spike``: instead of "why did I spike?", this
answers "what does my data show for *this period of day*?" for a chosen bucket
(overnight, dinner, a custom window). It is deterministic - the cards are fixed
code paths composing existing tools, never LLM-chosen - and it is observation
only: pump profile read-outs and historical patterns, never a dose, ratio, or
"raise/lower" directive (the treatment gate's bar).

Output schema is frozen: ``{bucket, cards, limitations, trace, safety}``. Each
card is ``{id, title, lines, n}``.
"""

from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, time, timedelta
from typing import TYPE_CHECKING, Any

from dexta_intelligence.agents.reason import ToolCall
from dexta_intelligence.agents.tools.toolkit import DiscoveryToolkit
from dexta_intelligence.agents.trace import render_trace

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dexta_intelligence.agents.base import AgentContext

logger = logging.getLogger(__name__)

__all__ = [
    "OUTPUT_KEYS",
    "PRESETS",
    "SAFETY_LINE",
    "Bucket",
    "TimingCard",
    "TimingContext",
    "gather_timing_context",
    "resolve_bucket",
    "timing_report",
]

OUTPUT_KEYS = ("bucket", "cards", "limitations", "trace", "safety")
SAFETY_LINE = "Pattern context for a time window. No dosing recommendation."

#: Local-time, hour-granular presets (half-open [start, end)).
PRESETS: dict[str, tuple[int, int]] = {
    "overnight": (0, 6),
    "breakfast": (6, 10),
    "lunch": (11, 14),
    "dinner": (17, 22),
    "bedtime": (21, 24),
}

#: A bolus this many minutes after the carb entry counts as the late-bolus signal.
_LATE_BOLUS_MIN = 15
#: Buckets whose midpoint falls in these hours get the meal cards by default.
_MEAL_HOURS = range(6, 22)

#: The provenance + safety stamp the oref0 card always carries. The algorithm's
#: own forecast, scored against what actually happened - never a forward dose.
_OREF_LABEL = "oref0 algorithm computation. Observation only, never a dose."
#: Horizon (minutes) at which the logged forecast is compared to realized CGM.
_FORECAST_HORIZON_MIN = 60
#: Curve preference when a cycle logged several oref0 scenarios.
_CURVE_PREFERENCE = ("cob", "loop", "iob", "uam", "zt")
#: How close a CGM reading must be to the forecast horizon to count as realized.
#: Real CGM is not on the algorithm's cycle grid, so an exact match is too strict.
_ALIGN_TOL = timedelta(minutes=2, seconds=30)


@dataclass(frozen=True, slots=True)
class Bucket:
    """A local-time, half-open hour window the briefing is scoped to."""

    name: str
    start_hour: int
    end_hour: int

    @property
    def label(self) -> str:
        return f"{self.start_hour:02d}:00-{self.end_hour:02d}:00"

    @property
    def midpoint_hour(self) -> int:
        return (self.start_hour + self.end_hour) // 2


@dataclass(frozen=True, slots=True)
class TimingCard:
    """One observation-only card. ``n`` is the supporting sample size, if any."""

    id: str
    title: str
    lines: list[str]
    n: int | None = None


@dataclass(frozen=True, slots=True)
class TimingContext:
    """Deterministically gathered briefing for one bucket. No LLM in here."""

    bucket: Bucket
    cards: list[TimingCard]
    limitations: list[str]
    steps: list[ToolCall]
    pool: dict[str, Any] = field(default_factory=dict)


def resolve_bucket(spec: str) -> Bucket | None:
    """A preset name (``dinner``) or an hour range (``17-22`` / ``17:00-22:00``).

    ``None`` when unparseable. Minute precision is truncated to the hour (the
    glucose bucketing is hour-granular in v1)."""
    raw = spec.strip().lower()
    if raw in PRESETS:
        start, end = PRESETS[raw]
        return Bucket(name=raw, start_hour=start, end_hour=end)
    if "-" not in raw:
        return None
    left, _, right = raw.partition("-")
    start_h = _parse_hour(left)
    end_h = _parse_hour(right)
    if start_h is None or end_h is None or not (0 <= start_h < end_h <= 24):
        return None
    return Bucket(name=raw, start_hour=start_h, end_hour=end_h)


def _parse_hour(text: str) -> int | None:
    head = text.strip().split(":", 1)[0]
    if not head.isdigit():
        return None
    hour = int(head)
    return hour if 0 <= hour <= 24 else None


def gather_timing_context(
    ctx: AgentContext,
    bucket: Bucket,
    *,
    intent: str = "general",
    target_low: int = 70,
    target_high: int = 180,
) -> TimingContext:
    """Compose the fixed cards for ``bucket``: profile read-out, glucose in the
    window, basal context, and meal patterns when the intent calls for it.

    Pure: no model, never raises on bad input. Mirrors ``gather_spike_evidence``
    so the audit trail is reproducible and the numbers stay guard-traceable."""
    toolkit = DiscoveryToolkit(ctx, target_low=target_low, target_high=target_high)
    caps = toolkit.capabilities()
    steps: list[ToolCall] = []
    pool: dict[str, Any] = {}
    limitations: list[str] = []
    cards: list[TimingCard] = []

    def step(name: str, args: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        ok = not result.get("error")
        steps.append(ToolCall(name=name, args=args, ok=ok, result=result))
        if ok:
            _collect(pool, name, len(steps), result)
        return result

    anchor_iso = datetime.combine(ctx.window[1], time(12), tzinfo=UTC).isoformat()
    cards.append(_profile_card(step, toolkit, bucket, anchor_iso, limitations))
    cards.append(_glucose_card(step, toolkit, bucket, limitations))

    want_basal = intent == "basal" or bucket.name == "overnight"
    if want_basal and caps.has_insulin:
        cards.append(_basal_card(step, toolkit))
    elif want_basal:
        limitations.append("no insulin data - basal context skipped")

    oref = _oref_card(step, ctx, toolkit, bucket, limitations)
    if oref is not None:
        cards.append(oref)

    if intent == "meal":
        if not caps.has_insulin:
            limitations.append("no insulin data - meal-timing cards skipped")
        elif bucket.midpoint_hour not in _MEAL_HOURS:
            limitations.append("bucket is outside meal hours - meal-timing cards skipped")
        else:
            cards.extend(_meal_cards(step, toolkit, bucket, limitations))

    return TimingContext(
        bucket=bucket, cards=cards, limitations=limitations, steps=steps, pool=pool
    )


def timing_report(
    ctx: AgentContext,
    bucket: Bucket,
    *,
    intent: str = "general",
    target_low: int = 70,
    target_high: int = 180,
) -> dict[str, Any]:
    """The frozen output dict for a bucket briefing (CLI / web / tests)."""
    tc = gather_timing_context(
        ctx, bucket, intent=intent, target_low=target_low, target_high=target_high
    )
    return {
        "bucket": {"name": tc.bucket.name, "label": tc.bucket.label, "intent": intent},
        "cards": [{"id": c.id, "title": c.title, "lines": c.lines, "n": c.n} for c in tc.cards],
        "limitations": tc.limitations,
        "trace": [line.text for line in render_trace(tc.steps)],
        "safety": SAFETY_LINE,
    }


# ── cards ──────────────────────────────────────────────────────────────────────


def _profile_card(
    step: Any, toolkit: DiscoveryToolkit, bucket: Bucket, anchor_iso: str, limitations: list[str]
) -> TimingCard:
    profile = step("get_insulin_profile", {}, toolkit.get_insulin_profile())
    if profile.get("error"):
        # Fall back to the versioned profile in effect (covers stores with version
        # history but no current-snapshot, like the demo).
        profile = step(
            "get_active_profile", {"timestamp": anchor_iso}, toolkit.get_active_profile(anchor_iso)
        )
    if profile.get("error"):
        limitations.append("no pump profile synced - segment read-out skipped")
        return TimingCard(id="P", title="Pump profile for this window", lines=[], n=None)
    segments = _segments_overlapping(_profile_segments(profile), bucket)
    if not segments:
        limitations.append("no profile segment overlaps this window")
        return TimingCard(id="P", title="Pump profile for this window", lines=[], n=None)
    lines = [
        f"{s.get('start', '??')} - basal {s.get('basal_u_hr', '?')} U/hr, "
        f"ISF {s.get('isf_mg_dl_u', '?')} mg/dL/U, CR {s.get('carb_ratio_g_u', '?')} g/U, "
        f"target {s.get('target_mg_dl', '?')} mg/dL  (on pump now, not a recommendation)"
        for s in segments
    ]
    return TimingCard(id="P", title="Pump profile for this window", lines=lines, n=len(segments))


def _glucose_card(
    step: Any, toolkit: DiscoveryToolkit, bucket: Bucket, limitations: list[str]
) -> TimingCard:
    g = step(
        "glucose_stats",
        {"hours": [bucket.start_hour, bucket.end_hour]},
        toolkit.glucose_stats(hours=(bucket.start_hour, bucket.end_hour)),
    )
    n = int(g.get("n") or 0)
    if not n:
        limitations.append("no glucose readings in this window over the active range")
        return TimingCard(id="G", title="Glucose in this window (recent)", lines=[], n=0)
    lines = [
        f"Mean {g.get('mean')} mg/dL, median {g.get('median')} mg/dL",
        f"Time in range {g.get('tir_pct')}% (low {g.get('tbr_pct')}%, high {g.get('tar_pct')}%)",
        f"Variability CV {g.get('cv_pct')}%",
    ]
    return TimingCard(id="G", title="Glucose in this window (recent)", lines=lines, n=n)


def _basal_card(step: Any, toolkit: DiscoveryToolkit) -> TimingCard:
    basal = step("get_basal_timeline", {}, toolkit.get_basal_timeline())
    if basal.get("basal_stable"):
        lines = ["Basal delivery was stable across the active window (no temp/suspend activity)."]
    else:
        n_temp = basal.get("n_temp_basal")
        n_susp = basal.get("n_suspend")
        parts = []
        if isinstance(n_temp, int) and n_temp:
            parts.append(f"{n_temp} temp-basal")
        if isinstance(n_susp, int) and n_susp:
            parts.append(f"{n_susp} suspend")
        suffix = f" ({', '.join(parts)})" if parts else ""
        lines = [f"Temp-basal or suspend activity occurred in the window{suffix}."]
    lines.append(
        "Observation only - basal changes are a decision for you, Loop, and your care team."
    )
    return TimingCard(id="B", title="Basal delivery in this window", lines=lines, n=None)


def _oref_card(
    step: Any,
    ctx: AgentContext,
    toolkit: DiscoveryToolkit,
    bucket: Bucket,
    limitations: list[str],
) -> TimingCard | None:
    """How oref0's own logged forecasts held up against realized CGM in this window.

    Pure backward observation: each cycle's predicted curve is scored against what
    glucose actually did ``_FORECAST_HORIZON_MIN`` minutes later. Never a forward
    dose. ``None`` when no forecast curves were logged for the window."""
    # Local-day bounds resolved in the patient timezone (match the toolkit's
    # window), then converted to UTC for the store query.
    win_start = datetime.combine(ctx.window[0], time.min, tzinfo=toolkit.tzinfo).astimezone(UTC)
    win_end = datetime.combine(ctx.window[1], time.max, tzinfo=toolkit.tzinfo).astimezone(UTC)
    try:
        predictions = ctx.store.get_predictions(win_start, win_end)
        glucose = ctx.store.get_glucose(win_start, win_end)
    except (AttributeError, NotImplementedError):  # minimal/partial stores
        predictions = []
        glucose = []
    if not predictions:
        limitations.append(
            "no oref0 forecast curves logged in this window - reconciliation skipped"
        )
        return None

    g_ts = [g.ts for g in glucose]  # store returns ascending by ts
    g_val = [g.mg_dl for g in glucose]
    cycles: dict[datetime, dict[str, list[float]]] = {}
    for p in predictions:
        cycles.setdefault(p.ts, {})[p.curve_kind] = p.values_mg_dl

    idx = _FORECAST_HORIZON_MIN // 5
    errors: list[float] = []
    for cycle_ts, curves in cycles.items():
        if not (bucket.start_hour <= cycle_ts.astimezone(toolkit.tzinfo).hour < bucket.end_hour):
            continue
        curve = next((curves[k] for k in _CURVE_PREFERENCE if k in curves), None)
        if curve is None or len(curve) <= idx:
            continue
        realized = _nearest_value(g_ts, g_val, cycle_ts + timedelta(minutes=_FORECAST_HORIZON_MIN))
        if realized is None:
            continue
        errors.append(abs(curve[idx] - realized))

    if not errors:
        limitations.append("oref0 forecasts present but none aligned to CGM in this window")
        return None

    result = step(
        "oref_reconcile",
        {"hours": [bucket.start_hour, bucket.end_hour], "horizon_min": _FORECAST_HORIZON_MIN},
        {"n_cycles": len(errors), "median_error_mg_dl": round(_median(errors), 1)},
    )
    lines = [
        f"{result['n_cycles']} oref0 forecast cycles in this window",
        f"Median {_FORECAST_HORIZON_MIN}-min forecast error {result['median_error_mg_dl']} mg/dL "
        "(the algorithm's prediction vs your CGM)",
        _OREF_LABEL,
    ]
    return TimingCard(
        id="O", title="oref0 forecast vs reality (this window)", lines=lines, n=len(errors)
    )


def _meal_cards(
    step: Any, toolkit: DiscoveryToolkit, bucket: Bucket, limitations: list[str]
) -> list[TimingCard]:
    boluses = step("get_boluses", {}, toolkit.get_boluses())
    delays = _bucket_bolus_delays(boluses, bucket, toolkit)
    if not delays:
        limitations.append("no bolused meals with timing in this window")
        return []
    usual = TimingCard(
        id="U",
        title="Your usual timing in this window",
        lines=[f"Median bolus delay {_median(delays):.0f} min after the carb entry"],
        n=len(delays),
    )
    late = [d for d in delays if d >= _LATE_BOLUS_MIN]
    cards = [usual]
    if late:
        cards.append(
            TimingCard(
                id="L",
                title="When you bolused late in this window",
                lines=[
                    f"{len(late)} of {len(delays)} boluses were >= {_LATE_BOLUS_MIN} min late",
                    f"Median late delay {_median(late):.0f} min after the carb entry",
                ],
                n=len(late),
            )
        )
    return cards


# ── helpers ────────────────────────────────────────────────────────────────────


def _profile_segments(profile: dict[str, Any]) -> list[Any]:
    """Segments from either profile shape: the snapshot's ``active_segments`` or a
    versioned profile's active (else first) entry in ``profiles``."""
    seg = profile.get("active_segments")
    if seg:
        return list(seg)
    profiles = profile.get("profiles") or []
    active = next((p for p in profiles if isinstance(p, dict) and p.get("active")), None)
    chosen = active or (profiles[0] if profiles else None)
    return list(chosen.get("segments") or []) if isinstance(chosen, dict) else []


def _segments_overlapping(segments: list[Any], bucket: Bucket) -> list[dict[str, Any]]:
    """Segments whose start hour falls within the bucket, else the one in effect at
    bucket start. Basal schedules wrap midnight, so when nothing starts at or before
    the bucket start the segment in effect is the day's last one (carried over)."""
    seg_dicts = [s for s in segments if isinstance(s, dict)]
    inside = [s for s in seg_dicts if bucket.start_hour <= _seg_hour(s) < bucket.end_hour]
    if inside:
        return inside
    before = [s for s in seg_dicts if _seg_hour(s) <= bucket.start_hour]
    if before:
        return [max(before, key=_seg_hour)]
    return [max(seg_dicts, key=_seg_hour)] if seg_dicts else []


def _nearest_value(
    ts_list: Sequence[datetime], val_list: Sequence[float], target: datetime
) -> float | None:
    """The CGM value nearest ``target`` within ``_ALIGN_TOL`` (``None`` if none).

    ``ts_list`` is ascending; real CGM is not on the algorithm's cycle grid, so an
    exact-timestamp match would drop most cycles."""
    if not ts_list:
        return None
    i = bisect.bisect_left(ts_list, target)
    best_val: float | None = None
    best_gap = _ALIGN_TOL
    for j in (i - 1, i):
        if 0 <= j < len(ts_list):
            gap = abs(ts_list[j] - target)
            if gap <= best_gap:
                best_gap = gap
                best_val = val_list[j]
    return best_val


def _seg_hour(segment: dict[str, Any]) -> int:
    head = str(segment.get("start", "0")).split(":", 1)[0]
    return int(head) if head.isdigit() else 0


def _bucket_bolus_delays(
    boluses: dict[str, Any], bucket: Bucket, toolkit: DiscoveryToolkit
) -> list[float]:
    out: list[float] = []
    for row in boluses.get("boluses") or []:
        delay = row.get("minutes_after_carb_entry")
        if not isinstance(delay, (int, float)):
            continue
        try:
            ts = datetime.fromisoformat(str(row.get("ts"))).astimezone(toolkit.tzinfo)
        except (TypeError, ValueError):
            continue
        if bucket.start_hour <= ts.hour < bucket.end_hour:
            out.append(float(delay))
    return out


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def _collect(pool: dict[str, Any], name: str, idx: int, result: dict[str, Any]) -> None:
    """Flatten a tool result's numbers (one list level deep) into the guard pool."""
    numbers: dict[str, Any] = {k: v for k, v in result.items() if isinstance(v, (int, float))}
    for k, v in result.items():
        if isinstance(v, list):
            for i, row in enumerate(v[:50]):
                if isinstance(row, dict):
                    nums = {kk: vv for kk, vv in row.items() if isinstance(vv, (int, float))}
                    if nums:
                        numbers[f"{k}_{i}"] = nums
    if numbers:
        pool[f"{name}_{idx}"] = numbers
