"""Deterministic tool belt the reasoning agents call.

Every tool is a pure read over the store. Results carry the exact numbers any
prose will be audited against, plus the raw day-level groups so the rigor
layer and skeptic can re-test claims independently.

Context policy: data enters the model's context only when a tool is called —
nothing here is injected eagerly. Every list-returning tool is bounded (see
``_MAX_LISTED_EVENTS`` / ``_MAX_RECALL_ITEMS``) with a truncation note, so a
single call can never blow the context budget. Descriptions are kept terse on
purpose: they ship in every prompt.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dexta_intelligence.agents.reason import ToolSpec
from dexta_intelligence.agents.time_tools import time_tool_specs
from dexta_intelligence.analytics.oref import carbs_on_board, insulin_totals
from dexta_intelligence.coldstart import CapabilitySet
from dexta_intelligence.connectors.tandem import PROFILE_SOURCE_ID
from dexta_intelligence.models import FindingStatus, HypothesisStatus, InsulinKind
from dexta_intelligence.stats.core import (
    cliffs_delta,
    cohen_d,
    mann_whitney_u,
    mean,
    pearson_r,
    spearman_rho,
    stdev,
    student_t_two_sided_p,
    summarize,
    welch_t_test,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dexta_intelligence.agents.base import AgentContext
    from dexta_intelligence.models import InsulinEvent, MealEvent

__all__ = [
    "TOOL_SCHEMA_FOR_LLM",
    "DiscoveryToolkit",
    "ToolResult",
    "evidence_backend",
    "tool_specs",
]

#: Per-day minimum readings before a daily aggregate is trusted.
_MIN_READINGS_PER_DAY = 12
#: Minimum readings on each side of an event for a pre/post pair.
_MIN_READINGS_PER_SIDE = 2
_WEEKEND = (5, 6)
#: Glucose Management Indicator (Bergenstal 2018): GMI% = 3.31 + 0.02392·mean(mg/dL).
_GMI_INTERCEPT = 3.31
_GMI_SLOPE = 0.02392


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolation quantile of an already-sorted, non-empty list (q in 0-100)."""
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (q / 100.0) * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] * (1.0 - (pos - lo)) + sorted_vals[hi] * (pos - lo)

#: Default spike threshold (mg/dL) for find_spikes / find_similar_events.
_SPIKE_THRESHOLD = 200.0
#: Max items returned in any treatment-tool event list (context budget).
_MAX_LISTED_EVENTS = 40
#: Max findings / connections / open-questions recall returns (context budget).
_MAX_RECALL_ITEMS = 8
#: Analysis-only oref profile defaults — mirrors the reconciliation agent.
#: Never used for dosing; tier-B labeling on every result that uses them.
_ANALYSIS_ISF = 50.0
_ANALYSIS_CARB_RATIO = 10.0

#: tool name → data stream it needs; tools whose stream is absent are HIDDEN
#: from the belt (capability filtering), not left to error at call time.
_TOOL_NEEDS: dict[str, str] = {
    "get_boluses": "insulin",
    "get_basal_timeline": "insulin",
    "get_iob": "insulin",
    "get_insulin_profile": "insulin",
    "correction_outcome": "insulin",
    "get_carb_entries": "meals",
    "get_cob": "meals",
    "meal_response": "meals",
}

TOOL_SCHEMA_FOR_LLM = """AVAILABLE TOOLS (call by exact name; args must match):

1. groupby_compare(group_by, target)
   group_by: "weekend" (weekend vs weekday days) | "sleep_bucket" (poorer-sleep
             vs better-sleep days, split at the median sleep score)
             | "workout_day" (days with any logged activity vs days without)
   target:   "mean_glucose" | "tir_pct"
   Compares the daily target value between the two groups of days.

2. tod_compare(hours_a, hours_b)
   hours_a/hours_b: [start_hour, end_hour) pairs, 0-24, e.g. [3, 7] vs [11, 15].
   Compares per-day mean glucose inside two time-of-day windows.

3. event_proximity(event_type, window_min)
   event_type: "meal" | "workout" | "bolus"
   window_min: minutes after the event to average (30-240).
   Compares glucose after each event vs the 60 minutes before it.

4. basal_overnight(hours)
   hours: [start_hour, end_hour) overnight window, default [0, 6].
   Per-night glucose drift (window-end minus window-start), first-half vs
   second-half nights. Nights with temp-basal/suspend are excluded when
   insulin is logged. Also returns n_nights.

5. meal_response(window_min)
   window_min: minutes after each meal to track (30-240, default 120).
   Per-meal excursion (post-meal peak minus pre-meal baseline) for bigger-carb
   vs smaller-carb meals (split at median logged carbs; meals without carbs
   excluded). Also returns mean_excursion_a/_b and n_meals.

6. correction_outcome(window_min)
   window_min: minutes after each correction bolus to track (30-240, default 180).
   Per-bolus glucose delta (window-end minus bolus-time baseline), newer vs
   older boluses. Also returns rebound_low_rate (% of boluses with a reading
   < 70 mg/dL inside the window) and n_boluses.

Every tool returns: ok, n_a, n_b, mean_a, mean_b, delta (mean_a - mean_b),
cohen_d, interpretation ("negligible"|"small"|"moderate"|"large"), and the
raw per-day/per-event groups used.

TIME-TRAVERSAL TOOLS (re-scope the window the analysis tools above operate on):

7. list_segments()
   Coarse structure of the whole record so you can decide where to drill: one
   row per month (or per week if the span is under 60 days), each with period,
   n_days, mean_glucose, tir_pct, n_lows. Cheap and deterministic. Call this
   FIRST to orient before narrowing.

8. set_window(start, end)
   start/end: ISO dates ("2026-03-01"). Narrows the ACTIVE window every later
   tool (tod_compare, groupby_compare, event_proximity, daily_series, ...) reads
   from. Clamps to available data and returns {active_start, active_end, n_days,
   n_readings} so you see exactly what you selected. Out-of-range dates clamp.

9. zoom_event(timestamp, pad_hours=12)
   timestamp: ISO datetime of a spike/event. Sets the active window tight around
   it (+/- pad_hours) and returns the minute-level trace: {readings:[{ts,mg_dl}],
   pre_mean, post_mean, peak, nadir}. The spike-drill primitive.

10. daily_series(metric)
    metric: "tir" | "mean_glucose" | "tbr" | "cv". Returns the per-day time
    series over the ACTIVE window as [{date, value}] so you can spot trends and
    change-points yourself before comparing groups.

TREATMENT TOOLS (insulin/carb context — exposed only when the data exists;
REQUIRED before claiming a likely cause for a spike/high/meal/correction):

11. get_carb_entries()
    Carb entries inside the ACTIVE window: [{ts, carbs_g, ...}], n_entries,
    total_carbs_g. An empty result around a spike is itself a signal
    (possible missing carb entry).

12. get_boluses()
    Boluses inside the ACTIVE window: [{ts, units, minutes_after_carb_entry}],
    n_boluses, total_units. minutes_after_carb_entry is the late-bolus signal.

13. get_basal_timeline()
    Basal / temp-basal / suspend events inside the ACTIVE window plus
    basal_stable (no temp-basal/suspend interruptions) — rules basal in or out.

14. get_iob(timestamp) / 15. get_cob(timestamp)
    Insulin-on-board / carbs-on-board at an ISO datetime (computed, oref0
    curves, tier B — analysis context only, never dosing).

16. get_insulin_profile()
    Pump-reported basal/ISF/carb-ratio/target segments for the active profile
    (and all stored profiles). Synced from Tandem on pull; tier B — analysis
    context only, never dosing.

17. find_spikes(threshold=200, top_n=10)
    Excursion peaks inside the ACTIVE window: [{ts, peak_mg_dl, duration_min}],
    largest first. Locates the spike when the user names a day but not a time.

18. find_similar_events(timestamp, threshold=200)
    Recurrence over the WHOLE record: same-time-of-day events (carb entries
    when logged) with per-event peak, spiked flag, and bolus_delay_min →
    n_similar, n_spiking, mean bolus delays spiking vs not.

CALENDAR TOOLS (always available — never compute dates in your head):

19. get_current_time(timezone)   what "now"/"today" is, with weekday
20. get_weekday(date)            weekday for any ISO date
21. parse_relative_date(expression, timezone)
    "last Tuesday" / "yesterday" / "3 days ago" → concrete ISO date for
    set_window / zoom_event arguments.

WORKFLOW: orient with list_segments, narrow with set_window, drill a spike with
zoom_event, read trends with daily_series, THEN compare with tod_compare /
groupby_compare / event_proximity. The analysis tools always honor the active
window; call set_window again (or with the full span) to widen back out.

SPIKE/CAUSE WORKFLOW: resolve dates (calendar tools) → list_segments →
set_window → find_spikes / zoom_event → get_carb_entries → get_boluses +
get_iob → get_basal_timeline → find_similar_events → only THEN state the most
consistent contributor. NEVER claim a likely cause from glucose shape alone
while treatment tools are available; if they are absent, say explicitly:
"Insulin/carb data unavailable. This is glucose-shape inference only."
Ground a confirmed pattern with search_evidence AFTER the data work, never
instead of it. Observation and discussion only — never dosing advice."""


@dataclass(frozen=True, slots=True)
class ToolResult:
    """One tool invocation's outcome — the evidence pool for any prose about it."""

    ok: bool
    tool: str
    args: dict[str, Any]
    summary: dict[str, Any]
    group_a: tuple[float, ...] = ()
    group_b: tuple[float, ...] = ()
    error: str | None = None

    def evidence(self) -> dict[str, Any]:
        """The numbers prose may cite (guard pool) plus skeptic re-check groups."""
        out: dict[str, Any] = {"tool": self.tool, "tool_args": dict(self.args)}
        out.update(self.summary)
        out["skeptic_group_a"] = list(self.group_a)
        out["skeptic_group_b"] = list(self.group_b)
        return out


class DiscoveryToolkit:
    """Window-scoped daily frames + the three two-group instruments."""

    def __init__(
        self,
        ctx: AgentContext,
        *,
        target_low: int = 70,
        target_high: int = 180,
    ) -> None:
        #: Patient-local zone for all date/time-of-day bucketing. Storage stays
        #: UTC; "dinner", "overnight", and per-day grouping are computed in this
        #: zone so they land at the patient's clock time, not UTC's.
        self._tz = _resolve_tz(getattr(ctx, "timezone", "UTC"))
        start = datetime.combine(ctx.window[0], time.min, tzinfo=self._tz).astimezone(UTC)
        end = datetime.combine(ctx.window[1], time.max, tzinfo=self._tz).astimezone(UTC)
        self._target = (target_low, target_high)
        #: Full ctx.window bounds — the active sub-window can never exceed these.
        self._full_start = start
        self._full_end = end
        #: Active sub-window (defaults to the full window: today's behavior).
        self._active_start = start
        self._active_end = end
        glucose = sorted(ctx.store.get_glucose(start, end), key=lambda g: g.ts)
        self._glucose_ts = [g.ts for g in glucose]
        self._glucose_vals = [float(g.mg_dl) for g in glucose]
        #: Full-window daily frame, built once; the active view filters it by date.
        self._daily_full = self._build_daily(glucose)
        self._sleep_score: dict[date, float] = {}
        for s in ctx.store.get_sleep(start, end):
            if s.score is not None:
                self._sleep_score[self._ld(s.ts_end)] = float(s.score)
        self._activity_ts = [a.ts for a in ctx.store.get_activity(start, end)]
        self._meals: list[MealEvent] = sorted(ctx.store.get_meals(start, end), key=lambda m: m.ts)
        self._meal_ts = [m.ts for m in self._meals]
        self._insulin: list[InsulinEvent] = sorted(
            ctx.store.get_insulin(start, end), key=lambda i: i.ts
        )
        self._bolus_ts = [i.ts for i in self._insulin if i.kind is InsulinKind.BOLUS]
        #: Nights touched by temp-basal / suspend events (excluded from drift when insulin exists).
        self._has_insulin = bool(self._insulin)
        self._last_insulin_ts = self._insulin[-1].ts if self._insulin else None
        self._last_meal_ts = self._meals[-1].ts if self._meals else None
        self._last_glucose_ts = self._glucose_ts[-1] if self._glucose_ts else None
        self._basal_intervention_dates: set[date] = {
            self._ld(i.ts)
            for i in self._insulin
            if i.kind in (InsulinKind.TEMP_BASAL, InsulinKind.SUSPEND)
        }
        try:
            self._n_predictions = len(ctx.store.get_predictions(start, end))
        except (AttributeError, NotImplementedError):  # minimal/partial stores
            self._n_predictions = 0
        self._insulin_profile: dict[str, Any] | None = None
        try:
            profile_raw = ctx.store.get_raw_event("tandem", PROFILE_SOURCE_ID)
        except (AttributeError, NotImplementedError):
            profile_raw = None
        if profile_raw is not None:
            self._insulin_profile = dict(profile_raw.payload)

    # ── local-time helpers (bucket in the patient's zone, slice the UTC arrays) ──

    def _ld(self, ts: datetime) -> date:
        """Calendar date of ``ts`` in the patient's local zone."""
        return ts.astimezone(self._tz).date()

    def _lh(self, ts: datetime) -> int:
        """Hour-of-day of ``ts`` in the patient's local zone (0-23)."""
        return ts.astimezone(self._tz).hour

    def _day_bounds(self, day: date) -> tuple[datetime, datetime]:
        """UTC instants bracketing one local calendar ``day`` (for bisecting)."""
        start = datetime.combine(day, time.min, tzinfo=self._tz).astimezone(UTC)
        end = datetime.combine(day, time.max, tzinfo=self._tz).astimezone(UTC)
        return start, end

    @property
    def tzinfo(self) -> ZoneInfo:
        """The patient-local zone used for all bucketing (callers localize for display)."""
        return self._tz

    def capabilities(self) -> CapabilitySet:
        """Which streams exist in this window — drives tool exposure."""
        return CapabilitySet(
            has_insulin=self._has_insulin,
            has_meals=bool(self._meals),
            has_sleep=bool(self._sleep_score),
            has_activity=bool(self._activity_ts),
            has_predictions=self._n_predictions > 0,
        )

    # ── active sub-window ────────────────────────────────────────────────────

    def set_active_window(self, start: datetime, end: datetime) -> tuple[datetime, datetime]:
        """Re-scope every analysis tool to ``[start, end]``, clamped to the full
        window. No re-query: the full ctx.window data stays loaded; sub-windowing
        is pure date filtering / bisect over the already-sorted arrays. Returns
        the clamped (start, end) actually applied."""
        lo = max(start, self._full_start)
        hi = min(end, self._full_end)
        if lo > hi:  # degenerate request — fall back to the full window
            lo, hi = self._full_start, self._full_end
        self._active_start = lo
        self._active_end = hi
        return (lo, hi)

    @property
    def _daily(self) -> dict[date, tuple[float, float]]:
        """The full-window daily frame filtered to the active sub-window (local dates)."""
        lo, hi = self._ld(self._active_start), self._ld(self._active_end)
        if lo == self._ld(self._full_start) and hi == self._ld(self._full_end):
            return self._daily_full
        return {d: v for d, v in self._daily_full.items() if lo <= d <= hi}

    def _active_glucose(self) -> tuple[list[datetime], list[float]]:
        """(timestamps, values) sliced to the active sub-window via bisect."""
        lo = bisect.bisect_left(self._glucose_ts, self._active_start)
        hi = bisect.bisect_right(self._glucose_ts, self._active_end)
        return self._glucose_ts[lo:hi], self._glucose_vals[lo:hi]

    def _active_event_ts(self, ts_list: list[datetime]) -> list[datetime]:
        """Event timestamps falling inside the active sub-window."""
        lo = bisect.bisect_left(ts_list, self._active_start)
        hi = bisect.bisect_right(ts_list, self._active_end)
        return ts_list[lo:hi]

    # ── context for the planner ──────────────────────────────────────────────

    def _insulin_in_active_window(self) -> bool:
        return any(self._active_start <= i.ts <= self._active_end for i in self._insulin)

    def _treatment_gap_note(self) -> str | None:
        """Explain when glucose spans the active window but pump data does not."""
        if not self._has_insulin or self._last_insulin_ts is None:
            return None
        if self._insulin_in_active_window():
            return None
        if self._last_insulin_ts >= self._active_start:
            return None
        local = self._last_insulin_ts.astimezone(self._tz).strftime("%b %d %H:%M")
        return (
            f"pump/insulin data in dexta ends {local} — before this window. "
            "CGM may be newer than Tandem uploads; open t:connect/Tandem Source on your "
            "phone to upload recent pump history, then Sync now."
        )

    def data_summary(self) -> str:
        """One block the planner reads before proposing hypotheses."""
        days = sorted(self._daily)
        span = f"{days[0].isoformat()} → {days[-1].isoformat()}" if days else "no data"
        lines = [
            f"- glucose days with enough readings: {len(self._daily)} ({span})",
            f"- sleep-scored days: {len(self._sleep_score)}",
            (
                f"- activity events: {len(self._activity_ts)}"
                f" · meals logged: {len(self._meal_ts)}"
                f" · boluses logged: {len(self._bolus_ts)}"
            ),
        ]
        if self._last_glucose_ts and self._last_insulin_ts:
            g = self._last_glucose_ts.astimezone(self._tz).strftime("%b %d %H:%M")
            ins = self._last_insulin_ts.astimezone(self._tz).strftime("%b %d %H:%M")
            lines.append(f"- latest glucose: {g} · latest pump/insulin: {ins}")
        gap = self._treatment_gap_note()
        if gap:
            lines.append(f"- treatment gap: {gap}")
        return "\n".join(lines)

    # ── dispatch ─────────────────────────────────────────────────────────────

    def run(self, tool: str, args: dict[str, Any]) -> ToolResult:
        """Validate and execute one tool call; never raises on bad LLM args."""
        dispatch: dict[str, Any] = {
            "groupby_compare": lambda: self._groupby_compare(
                str(args["group_by"]), str(args["target"])
            ),
            "tod_compare": lambda: self._tod_compare(
                _hours(args["hours_a"]), _hours(args["hours_b"])
            ),
            "event_proximity": lambda: self._event_proximity(
                str(args["event_type"]), int(args.get("window_min", 90))
            ),
            "basal_overnight": lambda: self._basal_overnight(_hours(args.get("hours", [0, 6]))),
            "meal_response": lambda: self._meal_response(int(args.get("window_min", 120))),
            "correction_outcome": lambda: self._correction_outcome(
                int(args.get("window_min", 180))
            ),
        }
        fn = dispatch.get(tool)
        if fn is None:
            return _error(tool, args, f"unknown tool {tool!r}")
        try:
            return fn()  # type: ignore[no-any-return]
        except (KeyError, TypeError, ValueError) as exc:
            return _error(tool, args, f"bad args: {exc}")

    # ── instruments ──────────────────────────────────────────────────────────

    def _groupby_compare(self, group_by: str, target: str) -> ToolResult:
        args = {"group_by": group_by, "target": target}
        if target not in ("mean_glucose", "tir_pct"):
            return _error("groupby_compare", args, f"unknown target {target!r}")
        series = {d: v[0] if target == "mean_glucose" else v[1] for d, v in self._daily.items()}

        if group_by == "weekend":
            in_a = {d for d in series if d.weekday() in _WEEKEND}
            labels = ("weekend", "weekday")
        elif group_by == "sleep_bucket":
            scored = {d: s for d, s in self._sleep_score.items() if d in series}
            if len(scored) < 4:
                return _error("groupby_compare", args, "fewer than 4 sleep-scored days")
            cutoff = sorted(scored.values())[len(scored) // 2]
            in_a = {d for d, s in scored.items() if s < cutoff}
            series = {d: v for d, v in series.items() if d in scored}
            labels = ("poorer_sleep", "better_sleep")
        elif group_by == "workout_day":
            workout_days = {self._ld(ts) for ts in self._activity_ts}
            in_a = {d for d in series if d in workout_days}
            labels = ("workout_day", "rest_day")
        else:
            return _error("groupby_compare", args, f"unknown group_by {group_by!r}")

        ordered = sorted(series)
        group_a = tuple(series[d] for d in ordered if d in in_a)
        group_b = tuple(series[d] for d in ordered if d not in in_a)
        return _two_group("groupby_compare", args, group_a, group_b, labels)

    def _active_day_count(self) -> int:
        lo, hi = self._ld(self._active_start), self._ld(self._active_end)
        return (hi - lo).days + 1

    def _tod_compare(self, hours_a: tuple[int, int], hours_b: tuple[int, int]) -> ToolResult:
        args = {"hours_a": list(hours_a), "hours_b": list(hours_b)}
        if self._active_day_count() < 2:
            return _error(
                "tod_compare",
                args,
                "need at least 2 days in the active window — widen with set_window first",
            )
        group_a = self._daily_window_means(hours_a)
        group_b = self._daily_window_means(hours_b)
        labels = (f"{hours_a[0]:02d}-{hours_a[1]:02d}h", f"{hours_b[0]:02d}-{hours_b[1]:02d}h")
        return _two_group("tod_compare", args, group_a, group_b, labels)

    def _event_proximity(self, event_type: str, window_min: int) -> ToolResult:
        args = {"event_type": event_type, "window_min": window_min}
        if not 30 <= window_min <= 240:
            return _error("event_proximity", args, "window_min must be 30-240")
        events = {"meal": self._meal_ts, "workout": self._activity_ts, "bolus": self._bolus_ts}
        if event_type not in events:
            return _error("event_proximity", args, f"unknown event_type {event_type!r}")
        post: list[float] = []
        pre: list[float] = []
        for ts in self._active_event_ts(events[event_type]):
            before = self._readings_between(ts - timedelta(minutes=60), ts)
            after = self._readings_between(ts, ts + timedelta(minutes=window_min))
            if len(before) >= _MIN_READINGS_PER_SIDE and len(after) >= _MIN_READINGS_PER_SIDE:
                pre.append(mean(before))
                post.append(mean(after))
        result = _two_group(
            "event_proximity", args, tuple(post), tuple(pre), ("post_event", "pre_event")
        )
        if result.ok:
            result.summary["n_events"] = len(post)
        return result

    # ── time-traversal instruments ───────────────────────────────────────────

    def set_window(self, start_iso: str, end_iso: str) -> dict[str, Any]:
        """Re-scope the active window to ISO dates; clamp to available data.

        Returns what was actually selected so the model sees the clamp. Never
        raises: bad ISO strings come back as an ``error`` dict. Dates are the
        patient's local calendar days."""
        try:
            d0, d1 = date.fromisoformat(start_iso), date.fromisoformat(end_iso)
        except (TypeError, ValueError) as exc:
            return {"error": f"bad date: {exc}"}
        if d0 > d1:
            d0, d1 = d1, d0
        start, _ = self._day_bounds(d0)
        _, end = self._day_bounds(d1)
        lo, hi = self.set_active_window(start, end)
        ts_list, _ = self._active_glucose()
        note = None
        if lo > start or hi < end:
            note = "clamped to available data"
        out: dict[str, Any] = {
            "active_start": self._ld(lo).isoformat(),
            "active_end": self._ld(hi).isoformat(),
            "n_days": (self._ld(hi) - self._ld(lo)).days + 1,
            "n_readings": len(ts_list),
        }
        if note:
            out["note"] = note
        return out

    def list_segments(self) -> dict[str, Any]:
        """Coarse per-month (or per-week if span < 60d) structure of the whole
        record so the model can decide where to drill. Deterministic and cheap;
        ignores the active window (always describes the full ctx.window)."""
        days = sorted(self._daily_full)
        if not days:
            return {"segments": [], "note": "no days with enough readings"}
        by_week = (days[-1] - days[0]).days < 60
        low_threshold = self._target[0]
        buckets: dict[str, list[date]] = {}
        for d in days:
            key = _week_key(d) if by_week else f"{d.year:04d}-{d.month:02d}"
            buckets.setdefault(key, []).append(d)
        segments: list[dict[str, Any]] = []
        for period in sorted(buckets):
            bucket_days = buckets[period]
            means = [self._daily_full[d][0] for d in bucket_days]
            tirs = [self._daily_full[d][1] for d in bucket_days]
            n_lows = sum(
                1 for d in bucket_days for v in self._day_values(d) if v < low_threshold
            )
            segments.append(
                {
                    "period": period,
                    "n_days": len(bucket_days),
                    "mean_glucose": round(mean(means), 1),
                    "tir_pct": round(mean(tirs), 1),
                    "n_lows": n_lows,
                }
            )
        return {"granularity": "week" if by_week else "month", "segments": segments}

    def zoom_event(self, timestamp_iso: str, pad_hours: int = 12) -> dict[str, Any]:
        """Set the active window tight around ``timestamp`` (+/- pad_hours) and
        return the minute-level glucose trace there — the spike-drill primitive.
        Never raises on bad args."""
        try:
            center = datetime.fromisoformat(timestamp_iso)
        except (TypeError, ValueError) as exc:
            return {"error": f"bad timestamp: {exc}"}
        if center.tzinfo is None:
            center = center.replace(tzinfo=UTC)
        pad = max(1, min(int(pad_hours), 72))
        self.set_active_window(center - timedelta(hours=pad), center + timedelta(hours=pad))
        ts_list, vals_list = self._active_glucose()
        if not ts_list:
            return {"error": "no readings in that window", "readings": []}
        readings = [
            {"ts": ts.isoformat(), "mg_dl": round(v, 1)}
            for ts, v in zip(ts_list, vals_list, strict=True)
        ]
        pre = [v for ts, v in zip(ts_list, vals_list, strict=True) if ts < center]
        post = [v for ts, v in zip(ts_list, vals_list, strict=True) if ts >= center]
        return {
            "center": center.isoformat(),
            "pad_hours": pad,
            "n_readings": len(readings),
            "readings": readings,
            "pre_mean": round(mean(pre), 1) if pre else None,
            "post_mean": round(mean(post), 1) if post else None,
            "peak": round(max(vals_list), 1),
            "nadir": round(min(vals_list), 1),
        }

    def daily_series(self, metric: str) -> dict[str, Any]:
        """Per-day time series of ``metric`` over the ACTIVE window as
        [{date, value}] so the model can spot trends/change-points. metric in
        tir | mean_glucose | tbr | cv. Never raises on bad args."""
        if metric not in ("tir", "mean_glucose", "tbr", "cv"):
            return {"error": f"unknown metric {metric!r}"}
        by_day: dict[date, list[float]] = {}
        ts_list, vals_list = self._active_glucose()
        for ts, val in zip(ts_list, vals_list, strict=True):
            by_day.setdefault(self._ld(ts), []).append(val)
        series: list[dict[str, Any]] = []
        values: list[float] = []
        for d in sorted(by_day):
            vals = by_day[d]
            if len(vals) < _MIN_READINGS_PER_DAY:
                continue
            value = self._daily_metric(metric, vals)
            series.append({"date": d.isoformat(), "value": round(value, 2)})
            values.append(value)
        return {
            "metric": metric,
            "n_days": len(series),
            "series": series,
            "mean_value": round(mean(values), 2) if values else None,
        }

    def _daily_metric(self, metric: str, vals: list[float]) -> float:
        lo, hi = self._target
        if metric == "mean_glucose":
            return mean(vals)
        if metric == "tir":
            return 100.0 * sum(1 for v in vals if lo <= v <= hi) / len(vals)
        if metric == "tbr":
            return 100.0 * sum(1 for v in vals if v < lo) / len(vals)
        m = mean(vals)
        return 100.0 * stdev(vals) / m if m else 0.0

    def _day_values(self, day: date) -> list[float]:
        """All readings for one local calendar day (over the full loaded series)."""
        start, end = self._day_bounds(day)
        lo = bisect.bisect_left(self._glucose_ts, start)
        hi = bisect.bisect_right(self._glucose_ts, end)
        return self._glucose_vals[lo:hi]

    def glucose_stats(
        self, *, day: str = "", hours: tuple[int, int] | None = None
    ) -> dict[str, Any]:
        """Descriptive stats over a window — the answer to "what was my mean / SD /
        variance / CV / TIR for <period>". Defaults to the ACTIVE window; ``day``
        (ISO date) scopes to one local calendar day; ``hours`` = [start, end) filters
        to a local time-of-day band. All bucketing is patient-local; never raises."""
        if day:
            try:
                d = date.fromisoformat(day)
            except ValueError:
                return {"error": f"bad day {day!r}; expected an ISO date like 2026-06-15"}
            start, end = self._day_bounds(d)
            scope: dict[str, Any] = {"day": d.isoformat()}
        else:
            start, end = self._active_start, self._active_end
            scope = {
                "start": start.astimezone(self._tz).date().isoformat(),
                "end": end.astimezone(self._tz).date().isoformat(),
            }
        lo = bisect.bisect_left(self._glucose_ts, start)
        hi = bisect.bisect_right(self._glucose_ts, end)
        ts_list = self._glucose_ts[lo:hi]
        vals = self._glucose_vals[lo:hi]
        if hours is not None:
            h0, h1 = hours
            vals = [v for ts, v in zip(ts_list, vals, strict=True) if h0 <= self._lh(ts) < h1]
            scope["hours"] = [h0, h1]

        n = len(vals)
        if n == 0:
            return {**scope, "n": 0, "note": "no readings in this window"}

        s = summarize(vals)
        ordered = sorted(vals)
        lo_t, hi_t = self._target
        out: dict[str, Any] = {
            **scope,
            "n": n,
            "mean": round(s.mean, 1) if s.mean is not None else None,
            "sd": round(s.sd, 1) if s.sd is not None else None,
            "variance": round(s.sd**2, 1) if s.sd is not None else None,
            "cv_pct": round(s.cv_pct, 1) if s.cv_pct is not None else None,
            "median": round(s.median, 1) if s.median is not None else None,
            "p10": round(_percentile(ordered, 10), 1),
            "p25": round(_percentile(ordered, 25), 1),
            "p75": round(_percentile(ordered, 75), 1),
            "p90": round(_percentile(ordered, 90), 1),
            "minimum": round(s.minimum, 1) if s.minimum is not None else None,
            "maximum": round(s.maximum, 1) if s.maximum is not None else None,
            "tir_pct": round(100.0 * sum(1 for v in vals if lo_t <= v <= hi_t) / n, 1),
            "tbr_pct": round(100.0 * sum(1 for v in vals if v < lo_t) / n, 1),
            "tar_pct": round(100.0 * sum(1 for v in vals if v > hi_t) / n, 1),
            "tbr54_pct": round(100.0 * sum(1 for v in vals if v < 54) / n, 1),
            "tar250_pct": round(100.0 * sum(1 for v in vals if v > 250) / n, 1),
            "target_range": [lo_t, hi_t],
        }
        if s.mean is not None:
            out["gmi_pct"] = round(_GMI_INTERCEPT + _GMI_SLOPE * s.mean, 1)
        return out

    # ── correlation ──────────────────────────────────────────────────────────

    _CORRELATE_METRICS = ("mean_glucose", "tir", "tbr", "cv", "sleep_score")

    def _per_day_metric(self, key: str) -> dict[date, float]:
        """Per-day series of ``key`` over the ACTIVE window, keyed by local date."""
        if key == "sleep_score":
            lo, hi = self._ld(self._active_start), self._ld(self._active_end)
            return {d: v for d, v in self._sleep_score.items() if lo <= d <= hi}
        by_day: dict[date, list[float]] = {}
        ts_list, vals_list = self._active_glucose()
        for ts, val in zip(ts_list, vals_list, strict=True):
            by_day.setdefault(self._ld(ts), []).append(val)
        out: dict[date, float] = {}
        for d, vals in by_day.items():
            if len(vals) < _MIN_READINGS_PER_DAY:
                continue
            out[d] = self._daily_metric(key, vals)
        return out

    def correlate(self, x: str, y: str) -> dict[str, Any]:
        """Correlate two per-day metrics across the ACTIVE window (inner-join on
        local date). Supported keys: mean_glucose|tir|tbr|cv|sleep_score. Requires
        n >= 4 paired days; never raises."""
        for key in (x, y):
            if key not in self._CORRELATE_METRICS:
                return {"x": x, "y": y, "error": f"unknown metric {key!r}"}
        sx, sy = self._per_day_metric(x), self._per_day_metric(y)
        days = sorted(set(sx) & set(sy))
        n = len(days)
        if n < 4:
            return {
                "x": x,
                "y": y,
                "n": n,
                "note": "fewer than 4 overlapping days — not enough to correlate",
            }
        xs = [sx[d] for d in days]
        ys = [sy[d] for d in days]
        r = pearson_r(xs, ys)
        rho = spearman_rho(xs, ys)
        if r is None or abs(r) >= 1.0 or n < 3:
            p: float | None = 0.0 if r is not None and abs(r) >= 1.0 else None
        else:
            t = r * ((n - 2) / (1.0 - r * r)) ** 0.5
            p = student_t_two_sided_p(t, n - 2)
        if r is None:
            direction = "none"
        elif r > 0:
            direction = "positive"
        elif r < 0:
            direction = "negative"
        else:
            direction = "none"
        return {
            "x": x,
            "y": y,
            "n": n,
            "pearson_r": round(r, 3) if r is not None else None,
            "spearman_rho": round(rho, 3) if rho is not None else None,
            "p": round(p, 4) if p is not None else None,
            "direction": direction,
        }

    # ── insulin / meal instruments ───────────────────────────────────────────

    def _basal_overnight(self, hours: tuple[int, int]) -> ToolResult:
        """Per-night glucose drift (night-end mean minus night-start mean), first vs
        second half of the run. Nights touched by temp-basal/suspend are excluded
        when insulin data exists (overnight basal effect, not algorithm overrides)."""
        args = {"hours": list(hours)}
        if self._active_day_count() < 4:
            return _error(
                "basal_overnight",
                args,
                "need at least 4 nights in the active window — widen with set_window "
                "(e.g. 14+ days)",
            )
        drift: list[tuple[date, float]] = []
        for night, vals in sorted(self._overnight_segments(hours).items()):
            if self._has_insulin and night in self._basal_intervention_dates:
                continue
            if len(vals) < 4:
                continue
            edge = max(2, len(vals) // 4)
            drift.append((night, mean(vals[-edge:]) - mean(vals[:edge])))
        if len(drift) < 4:
            return _error("basal_overnight", args, "fewer than 4 clean nights")
        mid = len(drift) // 2
        group_a = tuple(d for _, d in drift[:mid])
        group_b = tuple(d for _, d in drift[mid:])
        result = _two_group(
            "basal_overnight", args, group_a, group_b, ("first_half", "second_half")
        )
        if result.ok:
            result.summary["n_nights"] = len(drift)
        return result

    def _meal_response(self, window_min: int) -> ToolResult:
        """Per-meal excursion (post-meal peak minus pre-meal baseline). Meals split at
        the median logged carbs; meals without carbs are excluded. Reports the
        mean excursion of bigger-carb vs smaller-carb meals."""
        args = {"window_min": window_min}
        if not 30 <= window_min <= 240:
            return _error("meal_response", args, "window_min must be 30-240")
        rows: list[tuple[float, float]] = []  # (carbs_g, excursion)
        for meal in self._meals:
            if meal.carbs_g is None:
                continue
            pre = self._readings_between(meal.ts - timedelta(minutes=30), meal.ts)
            post = self._readings_between(meal.ts, meal.ts + timedelta(minutes=window_min))
            if len(pre) >= _MIN_READINGS_PER_SIDE and len(post) >= _MIN_READINGS_PER_SIDE:
                rows.append((float(meal.carbs_g), max(post) - mean(pre)))
        if len(rows) < 4:
            return _error("meal_response", args, "fewer than 4 carb-logged meals")
        cutoff = sorted(c for c, _ in rows)[len(rows) // 2]
        group_a = tuple(e for c, e in rows if c >= cutoff)
        group_b = tuple(e for c, e in rows if c < cutoff)
        result = _two_group(
            "meal_response", args, group_a, group_b, ("bigger_carb", "smaller_carb")
        )
        if result.ok:
            result.summary["mean_excursion_a"] = round(mean(group_a), 1)
            result.summary["mean_excursion_b"] = round(mean(group_b), 1)
            result.summary["n_meals"] = len(rows)
        return result

    def _correction_outcome(self, window_min: int) -> ToolResult:
        """Per-bolus glucose delta (window-end mean minus bolus-time baseline), older
        vs newer boluses. Also reports ``rebound_low_rate`` -- the share of boluses
        followed by a reading < 70 mg/dL inside the window."""
        args = {"window_min": window_min}
        if not 30 <= window_min <= 240:
            return _error("correction_outcome", args, "window_min must be 30-240")
        rows: list[float] = []
        rebound = 0
        for ts in self._bolus_ts:
            pre = self._readings_between(ts - timedelta(minutes=30), ts)
            post = self._readings_between(ts, ts + timedelta(minutes=window_min))
            if len(pre) >= _MIN_READINGS_PER_SIDE and len(post) >= _MIN_READINGS_PER_SIDE:
                rows.append(mean(post) - mean(pre))
                if any(v < 70 for v in post):
                    rebound += 1
        if len(rows) < 4:
            return _error("correction_outcome", args, "fewer than 4 boluses with coverage")
        mid = len(rows) // 2
        group_a = tuple(rows[mid:])  # post = newer boluses
        group_b = tuple(rows[:mid])  # pre = older boluses
        result = _two_group(
            "correction_outcome", args, group_a, group_b, ("newer", "older")
        )
        if result.ok:
            result.summary["rebound_low_rate"] = round(100.0 * rebound / len(rows), 1)
            result.summary["n_boluses"] = len(rows)
        return result

    # ── treatment inspection — read the events, not just aggregates ──

    def get_carb_entries(self) -> dict[str, Any]:
        """Carb entries inside the ACTIVE window, listed individually so the
        model can see what was (or wasn't) logged around an event."""
        meals = [
            m
            for m in self._meals
            if self._active_start <= m.ts <= self._active_end and m.carbs_g is not None
        ]
        if not meals:
            out: dict[str, Any] = {
                "n_entries": 0,
                "entries": [],
            }
            gap = self._treatment_gap_note()
            if gap:
                out["note"] = gap
            else:
                out["note"] = (
                    "no carb entries in the active window — widen with set_window "
                    "or treat this as a possible missing-carb-entry pattern"
                )
            return out
        entries = [
            {
                "ts": m.ts.isoformat(),
                "carbs_g": round(float(m.carbs_g), 1),  # type: ignore[arg-type]
                **({"protein_g": round(float(m.protein_g), 1)} if m.protein_g else {}),
                **({"fat_g": round(float(m.fat_g), 1)} if m.fat_g else {}),
                **({"note": m.note} if m.note else {}),
            }
            for m in meals[:_MAX_LISTED_EVENTS]
        ]
        out = {
            "n_entries": len(meals),
            "total_carbs_g": round(sum(float(m.carbs_g or 0.0) for m in meals), 1),
            "entries": entries,
        }
        if len(meals) > _MAX_LISTED_EVENTS:
            out["note"] = f"showing first {_MAX_LISTED_EVENTS} of {len(meals)}"
        return out

    def get_boluses(self) -> dict[str, Any]:
        """Boluses inside the ACTIVE window with their timing vs the nearest
        carb entry — ``minutes_after_carb_entry`` is the late-bolus signal."""
        boluses = [
            i
            for i in self._insulin
            if i.kind is InsulinKind.BOLUS and self._active_start <= i.ts <= self._active_end
        ]
        if not boluses:
            out: dict[str, Any] = {
                "n_boluses": 0,
                "boluses": [],
            }
            gap = self._treatment_gap_note()
            if gap:
                out["note"] = gap
            else:
                out["note"] = (
                    "no boluses in the active window — widen with set_window; "
                    "if a meal is present this may be a missed-bolus pattern"
                )
            return out
        rows: list[dict[str, Any]] = []
        for b in boluses[:_MAX_LISTED_EVENTS]:
            row: dict[str, Any] = {"ts": b.ts.isoformat()}
            if b.units is not None:
                row["units"] = round(float(b.units), 2)
            if b.automatic is not None:
                row["automatic"] = b.automatic
            delay = self._minutes_after_nearest_meal(b.ts)
            if delay is not None:
                row["minutes_after_carb_entry"] = delay
            rows.append(row)
        out = {
            "n_boluses": len(boluses),
            "total_units": round(sum(float(b.units) for b in boluses if b.units is not None), 2),
            "boluses": rows,
        }
        if len(boluses) > _MAX_LISTED_EVENTS:
            out["note"] = f"showing first {_MAX_LISTED_EVENTS} of {len(boluses)}"
        return out

    def get_basal_timeline(self) -> dict[str, Any]:
        """Basal / temp-basal / suspend state inside the ACTIVE window, plus a
        ``basal_stable`` flag (no temp-basal or suspend interruptions)."""
        events = [
            i
            for i in self._insulin
            if i.kind is not InsulinKind.BOLUS and self._active_start <= i.ts <= self._active_end
        ]
        kinds = {k: sum(1 for e in events if e.kind is k) for k in InsulinKind}
        rows = [
            {
                "ts": e.ts.isoformat(),
                "kind": e.kind.value,
                **({"units": round(float(e.units), 2)} if e.units is not None else {}),
                **(
                    {"duration_min": round(float(e.duration_min))}
                    if e.duration_min is not None
                    else {}
                ),
            }
            for e in events[:_MAX_LISTED_EVENTS]
        ]
        n_temp = kinds.get(InsulinKind.TEMP_BASAL, 0)
        n_susp = kinds.get(InsulinKind.SUSPEND, 0)
        out: dict[str, Any] = {
            "n_basal": kinds.get(InsulinKind.BASAL, 0),
            "n_temp_basal": n_temp,
            "n_suspend": n_susp,
            "basal_stable": n_temp == 0 and n_susp == 0,
            "events": rows,
        }
        if not events:
            gap = self._treatment_gap_note()
            out["note"] = gap or (
                "no basal records in the active window — basal context unavailable here"
            )
        if len(events) > _MAX_LISTED_EVENTS:
            out["note"] = f"showing first {_MAX_LISTED_EVENTS} of {len(events)}"
        return out

    def get_insulin_profile(self) -> dict[str, Any]:
        """Pump-reported basal/ISF/carb-ratio/target segments (Tandem sync).

        Tier B — analysis context only, never dosing."""
        if self._insulin_profile is None:
            return {
                "error": "no insulin profile synced",
                "note": (
                    "Connect Tandem in Settings and Sync now — profile is captured "
                    "from pump settings on each sync"
                ),
            }
        active = next(
            (p for p in self._insulin_profile.get("profiles") or [] if p.get("active")),
            None,
        )
        out: dict[str, Any] = {
            **self._insulin_profile,
            "active_segments": (active or {}).get("segments") or [],
            "tier": "B",
            "method": "pump-reported settings from Tandem Source (sync snapshot)",
        }
        return out

    def get_iob(self, timestamp_iso: str) -> dict[str, Any]:
        """Insulin-on-board at ``timestamp`` from logged boluses (oref0
        exponential curve, rapid-acting defaults). Tier B — computed for
        analysis context, never dosing. Never raises on bad args."""
        at = _parse_ts(timestamp_iso)
        if at is None:
            return {"error": f"bad timestamp: {timestamp_iso!r}"}
        doses = [
            (i.ts, float(i.units))
            for i in self._insulin
            if i.kind is InsulinKind.BOLUS and i.units is not None
        ]
        totals = insulin_totals(doses, at)
        n_recent = sum(1 for ts, _ in doses if timedelta(0) <= at - ts <= timedelta(hours=6))
        out: dict[str, Any] = {
            "timestamp": at.isoformat(),
            "iob_units": round(totals.iob, 2),
            "n_recent_boluses": n_recent,
            "tier": "B",
            "method": "computed from logged boluses (oref0 exponential curve)",
        }
        if any(
            i.kind in (InsulinKind.TEMP_BASAL, InsulinKind.SUSPEND)
            and timedelta(0) <= at - i.ts <= timedelta(hours=6)
            for i in self._insulin
        ):
            out["note"] = "temp-basal/suspend activity nearby is not included in this IOB"
        return out

    def get_cob(self, timestamp_iso: str) -> dict[str, Any]:
        """Carbs-on-board at ``timestamp`` from announced carb entries (oref0
        deviation-based decay, analysis-profile defaults). Tier B — computed
        for analysis context, never dosing. Never raises on bad args."""
        at = _parse_ts(timestamp_iso)
        if at is None:
            return {"error": f"bad timestamp: {timestamp_iso!r}"}
        window = timedelta(hours=6)
        recent = [
            m
            for m in self._meals
            if m.carbs_g is not None and timedelta(0) <= at - m.ts <= window
        ]
        if not recent:
            return {
                "timestamp": at.isoformat(),
                "cob_g": 0.0,
                "n_carb_entries": 0,
                "tier": "B",
                "note": "no announced carbs in the prior 6h — unannounced carbs "
                "would not appear here",
            }
        glucose = list(
            zip(self._glucose_ts, self._glucose_vals, strict=True)
        )
        doses = [
            (i.ts, float(i.units))
            for i in self._insulin
            if i.kind is InsulinKind.BOLUS and i.units is not None
        ]
        cob = absorbed = 0.0
        for m in recent:
            result = carbs_on_board(
                float(m.carbs_g),  # type: ignore[arg-type]
                m.ts,
                glucose,
                doses,
                _ANALYSIS_ISF,
                _ANALYSIS_CARB_RATIO,
                at,
            )
            cob += result.cob_g
            absorbed += result.absorbed_g
        return {
            "timestamp": at.isoformat(),
            "cob_g": round(cob, 1),
            "absorbed_g": round(absorbed, 1),
            "n_carb_entries": len(recent),
            "tier": "B",
            "method": "computed from announced carbs (oref0 deviation decay, "
            "analysis-profile defaults)",
        }

    def find_spikes(
        self, threshold: float = _SPIKE_THRESHOLD, top_n: int = 10
    ) -> dict[str, Any]:
        """Excursion peaks above ``threshold`` inside the ACTIVE window —
        contiguous above-threshold runs, one peak each, largest first."""
        threshold = max(140.0, min(float(threshold), 400.0))
        top_n = max(1, min(int(top_n), 25))
        ts_list, vals_list = self._active_glucose()
        spikes: list[dict[str, Any]] = []
        run_start: datetime | None = None
        run_peak, run_peak_ts = 0.0, None
        prev_ts: datetime | None = None
        for ts, v in zip(ts_list, vals_list, strict=True):
            if v >= threshold:
                if run_start is None:
                    run_start = ts
                    run_peak, run_peak_ts = v, ts
                elif v > run_peak:
                    run_peak, run_peak_ts = v, ts
                prev_ts = ts
            elif run_start is not None:
                assert run_peak_ts is not None and prev_ts is not None
                spikes.append(
                    {
                        "ts": run_peak_ts.isoformat(),
                        "peak_mg_dl": round(run_peak, 1),
                        "duration_min": round((prev_ts - run_start).total_seconds() / 60),
                    }
                )
                run_start, run_peak_ts = None, None
        if run_start is not None and run_peak_ts is not None and prev_ts is not None:
            spikes.append(
                {
                    "ts": run_peak_ts.isoformat(),
                    "peak_mg_dl": round(run_peak, 1),
                    "duration_min": round((prev_ts - run_start).total_seconds() / 60),
                }
            )
        spikes.sort(key=lambda s: -float(s["peak_mg_dl"]))
        out: dict[str, Any] = {
            "threshold": threshold,
            "n_spikes": len(spikes),
            "spikes": spikes[:top_n],
        }
        if not spikes:
            out["note"] = f"no readings ≥ {threshold:.0f} in the active window"
        return out

    def find_similar_events(
        self, timestamp_iso: str, threshold: float = _SPIKE_THRESHOLD
    ) -> dict[str, Any]:
        """Recurrence check over the WHOLE record (ignores the active window,
        like list_segments): events at the same time of day as ``timestamp`` —
        carb entries when meals are logged, otherwise daily clock anchors —
        each with its post-event peak and bolus timing. The '14 of 18 similar
        dinners' instrument. Never raises on bad args."""
        anchor = _parse_ts(timestamp_iso)
        if anchor is None:
            return {"error": f"bad timestamp: {timestamp_iso!r}"}
        band_min = 90
        anchor_local = anchor.astimezone(self._tz)
        anchor_min = anchor_local.hour * 60 + anchor_local.minute
        if self._meals:
            anchors = [
                m.ts
                for m in self._meals
                if m.carbs_g is not None
                and abs(self._lh(m.ts) * 60 + m.ts.astimezone(self._tz).minute - anchor_min)
                <= band_min
            ]
            basis = "carb entries at the same local time of day"
        else:
            days = sorted({self._ld(ts) for ts in self._glucose_ts})
            clock = time(anchor_local.hour, anchor_local.minute)
            anchors = [
                datetime.combine(d, clock, tzinfo=self._tz).astimezone(UTC) for d in days
            ]
            basis = "same local clock time each day (no carb entries logged)"
        events: list[dict[str, Any]] = []
        delays_spiking: list[float] = []
        delays_other: list[float] = []
        peaks_spiking: list[float] = []
        for ts in anchors:
            post = self._full_readings_between(ts, ts + timedelta(hours=2))
            if len(post) < _MIN_READINGS_PER_SIDE:
                continue
            peak = max(post)
            spiked = peak >= threshold
            row: dict[str, Any] = {
                "ts": ts.isoformat(),
                "peak_mg_dl": round(peak, 1),
                "spiked": spiked,
            }
            delay = self._minutes_after_nearest_meal_bolus(ts)
            if delay is not None:
                row["bolus_delay_min"] = delay
                (delays_spiking if spiked else delays_other).append(float(delay))
            if spiked:
                peaks_spiking.append(peak)
            events.append(row)
        if not events:
            return {
                "n_similar": 0,
                "events": [],
                "note": "no comparable events found — not enough history at this "
                "time of day",
            }
        n_spiking = sum(1 for e in events if e["spiked"])
        out: dict[str, Any] = {
            "basis": basis,
            "n_similar": len(events),
            "n_spiking": n_spiking,
            "events": events[:25],
        }
        if peaks_spiking:
            out["mean_peak_spiking"] = round(mean(peaks_spiking), 1)
        if delays_spiking:
            out["mean_bolus_delay_spiking_min"] = round(mean(delays_spiking), 1)
        if delays_other:
            out["mean_bolus_delay_other_min"] = round(mean(delays_other), 1)
        if len(events) > 25:
            out["note"] = f"showing first 25 of {len(events)}"
        return out

    def _minutes_after_nearest_meal(self, bolus_ts: datetime) -> int | None:
        """Signed minutes from the nearest carb entry (±2h) to this bolus."""
        if not self._meal_ts:
            return None
        idx = bisect.bisect_left(self._meal_ts, bolus_ts)
        best: timedelta | None = None
        for j in (idx - 1, idx):
            if 0 <= j < len(self._meal_ts):
                delta = bolus_ts - self._meal_ts[j]
                if best is None or abs(delta) < abs(best):
                    best = delta
        if best is None or abs(best) > timedelta(hours=2):
            return None
        return round(best.total_seconds() / 60)

    def _minutes_after_nearest_meal_bolus(self, meal_ts: datetime) -> int | None:
        """Minutes from ``meal_ts`` to the nearest bolus in [-60m, +120m]."""
        if not self._bolus_ts:
            return None
        idx = bisect.bisect_left(self._bolus_ts, meal_ts)
        best: timedelta | None = None
        for j in (idx - 1, idx):
            if 0 <= j < len(self._bolus_ts):
                delta = self._bolus_ts[j] - meal_ts
                if best is None or abs(delta) < abs(best):
                    best = delta
        if best is None or not (-60 <= best.total_seconds() / 60 <= 120):
            return None
        return round(best.total_seconds() / 60)

    def _full_readings_between(self, start: datetime, end: datetime) -> list[float]:
        """Readings between two datetimes over the FULL record (no active clamp)."""
        lo = bisect.bisect_left(self._glucose_ts, start)
        hi = bisect.bisect_right(self._glucose_ts, end)
        return self._glucose_vals[lo:hi]

    # ── frame builders ───────────────────────────────────────────────────────

    def _build_daily(self, glucose: Sequence[Any]) -> dict[date, tuple[float, float]]:
        """local date → (mean mg/dL, TIR %) for days with enough readings."""
        by_day: dict[date, list[float]] = {}
        for g in glucose:
            by_day.setdefault(self._ld(g.ts), []).append(float(g.mg_dl))
        lo, hi = self._target
        out: dict[date, tuple[float, float]] = {}
        for day, vals in by_day.items():
            if len(vals) < _MIN_READINGS_PER_DAY:
                continue
            tir = 100.0 * sum(1 for v in vals if lo <= v <= hi) / len(vals)
            out[day] = (mean(vals), tir)
        return out

    def _daily_window_means(self, hours: tuple[int, int]) -> tuple[float, ...]:
        ts_list, vals_list = self._active_glucose()
        by_day: dict[date, list[float]] = {}
        for ts, val in zip(ts_list, vals_list, strict=True):
            local = ts.astimezone(self._tz)
            if hours[0] <= local.hour < hours[1]:
                by_day.setdefault(local.date(), []).append(val)
        return tuple(
            mean(vals) for _, vals in sorted(by_day.items()) if len(vals) >= 3
        )

    def _readings_between(self, start: datetime, end: datetime) -> list[float]:
        lo = bisect.bisect_left(self._glucose_ts, max(start, self._active_start))
        hi = bisect.bisect_right(self._glucose_ts, min(end, self._active_end))
        return self._glucose_vals[lo:hi]

    def _overnight_segments(self, hours: tuple[int, int]) -> dict[date, list[float]]:
        """night-date → time-ordered readings inside the [start,end) overnight window.

        Each segment is keyed by the date its window opens, so one night never
        spans two keys even when ``hours`` straddles midnight."""
        out: dict[date, list[float]] = {}
        ts_list, vals_list = self._active_glucose()
        for ts, val in zip(ts_list, vals_list, strict=True):
            local = ts.astimezone(self._tz)
            if hours[0] <= local.hour < hours[1]:
                out.setdefault(local.date(), []).append(val)
        return out


# ── shared shaping ───────────────────────────────────────────────────────────


def _two_group(
    tool: str,
    args: dict[str, Any],
    group_a: tuple[float, ...],
    group_b: tuple[float, ...],
    labels: tuple[str, str],
) -> ToolResult:
    if len(group_a) < 2 or len(group_b) < 2:
        msg = f"too few samples ({labels[0]}={len(group_a)}, {labels[1]}={len(group_b)})"
        return _error(tool, args, msg)
    mean_a, mean_b = mean(group_a), mean(group_b)
    d = cohen_d(group_a, group_b)
    welch = welch_t_test(group_a, group_b)
    mw = mann_whitney_u(group_a, group_b)
    delta_c = cliffs_delta(group_a, group_b)
    p_welch = round(welch.p_two_sided, 4) if welch is not None else None
    summary: dict[str, Any] = {
        "label_a": labels[0],
        "label_b": labels[1],
        "n_a": len(group_a),
        "n_b": len(group_b),
        "mean_a": round(mean_a, 1),
        "mean_b": round(mean_b, 1),
        "delta": round(mean_a - mean_b, 1),
        "cohen_d": round(d, 3) if d is not None else None,
        "interpretation": _interpret(d),
        "welch_t": round(welch.t, 3) if welch is not None else None,
        "welch_df": round(welch.df, 1) if welch is not None else None,
        "p_welch": p_welch,
        "mann_whitney_p": round(mw.p_two_sided, 4) if mw is not None else None,
        "rank_biserial": round(mw.rank_biserial, 3) if mw is not None else None,
        "cliffs_delta": round(delta_c, 3) if delta_c is not None else None,
        "significant": p_welch is not None and p_welch < 0.05,
    }
    return ToolResult(
        ok=True, tool=tool, args=args, summary=summary, group_a=group_a, group_b=group_b
    )


def _interpret(d: float | None) -> str:
    if d is None:
        return "negligible"
    magnitude = abs(d)
    if magnitude >= 0.8:
        return "large"
    if magnitude >= 0.5:
        return "moderate"
    if magnitude >= 0.2:
        return "small"
    return "negligible"


def _hours(raw: Any) -> tuple[int, int]:
    start, end = int(raw[0]), int(raw[1])
    if not (0 <= start < end <= 24):
        msg = f"hours must satisfy 0 <= start < end <= 24, got {raw!r}"
        raise ValueError(msg)
    return (start, end)


def _numbers(result: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """Pull the guard-auditable numeric fields out of a tool result dict."""
    return {k: result[k] for k in keys if isinstance(result.get(k), (int, float))}


def _resolve_tz(name: str) -> ZoneInfo:
    """Resolve an IANA zone name, falling back to UTC on anything empty/unknown."""
    try:
        return ZoneInfo(name or "UTC")
    except (ZoneInfoNotFoundError, ValueError, OSError):
        return ZoneInfo("UTC")


def _parse_ts(timestamp_iso: str) -> datetime | None:
    """ISO datetime → aware UTC datetime, or None on garbage (never raises)."""
    try:
        ts = datetime.fromisoformat(str(timestamp_iso))
    except (TypeError, ValueError):
        return None
    return ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts


def _week_key(d: date) -> str:
    """ISO-year + ISO-week label, e.g. ``2026-W11`` — sorts chronologically."""
    iso = d.isocalendar()
    return f"{iso.year:04d}-W{iso.week:02d}"


def _error(tool: str, args: dict[str, Any], message: str) -> ToolResult:
    return ToolResult(
        ok=False, tool=tool, args=args, summary={"error": message}, error=message
    )


# ── clinical-evidence grounding (reasoning-loop only) ────────────────────────


def evidence_backend() -> Any:
    """Build the configured evidence backend (lazy; PubMed default).

    Imports inside the function so the deterministic toolkit never pulls in
    ``httpx`` / the evidence package unless a reasoning loop actually grounds a
    pattern. Falls back to the zero-auth PubMed backend if the config names an
    unknown backend, so a bad ``[evidence].backend`` value never breaks search.
    """
    from dexta_intelligence.config import load_config  # noqa: PLC0415
    from dexta_intelligence.evidence.pubmed import PubMedBackend  # noqa: PLC0415

    cfg = load_config().evidence
    if cfg.backend == "openevidence":
        from dexta_intelligence.evidence.openevidence import OpenEvidenceBackend  # noqa: PLC0415

        return OpenEvidenceBackend()
    return PubMedBackend(email=cfg.email)


def _search_evidence(args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Ground a confirmed pattern in published literature; numbers carry PMIDs/years."""
    query = str(args.get("query", "")).strip()
    limit = max(1, min(int(args.get("limit", 5)), 8))
    if not query:
        return {"hits": [], "note": "empty query"}, {}
    try:
        hits = evidence_backend().search(query, limit=limit)
    except Exception:  # backend build or search fault must not kill the loop
        return {"hits": [], "note": "evidence search unavailable"}, {}

    public = [
        {"title": h.title, "source": h.source, "id": h.id, "year": h.year, "snippet": h.snippet}
        for h in hits
    ]
    numbers: dict[str, Any] = {}
    for i, h in enumerate(hits):
        entry: dict[str, Any] = {}
        digits = "".join(c for c in h.id if c.isdigit())
        if digits:
            entry["pmid"] = int(digits)
        if h.year is not None:
            entry["year"] = h.year
        if entry:
            numbers[f"hit_{i}"] = entry
    return {"hits": public}, numbers


# ── reasoning-loop tool specs (shared by chat and goal agents) ───────────────

_HOURS_PAIR = {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}


def tool_specs(ctx: AgentContext, toolkit: DiscoveryToolkit) -> list[ToolSpec]:
    """The read-only instruments a reasoning loop may call."""

    def run(tool: str) -> Any:
        def call(args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
            result = toolkit.run(tool, args)
            return result.summary, result.evidence()

        return call

    def recall(args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        return _recall(ctx, str(args.get("query", "")))

    def set_window(args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        result = toolkit.set_window(str(args.get("start", "")), str(args.get("end", "")))
        return result, _numbers(result, ("n_days", "n_readings"))

    def list_segments(_args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        result = toolkit.list_segments()
        numbers: dict[str, Any] = {}
        for seg in result.get("segments", []):
            numbers[seg["period"]] = {
                "n_days": seg["n_days"],
                "mean_glucose": seg["mean_glucose"],
                "tir_pct": seg["tir_pct"],
                "n_lows": seg["n_lows"],
            }
        return result, numbers

    def zoom_event(args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        result = toolkit.zoom_event(
            str(args.get("timestamp", "")), int(args.get("pad_hours", 12))
        )
        return result, _numbers(result, ("pre_mean", "post_mean", "peak", "nadir"))

    def daily_series(args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        result = toolkit.daily_series(str(args.get("metric", "")))
        numbers: dict[str, Any] = {}
        for row in result.get("series", []):
            numbers[row["date"]] = row["value"]
        if result.get("mean_value") is not None:
            numbers["mean_value"] = result["mean_value"]
        return result, numbers

    def glucose_stats(args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        raw = args.get("hours")
        hours = (
            (int(raw[0]), int(raw[1]))
            if isinstance(raw, (list, tuple)) and len(raw) == 2
            else None
        )
        result = toolkit.glucose_stats(day=str(args.get("day", "")), hours=hours)
        return result, _numbers(
            result,
            (
                "n", "mean", "sd", "variance", "cv_pct", "median", "p10", "p25",
                "p75", "p90", "minimum", "maximum", "tir_pct", "tbr_pct", "tar_pct",
                "tbr54_pct", "tar250_pct", "gmi_pct",
            ),
        )

    def correlate(args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        result = toolkit.correlate(str(args.get("x", "")), str(args.get("y", "")))
        return result, _numbers(result, ("n", "pearson_r", "spearman_rho", "p"))

    def coverage(_args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        cov = ctx.store.coverage()
        numbers = {
            "span_days": cov.span_days,
            "glucose_coverage_pct": cov.glucose_coverage_pct,
            "n_meals": cov.n_meals,
            "n_insulin": cov.n_insulin,
            "n_sleep": cov.n_sleep,
            "n_activity": cov.n_activity,
        }
        out: dict[str, Any] = {"summary": toolkit.data_summary(), **numbers}
        if toolkit._last_glucose_ts:
            out["last_glucose_ts"] = toolkit._last_glucose_ts.isoformat()
        if toolkit._last_insulin_ts:
            out["last_insulin_ts"] = toolkit._last_insulin_ts.isoformat()
        gap = toolkit._treatment_gap_note()
        if gap:
            out["treatment_gap"] = gap
        missing = toolkit.capabilities().missing_notes()
        if missing:
            out["unavailable"] = missing
        return out, numbers

    specs = [
        ToolSpec(
            name="recall",
            description=(
                "What dexta already believes: prior findings (with status, confidence "
                "and the skeptic's confound notes), open questions, and cross-finding "
                "connections. Call FIRST for known patterns — it tells you what was "
                "already doubted so you pick better tools."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "topic, e.g. 'overnight'"}
                },
                "required": ["query"],
            },
            fn=recall,
        ),
        ToolSpec(
            name="coverage",
            description="How much data exists (days, coverage %, streams).",
            parameters={"type": "object", "properties": {}},
            fn=coverage,
        ),
        ToolSpec(
            name="list_segments",
            description=(
                "Coarse structure of the whole record: one row per month (per week "
                "if span < 60d) with n_days, mean_glucose, tir_pct, n_lows. Call FIRST "
                "to orient before set_window."
            ),
            parameters={"type": "object", "properties": {}},
            fn=list_segments,
        ),
        ToolSpec(
            name="set_window",
            description=(
                "Narrow the ACTIVE window all later tools read from to [start, end] "
                "ISO dates. Clamps to available data; returns active_start/end, "
                "n_days, n_readings. Re-call with the full span to widen back out."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "start": {"type": "string", "description": "ISO date, e.g. 2026-03-01"},
                    "end": {"type": "string", "description": "ISO date, e.g. 2026-03-31"},
                },
                "required": ["start", "end"],
            },
            fn=set_window,
        ),
        ToolSpec(
            name="zoom_event",
            description=(
                "Drill a spike: window tight around an ISO timestamp (+/- pad_hours, "
                "default 12); returns the minute-level trace with pre_mean, post_mean, "
                "peak, nadir."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string", "description": "ISO datetime of the event"},
                    "pad_hours": {"type": "integer", "minimum": 1, "maximum": 72},
                },
                "required": ["timestamp"],
            },
            fn=zoom_event,
        ),
        ToolSpec(
            name="daily_series",
            description=(
                "Per-day time series over the ACTIVE window as [{date, value}] to spot "
                "trends/change-points. metric: tir|mean_glucose|tbr|cv."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "enum": ["tir", "mean_glucose", "tbr", "cv"],
                    },
                },
                "required": ["metric"],
            },
            fn=daily_series,
        ),
        ToolSpec(
            name="glucose_stats",
            description=(
                "Descriptive stats over one window: n, mean, SD, variance, CV, median, "
                "p10/25/75/90, min, max, TIR/TBR/TAR %, GMI. Use this for any "
                "'what was my mean/variance/SD/TIR/GMI for <period>' question — never "
                "estimate these from a raw trace. Defaults to the ACTIVE window; pass "
                "day (ISO date) for one local day and/or hours [start,end) for a "
                "time-of-day band (evening=[17,24], overnight=[0,6], morning=[6,11])."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "day": {
                        "type": "string",
                        "description": "ISO date for one local day, e.g. 2026-06-15",
                    },
                    "hours": _HOURS_PAIR,
                },
            },
            fn=glucose_stats,
        ),
        ToolSpec(
            name="correlate",
            description=(
                "Correlate two per-day metrics across the active window "
                "(Pearson + Spearman + p). metrics: mean_glucose|tir|tbr|cv|sleep_score."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "x": {
                        "type": "string",
                        "enum": ["mean_glucose", "tir", "tbr", "cv", "sleep_score"],
                    },
                    "y": {
                        "type": "string",
                        "enum": ["mean_glucose", "tir", "tbr", "cv", "sleep_score"],
                    },
                },
                "required": ["x", "y"],
            },
            fn=correlate,
        ),
        ToolSpec(
            name="groupby_compare",
            description=(
                "Compare a daily metric between two groups of days. "
                "group_by: weekend|sleep_bucket|workout_day. target: mean_glucose|tir_pct. "
                "Returns a p-value (p_welch) and effect sizes (cohen_d, cliffs_delta)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "group_by": {
                        "type": "string",
                        "enum": ["weekend", "sleep_bucket", "workout_day"],
                    },
                    "target": {"type": "string", "enum": ["mean_glucose", "tir_pct"]},
                },
                "required": ["group_by", "target"],
            },
            fn=run("groupby_compare"),
        ),
        ToolSpec(
            name="tod_compare",
            description=(
                "Compare mean glucose between two time-of-day windows. "
                "hours_a/hours_b are [start,end) hours 0-24. "
                "Returns a p-value (p_welch) and effect sizes (cohen_d, cliffs_delta)."
            ),
            parameters={
                "type": "object",
                "properties": {"hours_a": _HOURS_PAIR, "hours_b": _HOURS_PAIR},
                "required": ["hours_a", "hours_b"],
            },
            fn=run("tod_compare"),
        ),
        ToolSpec(
            name="event_proximity",
            description=(
                "Average glucose after an event vs the hour before it. "
                "event_type: meal|workout|bolus. window_min: 30-240."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "event_type": {"type": "string", "enum": ["meal", "workout", "bolus"]},
                    "window_min": {"type": "integer", "minimum": 30, "maximum": 240},
                },
                "required": ["event_type"],
            },
            fn=run("event_proximity"),
        ),
        ToolSpec(
            name="basal_overnight",
            description=(
                "Per-night overnight glucose drift, first-half vs second-half nights. "
                "hours is the [start,end) overnight window (default [0,6])."
            ),
            parameters={
                "type": "object",
                "properties": {"hours": _HOURS_PAIR},
            },
            fn=run("basal_overnight"),
        ),
        ToolSpec(
            name="meal_response",
            description=(
                "Per-meal excursion (peak minus pre-meal baseline) for bigger-carb vs "
                "smaller-carb meals, split at median carbs. window_min: 30-240."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "window_min": {"type": "integer", "minimum": 30, "maximum": 240},
                },
            },
            fn=run("meal_response"),
        ),
        ToolSpec(
            name="correction_outcome",
            description=(
                "Per-bolus glucose delta (window-end minus baseline), newer vs older "
                "boluses, plus rebound_low_rate (% with a <70 reading). window_min: 30-240."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "window_min": {"type": "integer", "minimum": 30, "maximum": 240},
                },
            },
            fn=run("correction_outcome"),
        ),
        ToolSpec(
            name="search_evidence",
            description=(
                "Search clinical literature (PubMed). Use to ground a confirmed personal "
                "pattern in published evidence or note contradiction. Cite only returned PMIDs."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "clinical search terms"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 8},
                },
                "required": ["query"],
            },
            fn=_search_evidence,
        ),
    ]
    specs.extend(_treatment_specs(toolkit))
    caps = toolkit.capabilities()
    specs = [spec for spec in specs if caps.allows(_TOOL_NEEDS.get(spec.name))]
    specs.extend(time_tool_specs())
    return specs


def _treatment_specs(toolkit: DiscoveryToolkit) -> list[ToolSpec]:
    """Treatment-inspection ToolSpecs. Capability filtering happens in
    :func:`tool_specs`; these are built unconditionally."""

    def _item_numbers(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
        return {
            f"{prefix}_{i}": {k: v for k, v in row.items() if isinstance(v, (int, float))}
            for i, row in enumerate(rows)
        }

    def get_carb_entries(_args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        result = toolkit.get_carb_entries()
        numbers = _numbers(result, ("n_entries", "total_carbs_g"))
        numbers.update(_item_numbers(result.get("entries", []), "entry"))
        return result, numbers

    def get_boluses(_args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        result = toolkit.get_boluses()
        numbers = _numbers(result, ("n_boluses", "total_units"))
        numbers.update(_item_numbers(result.get("boluses", []), "bolus"))
        return result, numbers

    def get_basal_timeline(_args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        result = toolkit.get_basal_timeline()
        numbers = _numbers(result, ("n_basal", "n_temp_basal", "n_suspend"))
        numbers.update(_item_numbers(result.get("events", []), "event"))
        return result, numbers

    def get_iob(args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        result = toolkit.get_iob(str(args.get("timestamp", "")))
        return result, _numbers(result, ("iob_units", "n_recent_boluses"))

    def get_insulin_profile(_args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        result = toolkit.get_insulin_profile()
        active = next(
            (p for p in result.get("profiles") or [] if p.get("active")),
            None,
        )
        numbers = _numbers(result, ("pump_serial",))
        if active:
            numbers["active_dia_hr"] = active.get("dia_hr")
            numbers["n_active_segments"] = len(active.get("segments") or [])
        return result, numbers

    def get_cob(args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        result = toolkit.get_cob(str(args.get("timestamp", "")))
        return result, _numbers(result, ("cob_g", "absorbed_g", "n_carb_entries"))

    def find_spikes(args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        result = toolkit.find_spikes(
            float(args.get("threshold", _SPIKE_THRESHOLD)), int(args.get("top_n", 10))
        )
        numbers = _numbers(result, ("threshold", "n_spikes"))
        numbers.update(_item_numbers(result.get("spikes", []), "spike"))
        return result, numbers

    def find_similar_events(args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        result = toolkit.find_similar_events(
            str(args.get("timestamp", "")), float(args.get("threshold", _SPIKE_THRESHOLD))
        )
        numbers = _numbers(
            result,
            (
                "n_similar",
                "n_spiking",
                "mean_peak_spiking",
                "mean_bolus_delay_spiking_min",
                "mean_bolus_delay_other_min",
            ),
        )
        numbers.update(_item_numbers(result.get("events", []), "similar"))
        return result, numbers

    return [
        ToolSpec(
            name="get_carb_entries",
            description=(
                "Carb entries in the ACTIVE window ({ts, carbs_g, ...}, n_entries, "
                "total_carbs_g). Call when explaining a spike/meal — an empty result "
                "around a spike is itself a signal (possible missing carb entry)."
            ),
            parameters={"type": "object", "properties": {}},
            fn=get_carb_entries,
        ),
        ToolSpec(
            name="get_boluses",
            description=(
                "Boluses in the ACTIVE window ({ts, units, minutes_after_carb_entry}). "
                "minutes_after_carb_entry is the late-bolus signal. Call when explaining "
                "a spike/meal/correction."
            ),
            parameters={"type": "object", "properties": {}},
            fn=get_boluses,
        ),
        ToolSpec(
            name="get_basal_timeline",
            description=(
                "Basal / temp-basal / suspend events in the ACTIVE window plus "
                "basal_stable (no temp-basal/suspend). Rules basal in or out as a "
                "spike contributor."
            ),
            parameters={"type": "object", "properties": {}},
            fn=get_basal_timeline,
        ),
        ToolSpec(
            name="get_iob",
            description=(
                "Insulin-on-board at an ISO datetime, computed from logged boluses "
                "(oref0 curve, tier B — analysis context only, never dosing)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string", "description": "ISO datetime"},
                },
                "required": ["timestamp"],
            },
            fn=get_iob,
        ),
        ToolSpec(
            name="get_insulin_profile",
            description=(
                "Pump-reported basal/ISF/carb-ratio/target segments for the active "
                "profile (and all stored profiles). Synced from Tandem; tier B — "
                "analysis context only, never dosing."
            ),
            parameters={"type": "object", "properties": {}},
            fn=get_insulin_profile,
        ),
        ToolSpec(
            name="get_cob",
            description=(
                "Carbs-on-board at an ISO datetime from announced carb entries "
                "(oref0 decay, tier B — analysis context only). Unannounced carbs "
                "do not appear here."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string", "description": "ISO datetime"},
                },
                "required": ["timestamp"],
            },
            fn=get_cob,
        ),
        ToolSpec(
            name="find_spikes",
            description=(
                "Excursion peaks above threshold in the ACTIVE window ({ts, peak_mg_dl, "
                "duration_min}, largest first). Locates the spike to zoom_event when the "
                "user names a day but not a time."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "threshold": {"type": "number", "minimum": 140, "maximum": 400},
                    "top_n": {"type": "integer", "minimum": 1, "maximum": 25},
                },
            },
            fn=find_spikes,
        ),
        ToolSpec(
            name="find_similar_events",
            description=(
                "Recurrence over the WHOLE record: events at the same time of day as the "
                "timestamp (carb entries when logged), each with post-event peak, spiked "
                "flag, bolus_delay_min; plus n_similar, n_spiking, mean spiking vs "
                "non-spiking bolus delays. The 'N of M similar dinners' instrument."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string", "description": "ISO datetime of the event"},
                    "threshold": {"type": "number", "minimum": 140, "maximum": 400},
                },
                "required": ["timestamp"],
            },
            fn=find_similar_events,
        ),
    ]


def _recall(ctx: AgentContext, query: str) -> tuple[Any, dict[str, Any]]:
    """The structured shared-context channel: what other agents already believe.

    Returns each relevant finding's headline AND the reasoning left behind —
    ``status``, ``confidence`` and the skeptic's ``skeptic_notes`` (why it was
    doubted / what confound was flagged) when present — plus the open
    hypotheses (what is suspected) and synthesis ``connections`` (what is
    contested across agents). The caller reads these to pick better tools.

    Shape is additive/backward-compatible: keys ``findings`` /
    ``open_questions`` / ``connections`` and the per-finding numbers tuple are
    preserved; new fields only extend each entry. Findings, connections and
    open questions are each capped at :data:`_MAX_RECALL_ITEMS` with a note.
    """
    from dexta_intelligence.memory import embeddings  # noqa: PLC0415
    from dexta_intelligence.memory.synthesis import load_latest  # noqa: PLC0415

    candidates = [
        f
        for f in ctx.store.get_findings(limit=50)
        if f.agent != "synthesis"
        and f.kind != "investigation"
        and f.status != FindingStatus.STALE
    ]
    q = query.strip()
    if q and candidates:
        scored = embeddings.rank_findings(q, candidates, top_k=_MAX_RECALL_ITEMS)
        findings = [f for _score, f in scored] or candidates[:5]
    else:
        findings = candidates[:5]

    numbers: dict[str, Any] = {}
    items: list[dict[str, Any]] = []
    for f in findings[:_MAX_RECALL_ITEMS]:
        item: dict[str, Any] = {
            "headline": f.headline,
            "effect_size": f.stats.effect_size,
            "n": f.stats.n,
            "confidence": f.confidence,
            "status": f.status.value,
        }
        if f.skeptic_notes:  # the cross-agent "why this was doubted" signal
            item["skeptic_notes"] = f.skeptic_notes
        items.append(item)
        numbers[f"finding_{len(items)}"] = {
            "effect_size": f.stats.effect_size,
            "n": f.stats.n,
            "confidence": f.confidence,
        }

    try:
        open_q = ctx.store.get_hypotheses(status=HypothesisStatus.OPEN.value)
    except Exception:
        open_q = []

    payload: dict[str, Any] = {
        "findings": items,
        "open_questions": [h.statement for h in open_q[:_MAX_RECALL_ITEMS]],
    }
    if len(open_q) > _MAX_RECALL_ITEMS:
        payload["open_questions_note"] = (
            f"showing first {_MAX_RECALL_ITEMS} of {len(open_q)}"
        )

    synthesis = load_latest(ctx.store)
    if synthesis is not None and synthesis.connections:
        if q:
            ranked = embeddings.rank(
                q,
                [(line, line) for line in synthesis.connections],
                top_k=_MAX_RECALL_ITEMS,
                synonyms=True,
            )
            connections = [line for _score, line in ranked]
        else:
            connections = list(synthesis.connections[:_MAX_RECALL_ITEMS])
        payload["connections"] = connections
        if len(synthesis.connections) > _MAX_RECALL_ITEMS:
            payload["connections_note"] = (
                f"showing first {_MAX_RECALL_ITEMS} of {len(synthesis.connections)}"
            )

    return (payload, numbers)
