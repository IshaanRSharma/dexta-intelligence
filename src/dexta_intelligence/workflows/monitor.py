"""Monitoring pipeline - deterministic anomaly detectors over a recent window.

Detection is an instrument, not a reasoning step: each detector reads the
last N hours of timeline data through the :class:`~dexta_intelligence.store.port.StoragePort`
and reports structured anomalies straight from the numbers - no LLM, no
interpretation, no rigor gate. This is the foundation for monitoring agents
that notify on anomalies; an optional LLM summary can layer on later.

Detectors (clinical thresholds documented inline)
-------------------------------------------------
- ``severe_low`` - any reading < 54 mg/dL (level-2 hypoglycemia). **urgent**.
- ``severe_high`` - readings > 250 mg/dL (level-2 hyperglycemia) sustained for
  at least :data:`SEVERE_HIGH_SUSTAIN_MIN` minutes. **warning**.
- ``time_in_range_cliff`` - recent-window TIR materially below the trailing
  baseline TIR (drop ≥ :data:`TIR_CLIFF_DROP_PCT` points). **warning**.
- ``sensor_gap`` - a contiguous gap between consecutive readings longer than
  :data:`SENSOR_GAP_MIN` minutes (CGM cadence is 5 min). **info/warning**.
- ``rapid_rise`` - glucose climbing ≥ :data:`RAPID_RISE_MG_DL` mg/dL within
  :data:`RAPID_RISE_WINDOW_MIN` minutes, flagging whether carbs were logged
  beforehand (unannounced meal vs. announced). **warning**.
- ``correction_not_working`` - a correction bolus given while high that fails to
  bring glucose back below target within :data:`CORRECTION_WAIT_MIN` minutes
  (possible failed correction / occlusion / resistance). **warning**.
- ``low_after_correction`` - a bolus followed by a low within
  :data:`LOW_AFTER_HOURS` hours (possible over-correction / stacking). **warning**.

Each anomaly surfaces as a ``Finding(kind="anomaly")`` (guard-safe: every
number comes straight from the data) and/or is pushed through a
:class:`~dexta_intelligence.notifications.Notifier`. The pipeline is pure and
deterministic and never raises on thin data - it degrades and returns ``[]``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal

from dexta_intelligence.analytics.rollups import (
    TARGET_HIGH_MG_DL,
    TARGET_LOW_MG_DL,
    VERY_HIGH_MG_DL,
    VERY_LOW_MG_DL,
)
from dexta_intelligence.models import Finding, FindingStatus, InsulinKind

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dexta_intelligence.agents.base import AgentContext
    from dexta_intelligence.models import GlucoseEvent, InsulinEvent, MealEvent
    from dexta_intelligence.notifications import Notifier

logger = logging.getLogger(__name__)

__all__ = [
    "Anomaly",
    "Severity",
    "run_monitor",
]

Severity = Literal["info", "warning", "urgent"]

AGENT_NAME = "monitor"

#: Default recent window the detectors scan (hours).
DEFAULT_WINDOW_HOURS = 24
#: Trailing baseline used by the TIR-cliff detector (hours, ending at window start).
DEFAULT_BASELINE_HOURS = 24 * 14
#: Native CGM cadence (minutes). A gap is measured against this.
CGM_CADENCE_MIN = 5
#: A contiguous reading gap longer than this is a sensor gap (minutes).
SENSOR_GAP_MIN = 30
#: A sensor gap at/above this length escalates info → warning (minutes).
SENSOR_GAP_WARN_MIN = 120
#: severe_high requires this many sustained minutes above 250 to fire.
SEVERE_HIGH_SUSTAIN_MIN = 30
#: TIR drop (percentage points) vs baseline that counts as a cliff.
TIR_CLIFF_DROP_PCT = 15.0
#: Minimum readings in each window before TIR-cliff is trustworthy.
TIR_CLIFF_MIN_READINGS = 24

#: A rise of this many mg/dL within RAPID_RISE_WINDOW_MIN is a rapid rise. A jump
#: this steep this fast is clinically meaningful (a meal/correction-needed spike).
RAPID_RISE_MG_DL = 70
#: The sliding window (minutes) the rapid-rise climb is measured across.
RAPID_RISE_WINDOW_MIN = 30
#: Carbs logged within this many minutes before a rise mark it as announced,
#: which distinguishes an expected meal spike from an unannounced one.
RAPID_RISE_CARB_LOOKBACK_MIN = 60
#: A correction bolus has this long to pull glucose back under target. If every
#: reading stays high for this whole window, the correction is not working
#: (stale insulin, occlusion, illness-driven resistance).
CORRECTION_WAIT_MIN = 90
#: A bolus followed by a sub-target low inside this many hours suggests an
#: over-correction or insulin stacking and warrants attention.
LOW_AFTER_HOURS = 4


@dataclass(frozen=True, slots=True)
class Anomaly:
    """One detected anomaly. ``numbers`` carries the triggering measurements.

    ``key`` is a stable signature of the *specific* anomaly (e.g. the nadir of a
    low, the timestamp of a gap) - stable across runs while the same underlying
    event persists, so a polling loop dedupes the same anomaly instead of
    re-recording it every cycle as the scan window slides.
    """

    name: str
    severity: Severity
    headline: str
    window: tuple[datetime, datetime]
    numbers: dict[str, float | int] = field(default_factory=dict)
    key: str = ""


def run_monitor(
    ctx: AgentContext,
    *,
    notify: Notifier | None = None,
    persist: bool = True,
    window_hours: int = DEFAULT_WINDOW_HOURS,
    baseline_hours: int = DEFAULT_BASELINE_HOURS,
    now: datetime | None = None,
) -> list[Anomaly]:
    """Run all detectors over the recent window; persist and/or notify.

    The window is anchored on the most recent reading (``coverage().last_ts``)
    so the monitor always inspects the live tail of the data regardless of
    ``ctx.window``. Returns the detected anomalies (possibly empty). Pure and
    deterministic; never raises on thin data.
    """
    anchor = _anchor(ctx, now)
    if anchor is None:
        return []

    window_start = anchor - timedelta(hours=window_hours)
    recent = ctx.store.get_glucose(window_start, anchor)
    if not recent:
        return []
    recent = sorted(recent, key=lambda g: g.ts)
    window = (window_start, anchor)

    baseline = ctx.store.get_glucose(window_start - timedelta(hours=baseline_hours), window_start)

    insulin = sorted(ctx.store.get_insulin(window_start, anchor), key=lambda i: i.ts)
    meals = sorted(ctx.store.get_meals(window_start, anchor), key=lambda m: m.ts)

    anomalies: list[Anomaly] = []
    anomalies.extend(_severe_low(recent, window))
    anomalies.extend(_severe_high(recent, window))
    anomalies.extend(_time_in_range_cliff(recent, baseline, window))
    anomalies.extend(_sensor_gap(recent, window))
    anomalies.extend(_rapid_rise(recent, meals, window))
    anomalies.extend(_correction_not_working(recent, insulin, window))
    anomalies.extend(_low_after_correction(recent, insulin, window))

    # Dedup against what's already recorded: the same ongoing anomaly (same key)
    # must not be re-persisted or re-notified each polling cycle. Dedup needs the
    # store as its ledger, so it only applies when persisting.
    to_notify = anomalies
    if persist:
        to_notify = _persist_new(ctx, anomalies)

    if notify is not None:
        for anomaly in to_notify:
            try:
                notify.send(anomaly)
            except Exception:
                logger.exception("monitor: notifier failed for anomaly %s", anomaly.name)

    return anomalies


def _persist_new(ctx: AgentContext, anomalies: list[Anomaly]) -> list[Anomaly]:
    """Insert only anomalies not already active (by ``key``); supersede a
    detector's stale finding when its state changes. Returns the newly recorded
    anomalies (what a notifier should fire on)."""
    try:
        existing = ctx.store.get_findings(
            agent=AGENT_NAME, kind="anomaly", status=FindingStatus.ACTIVE, limit=1000
        )
    except Exception:
        logger.exception("monitor: failed to read existing anomalies; skipping dedup")
        existing = []
    active_keys = {str(f.evidence.get("key", "")) for f in existing}
    stale_by_scope: dict[str, list[Finding]] = {}
    for f in existing:
        stale_by_scope.setdefault(f.scope, []).append(f)

    recorded: list[Anomaly] = []
    for anomaly in anomalies:
        if anomaly.key and anomaly.key in active_keys:
            continue  # same ongoing anomaly - already recorded, don't duplicate
        try:
            new_id = ctx.store.insert_finding(_to_finding(anomaly))
        except Exception:
            logger.exception("monitor: failed to persist anomaly %s", anomaly.name)
            continue
        for stale in stale_by_scope.get(anomaly.name, []):
            if stale.id is not None and str(stale.evidence.get("key", "")) != anomaly.key:
                ctx.store.supersede_finding(stale.id, new_id)
        recorded.append(anomaly)
    return recorded


def _anchor(ctx: AgentContext, now: datetime | None) -> datetime | None:
    if now is not None:
        return now if now.tzinfo else now.replace(tzinfo=UTC)
    try:
        last = ctx.store.coverage().last_ts
    except Exception:
        logger.exception("monitor: coverage() failed")
        return None
    return last


# ── detectors ─────────────────────────────────────────────────────────────────


def _severe_low(recent: Sequence[GlucoseEvent], window: tuple[datetime, datetime]) -> list[Anomaly]:
    """Any reading < 54 mg/dL (level-2 hypoglycemia) - always urgent."""
    lows = [g for g in recent if g.mg_dl < VERY_LOW_MG_DL]
    if not lows:
        return []
    nadir = min(g.mg_dl for g in lows)
    return [
        Anomaly(
            name="severe_low",
            severity="urgent",
            headline=(
                f"Severe low: {len(lows)} reading(s) below {VERY_LOW_MG_DL} mg/dL "
                f"(nadir {nadir})"
            ),
            window=window,
            numbers={
                "n_readings": len(lows),
                "nadir_mg_dl": nadir,
                "threshold_mg_dl": VERY_LOW_MG_DL,
            },
            key=f"severe_low:{nadir}",
        )
    ]


def _severe_high(
    recent: Sequence[GlucoseEvent], window: tuple[datetime, datetime]
) -> list[Anomaly]:
    """Sustained run > 250 mg/dL for ≥ SEVERE_HIGH_SUSTAIN_MIN minutes - warning."""
    run_start: datetime | None = None
    run_end: datetime | None = None
    peak = 0
    longest_min = 0.0
    peak_overall = 0
    total_high = 0

    def close() -> None:
        nonlocal run_start, run_end, peak, longest_min, peak_overall
        if run_start is not None and run_end is not None:
            duration = (run_end - run_start).total_seconds() / 60.0
            if duration >= SEVERE_HIGH_SUSTAIN_MIN:
                longest_min = max(longest_min, duration)
                peak_overall = max(peak_overall, peak)
        run_start = run_end = None
        peak = 0

    gap = timedelta(minutes=SENSOR_GAP_MIN)
    for g in recent:
        if g.mg_dl > VERY_HIGH_MG_DL:
            total_high += 1
            # A sensor gap breaks the "sustained" claim: we cannot assert the run
            # stayed high across minutes with no readings, so bank it and restart.
            if run_end is not None and g.ts - run_end > gap:
                close()
            if run_start is None:
                run_start = g.ts
            run_end = g.ts
            peak = max(peak, g.mg_dl)
        else:
            close()
    close()

    if longest_min < SEVERE_HIGH_SUSTAIN_MIN:
        return []
    return [
        Anomaly(
            name="severe_high",
            severity="warning",
            headline=(
                f"Severe high: sustained {longest_min:.0f} min above {VERY_HIGH_MG_DL} mg/dL "
                f"(peak {peak_overall})"
            ),
            window=window,
            numbers={
                "longest_run_min": round(longest_min, 1),
                "peak_mg_dl": peak_overall,
                "n_readings_above": total_high,
                "threshold_mg_dl": VERY_HIGH_MG_DL,
            },
            key=f"severe_high:{peak_overall}",
        )
    ]


def _tir(values: Sequence[int]) -> float:
    in_range = sum(1 for v in values if TARGET_LOW_MG_DL <= v <= TARGET_HIGH_MG_DL)
    return 100.0 * in_range / len(values)


def _time_in_range_cliff(
    recent: Sequence[GlucoseEvent],
    baseline: Sequence[GlucoseEvent],
    window: tuple[datetime, datetime],
) -> list[Anomaly]:
    """Recent TIR materially below the trailing baseline TIR - warning."""
    if len(recent) < TIR_CLIFF_MIN_READINGS or len(baseline) < TIR_CLIFF_MIN_READINGS:
        return []
    recent_tir = _tir([g.mg_dl for g in recent])
    baseline_tir = _tir([g.mg_dl for g in baseline])
    drop = baseline_tir - recent_tir
    if drop < TIR_CLIFF_DROP_PCT:
        return []
    return [
        Anomaly(
            name="time_in_range_cliff",
            severity="warning",
            headline=(
                f"Time-in-range cliff: TIR fell {drop:.1f} pts "
                f"({baseline_tir:.1f}% → {recent_tir:.1f}%)"
            ),
            window=window,
            numbers={
                "recent_tir_pct": round(recent_tir, 1),
                "baseline_tir_pct": round(baseline_tir, 1),
                "drop_pct": round(drop, 1),
                "threshold_drop_pct": TIR_CLIFF_DROP_PCT,
            },
            key=f"tir_cliff:{round(recent_tir / 5) * 5}",
        )
    ]


def _sensor_gap(recent: Sequence[GlucoseEvent], window: tuple[datetime, datetime]) -> list[Anomaly]:
    """Largest contiguous reading gap > SENSOR_GAP_MIN minutes - info/warning."""
    max_gap_min = 0.0
    gap_at: datetime | None = None
    n_gaps = 0
    prev: datetime | None = None
    for g in recent:
        if prev is not None:
            gap = (g.ts - prev).total_seconds() / 60.0
            if gap > SENSOR_GAP_MIN:
                n_gaps += 1
                if gap > max_gap_min:
                    max_gap_min = gap
                    gap_at = prev
        prev = g.ts

    if max_gap_min <= SENSOR_GAP_MIN or gap_at is None:
        return []
    severity: Severity = "warning" if max_gap_min >= SENSOR_GAP_WARN_MIN else "info"
    return [
        Anomaly(
            name="sensor_gap",
            severity=severity,
            headline=f"Sensor gap: {max_gap_min:.0f} min without readings ({n_gaps} gap(s))",
            window=window,
            numbers={
                "max_gap_min": round(max_gap_min, 1),
                "n_gaps": n_gaps,
                "threshold_min": SENSOR_GAP_MIN,
            },
            key=f"sensor_gap:{gap_at.isoformat()}",
        )
    ]


# ── treatment-aware detectors ───────────────────────────────────────────────────


def _rapid_rise(
    recent: Sequence[GlucoseEvent],
    meals: Sequence[MealEvent],
    window: tuple[datetime, datetime],
) -> list[Anomaly]:
    """Steepest glucose climb >= RAPID_RISE_MG_DL within RAPID_RISE_WINDOW_MIN - warning.

    Flags how fast glucose is going up, not just how high it gets, and records
    whether carbs were logged just before the climb so an announced meal spike is
    distinguishable from an unannounced one.
    """
    if len(recent) < 2:
        return []
    best_rise = 0
    rise_start: GlucoseEvent | None = None
    rise_end: GlucoseEvent | None = None
    span = timedelta(minutes=RAPID_RISE_WINDOW_MIN)
    for i, start in enumerate(recent):
        for end in recent[i + 1 :]:
            if end.ts - start.ts > span:
                break
            rise = end.mg_dl - start.mg_dl
            if rise > best_rise:
                best_rise = rise
                rise_start = start
                rise_end = end

    if best_rise < RAPID_RISE_MG_DL or rise_start is None or rise_end is None:
        return []

    lookback_start = rise_start.ts - timedelta(minutes=RAPID_RISE_CARB_LOOKBACK_MIN)
    carb_logged = any(
        m.carbs_g is not None
        and m.carbs_g > 0
        and lookback_start <= m.ts <= rise_start.ts
        for m in meals
    )
    window_min = (rise_end.ts - rise_start.ts).total_seconds() / 60.0
    return [
        Anomaly(
            name="rapid_rise",
            severity="warning",
            headline=(
                f"Rapid rise: +{best_rise} mg/dL in {window_min:.0f} min "
                f"({rise_start.mg_dl} → {rise_end.mg_dl}"
                f"{'' if carb_logged else ', no carbs logged'})"
            ),
            window=window,
            numbers={
                "rise_mg_dl": best_rise,
                "window_min": round(window_min, 1),
                "from_mg_dl": rise_start.mg_dl,
                "to_mg_dl": rise_end.mg_dl,
                "carb_logged": 1 if carb_logged else 0,
            },
            key=f"rapid_rise:{int(rise_start.ts.timestamp())}",
        )
    ]


def _glucose_at(recent: Sequence[GlucoseEvent], ts: datetime) -> GlucoseEvent | None:
    """Nearest reading at or before ``ts`` (``recent`` is sorted by ts)."""
    found: GlucoseEvent | None = None
    for g in recent:
        if g.ts <= ts:
            found = g
        else:
            break
    return found


def _correction_not_working(
    recent: Sequence[GlucoseEvent],
    insulin: Sequence[InsulinEvent],
    window: tuple[datetime, datetime],
) -> list[Anomaly]:
    """A correction bolus given while high that leaves glucose above target - warning.

    A correction bolus is a BOLUS with units > 0 whose nearest preceding reading
    is above target. If every reading for CORRECTION_WAIT_MIN after it stays above
    target (with coverage spanning that whole window), the correction is failing.
    """
    if not insulin or not recent:
        return []
    out: list[Anomaly] = []
    wait = timedelta(minutes=CORRECTION_WAIT_MIN)
    for shot in insulin:
        if shot.kind != InsulinKind.BOLUS or shot.units is None or shot.units <= 0:
            continue
        at_bolus = _glucose_at(recent, shot.ts)
        if at_bolus is None or at_bolus.mg_dl <= TARGET_HIGH_MG_DL:
            continue
        after = [g for g in recent if shot.ts <= g.ts <= shot.ts + wait]
        if not after:
            continue
        span_min = (after[-1].ts - shot.ts).total_seconds() / 60.0
        if span_min < CORRECTION_WAIT_MIN:
            continue
        if any(g.mg_dl <= TARGET_HIGH_MG_DL for g in after):
            continue
        out.append(
            Anomaly(
                name="correction_not_working",
                severity="warning",
                headline=(
                    f"Correction not working: {shot.units:g}U bolus at "
                    f"{at_bolus.mg_dl} mg/dL, still above {TARGET_HIGH_MG_DL} "
                    f"after {span_min:.0f} min"
                ),
                window=window,
                numbers={
                    "bolus_units": shot.units,
                    "minutes_high_after": round(span_min, 1),
                    "glucose_at_bolus": at_bolus.mg_dl,
                    "glucose_after": after[-1].mg_dl,
                },
                key=f"correction_not_working:{int(shot.ts.timestamp())}",
            )
        )
    return out


def _low_after_correction(
    recent: Sequence[GlucoseEvent],
    insulin: Sequence[InsulinEvent],
    window: tuple[datetime, datetime],
) -> list[Anomaly]:
    """A bolus followed by a sub-target low within LOW_AFTER_HOURS - warning.

    A low arriving within the action window of a bolus suggests over-correction
    or insulin stacking rather than an unrelated low.
    """
    if not insulin or not recent:
        return []
    out: list[Anomaly] = []
    horizon = timedelta(hours=LOW_AFTER_HOURS)
    for shot in insulin:
        if shot.kind != InsulinKind.BOLUS or shot.units is None or shot.units <= 0:
            continue
        after = [
            g
            for g in recent
            if shot.ts < g.ts <= shot.ts + horizon and g.mg_dl < TARGET_LOW_MG_DL
        ]
        if not after:
            continue
        nadir = min(after, key=lambda g: g.mg_dl)
        minutes_to_low = (nadir.ts - shot.ts).total_seconds() / 60.0
        out.append(
            Anomaly(
                name="low_after_correction",
                severity="warning",
                headline=(
                    f"Low after correction: {shot.units:g}U bolus then "
                    f"{nadir.mg_dl} mg/dL after {minutes_to_low:.0f} min"
                ),
                window=window,
                numbers={
                    "bolus_units": shot.units,
                    "nadir_mg_dl": nadir.mg_dl,
                    "minutes_to_low": round(minutes_to_low, 1),
                },
                key=f"low_after_correction:{int(shot.ts.timestamp())}",
            )
        )
    return out


# ── finding adapter ─────────────────────────────────────────────────────────────


def _to_finding(anomaly: Anomaly) -> Finding:
    return Finding(
        agent=AGENT_NAME,
        kind="anomaly",
        scope=anomaly.name,
        headline=anomaly.headline,
        body_md=anomaly.headline,
        evidence={"severity": anomaly.severity, "key": anomaly.key, **anomaly.numbers},
        confidence=1.0,
        status=FindingStatus.ACTIVE,
        window_start=anomaly.window[0],
        window_end=anomaly.window[1],
    )
