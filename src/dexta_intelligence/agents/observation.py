"""Observation Agent — deterministic facts-only window summary.

Computes glycemic metrics, coverage, episode counts, optional insulin totals,
and wearable presence from the analysis window. Emits direct measurements only —
no interpretation, no LLM, no rigor gate.

Finding kinds (documented contract)
-----------------------------------
- ``observation_glycemic`` — TIR/TBR/TAR, mean, GMI, CV, coverage, hypo/hyper
  episode counts for the window.
- ``observation_insulin`` — mean daily bolus and basal totals when insulin
  events exist in the window.
- ``observation_wearables`` — sleep and recovery event counts (presence facts).

Spec §7. No LLM imports in this module.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Literal

from dexta_intelligence.agents.base import (
    AgentContext,
    AgentRegistry,
    DataRequirement,
)
from dexta_intelligence.analytics.rollups import (
    EXPECTED_READINGS_PER_DAY,
    TARGET_HIGH_MG_DL,
    TARGET_LOW_MG_DL,
    VERY_HIGH_MG_DL,
    VERY_LOW_MG_DL,
    coverage_fraction,
)
from dexta_intelligence.models import Finding, FindingStatus, InsulinKind
from dexta_intelligence.stats.core import summarize

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dexta_intelligence.models import GlucoseEvent, InsulinEvent

__all__ = [
    "AGENT_NAME",
    "EPISODE_MIN_DURATION_MINUTES",
    "ObservationAgent",
    "observation_agent",
    "register_observation",
]

AGENT_NAME = "observation"

#: Minimum contiguous out-of-range duration to count as an episode (MCP contract).
EPISODE_MIN_DURATION_MINUTES = 15

#: GMI affine map (Bergenstal et al. 2018) — mirrors rollups daily kernel.
_GMI_INTERCEPT = 3.31
_GMI_SLOPE = 0.02392

_EpisodeKind = Literal["hypo", "hyper"]


class ObservationAgent:
    """Facts-only agent implementing spec §7 Observation row."""

    name: str
    requires: DataRequirement

    def __init__(
        self,
        *,
        name: str = AGENT_NAME,
        requires: DataRequirement | None = None,
    ) -> None:
        self.name = name
        self.requires = requires or DataRequirement(
            min_span_days=7.0,
            min_glucose_coverage_pct=70.0,
        )

    def run(self, ctx: AgentContext) -> list[Finding]:
        window_start, window_end = _window_datetimes(ctx)
        glucose = ctx.store.get_glucose(window_start, window_end)
        if not glucose:
            return []

        findings: list[Finding] = []
        glycemic = _glycemic_finding(glucose, window_start, window_end)
        if glycemic is not None:
            findings.append(glycemic)

        insulin = ctx.store.get_insulin(window_start, window_end)
        if insulin:
            ins_finding = _insulin_finding(insulin, window_start, window_end)
            if ins_finding is not None:
                findings.append(ins_finding)

        sleep = ctx.store.get_sleep(window_start, window_end)
        recovery = ctx.store.get_recovery(window_start, window_end)
        if sleep or recovery:
            findings.append(_wearables_finding(sleep, recovery, window_start, window_end))

        return findings


observation_agent = ObservationAgent()


def register_observation(registry: AgentRegistry) -> None:
    """Register :data:`observation_agent` on ``registry``."""
    registry.register(observation_agent)


def _window_datetimes(ctx: AgentContext) -> tuple[datetime, datetime]:
    start_day, end_day = ctx.window
    start = datetime(start_day.year, start_day.month, start_day.day, tzinfo=UTC)
    end = datetime(end_day.year, end_day.month, end_day.day, tzinfo=UTC) + timedelta(days=1)
    return start, end


def _span_days(start: datetime, end: datetime) -> float:
    return max((end - start).total_seconds() / 86400.0, 0.0)


def _pct(count: int, total: int) -> float:
    return 100.0 * count / total


def _gmi(mean_mg_dl: float) -> float:
    return round(_GMI_INTERCEPT + _GMI_SLOPE * mean_mg_dl, 1)


def _count_episodes(
    glucose: Sequence[GlucoseEvent],
    *,
    target_low: int = TARGET_LOW_MG_DL,
    target_high: int = TARGET_HIGH_MG_DL,
) -> dict[str, int]:
    """Count hypo/hyper episodes with minimum duration (15 min at 5-min cadence)."""
    readings = sorted(glucose, key=lambda g: g.ts)
    hypo = 0
    hyper = 0
    current_kind: _EpisodeKind | None = None
    run_start: datetime | None = None
    run_end: datetime | None = None

    def _close_run() -> None:
        nonlocal hypo, hyper, current_kind, run_start, run_end
        if current_kind is None or run_start is None or run_end is None:
            return
        duration = (run_end - run_start).total_seconds() / 60.0
        if duration >= EPISODE_MIN_DURATION_MINUTES:
            if current_kind == "hypo":
                hypo += 1
            else:
                hyper += 1
        current_kind = None
        run_start = None
        run_end = None

    for reading in readings:
        mg_dl = reading.mg_dl
        if mg_dl < target_low:
            kind: _EpisodeKind | None = "hypo"
        elif mg_dl > target_high:
            kind = "hyper"
        else:
            kind = None

        if kind is None:
            _close_run()
            continue

        if current_kind == kind:
            run_end = reading.ts
        else:
            _close_run()
            current_kind = kind
            run_start = reading.ts
            run_end = reading.ts

    _close_run()
    return {"hypo": hypo, "hyper": hyper}


def _glycemic_finding(
    glucose: Sequence[GlucoseEvent],
    window_start: datetime,
    window_end: datetime,
) -> Finding | None:
    values = [float(g.mg_dl) for g in sorted(glucose, key=lambda g: g.ts)]
    stats = summarize(values)
    if stats.n == 0 or stats.mean is None:
        return None

    n = stats.n
    tbr_count = sum(1 for v in values if v < TARGET_LOW_MG_DL)
    tar_count = sum(1 for v in values if v > TARGET_HIGH_MG_DL)
    tbr2_count = sum(1 for v in values if v < VERY_LOW_MG_DL)
    tar2_count = sum(1 for v in values if v > VERY_HIGH_MG_DL)
    tir = _pct(n - tbr_count - tar_count, n)

    expected = max(1, round(_span_days(window_start, window_end) * EXPECTED_READINGS_PER_DAY))
    coverage_pct = round(coverage_fraction(n, expected=expected) * 100.0, 1)
    episodes = _count_episodes(glucose)

    evidence = {
        "n_readings": n,
        "coverage_pct": coverage_pct,
        "mean_mg_dl": round(stats.mean, 1),
        "gmi_pct": _gmi(stats.mean),
        "cv_pct": round(stats.cv_pct, 1) if stats.cv_pct is not None else None,
        "tir_pct": round(tir, 1),
        "tbr_pct": round(_pct(tbr_count, n), 1),
        "tbr2_pct": round(_pct(tbr2_count, n), 1),
        "tar_pct": round(_pct(tar_count, n), 1),
        "tar2_pct": round(_pct(tar2_count, n), 1),
        "hypo_episode_count": episodes["hypo"],
        "hyper_episode_count": episodes["hyper"],
        "target_low_mg_dl": TARGET_LOW_MG_DL,
        "target_high_mg_dl": TARGET_HIGH_MG_DL,
    }

    return Finding(
        agent=AGENT_NAME,
        kind="observation_glycemic",
        scope="observation",
        headline=(
            f"Window glycemic summary: {evidence['tir_pct']:.1f}% TIR, "
            f"mean {evidence['mean_mg_dl']:.1f} mg/dL, GMI {evidence['gmi_pct']:.1f}%"
        ),
        body_md=(
            f"Computed over {n} readings ({coverage_pct:.1f}% coverage). "
            f"TIR {evidence['tir_pct']:.1f}%, TBR {evidence['tbr_pct']:.1f}%, "
            f"TAR {evidence['tar_pct']:.1f}%. "
            f"Mean {evidence['mean_mg_dl']:.1f} mg/dL, CV "
            f"{evidence['cv_pct'] if evidence['cv_pct'] is not None else 'n/a'}%. "
            f"Hypo episodes (≥{EPISODE_MIN_DURATION_MINUTES} min): {episodes['hypo']}; "
            f"hyper episodes: {episodes['hyper']}."
        ),
        evidence=evidence,
        confidence=1.0,
        status=FindingStatus.ACTIVE,
        window_start=window_start,
        window_end=window_end,
    )


def _insulin_finding(
    insulin: Sequence[InsulinEvent],
    window_start: datetime,
    window_end: datetime,
) -> Finding | None:
    bolus_by_day: dict[date, float] = defaultdict(float)
    basal_by_day: dict[date, float] = defaultdict(float)
    for event in insulin:
        if event.units is None:
            continue
        day = event.ts.date()
        if event.kind is InsulinKind.BOLUS:
            bolus_by_day[day] += event.units
        elif event.kind in (InsulinKind.BASAL, InsulinKind.TEMP_BASAL):
            basal_by_day[day] += event.units

    bolus_days = [v for v in bolus_by_day.values() if v > 0]
    basal_days = [v for v in basal_by_day.values() if v > 0]
    if not bolus_days and not basal_days:
        return None

    mean_bolus = round(statistics.fmean(bolus_days), 2) if bolus_days else None
    mean_basal = round(statistics.fmean(basal_days), 2) if basal_days else None
    evidence: dict[str, float | int | None] = {
        "days_with_bolus": len(bolus_days),
        "days_with_basal": len(basal_days),
        "mean_bolus_units_per_day": mean_bolus,
        "mean_basal_units_per_day": mean_basal,
    }

    parts = []
    if mean_bolus is not None:
        parts.append(f"mean bolus {mean_bolus:.2f} U/day over {len(bolus_days)} day(s)")
    if mean_basal is not None:
        parts.append(f"mean basal {mean_basal:.2f} U/day over {len(basal_days)} day(s)")

    return Finding(
        agent=AGENT_NAME,
        kind="observation_insulin",
        scope="observation",
        headline=f"Insulin totals: {', '.join(parts)}",
        body_md=f"Daily insulin totals in window. {', '.join(parts)}.",
        evidence=evidence,
        confidence=1.0,
        status=FindingStatus.ACTIVE,
        window_start=window_start,
        window_end=window_end,
    )


def _wearables_finding(
    sleep: Sequence[object],
    recovery: Sequence[object],
    window_start: datetime,
    window_end: datetime,
) -> Finding:
    n_sleep = len(sleep)
    n_recovery = len(recovery)
    evidence = {
        "n_sleep_events": n_sleep,
        "n_recovery_events": n_recovery,
    }
    return Finding(
        agent=AGENT_NAME,
        kind="observation_wearables",
        scope="observation",
        headline=(
            f"Wearable events in window: {n_sleep} sleep, {n_recovery} recovery"
        ),
        body_md=(
            f"Sleep events: {n_sleep}. Recovery events: {n_recovery}."
        ),
        evidence=evidence,
        confidence=1.0,
        status=FindingStatus.ACTIVE,
        window_start=window_start,
        window_end=window_end,
    )
