"""Prediction Reconciliation Agent — deterministic forecast-vs-realized analysis.

Compares expected glucose trajectories (logged predBG curves for looping users,
or oref0-math expectations otherwise) against realized CGM, attributes the error,
counts recurrence, and gates claims through :mod:`dexta_intelligence.stats.rigor`.

Spec §7.1. No LLM imports in this module.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from dexta_intelligence.agents.base import (
    AgentContext,
    AgentRegistry,
    DataRequirement,
)
from dexta_intelligence.analytics.oref import (
    carbs_on_board,
    deviation_series,
    predict_glucose,
    temp_basal_to_microboluses,
)
from dexta_intelligence.models import (
    Finding,
    FindingStats,
    FindingStatus,
    InsulinKind,
    PredictionEvent,
)
from dexta_intelligence.stats.rigor import assess, mean_difference

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dexta_intelligence.models import GlucoseEvent, InsulinEvent, MealEvent

__all__ = [
    "AGENT_NAME",
    "EPISODE_ERROR_THRESHOLD_MG_DL",
    "EPISODE_MIN_HORIZON_MIN",
    "ContributorKind",
    "PredictionReconciliationAgent",
    "ReconciliationEpisode",
    "ReconciliationTier",
    "reconciliation_agent",
    "register_reconciliation",
]

AGENT_NAME = "reconciliation"

#: Signed error (actual - predicted) must exceed this at or beyond
#: :data:`EPISODE_MIN_HORIZON_MIN` to count as a reconciliation episode.
EPISODE_ERROR_THRESHOLD_MG_DL = 30.0

#: Minimum forecast horizon (minutes from cycle time) for episode detection.
EPISODE_MIN_HORIZON_MIN = 30

#: IOB curve mean absolute error below this (mg/dL) is treated as "IOB fine".
IOB_FINE_THRESHOLD_MG_DL = 15.0

#: UAM must beat COB by at least this margin (mg/dL) to attribute carb error.
CURVE_FIT_MARGIN_MG_DL = 5.0

#: Default oref profile for Tier B expectations (analysis only, never dosing).
_DEFAULT_ISF = 50.0
_DEFAULT_CARB_RATIO = 10.0
_DEFAULT_HORIZON_MIN = 120.0
_RIGOR_SEED = 42
_STEP = timedelta(minutes=5)

ContributorKind = Literal[
    "carb_underestimate",
    "sensitivity_shift",
    "absorption_timing",
    "unclassified_mismatch",
]
ReconciliationTier = Literal["A", "B"]

_OREF_CURVES = ("iob", "cob", "uam", "zt")


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ReconciliationEpisode(_FrozenModel):
    """One forecast miss localized to a cycle and horizon."""

    cycle_ts: datetime
    horizon_min: int
    signed_error_mg_dl: float
    contributor: ContributorKind
    tier: ReconciliationTier
    best_curve: str
    curve_errors: dict[str, float] = Field(default_factory=dict)


class PredictionReconciliationAgent:
    """Deterministic agent implementing spec §7.1."""

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
            min_span_days=1.0,
            min_glucose_coverage_pct=50.0,
        )

    def run(self, ctx: AgentContext) -> list[Finding]:
        window_start, window_end = _window_datetimes(ctx)
        glucose = ctx.store.get_glucose(window_start, window_end)
        if len(glucose) < 2:
            return []

        predictions = ctx.store.get_predictions(window_start, window_end)
        tier: ReconciliationTier = "A" if predictions else "B"
        if tier == "B":
            insulin = ctx.store.get_insulin(window_start, window_end)
            if not insulin:
                return []
            meals = ctx.store.get_meals(window_start, window_end)
            cycles = _tier_b_cycles(glucose, insulin, meals)
        else:
            cycles = _group_prediction_cycles(predictions)

        if not cycles:
            return []

        glucose_map = {g.ts: g.mg_dl for g in glucose}
        episodes = _detect_episodes(cycles, glucose_map, tier=tier)
        if not episodes:
            return []

        baseline_errors = _baseline_errors(cycles, glucose_map)
        return _episodes_to_findings(
            episodes,
            baseline_errors,
            ctx,
            window_start,
            window_end,
            tier=tier,
        )


reconciliation_agent = PredictionReconciliationAgent()


def register_reconciliation(registry: AgentRegistry) -> None:
    """Register :data:`reconciliation_agent` on ``registry``."""
    registry.register(reconciliation_agent)


def _window_datetimes(ctx: AgentContext) -> tuple[datetime, datetime]:
    start_day, end_day = ctx.window
    start = datetime(start_day.year, start_day.month, start_day.day, tzinfo=UTC)
    end = datetime(end_day.year, end_day.month, end_day.day, tzinfo=UTC) + timedelta(days=1)
    return start, end


def _group_prediction_cycles(
    predictions: Iterable[PredictionEvent],
) -> dict[datetime, dict[str, PredictionEvent]]:
    cycles: dict[datetime, dict[str, PredictionEvent]] = defaultdict(dict)
    for pred in predictions:
        cycles[pred.ts][pred.curve_kind] = pred
    return dict(cycles)


def _build_doses(insulin: Iterable[InsulinEvent]) -> list[tuple[datetime, float]]:
    doses: list[tuple[datetime, float]] = []
    for event in insulin:
        if event.kind is InsulinKind.BOLUS and event.units:
            doses.append((event.ts, event.units))
        elif (
            event.kind is InsulinKind.TEMP_BASAL
            and event.units is not None
            and event.duration_min
        ):
            end = event.ts + timedelta(minutes=event.duration_min)
            doses.extend(temp_basal_to_microboluses(event.ts, end, event.units, 0.0))
    return doses


def _tier_b_cycles(
    glucose: list[GlucoseEvent],
    insulin: list[InsulinEvent],
    meals: list[MealEvent],
) -> dict[datetime, dict[str, PredictionEvent]]:
    doses = _build_doses(insulin)
    series = [(g.ts, float(g.mg_dl)) for g in glucose]
    deviations = dict(deviation_series(series, doses, _DEFAULT_ISF))
    cycles: dict[datetime, dict[str, PredictionEvent]] = {}

    for idx, g in enumerate(glucose):
        if idx + EPISODE_MIN_HORIZON_MIN // 5 >= len(glucose):
            continue
        ts = g.ts
        bg = float(g.mg_dl)
        cob_g = _total_cob_at(meals, series[: idx + 1], doses, ts)
        dev = deviations.get(ts, 0.0)
        curves = predict_glucose(
            bg,
            doses,
            ts,
            _DEFAULT_ISF,
            horizon_min=_DEFAULT_HORIZON_MIN,
            carb_ratio=_DEFAULT_CARB_RATIO,
            cob_g=cob_g,
            deviation_5m=dev,
        )
        cycles[ts] = {
            kind: PredictionEvent(
                ts=ts,
                source="oref",
                curve_kind=kind,  # type: ignore[arg-type]
                values_mg_dl=getattr(curves, kind),
            )
            for kind in _OREF_CURVES
        }
    return cycles


def _total_cob_at(
    meals: list[MealEvent],
    glucose_prefix: list[tuple[datetime, float]],
    doses: list[tuple[datetime, float]],
    at: datetime,
) -> float:
    total = 0.0
    for meal in meals:
        if meal.carbs_g and meal.ts <= at <= meal.ts + timedelta(minutes=360):
            result = carbs_on_board(
                meal.carbs_g,
                meal.ts,
                glucose_prefix,
                doses,
                _DEFAULT_ISF,
                _DEFAULT_CARB_RATIO,
                at,
            )
            total += result.cob_g
    return total


def _horizon_errors(
    pred: PredictionEvent,
    glucose_map: dict[datetime, int],
) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    for step, predicted in enumerate(pred.values_mg_dl):
        ts = pred.ts + _STEP * step
        actual = glucose_map.get(ts)
        if actual is not None:
            out.append((step * 5, float(actual) - predicted))
    return out


def _detect_episodes(
    cycles: dict[datetime, dict[str, PredictionEvent]],
    glucose_map: dict[datetime, int],
    *,
    tier: ReconciliationTier,
) -> list[ReconciliationEpisode]:
    episodes: list[ReconciliationEpisode] = []

    for cycle_ts, curves in sorted(cycles.items()):
        errors_by_curve: dict[str, list[tuple[int, float]]] = {
            kind: _horizon_errors(curves[kind], glucose_map) for kind in curves
        }
        if not errors_by_curve:
            continue

        peak_horizon = 0
        peak_abs = 0.0
        peak_errors: dict[str, float] = {}
        for _kind, pairs in errors_by_curve.items():
            for horizon, err in pairs:
                if horizon < EPISODE_MIN_HORIZON_MIN:
                    continue
                if abs(err) > peak_abs:
                    peak_abs = abs(err)
                    peak_horizon = horizon
                    peak_errors = _errors_at_horizon(errors_by_curve, horizon)

        if peak_abs < EPISODE_ERROR_THRESHOLD_MG_DL:
            continue

        contributor = _attribute(peak_errors)
        worst_curve = max(peak_errors, key=lambda k: abs(peak_errors[k]))
        signed = peak_errors[worst_curve]
        episodes.append(
            ReconciliationEpisode(
                cycle_ts=cycle_ts,
                horizon_min=peak_horizon,
                signed_error_mg_dl=signed,
                contributor=contributor,
                tier=tier,
                best_curve=worst_curve,
                curve_errors=peak_errors,
            )
        )
    return episodes


def _errors_at_horizon(
    errors_by_curve: dict[str, list[tuple[int, float]]],
    horizon: int,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for kind, pairs in errors_by_curve.items():
        for h, err in pairs:
            if h == horizon:
                out[kind] = err
                break
    return out


def _attribute(errors: dict[str, float]) -> ContributorKind:
    iob = errors.get("iob")
    cob = errors.get("cob")
    uam = errors.get("uam")
    zt = errors.get("zt")

    if (
        cob is not None
        and uam is not None
        and abs(uam) + CURVE_FIT_MARGIN_MG_DL < abs(cob)
        and cob >= EPISODE_ERROR_THRESHOLD_MG_DL
    ):
        return "carb_underestimate"

    if (
        iob is not None
        and cob is not None
        and abs(iob) < IOB_FINE_THRESHOLD_MG_DL
        and abs(cob) >= EPISODE_ERROR_THRESHOLD_MG_DL
    ):
        return "absorption_timing"

    significant = [
        e for e in (iob, cob, uam, zt) if e is not None and abs(e) >= EPISODE_ERROR_THRESHOLD_MG_DL
    ]
    if len(significant) >= 3:
        signs = [math.copysign(1.0, e) for e in significant]
        if all(s == signs[0] for s in signs):
            return "sensitivity_shift"

    return "unclassified_mismatch"


def _baseline_errors(
    cycles: dict[datetime, dict[str, PredictionEvent]],
    glucose_map: dict[datetime, int],
) -> list[float]:
    """Near-null forecast errors used as the rigor comparison group."""
    baseline: list[float] = []
    for curves in cycles.values():
        ref = curves.get("iob") or next(iter(curves.values()))
        for horizon, err in _horizon_errors(ref, glucose_map):
            if horizon >= EPISODE_MIN_HORIZON_MIN and abs(err) < EPISODE_ERROR_THRESHOLD_MG_DL:
                baseline.append(err)
    return baseline


def _prior_recurrence_count(
    ctx: AgentContext,
    contributor: ContributorKind,
) -> int:
    prior = ctx.store.get_findings(
        agent=AGENT_NAME,
        kind=f"prediction_miss_{contributor}",
        status=FindingStatus.ACTIVE,
        limit=500,
    )
    return len(prior)


def _episodes_to_findings(
    episodes: list[ReconciliationEpisode],
    baseline_errors: list[float],
    ctx: AgentContext,
    window_start: datetime,
    window_end: datetime,
    *,
    tier: ReconciliationTier,
) -> list[Finding]:
    by_contributor: dict[ContributorKind, list[ReconciliationEpisode]] = defaultdict(list)
    for ep in episodes:
        if ep.contributor == "unclassified_mismatch":
            continue
        by_contributor[ep.contributor].append(ep)

    findings: list[Finding] = []
    rng = random.Random(_RIGOR_SEED)

    for contributor, group in by_contributor.items():
        episode_errors = [ep.signed_error_mg_dl for ep in group]
        null_group = list(baseline_errors) if baseline_errors else [0.0] * len(episode_errors)
        if len(null_group) < 8:
            null_group.extend([0.0] * (8 - len(null_group)))

        verdict = assess(
            episode_errors,
            null_group,
            rng=rng,
            min_per_group=min(8, len(episode_errors), len(null_group)),
            min_total=min(16, len(episode_errors) + len(null_group)),
            min_per_half=min(3, max(1, len(episode_errors) // 2)),
        )
        if verdict.verdict == "fail":
            continue

        recurrence = _prior_recurrence_count(ctx, contributor)
        total_occurrences = recurrence + 1
        tier_label = "Tier A (logged predBG curves)" if tier == "A" else (
            "Tier B (oref-computed expectations; weaker evidence)"
        )
        representative = max(group, key=lambda e: abs(e.signed_error_mg_dl))
        sign = "+" if representative.signed_error_mg_dl >= 0 else ""
        headline = (
            f"Forecast miss: {contributor.replace('_', ' ')} "
            f"({sign}{representative.signed_error_mg_dl:.0f} mg/dL at "
            f"{representative.horizon_min} min, {tier_label})"
        )
        recurrence_note = (
            f"Similar pattern, {total_occurrences} occurrence(s) including this run."
            if total_occurrences > 1
            else ""
        )
        body = (
            f"Reconciled realized CGM against {'logged' if tier == 'A' else 'computed'} "
            f"prediction curves. {tier_label}. "
            f"Representative episode at {representative.cycle_ts.isoformat()}: "
            f"{sign}{representative.signed_error_mg_dl:.0f} mg/dL error at "
            f"{representative.horizon_min} min (contributor={contributor}). "
            f"{recurrence_note} "
            "Retrospective analysis only — not dosing advice."
        ).strip()

        confidence = 0.75 if verdict.verdict == "pass" else 0.55
        if tier == "B":
            confidence = min(confidence, 0.6)

        findings.append(
            Finding(
                agent=AGENT_NAME,
                kind=f"prediction_miss_{contributor}",
                scope="prediction_reconciliation",
                headline=headline,
                body_md=body,
                evidence={
                    "tier": tier,
                    "contributor": contributor,
                    "n_episodes": len(group),
                    "recurrence_count": total_occurrences,
                    "episodes": [ep.model_dump(mode="json") for ep in group],
                    "rigor_verdict": verdict.verdict,
                    "rigor_reasons": list(verdict.reasons),
                },
                stats=FindingStats(
                    effect_size=mean_difference(
                        episode_errors, null_group[: len(episode_errors)]
                    ),
                    n=len(episode_errors) + len(null_group),
                    p_perm=verdict.p,
                    q_fdr=verdict.q,
                    replicated=verdict.replicated,
                ),
                confidence=confidence,
                window_start=window_start,
                window_end=window_end,
            )
        )
    return findings
