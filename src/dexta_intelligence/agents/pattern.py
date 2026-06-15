"""Pattern Agent — deterministic correlators and detectors, rigor-gated.

Sub-checks (each skipped when required data is absent):
1. ``pattern_tod_drift`` - overnight (00-06 UTC) mean deviation, first vs second
   window half.
2. ``pattern_weekday_weekend`` — weekday vs weekend TIR (requires
   ``patterns_weekday_tod`` capability).
3. ``pattern_post_meal_outlier`` — 2h post-meal rise by time-of-day bucket.
4. ``pattern_episode_frequency`` — hypo episode rate, first vs second window half.
5. ``pattern_sleep_glucose`` — sleep duration or recovery score vs next-day glucose.

FDR family correction
---------------------
Every sub-check that clears the power gate contributes one hypothesis. Raw
permutation p-values are collected, then :func:`~dexta_intelligence.stats.rigor.benjamini_hochberg`
is applied once across that family. A finding is emitted only when its adjusted
``q`` ≤ alpha **and** split-half replication agrees on direction — matching
reconciliation's ``verdict == "pass"`` bar.

No LLM imports in this module.
"""

from __future__ import annotations

import logging
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

from dexta_intelligence.agents.base import (
    AgentContext,
    AgentRegistry,
    DataRequirement,
)
from dexta_intelligence.agents.observation import EPISODE_MIN_DURATION_MINUTES
from dexta_intelligence.analytics.rollups import TARGET_LOW_MG_DL, daily_rollup
from dexta_intelligence.models import Finding, FindingStats
from dexta_intelligence.stats.rigor import (
    benjamini_hochberg,
    mean_difference,
    permutation_pvalue,
    power_gate,
    split_half_replication,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dexta_intelligence.models import GlucoseEvent

logger = logging.getLogger(__name__)

__all__ = [
    "AGENT_NAME",
    "PatternAgent",
    "pattern_agent",
    "register_pattern",
]

AGENT_NAME = "pattern"
_RIGOR_SEED = 42
_FDR_ALPHA = 0.10
_MIN_PER_GROUP = 8
_MIN_TOTAL = 16
_MIN_PER_HALF = 3
_OVERNIGHT_START_HOUR = 0
_OVERNIGHT_END_HOUR = 6
_POST_MEAL_HOURS = 2.0
_MEAL_BASELINE_MINUTES = 15
_MIN_MEALS_PER_BUCKET = 8
_POOR_SLEEP_SCORE = 55.0


@dataclass(frozen=True, slots=True)
class _PatternCandidate:
    kind: str
    group_a: tuple[float, ...]
    group_b: tuple[float, ...]
    headline: str
    body_md: str
    evidence: dict[str, object]
    scope: str = "pattern_analysis"


class PatternAgent:
    """Deterministic pattern agent."""

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
            min_glucose_coverage_pct=50.0,
        )

    def run(self, ctx: AgentContext) -> list[Finding]:
        window_start, window_end = _window_datetimes(ctx)
        glucose = ctx.store.get_glucose(window_start, window_end)
        if len(glucose) < 2:
            return []

        candidates: list[_PatternCandidate] = []
        _collect(candidates, _check_tod_drift(glucose, window_start, window_end))
        _collect(candidates, _check_weekday_weekend(ctx, glucose, window_start, window_end))
        _collect(candidates, _check_post_meal_outlier(ctx, glucose, window_start, window_end))
        _collect(candidates, _check_episode_frequency(glucose, window_start, window_end))
        _collect(candidates, _check_sleep_glucose(ctx, glucose, window_start, window_end))

        return _candidates_to_findings(candidates, window_start, window_end)


pattern_agent = PatternAgent()


def register_pattern(registry: AgentRegistry) -> None:
    """Register :data:`pattern_agent` on ``registry``."""
    registry.register(pattern_agent)


def _window_datetimes(ctx: AgentContext) -> tuple[datetime, datetime]:
    start_day, end_day = ctx.window
    start = datetime(start_day.year, start_day.month, start_day.day, tzinfo=UTC)
    end = datetime(end_day.year, end_day.month, end_day.day, tzinfo=UTC) + timedelta(days=1)
    return start, end


def _collect(target: list[_PatternCandidate], candidate: _PatternCandidate | None) -> None:
    if candidate is not None:
        target.append(candidate)


def _midpoint_datetime(start: datetime, end: datetime) -> datetime:
    return start + (end - start) / 2


def _overnight_means_by_day(
    glucose: Sequence[GlucoseEvent],
) -> dict[date, float]:
    by_day: dict[date, list[int]] = defaultdict(list)
    for g in glucose:
        if _OVERNIGHT_START_HOUR <= g.ts.hour < _OVERNIGHT_END_HOUR:
            by_day[g.ts.date()].append(g.mg_dl)
    return {day: statistics.fmean(vals) for day, vals in by_day.items() if vals}


def _daily_tir(
    glucose: Sequence[GlucoseEvent],
    window_start: datetime,
    window_end: datetime,
) -> dict[date, float]:
    out: dict[date, float] = {}
    day = window_start.date()
    end_day = (window_end - timedelta(seconds=1)).date()
    while day <= end_day:
        rollup = daily_rollup(day, glucose)
        if rollup is not None and rollup.tir is not None:
            out[day] = rollup.tir
        day += timedelta(days=1)
    return out


def _split_by_half(
    series: dict[date, float],
    midpoint: datetime,
) -> tuple[list[float], list[float]]:
    mid_day = midpoint.date()
    first = [v for d, v in sorted(series.items()) if d < mid_day]
    second = [v for d, v in sorted(series.items()) if d >= mid_day]
    return first, second


def _check_tod_drift(
    glucose: Sequence[GlucoseEvent],
    window_start: datetime,
    window_end: datetime,
) -> _PatternCandidate | None:
    overnight = _overnight_means_by_day(glucose)
    if len(overnight) < 4:
        logger.debug("pattern_tod_drift skipped: insufficient overnight nights")
        return None

    window_mean = statistics.fmean(g.mg_dl for g in glucose)
    deviations = {day: mean - window_mean for day, mean in overnight.items()}
    midpoint = _midpoint_datetime(window_start, window_end)
    first, second = _split_by_half(deviations, midpoint)
    if len(first) < 2 or len(second) < 2:
        return None

    effect = mean_difference(first, second)
    return _PatternCandidate(
        kind="pattern_tod_drift",
        group_a=tuple(first),
        group_b=tuple(second),
        headline=(
            f"Overnight glucose deviation shifted {effect:+.1f} mg/dL "
            f"(first vs second window half)"
        ),
        body_md=(
            f"Overnight (00-06 UTC) mean deviation from window mean: "
            f"first half {statistics.fmean(first):+.1f} mg/dL, "
            f"second half {statistics.fmean(second):+.1f} mg/dL "
            f"(Δ {effect:+.1f} mg/dL across {len(first)} vs {len(second)} nights)."
        ),
        evidence={
            "check": "tod_drift",
            "overnight_window_utc": f"{_OVERNIGHT_START_HOUR:02d}-{_OVERNIGHT_END_HOUR:02d}",
            "window_mean_mg_dl": round(window_mean, 1),
            "first_half_mean_deviation_mg_dl": round(statistics.fmean(first), 1),
            "second_half_mean_deviation_mg_dl": round(statistics.fmean(second), 1),
            "effect_mg_dl": round(effect, 1),
            "n_first_half_nights": len(first),
            "n_second_half_nights": len(second),
        },
    )


def _check_weekday_weekend(
    ctx: AgentContext,
    glucose: Sequence[GlucoseEvent],
    window_start: datetime,
    window_end: datetime,
) -> _PatternCandidate | None:
    if not ctx.gates.allows("patterns_weekday_tod"):
        logger.debug("pattern_weekday_weekend skipped: patterns_weekday_tod locked")
        return None

    daily = _daily_tir(glucose, window_start, window_end)
    weekday: list[float] = []
    weekend: list[float] = []
    for day, tir in daily.items():
        if day.weekday() < 5:
            weekday.append(tir)
        else:
            weekend.append(tir)

    if len(weekday) < 3 or len(weekend) < 2:
        logger.debug("pattern_weekday_weekend skipped: insufficient weekday/weekend days")
        return None

    effect = mean_difference(weekday, weekend)
    return _PatternCandidate(
        kind="pattern_weekday_weekend",
        group_a=tuple(weekday),
        group_b=tuple(weekend),
        headline=(
            f"Weekday TIR differs from weekend by {effect:+.1f} percentage points"
        ),
        body_md=(
            f"Weekday mean TIR {statistics.fmean(weekday):.1f}% "
            f"({len(weekday)} days) vs weekend {statistics.fmean(weekend):.1f}% "
            f"({len(weekend)} days), Δ {effect:+.1f} points."
        ),
        evidence={
            "check": "weekday_weekend",
            "weekday_mean_tir_pct": round(statistics.fmean(weekday), 1),
            "weekend_mean_tir_pct": round(statistics.fmean(weekend), 1),
            "effect_tir_pct": round(effect, 1),
            "n_weekday_days": len(weekday),
            "n_weekend_days": len(weekend),
        },
    )


def _glucose_index(
    glucose: Sequence[GlucoseEvent],
) -> list[tuple[datetime, int]]:
    return sorted((g.ts, g.mg_dl) for g in glucose)


def _reading_near(
    index: list[tuple[datetime, int]],
    target: datetime,
    *,
    tolerance_min: float,
) -> float | None:
    tol = timedelta(minutes=tolerance_min)
    candidates = [v for ts, v in index if target - tol <= ts <= target + tol]
    if not candidates:
        return None
    return statistics.fmean(candidates)


def _post_meal_rise(
    index: list[tuple[datetime, int]],
    meal_ts: datetime,
) -> float | None:
    baseline = _reading_near(index, meal_ts, tolerance_min=_MEAL_BASELINE_MINUTES)
    if baseline is None:
        return None
    post_ts = meal_ts + timedelta(hours=_POST_MEAL_HOURS)
    post_val = _reading_near(index, post_ts, tolerance_min=_MEAL_BASELINE_MINUTES)
    if post_val is None:
        return None
    return post_val - baseline


def _meal_hour_bucket(hour: int) -> str:
    if hour < 11:
        return "morning"
    if hour < 16:
        return "midday"
    return "evening"


def _check_post_meal_outlier(
    ctx: AgentContext,
    glucose: Sequence[GlucoseEvent],
    window_start: datetime,
    window_end: datetime,
) -> _PatternCandidate | None:
    meals = ctx.store.get_meals(window_start, window_end)
    if len(meals) < _MIN_MEALS_PER_BUCKET:
        logger.debug("pattern_post_meal_outlier skipped: no meals")
        return None

    index = _glucose_index(glucose)
    rises_by_bucket: dict[str, list[float]] = defaultdict(list)
    all_rises: list[float] = []

    for meal in meals:
        rise = _post_meal_rise(index, meal.ts)
        if rise is None:
            continue
        bucket = _meal_hour_bucket(meal.ts.hour)
        rises_by_bucket[bucket].append(rise)
        all_rises.append(rise)

    if len(all_rises) < _MIN_MEALS_PER_BUCKET:
        return None

    best_bucket: str | None = None
    best_delta = 0.0
    best_a: list[float] = []
    best_b: list[float] = []

    for bucket, bucket_rises in rises_by_bucket.items():
        if len(bucket_rises) < 4:
            continue
        other = [r for b, rs in rises_by_bucket.items() if b != bucket for r in rs]
        if len(other) < 4:
            continue
        delta = mean_difference(bucket_rises, other)
        if best_bucket is None or abs(delta) > abs(best_delta):
            best_bucket = bucket
            best_delta = delta
            best_a = bucket_rises
            best_b = other

    if best_bucket is None:
        return None

    return _PatternCandidate(
        kind="pattern_post_meal_outlier",
        group_a=tuple(best_a),
        group_b=tuple(best_b),
        headline=(
            f"{best_bucket.capitalize()} meals: 2h rise {best_delta:+.1f} mg/dL "
            f"vs other meal windows"
        ),
        body_md=(
            f"2h post-meal glucose rise in {best_bucket} bucket "
            f"(mean {statistics.fmean(best_a):.1f} mg/dL, n={len(best_a)}) "
            f"vs other buckets (mean {statistics.fmean(best_b):.1f} mg/dL, n={len(best_b)})."
        ),
        evidence={
            "check": "post_meal_outlier",
            "meal_bucket": best_bucket,
            "bucket_mean_rise_mg_dl": round(statistics.fmean(best_a), 1),
            "other_mean_rise_mg_dl": round(statistics.fmean(best_b), 1),
            "effect_mg_dl": round(best_delta, 1),
            "n_bucket_meals": len(best_a),
            "n_other_meals": len(best_b),
            "post_meal_hours": _POST_MEAL_HOURS,
        },
    )


def _hypo_episodes_by_day(
    glucose: Sequence[GlucoseEvent],
) -> dict[date, int]:
    """Approximate daily hypo counts via contiguous low runs ≥ 15 min."""
    readings = sorted(glucose, key=lambda g: g.ts)
    counts: dict[date, int] = defaultdict(int)
    in_run = False
    run_start: datetime | None = None
    run_end: datetime | None = None

    def _close() -> None:
        nonlocal in_run, run_start, run_end
        if not in_run or run_start is None or run_end is None:
            return
        duration = (run_end - run_start).total_seconds() / 60.0
        if duration >= EPISODE_MIN_DURATION_MINUTES:
            counts[run_start.date()] += 1
        in_run = False
        run_start = None
        run_end = None

    for reading in readings:
        if reading.mg_dl < TARGET_LOW_MG_DL:
            if in_run:
                run_end = reading.ts
            else:
                in_run = True
                run_start = reading.ts
                run_end = reading.ts
        else:
            _close()
    _close()
    return dict(counts)


def _check_episode_frequency(
    glucose: Sequence[GlucoseEvent],
    window_start: datetime,
    window_end: datetime,
) -> _PatternCandidate | None:
    episode_days = _hypo_episodes_by_day(glucose)
    midpoint = _midpoint_datetime(window_start, window_end)
    mid_day = midpoint.date()

    first: list[float] = []
    second: list[float] = []
    day = window_start.date()
    end_day = (window_end - timedelta(seconds=1)).date()
    while day <= end_day:
        count = float(episode_days.get(day, 0))
        if day < mid_day:
            first.append(count)
        else:
            second.append(count)
        day += timedelta(days=1)

    total_first = int(sum(first))
    total_second = int(sum(second))
    if total_first == 0 and total_second == 0:
        logger.debug("pattern_episode_frequency skipped: no hypo episodes")
        return None

    effect = mean_difference(second, first)

    return _PatternCandidate(
        kind="pattern_episode_frequency",
        group_a=tuple(first),
        group_b=tuple(second),
        headline=(
            f"Hypo episode rate changed {effect:+.3f}/day (second vs first half)"
        ),
        body_md=(
            f"Hypo episodes (≥{EPISODE_MIN_DURATION_MINUTES} min): "
            f"{total_first} in first half, {total_second} in second half. "
            f"Daily rate shift {effect:+.3f} episodes/day."
        ),
        evidence={
            "check": "episode_frequency",
            "hypo_episodes_first_half": total_first,
            "hypo_episodes_second_half": total_second,
            "first_half_rate_per_day": round(statistics.fmean(first), 4),
            "second_half_rate_per_day": round(statistics.fmean(second), 4),
            "effect_rate_per_day": round(effect, 4),
            "n_first_half_days": len(first),
            "n_second_half_days": len(second),
        },
    )


def _day_mean_glucose(glucose: Sequence[GlucoseEvent], day: date) -> float | None:
    day_start = datetime(day.year, day.month, day.day, tzinfo=UTC)
    day_end = day_start + timedelta(days=1)
    vals = [g.mg_dl for g in glucose if day_start <= g.ts < day_end]
    if len(vals) < 12:
        return None
    return statistics.fmean(vals)


def _pair_sleep_signal(
    glucose: Sequence[GlucoseEvent],
    *,
    waking_day: date,
    signal: float,
    threshold: float,
    poor: list[float],
    good: list[float],
) -> None:
    next_mean = _day_mean_glucose(glucose, waking_day)
    if next_mean is None:
        return
    if signal < threshold:
        poor.append(next_mean)
    else:
        good.append(next_mean)


def _check_sleep_glucose(
    ctx: AgentContext,
    glucose: Sequence[GlucoseEvent],
    window_start: datetime,
    window_end: datetime,
) -> _PatternCandidate | None:
    sleep = ctx.store.get_sleep(window_start, window_end)
    recovery = ctx.store.get_recovery(window_start, window_end)
    if not sleep and not recovery:
        logger.debug("pattern_sleep_glucose skipped: no sleep/recovery data")
        return None

    poor: list[float] = []
    good: list[float] = []

    for event in sleep:
        if event.score is not None:
            signal = event.score
            threshold = _POOR_SLEEP_SCORE
        else:
            signal = event.duration_min
            threshold = 360.0
        _pair_sleep_signal(
            glucose,
            waking_day=event.ts_end.date(),
            signal=signal,
            threshold=threshold,
            poor=poor,
            good=good,
        )

    for rec in recovery:
        if rec.score is None:
            continue
        next_day = rec.ts.date() + timedelta(days=1)
        _pair_sleep_signal(
            glucose,
            waking_day=next_day,
            signal=rec.score,
            threshold=_POOR_SLEEP_SCORE,
            poor=poor,
            good=good,
        )

    if len(poor) < 4 or len(good) < 4:
        return None

    effect = mean_difference(poor, good)
    return _PatternCandidate(
        kind="pattern_sleep_glucose",
        group_a=tuple(poor),
        group_b=tuple(good),
        headline=(
            f"Next-day mean glucose {effect:+.1f} mg/dL after poorer sleep/recovery"
        ),
        body_md=(
            f"Mean next-day glucose after poorer nights/scores "
            f"{statistics.fmean(poor):.1f} mg/dL (n={len(poor)}) vs "
            f"{statistics.fmean(good):.1f} mg/dL (n={len(good)})."
        ),
        evidence={
            "check": "sleep_glucose",
            "poor_next_day_mean_mg_dl": round(statistics.fmean(poor), 1),
            "good_next_day_mean_mg_dl": round(statistics.fmean(good), 1),
            "effect_mg_dl": round(effect, 1),
            "n_poor": len(poor),
            "n_good": len(good),
            "poor_sleep_score_threshold": _POOR_SLEEP_SCORE,
        },
    )


def _candidates_to_findings(
    candidates: list[_PatternCandidate],
    window_start: datetime,
    window_end: datetime,
) -> list[Finding]:
    if not candidates:
        return []

    rng = random.Random(_RIGOR_SEED)
    scored: list[tuple[_PatternCandidate, float, str, str]] = []

    for candidate in candidates:
        gate = power_gate(
            [len(candidate.group_a), len(candidate.group_b)],
            min_per_group=_MIN_PER_GROUP,
            min_total=_MIN_TOTAL,
        )
        if not gate.passed:
            logger.debug("pattern %s skipped: %s", candidate.kind, gate.reason)
            continue

        observed = mean_difference(candidate.group_a, candidate.group_b)
        p = permutation_pvalue(
            observed,
            mean_difference,
            candidate.group_a,
            candidate.group_b,
            rng=rng,
        )
        split = split_half_replication(
            mean_difference,
            candidate.group_a,
            candidate.group_b,
            min_per_half=_MIN_PER_HALF,
        )
        scored.append((candidate, p, gate.reason, split.reason))

    if not scored:
        return []

    ps = [p for _, p, _, _ in scored]
    bh = benjamini_hochberg(ps, alpha=_FDR_ALPHA)

    findings: list[Finding] = []
    for idx, (candidate, p, gate_reason, split_reason) in enumerate(scored):
        q = bh.qvalues[idx]
        split = split_half_replication(
            mean_difference,
            candidate.group_a,
            candidate.group_b,
            min_per_half=_MIN_PER_HALF,
        )
        if q > _FDR_ALPHA:
            continue
        if not split.replicated:
            continue

        effect = mean_difference(candidate.group_a, candidate.group_b)
        evidence = dict(candidate.evidence)
        evidence.update(
            {
                "rigor_verdict": "pass",
                "rigor_reasons": [gate_reason, split_reason, split.reason],
                "p_perm": p,
                "q_fdr": q,
                "fdr_family_size": len(ps),
                "fdr_alpha": _FDR_ALPHA,
                "skeptic_group_a": list(candidate.group_a),
                "skeptic_group_b": list(candidate.group_b),
            }
        )
        confidence = 0.75 if split.replicated else 0.55
        findings.append(
            Finding(
                agent=AGENT_NAME,
                kind=candidate.kind,
                scope=candidate.scope,
                headline=candidate.headline,
                body_md=candidate.body_md,
                evidence=evidence,
                stats=FindingStats(
                    effect_size=effect,
                    n=len(candidate.group_a) + len(candidate.group_b),
                    p_perm=p,
                    q_fdr=q,
                    replicated=split.replicated,
                ),
                confidence=confidence,
                window_start=window_start,
                window_end=window_end,
            )
        )
    return findings
