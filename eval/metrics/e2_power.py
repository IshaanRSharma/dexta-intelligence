"""E2 statistical power - true-discovery rate of the rigor-gated pattern agent.

The twin of E4's false-discovery rate. E4 plants nothing and asks "does the
agent stay silent?"; E2 plants a known effect at a controlled size and asks
"does the agent surface it?". Ground truth is non-LLM: the synthetic generator
records exactly what it planted (the scenario manifest), so a planted-effect's
target finding ``kind`` is known in advance.

For each (effect-size, data-span) cell we run several seeds, count how often the
agent emits the matching finding kind, and report that recall. A useful,
clinically-valid harness must have *high recall at a large effect and adequate
span* - otherwise it is merely conservative, not informative.

Effect → expected finding kind:

- insulin-sensitivity shift → ``pattern_tod_drift`` (first-vs-second-half drift)
- all-weekday breakfast spike → ``pattern_weekday_weekend`` (weekday TIR worse
  than weekend; the spike is planted on every weekday so the weekday/weekend
  contrast the agent computes is real)

Both are detectable from glucose alone (no gated capabilities required) and map
onto group comparisons the pattern agent already runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.agents.pattern import pattern_agent
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.models import CoverageStats
from dexta_intelligence.store import SQLiteStore
from dexta_intelligence.testing.synthetic import (
    DEFAULT_START,
    SensitivityRegimeShift,
    WeekdayBreakfastSpike,
    generate_dataset,
)
from dexta_intelligence.testing.synthetic import (
    Effect as _Effect,
)

__all__ = ["E2CellResult", "E2PowerResult", "run_e2_power"]

#: Recall the harness must clear in its strongest cell (large effect, long span).
_RECALL_TARGET = 0.8

#: (effect_size mg/dL, n_days) cells swept, weakest to strongest.
_DEFAULT_CELLS: tuple[tuple[float, int], ...] = (
    (15.0, 30),
    (30.0, 60),
    (50.0, 90),
)


@dataclass(frozen=True, slots=True)
class E2CellResult:
    """Recall for one (effect, scenario, span) cell across seeds."""

    scenario: str
    target_kind: str
    effect_size: float
    n_days: int
    n_seeds: int
    n_recovered: int
    recall: float


@dataclass(frozen=True, slots=True)
class E2PowerResult:
    """Outcome of one E2 sweep over effect sizes and spans."""

    cells: tuple[E2CellResult, ...]
    best_recall: float
    recall_target: float
    passed: bool


def _gates(n_days: int) -> ColdStartReport:
    """Coverage that unlocks the pattern gates for an eval run."""
    end = DEFAULT_START + timedelta(days=n_days)
    return ColdStartReport.from_coverage(
        CoverageStats(
            first_ts=DEFAULT_START,
            last_ts=end,
            span_days=float(n_days),
            n_glucose=n_days * 288,
            glucose_coverage_pct=95.0,
            n_insulin=0,
            days_with_insulin_pct=0.0,
            n_meals=n_days * 3,
            n_sleep=0,
            n_activity=0,
        )
    )


def _recovered(
    *,
    scenario: str,
    target_kind: str,
    effect_size: float,
    n_days: int,
    seed: int,
) -> bool:
    """True iff the pattern agent emits ``target_kind`` for one planted seed."""
    effects: tuple[_Effect, ...]
    if scenario == "weekday_breakfast":
        # Plant on every weekday so the weekday-vs-weekend contrast is genuine.
        effects = tuple(WeekdayBreakfastSpike(weekday=w, effect_size=effect_size) for w in range(5))
    elif scenario == "sensitivity_shift":
        effects = (SensitivityRegimeShift(effect_size=effect_size, after_day=n_days // 2),)
    else:  # pragma: no cover - guarded by caller
        msg = f"unknown E2 scenario {scenario!r}"
        raise ValueError(msg)

    events, _manifest = generate_dataset(seed=seed, n_days=n_days, effects=effects, name=scenario)
    window_end = DEFAULT_START + timedelta(days=n_days - 1)
    window = (DEFAULT_START.date(), window_end.date())

    store = SQLiteStore(":memory:")
    store.migrate()
    try:
        store.insert_glucose(events["glucose"])
        store.insert_meals(events["meal"])
        ctx = AgentContext(
            store=store,
            window=window,
            gates=_gates(n_days),
            run_id=f"e2-{scenario}-{seed}",
        )
        kinds = {f.kind for f in pattern_agent.run(ctx)}
        return target_kind in kinds
    finally:
        store.close()


def run_e2_power(
    *,
    n_seeds: int = 5,
    seed_base: int = 7700,
    cells: tuple[tuple[float, int], ...] = _DEFAULT_CELLS,
) -> E2PowerResult:
    """Sweep planted-effect recall across effect sizes and data spans.

    Two scenarios are planted per cell (weekday breakfast spike →
    ``pattern_weekday_weekend``; sensitivity regime shift → ``pattern_tod_drift``).
    Recall is the fraction of seeds for which the matching finding kind is
    surfaced. The sweep passes when the strongest cell clears the recall target.
    """
    if n_seeds < 1:
        msg = "n_seeds must be >= 1"
        raise ValueError(msg)

    scenarios = (
        ("weekday_breakfast", "pattern_weekday_weekend"),
        ("sensitivity_shift", "pattern_tod_drift"),
    )

    results: list[E2CellResult] = []
    for scenario, target_kind in scenarios:
        for cell_idx, (effect_size, n_days) in enumerate(cells):
            recovered = 0
            for s in range(n_seeds):
                seed = seed_base + cell_idx * 1000 + s
                if _recovered(
                    scenario=scenario,
                    target_kind=target_kind,
                    effect_size=effect_size,
                    n_days=n_days,
                    seed=seed,
                ):
                    recovered += 1
            results.append(
                E2CellResult(
                    scenario=scenario,
                    target_kind=target_kind,
                    effect_size=effect_size,
                    n_days=n_days,
                    n_seeds=n_seeds,
                    n_recovered=recovered,
                    recall=recovered / n_seeds,
                )
            )

    best = max((r.recall for r in results), default=0.0)
    return E2PowerResult(
        cells=tuple(results),
        best_recall=best,
        recall_target=_RECALL_TARGET,
        passed=best >= _RECALL_TARGET,
    )
