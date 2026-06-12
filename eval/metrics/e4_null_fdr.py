"""E4 null-set false discovery rate — pattern agent on pure-null synthetic data.

Ground truth: zero planted effects → any surfaced pattern finding is a
false discovery. Reports the empirical FDR at the project default alpha=0.10.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.agents.pattern import pattern_agent
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.models import CoverageStats
from dexta_intelligence.store import SQLiteStore
from dexta_intelligence.testing.synthetic import DEFAULT_START, generate_null

__all__ = ["E4NullResult", "run_e4_null_fdr"]

_FDR_ALPHA = 0.10


@dataclass(frozen=True, slots=True)
class E4NullResult:
    """Outcome of one E4 null-set sweep."""

    n_datasets: int
    n_false_discoveries: int
    empirical_fdr: float
    alpha: float
    passed: bool
    details: tuple[tuple[int, tuple[str, ...]], ...]


def run_e4_null_fdr(
    *,
    n_datasets: int = 20,
    n_days: int = 90,
    seed_base: int = 9000,
    alpha: float = _FDR_ALPHA,
) -> E4NullResult:
    """Run the pattern agent on ``n_datasets`` null synthetic sets."""
    if n_datasets < 1:
        msg = "n_datasets must be >= 1"
        raise ValueError(msg)

    window_end = DEFAULT_START + timedelta(days=n_days - 1)
    window = (DEFAULT_START.date(), window_end.date())
    false_discoveries = 0
    details: list[tuple[int, tuple[str, ...]]] = []

    for idx in range(n_datasets):
        store = SQLiteStore(":memory:")
        store.migrate()
        try:
            events, _manifest = generate_null(seed=seed_base + idx, n_days=n_days)
            store.insert_glucose(events["glucose"])
            coverage = store.coverage()
            gates = ColdStartReport.from_coverage(_boost_coverage(coverage, n_days))
            ctx = AgentContext(
                store=store,
                window=window,
                gates=gates,
                run_id=f"e4-null-{idx}",
            )
            findings = pattern_agent.run(ctx)
            kinds = tuple(f.kind for f in findings)
            if kinds:
                false_discoveries += 1
            details.append((idx, kinds))
        finally:
            store.close()

    empirical_fdr = false_discoveries / n_datasets
    return E4NullResult(
        n_datasets=n_datasets,
        n_false_discoveries=false_discoveries,
        empirical_fdr=empirical_fdr,
        alpha=alpha,
        passed=empirical_fdr <= alpha,
        details=tuple(details),
    )


def _boost_coverage(coverage: CoverageStats, n_days: int) -> CoverageStats:
    """Ensure cold-start gates unlock for eval runs on sparse null data."""
    end = DEFAULT_START + timedelta(days=n_days)
    return CoverageStats(
        first_ts=coverage.first_ts or DEFAULT_START,
        last_ts=coverage.last_ts or end,
        span_days=float(n_days),
        n_glucose=max(coverage.n_glucose, n_days * 288),
        glucose_coverage_pct=max(coverage.glucose_coverage_pct, 95.0),
        n_insulin=coverage.n_insulin,
        days_with_insulin_pct=coverage.days_with_insulin_pct,
        n_meals=coverage.n_meals,
        n_sleep=coverage.n_sleep,
        n_activity=coverage.n_activity,
    )
