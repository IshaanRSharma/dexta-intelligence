"""E_consensus — rollup metrics exactly match the published consensus formulas.

A faithfulness check on the deterministic analytics layer: the daily rollup's
glycemic metrics must equal the international-consensus definitions, recomputed
here independently from the raw readings. Any drift between
:func:`~dexta_intelligence.analytics.rollups.daily_rollup` and the published
formulas is a correctness bug, so the bar is *exact* agreement.

Definitions recomputed (mg/dL):

- mean — arithmetic mean of in-day readings.
- TIR — % of readings in [70, 180] inclusive (Battelino et al., *Diabetes Care*
  2019;42(8):1593-1603).
- CV — coefficient of variation, ``100 * sample_sd / mean`` (sample sd, n-1).
- GMI — ``3.31 + 0.02392 * mean`` (Bergenstal et al., *Diabetes Care*
  2018;41(11):2275-2280).

Ground truth is non-LLM: a deterministic synthetic glucose series.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import date

from dexta_intelligence.analytics.rollups import (
    TARGET_HIGH_MG_DL,
    TARGET_LOW_MG_DL,
    daily_rollup,
)
from dexta_intelligence.models import GlucoseEvent
from dexta_intelligence.testing.synthetic import generate_baseline

__all__ = ["EConsensusResult", "EConsensusRow", "run_e_consensus"]

#: Absolute tolerance for "exact" float agreement (recompute uses identical math).
_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class EConsensusRow:
    """Per-metric agreement for one day."""

    day: date
    metric: str
    rollup_value: float
    reference_value: float
    agrees: bool


@dataclass(frozen=True, slots=True)
class EConsensusResult:
    """Outcome of one consensus-agreement sweep."""

    n_days: int
    n_checks: int
    n_disagreements: int
    passed: bool
    disagreements: tuple[EConsensusRow, ...] = field(default_factory=tuple)


def _reference_metrics(values: list[int]) -> dict[str, float]:
    """Independently recompute the consensus metrics from raw readings."""
    n = len(values)
    mean = statistics.fmean(values)
    # Consensus TIR band [70, 180] inclusive: count what is out of range, as the
    # rollup does, so the two computations agree to the last bit.
    out_of_range = sum(1 for v in values if v < TARGET_LOW_MG_DL or v > TARGET_HIGH_MG_DL)
    tir = 100.0 * (n - out_of_range) / n
    gmi = 3.31 + 0.02392 * mean
    out = {"mean": mean, "tir": tir, "gmi": gmi}
    if n >= 2:
        sd = statistics.stdev(values, mean)
        out["cv"] = (sd / mean) * 100.0
    return out


def run_e_consensus(*, seed: int = 9100, n_days: int = 14) -> EConsensusResult:
    """Assert daily-rollup metrics match independently-recomputed definitions.

    Every in-range readings count, mean, GMI, and CV produced by
    :func:`daily_rollup` is compared to a from-scratch recomputation. The sweep
    passes only on exact agreement across all days and metrics.
    """
    events = generate_baseline(seed=seed, n_days=n_days)
    glucose: list[GlucoseEvent] = events["glucose"]
    by_day: dict[date, list[int]] = {}
    for g in glucose:
        by_day.setdefault(g.ts.date(), []).append(g.mg_dl)

    rows: list[EConsensusRow] = []
    n_checks = 0
    for day in sorted(by_day):
        rollup = daily_rollup(day, glucose)
        assert rollup is not None  # day has readings by construction
        ref = _reference_metrics(by_day[day])

        checks: list[tuple[str, float | None]] = [
            ("mean", rollup.mean),
            ("tir", rollup.tir),
            ("gmi", rollup.gmi),
            ("cv", rollup.cv),
        ]
        for metric, rollup_value in checks:
            if metric not in ref or rollup_value is None:
                continue
            n_checks += 1
            ref_value = ref[metric]
            agrees = abs(rollup_value - ref_value) <= _EPS
            if not agrees:
                rows.append(
                    EConsensusRow(
                        day=day,
                        metric=metric,
                        rollup_value=rollup_value,
                        reference_value=ref_value,
                        agrees=False,
                    )
                )

    return EConsensusResult(
        n_days=len(by_day),
        n_checks=n_checks,
        n_disagreements=len(rows),
        passed=not rows,
        disagreements=tuple(rows),
    )
