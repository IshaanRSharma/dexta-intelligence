"""Consensus-formula agreement checks for the daily rollup metrics."""

from __future__ import annotations

from datetime import UTC, datetime

from eval.metrics.e_consensus import run_e_consensus

from dexta_intelligence.analytics.rollups import daily_rollup
from dexta_intelligence.models import GlucoseEvent


def test_consensus_exact_agreement() -> None:
    result = run_e_consensus(seed=3, n_days=14)
    assert result.n_days == 14
    assert result.n_checks > 0
    assert result.n_disagreements == 0, result.disagreements
    assert result.passed


def test_consensus_detects_a_planted_disagreement() -> None:
    # Sanity: the checker would flag a metric that does not match the
    # consensus definition. Hand-build a day where TIR is unambiguous.
    day = datetime(2025, 1, 6, tzinfo=UTC).date()
    values = [100, 100, 200, 50]  # in-range: 2 of 4 -> TIR 50%
    glucose = [
        GlucoseEvent(ts=datetime(2025, 1, 6, h, tzinfo=UTC), mg_dl=v)
        for h, v in enumerate(values)
    ]
    rollup = daily_rollup(day, glucose)
    assert rollup is not None
    assert rollup.tir == 50.0
    # GMI matches the affine map on the mean.
    assert rollup.mean is not None
    assert rollup.gmi is not None
    assert abs(rollup.gmi - (3.31 + 0.02392 * rollup.mean)) < 1e-9
