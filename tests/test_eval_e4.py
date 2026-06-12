"""Smoke tests for the eval harness (E4)."""

from __future__ import annotations

from eval.metrics.e4_null_fdr import run_e4_null_fdr


def test_e4_null_small_sweep() -> None:
    result = run_e4_null_fdr(n_datasets=3, n_days=30, seed_base=100)
    assert result.n_datasets == 3
    assert 0.0 <= result.empirical_fdr <= 1.0
    assert len(result.details) == 3
