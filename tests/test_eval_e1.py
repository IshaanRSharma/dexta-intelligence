"""Smoke tests for the eval harness (E1 numeric faithfulness)."""

from __future__ import annotations

from eval.metrics.e1_faithfulness import run_e1


def test_e1_small_sweep() -> None:
    result = run_e1(seed=1, n_texts=5)
    assert result.n_faithful == 5
    assert result.n_fabricated == 5
    assert result.n_miscontextualized == 5
    # Obviously-fabricated prose must always be caught.
    assert result.catch_rate_fabricated == 1.0
    assert result.catch_rate_miscontextualized == 1.0
    assert 0.0 <= result.false_rejection_rate < 0.05
    assert result.passed


def test_e1_metrics_in_unit_interval() -> None:
    result = run_e1(seed=2, n_texts=4)
    for rate in (
        result.catch_rate_fabricated,
        result.catch_rate_miscontextualized,
        result.false_rejection_rate,
    ):
        assert 0.0 <= rate <= 1.0
