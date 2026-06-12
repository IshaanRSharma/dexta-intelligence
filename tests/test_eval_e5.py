"""Smoke tests for the eval harness (E5 perturbation robustness)."""

from __future__ import annotations

from eval.metrics.e5_perturbation import run_e5


def test_e5_small_sweep() -> None:
    result = run_e5(seed=11, n_days=30)
    assert result.n_days == 30
    assert {r.name for r in result.corruptions} == {
        "dropout",
        "duplicates",
        "gap",
        "tz_shift",
    }
    for row in result.corruptions:
        assert 0.0 <= row.jaccard <= 1.0
        assert row.n_clean_kinds >= 0
    assert 0.0 <= result.min_jaccard <= 1.0


def test_e5_clean_kinds_present() -> None:
    result = run_e5(seed=12, n_days=30)
    # The clean run should surface at least the observation glycemic finding.
    assert result.clean_kinds
    assert "observation_glycemic" in result.clean_kinds
