"""Smoke tests for the E3 clinical-accuracy eval."""

from __future__ import annotations

from eval.metrics.e3_accuracy import run_e3_accuracy


def test_e3_emits_numbers() -> None:
    result = run_e3_accuracy(seed=1, n_days=14)
    assert result.n_pairs > 0
    assert result.mard_pct >= 0.0
    assert set(result.clarke) == {"A", "B", "C", "D", "E"}
    assert set(result.parkes) == {"A", "B", "C", "D", "E"}
    # Zone fractions are a distribution.
    assert abs(sum(result.clarke.values()) - 1.0) < 1e-3
    assert abs(sum(result.parkes.values()) - 1.0) < 1e-3
    assert 0.0 <= result.clarke_ab_pct <= 100.0
    assert 0.0 <= result.parkes_ab_pct <= 100.0


def test_e3_deterministic() -> None:
    a = run_e3_accuracy(seed=5, n_days=10)
    b = run_e3_accuracy(seed=5, n_days=10)
    assert a.mard_pct == b.mard_pct
    assert a.clarke == b.clarke
