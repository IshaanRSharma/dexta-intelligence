"""Smoke tests for the E2 statistical-power (true-discovery) eval."""

from __future__ import annotations

from eval.metrics.e2_power import run_e2_power


def test_e2_small_sweep_recovers_strong_effect() -> None:
    result = run_e2_power(n_seeds=3, cells=((50.0, 90),))
    assert result.cells
    for cell in result.cells:
        assert 0.0 <= cell.recall <= 1.0
        assert cell.n_seeds == 3
    # A large planted effect over a long span must be recovered with high recall.
    assert result.best_recall >= 0.8
    assert result.passed


def test_e2_reports_a_power_curve() -> None:
    # Recall should be monotone-ish in effect size: the strongest cell beats
    # (or ties) the weakest for at least one scenario.
    result = run_e2_power(n_seeds=3, cells=((10.0, 30), (50.0, 90)))
    by_scenario: dict[str, list[float]] = {}
    for cell in result.cells:
        by_scenario.setdefault(cell.scenario, []).append(cell.recall)
    assert any(recalls[-1] >= recalls[0] for recalls in by_scenario.values())
