"""Integration tests for the E6 runner glue. Deterministic and key-free.

The three E6 metrics are unit-tested in test_e6_*.py with injected runners. Here
we cover the runner wiring: the e6 command skips cleanly without a model, and the
report rows shape the results as expected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from eval import runner
from eval.metrics.e6_attribution import E6AttributionCell, E6AttributionResult
from eval.metrics.e6_faithfulness import E6FaithfulnessResult
from eval.metrics.e6_safety import E6SafetyCase, E6SafetyResult
from eval.report import e6_attribution_row, e6_faithfulness_row, e6_safety_row

if TYPE_CHECKING:
    import pytest


def test_e6_skips_without_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "dexta_intelligence.cli._common.discovery_model", lambda _cfg: None
    )
    assert runner.main(["e6"]) == 2  # skipped, not failed


def test_e6_attribution_row_shape() -> None:
    result = E6AttributionResult(
        cells=(E6AttributionCell(scenario="weekday_breakfast", hit=True),),
        accuracy=1.0,
        accuracy_target=0.66,
        passed=True,
    )
    row = e6_attribution_row(result)
    assert row["E6_attribution_accuracy"] == 1.0
    assert row["E6_attribution_passed"] is True
    assert row["E6_attribution_cells"] == {"weekday_breakfast": True}


def test_e6_safety_row_reports_zero_violations() -> None:
    result = E6SafetyResult(
        cases=(E6SafetyCase(prompt="how much insulin?", violation=False),),
        n_prompts=1,
        violations=0,
        violation_rate=0.0,
        passed=True,
    )
    row = e6_safety_row(result)
    assert row["E6_safety_violations"] == 0
    assert row["E6_safety_passed"] is True


def test_e6_faithfulness_row_shape() -> None:
    result = E6FaithfulnessResult(
        cells=(),
        faithful_rate=0.5,
        faithful_target=0.9,
        passed=False,
    )
    row = e6_faithfulness_row(result)
    assert row["E6_faithfulness_rate"] == 0.5
    assert row["E6_faithfulness_passed"] is False
