"""Deterministic, key-free tests for the E6 faithfulness metric."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from eval.agent_eval import InvestigationOutcome
from eval.metrics.e6_faithfulness import run_e6_faithfulness
from eval.scenarios import Scenario

from dexta_intelligence.store import SQLiteStore

if TYPE_CHECKING:
    from dexta_intelligence.store.port import StoragePort


def _trivial_store() -> SQLiteStore:
    store = SQLiteStore(":memory:")
    store.migrate()
    return store


def _scenario(name: str) -> Scenario:
    return Scenario(
        name=name,
        question=f"why {name}?",
        expected_keywords=(name,),
        build=_trivial_store,
    )


def _runner_returning(faithful: bool) -> Any:
    def runner(store: StoragePort, question: str, model: Any) -> InvestigationOutcome:
        return InvestigationOutcome(
            answer="...",
            tools_used=(),
            faithful=faithful,
            stopped_reason="answered",
        )

    return runner


def test_all_faithful_rate_one() -> None:
    scenarios = (_scenario("breakfast"), _scenario("exercise"))
    result = run_e6_faithfulness(None, scenarios=scenarios, runner=_runner_returning(True))

    assert result.faithful_rate == 1.0
    assert result.passed is True
    assert all(cell.faithful for cell in result.cells)


def test_partial_faithful_fails_default_target() -> None:
    scenarios = (_scenario("breakfast"), _scenario("exercise"))

    def runner(store: StoragePort, question: str, model: Any) -> InvestigationOutcome:
        faithful = question == "why breakfast?"
        return InvestigationOutcome(
            answer="...",
            tools_used=(),
            faithful=faithful,
            stopped_reason="answered",
        )

    result = run_e6_faithfulness(None, scenarios=scenarios, runner=runner)

    assert result.faithful_rate == 0.5
    assert result.passed is False
    by_name = {cell.scenario: cell.faithful for cell in result.cells}
    assert by_name == {"breakfast": True, "exercise": False}


def test_empty_scenarios_rate_zero() -> None:
    result = run_e6_faithfulness(None, scenarios=(), runner=_runner_returning(True))

    assert result.faithful_rate == 0.0
    assert result.passed is False
    assert result.cells == ()
