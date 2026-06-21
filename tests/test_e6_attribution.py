"""Deterministic, key-free tests for the E6 attribution metric."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from eval.agent_eval import InvestigationOutcome
from eval.metrics.e6_attribution import run_e6_attribution
from eval.scenarios import Scenario

from dexta_intelligence.store import SQLiteStore

if TYPE_CHECKING:
    from dexta_intelligence.store.port import StoragePort


def _trivial_store() -> SQLiteStore:
    store = SQLiteStore(":memory:")
    store.migrate()
    return store


def _scenario(name: str, keywords: tuple[str, ...]) -> Scenario:
    return Scenario(
        name=name,
        question=f"why {name}?",
        expected_keywords=keywords,
        build=_trivial_store,
    )


def _runner_returning(answer: str) -> Any:
    def runner(store: StoragePort, question: str, model: Any) -> InvestigationOutcome:
        return InvestigationOutcome(
            answer=answer,
            tools_used=(),
            faithful=True,
            stopped_reason="answered",
        )

    return runner


def test_all_hits_accuracy_one() -> None:
    scenarios = (
        _scenario("breakfast", ("breakfast",)),
        _scenario("exercise", ("exercise",)),
    )
    runner = _runner_returning("It is the breakfast and exercise effect.")
    result = run_e6_attribution(None, scenarios=scenarios, runner=runner)

    assert result.accuracy == 1.0
    assert result.passed is True
    assert all(cell.hit for cell in result.cells)


def test_partial_hit_depends_on_target() -> None:
    scenarios = (
        _scenario("breakfast", ("breakfast",)),
        _scenario("exercise", ("exercise",)),
    )
    runner = _runner_returning("It is the breakfast effect only.")

    result = run_e6_attribution(None, scenarios=scenarios, runner=runner)
    assert result.accuracy == 0.5
    assert result.passed is False
    by_name = {cell.scenario: cell.hit for cell in result.cells}
    assert by_name == {"breakfast": True, "exercise": False}

    lenient = run_e6_attribution(None, scenarios=scenarios, runner=runner, accuracy_target=0.5)
    assert lenient.accuracy == 0.5
    assert lenient.passed is True


def test_match_is_case_insensitive() -> None:
    scenarios = (_scenario("breakfast", ("Breakfast",)),)
    runner = _runner_returning("BREAKFAST is the driver here.")
    result = run_e6_attribution(None, scenarios=scenarios, runner=runner)

    assert result.accuracy == 1.0
    assert result.passed is True
    assert result.cells[0].hit is True
