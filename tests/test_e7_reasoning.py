"""Deterministic, key-free tests for the E7 reasoning-process metric."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from eval.agent_eval import InvestigationOutcome
from eval.metrics.e7_reasoning import (
    IDEAL_PROBES,
    gap_flagged,
    modality_coverage,
    path_sound,
    probe_efficiency,
    run_e7_reasoning,
)
from eval.scenarios import CARB, TEMPORAL, Scenario

from dexta_intelligence.store import SQLiteStore

if TYPE_CHECKING:
    from dexta_intelligence.store.port import StoragePort


def _trivial_store() -> SQLiteStore:
    store = SQLiteStore(":memory:")
    store.migrate()
    return store


def _scenario(
    name: str,
    *,
    keywords: tuple[str, ...] = ("breakfast",),
    required: tuple[frozenset[str], ...] = (),
    is_gap: bool = False,
) -> Scenario:
    return Scenario(
        name=name,
        question=f"why {name}?",
        expected_keywords=keywords,
        build=_trivial_store,
        required_evidence=required,
        is_gap=is_gap,
    )


def _outcome(
    answer: str = "It is breakfast.",
    tools: tuple[str, ...] = (),
    *,
    faithful: bool = True,
) -> InvestigationOutcome:
    return InvestigationOutcome(
        answer=answer, tools_used=tools, faithful=faithful, stopped_reason="answered"
    )


def _runner(outcome: InvestigationOutcome) -> Any:
    def runner(store: StoragePort, question: str, model: Any) -> InvestigationOutcome:
        return outcome

    return runner


def test_coverage_full_when_each_required_class_probed() -> None:
    scenario = _scenario("wk", required=(TEMPORAL, CARB))
    outcome = _outcome(tools=("get_weekday", "get_cob"))
    assert modality_coverage(outcome, scenario) == 1.0


def test_coverage_partial_when_one_class_missing() -> None:
    scenario = _scenario("wk", required=(TEMPORAL, CARB))
    outcome = _outcome(tools=("get_weekday",))
    assert modality_coverage(outcome, scenario) == 0.5


def test_coverage_one_when_nothing_required() -> None:
    assert modality_coverage(_outcome(), _scenario("none")) == 1.0


def test_probe_efficiency_full_at_or_under_budget() -> None:
    assert probe_efficiency(_outcome(tools=("a",) * IDEAL_PROBES)) == 1.0


def test_probe_efficiency_decays_above_budget() -> None:
    outcome = _outcome(tools=("a",) * (IDEAL_PROBES * 2))
    assert probe_efficiency(outcome) == 0.5


def test_probe_efficiency_fractional_decay() -> None:
    outcome = _outcome(tools=("a",) * (IDEAL_PROBES + 2))
    assert probe_efficiency(outcome) == IDEAL_PROBES / (IDEAL_PROBES + 2)


def test_gap_flagged_detects_missing_context_language() -> None:
    assert gap_flagged(_outcome("I have no logged meal to explain this; please log it."))
    assert not gap_flagged(_outcome("It is clearly breakfast carbohydrates."))


def test_gap_flagged_ignores_ordinary_attribution_prose() -> None:
    assert not gap_flagged(_outcome("There is no carb-ratio issue; it is breakfast."))


def test_path_sound_requires_coverage_and_attribution() -> None:
    scenario = _scenario("wk", keywords=("breakfast",), required=(TEMPORAL, CARB))
    sound = _outcome("It is breakfast.", tools=("get_weekday", "get_cob"))
    assert path_sound(sound, scenario) is True

    missing_evidence = _outcome("It is breakfast.", tools=("get_weekday",))
    assert path_sound(missing_evidence, scenario) is False

    wrong_answer = _outcome("Unknown cause.", tools=("get_weekday", "get_cob"))
    assert path_sound(wrong_answer, scenario) is False


def test_path_sound_unfaithful_is_never_sound() -> None:
    scenario = _scenario("wk", required=(TEMPORAL, CARB))
    outcome = _outcome("It is breakfast.", tools=("get_weekday", "get_cob"), faithful=False)
    assert path_sound(outcome, scenario) is False


def test_path_sound_gap_scenario_needs_gap_flag() -> None:
    scenario = _scenario("gap", required=(CARB,), is_gap=True)
    flagged = _outcome("I have no logged meal near these highs.")
    assert path_sound(flagged, scenario) is True

    fabricated = _outcome("It is breakfast carbohydrates.")
    assert path_sound(fabricated, scenario) is False


def test_path_sound_gap_rejects_flagged_but_fabricated_answer() -> None:
    scenario = _scenario("gap", keywords=("breakfast",), required=(CARB,), is_gap=True)
    hedged = _outcome("It is breakfast carbohydrates, though no meal was logged.")
    assert gap_flagged(hedged) is True
    assert path_sound(hedged, scenario) is False


def test_run_aggregates_means_and_gates_on_targets() -> None:
    scenarios = (
        _scenario("a", keywords=("breakfast",), required=(TEMPORAL,)),
        _scenario("gap", required=(CARB,), is_gap=True),
    )

    def runner(store: StoragePort, question: str, model: Any) -> InvestigationOutcome:
        if question.startswith("why gap"):
            return _outcome("no logged meal here", tools=("get_cob",))
        return _outcome("It is breakfast.", tools=("get_weekday",))

    result = run_e7_reasoning(None, scenarios=scenarios, runner=runner)

    assert result.mean_coverage == 1.0
    assert result.soundness_rate == 1.0
    assert result.gap_handling_rate == 1.0
    assert result.passed is True


def test_run_fails_when_below_targets() -> None:
    scenario = _scenario("a", keywords=("breakfast",), required=(TEMPORAL, CARB))
    result = run_e7_reasoning(
        None, scenarios=(scenario,), runner=_runner(_outcome("unknown", tools=()))
    )
    assert result.mean_coverage == 0.0
    assert result.soundness_rate == 0.0
    assert result.passed is False


def test_gap_handling_rate_zero_when_no_gap_scenarios() -> None:
    scenario = _scenario("a", keywords=("breakfast",), required=(TEMPORAL,))
    result = run_e7_reasoning(
        None,
        scenarios=(scenario,),
        runner=_runner(_outcome("It is breakfast.", tools=("get_weekday",))),
    )
    assert result.gap_handling_rate == 0.0
