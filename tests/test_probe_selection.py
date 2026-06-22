"""Phase 3 - the belief state suggests the most discriminating next probe.

A light information-gain heuristic: the most useful next probe gathers evidence a
live hypothesis depends on but the run has not collected. It is advisory (folded
into the belief snapshot the model reads), never a controller, and the loop feeds
it the tools actually called.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dexta_intelligence.agents.investigation import (
    BeliefState,
    Hypothesis,
    HypothesisStatus,
)
from dexta_intelligence.agents.reason import ToolSpec, run_reasoning_loop


def _belief_with(statement: str, status: HypothesisStatus = HypothesisStatus.OPEN) -> BeliefState:
    state = BeliefState()
    state.hypotheses["h1"] = Hypothesis(id="h1", statement=statement, status=status)
    return state


def test_suggests_a_modality_implied_by_an_open_hypothesis() -> None:
    state = _belief_with("a high-carb breakfast drives the spike")
    suggestion = state.suggested_probe()
    assert suggestion.startswith("carbs")


def test_suggestion_clears_once_that_modality_is_probed() -> None:
    state = _belief_with("a high-carb breakfast drives the spike")
    state.note_probe("get_cob")
    # carbs is the only implied modality here; once examined, no suggestion remains.
    assert state.suggested_probe() == ""


def test_suggestion_advances_to_the_next_unexamined_modality() -> None:
    state = _belief_with("the morning bolus is too small for breakfast")
    # implies carbs (breakfast), insulin (bolus), temporal (morning); carbs first.
    first = state.suggested_probe()
    assert first.startswith("carbs")
    state.note_probe("get_cob")  # examine carbs
    second = state.suggested_probe()
    assert second.startswith("insulin")


def test_tie_break_skips_carbs_when_only_later_modalities_apply() -> None:
    state = _belief_with("the post-exercise correction was too aggressive")
    # implies insulin (correction) and activity (exercise), not carbs; insulin first.
    assert state.suggested_probe().startswith("insulin")


def test_unknown_probed_tool_clears_no_modality() -> None:
    state = _belief_with("a high-carb breakfast drives the spike")
    state.note_probe("get_glucose_stats")  # not in any modality's tool set
    assert state.probed == ["get_glucose_stats"]
    assert state.suggested_probe().startswith("carbs")


def test_no_open_hypotheses_means_no_suggestion() -> None:
    state = _belief_with("resolved", status=HypothesisStatus.SUPPORTED)
    assert state.suggested_probe() == ""


def test_unrecognized_hypothesis_yields_no_suggestion() -> None:
    state = _belief_with("something inexplicable is going on")
    assert state.suggested_probe() == ""


def test_note_probe_ignores_the_belief_tool_itself() -> None:
    state = BeliefState()
    state.note_probe("update_belief")
    state.note_probe("get_cob")
    assert state.probed == ["get_cob"]


def test_snapshot_carries_the_suggestion() -> None:
    state = _belief_with("dinner carbs spike me")
    assert state.snapshot()["suggested_probe"].startswith("carbs")


@dataclass
class _AIMessage:
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class _FakeModel:
    def __init__(self, turns: list[Any]) -> None:
        self._turns = turns

    def bind_tools(self, _schemas: list[dict[str, Any]]) -> _FakeModel:
        return self

    def invoke(self, _messages: list[Any]) -> _AIMessage:
        turn = self._turns.pop(0)
        return _AIMessage(content=turn) if isinstance(turn, str) else _AIMessage(tool_calls=turn)


def _cob_tool() -> ToolSpec:
    return ToolSpec(
        name="get_cob",
        description="carbs on board",
        parameters={"type": "object", "properties": {}},
        fn=lambda _args: ({"cob": 20}, {"cob": 20}),
    )


def test_loop_feeds_called_tools_into_probe_tracking() -> None:
    belief = _belief_with("a high-carb breakfast drives the spike")
    assert belief.suggested_probe().startswith("carbs")
    model = _FakeModel([[{"name": "get_cob", "args": {}, "id": "1"}], "It was breakfast carbs."])
    run_reasoning_loop(model, [_cob_tool()], system="s", user="why?", belief=belief)
    assert "get_cob" in belief.probed
    assert not belief.suggested_probe().startswith("carbs")  # carbs now examined
