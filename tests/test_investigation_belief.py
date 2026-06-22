"""The working belief state and its loop hook.

Unit-tests the merge semantics of ``BeliefState`` and that ``run_reasoning_loop``
threads it through: the update_belief tool is offered, the state evolves as the
model calls it, belief events stream, and the final state rides home on the
result. Key-free: a scripted fake model drives the loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dexta_intelligence.agents.investigation import (
    BeliefState,
    Hypothesis,
    HypothesisStatus,
)
from dexta_intelligence.agents.reason import (
    ReasoningEvent,
    ToolSpec,
    run_reasoning_loop,
)


def test_apply_adds_hypotheses_with_auto_ids() -> None:
    state = BeliefState()
    state.apply({"hypotheses": [{"statement": "late bolus"}, {"statement": "high-carb meal"}]})
    ids = list(state.hypotheses)
    assert ids == ["h1", "h2"]
    assert state.hypotheses["h1"].status is HypothesisStatus.OPEN


def test_apply_updates_existing_hypothesis_in_place() -> None:
    state = BeliefState()
    state.apply({"hypotheses": [{"id": "h1", "statement": "late bolus"}]})
    state.apply({"hypotheses": [{"id": "h1", "status": "supported", "note": "bolus 40m late"}]})
    h = state.hypotheses["h1"]
    assert h.statement == "late bolus"
    assert h.status is HypothesisStatus.SUPPORTED
    assert h.note == "bolus 40m late"


def test_invalid_status_becomes_undetermined() -> None:
    state = BeliefState()
    state.apply({"hypotheses": [{"statement": "x", "status": "definitely"}]})
    assert state.hypotheses["h1"].status is HypothesisStatus.UNDETERMINED


def test_empty_status_on_update_does_not_clobber_existing() -> None:
    state = BeliefState()
    state.apply({"hypotheses": [{"id": "h1", "statement": "x", "status": "supported"}]})
    state.apply({"hypotheses": [{"id": "h1", "status": "", "note": "more detail"}]})
    h = state.hypotheses["h1"]
    assert h.status is HypothesisStatus.SUPPORTED
    assert h.note == "more detail"


def test_evidence_appends_and_dedupes() -> None:
    state = BeliefState()
    state.apply({"evidence": ["spike at 8am", "spike at 8am", " "]})
    state.apply({"evidence": ["no meal logged"]})
    assert state.evidence == ["spike at 8am", "no meal logged"]


def test_gaps_replace_when_supplied() -> None:
    state = BeliefState()
    state.apply({"gaps": ["need meal log"]})
    state.apply({"gaps": []})
    assert state.gaps == []


def test_gaps_unchanged_when_key_absent() -> None:
    state = BeliefState()
    state.apply({"gaps": ["need meal log"]})
    state.apply({"summary": "still unclear"})
    assert state.gaps == ["need meal log"]


def test_confidence_is_clamped() -> None:
    state = BeliefState()
    state.apply({"confidence": 5})
    assert state.confidence == 1.0
    state.apply({"confidence": -2})
    assert state.confidence == 0.0
    state.apply({"confidence": "nan-ish"})
    assert state.confidence == 0.0


def test_snapshot_is_json_serializable_shape() -> None:
    state = BeliefState(
        hypotheses={"h1": Hypothesis("h1", "late bolus", HypothesisStatus.SUPPORTED)},
        evidence=["e"],
        gaps=["g"],
        confidence=0.7,
        summary="leaning late bolus",
    )
    snap = state.snapshot()
    assert snap["hypotheses"] == [
        {"id": "h1", "statement": "late bolus", "status": "supported", "note": ""}
    ]
    assert snap["confidence"] == 0.7
    assert snap["summary"] == "leaning late bolus"


def test_tool_merges_and_returns_snapshot_without_evidence_numbers() -> None:
    state = BeliefState()
    spec = state.tool()
    assert spec.name == "update_belief"
    result, numbers = spec.fn({"summary": "checking carbs", "confidence": 0.4})
    assert numbers == {}  # belief is meta, never the guard's evidence pool
    assert result["summary"] == "checking carbs"
    assert state.confidence == 0.4


@dataclass
class _AIMessage:
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class _FakeModel:
    def __init__(self, turns: list[Any]) -> None:
        self._turns = turns
        self.bound_names: list[str] = []

    def bind_tools(self, schemas: list[dict[str, Any]]) -> _FakeModel:
        self.bound_names = [s["function"]["name"] for s in schemas]
        return self

    def invoke(self, _messages: list[Any]) -> _AIMessage:
        turn = self._turns.pop(0)
        return _AIMessage(content=turn) if isinstance(turn, str) else _AIMessage(tool_calls=turn)


def _echo_tool() -> ToolSpec:
    return ToolSpec(
        name="echo",
        description="echo",
        parameters={"type": "object", "properties": {}},
        fn=lambda _args: ({"value": 1}, {"value": 1}),
    )


def test_loop_without_belief_offers_no_update_tool() -> None:
    model = _FakeModel(["done"])
    run_reasoning_loop(model, [_echo_tool()], system="s", user="u")
    assert "update_belief" not in model.bound_names


def test_loop_threads_belief_through_and_returns_it() -> None:
    belief = BeliefState()
    update = [
        {
            "name": "update_belief",
            "args": {"hypotheses": [{"statement": "late bolus"}], "confidence": 0.6},
            "id": "1",
        }
    ]
    model = _FakeModel([update, "It was a late bolus."])
    events: list[ReasoningEvent] = []
    result = run_reasoning_loop(
        model,
        [_echo_tool()],
        system="s",
        user="why the spike?",
        belief=belief,
        on_event=events.append,
    )

    assert "update_belief" in model.bound_names
    assert result.belief is belief
    assert belief.confidence == 0.6
    assert belief.hypotheses["h1"].statement == "late bolus"
    belief_events = [e for e in events if e.kind == "belief"]
    assert belief_events and belief_events[-1].payload["confidence"] == 0.6


def test_belief_updates_do_not_enter_the_evidence_pool() -> None:
    belief = BeliefState()
    update = [
        {"name": "update_belief", "args": {"summary": "checking"}, "id": "1"},
    ]
    model = _FakeModel([update, "Answer."])
    result = run_reasoning_loop(model, [_echo_tool()], system="s", user="u", belief=belief)
    assert not any(key.startswith("update_belief") for key in result.evidence)
