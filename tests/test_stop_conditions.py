"""Phase 4 - deterministic stop conditions nudge the loop to conclude.

probe -> update -> decide, with explicit conditions: when confidence is high or
the last probes added no new information, the loop injects a one-shot nudge to
conclude and emits a stop_signal. The nudges are advisory (the model still writes
the answer) and max_steps stays the only hard budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dexta_intelligence.agents.investigation import BeliefState
from dexta_intelligence.agents.reason import (
    ReasoningEvent,
    ToolSpec,
    run_reasoning_loop,
)


@dataclass
class _AIMessage:
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class _FakeModel:
    """Plays a fixed script and records the messages handed to each invoke."""

    def __init__(self, turns: list[Any]) -> None:
        self._turns = turns
        self.seen: list[list[Any]] = []

    def bind_tools(self, _schemas: list[dict[str, Any]]) -> _FakeModel:
        return self

    def invoke(self, messages: list[Any]) -> _AIMessage:
        self.seen.append(list(messages))
        turn = self._turns.pop(0) if self._turns else "Done."
        return _AIMessage(content=turn) if isinstance(turn, str) else _AIMessage(tool_calls=turn)


def _echo() -> ToolSpec:
    return ToolSpec(
        name="echo",
        description="echo",
        parameters={"type": "object", "properties": {}},
        fn=lambda _args: ({"v": 1}, {"v": 1}),
    )


def _nudges(events: list[ReasoningEvent]) -> list[str]:
    return [e.payload["reason"] for e in events if e.kind == "stop_signal"]


def _user_texts(messages: list[Any]) -> list[str]:
    return [m["content"] for m in messages if isinstance(m, dict) and m.get("role") == "user"]


def test_high_confidence_emits_a_conclude_nudge() -> None:
    belief = BeliefState()
    raise_conf = [{"name": "update_belief", "args": {"confidence": 0.9}, "id": "1"}]
    model = _FakeModel([raise_conf, "It is the late bolus."])
    events: list[ReasoningEvent] = []
    run_reasoning_loop(
        model, [_echo()], system="s", user="u", belief=belief, on_event=events.append
    )
    assert "confidence" in _nudges(events)
    # the nudge reached the model, appended after the tool result (ordering contract)
    last = model.seen[-1]
    nudge_idx = next(
        i
        for i, m in enumerate(last)
        if isinstance(m, dict)
        and m.get("role") == "user"
        and "Give your answer now" in str(m["content"])
    )
    tool_idx = max(i for i, m in enumerate(last) if isinstance(m, dict) and m.get("role") == "tool")
    assert nudge_idx > tool_idx


def test_confidence_nudges_at_most_once() -> None:
    belief = BeliefState()
    hi = [{"name": "update_belief", "args": {"confidence": 0.9}, "id": "1"}]
    probe = [{"name": "echo", "args": {}, "id": "p"}]
    model = _FakeModel([hi, probe, probe, "done"])
    events: list[ReasoningEvent] = []
    run_reasoning_loop(
        model, [_echo()], system="s", user="u", max_steps=8, belief=belief, on_event=events.append
    )
    assert _nudges(events).count("confidence") == 1


def test_statement_refinement_is_not_a_stall() -> None:
    belief = BeliefState()
    seed = [
        {
            "name": "update_belief",
            "args": {"hypotheses": [{"id": "h1", "statement": "late bolus"}]},
            "id": "0",
        }
    ]
    refine_stmt = [
        {
            "name": "update_belief",
            "args": {"hypotheses": [{"id": "h1", "statement": "late bolus by 40m"}]},
            "id": "1",
        }
    ]
    refine_note = [
        {
            "name": "update_belief",
            "args": {"hypotheses": [{"id": "h1", "note": "confirmed via IOB"}]},
            "id": "2",
        }
    ]
    model = _FakeModel([seed, refine_stmt, refine_note, "Concluded."])
    events: list[ReasoningEvent] = []
    run_reasoning_loop(
        model, [_echo()], system="s", user="u", max_steps=8, belief=belief, on_event=events.append
    )
    assert "stall" not in _nudges(events)  # refining a hypothesis is progress, not a stall


def test_stagnation_emits_a_wrap_up_nudge() -> None:
    belief = BeliefState()
    probe = [{"name": "echo", "args": {}, "id": "p"}]
    # three identical probe rounds with no belief change, then an answer.
    model = _FakeModel([probe, probe, probe, "Nothing new to add."])
    events: list[ReasoningEvent] = []
    run_reasoning_loop(
        model, [_echo()], system="s", user="u", max_steps=8, belief=belief, on_event=events.append
    )
    assert "stall" in _nudges(events)


def test_each_stop_condition_nudges_at_most_once() -> None:
    belief = BeliefState()
    probe = [{"name": "echo", "args": {}, "id": "p"}]
    model = _FakeModel([probe, probe, probe, probe, "done"])
    events: list[ReasoningEvent] = []
    run_reasoning_loop(
        model, [_echo()], system="s", user="u", max_steps=10, belief=belief, on_event=events.append
    )
    assert _nudges(events).count("stall") == 1


def test_no_belief_means_no_stop_signals() -> None:
    probe = [{"name": "echo", "args": {}, "id": "p"}]
    model = _FakeModel([probe, probe, probe, "done"])
    events: list[ReasoningEvent] = []
    run_reasoning_loop(model, [_echo()], system="s", user="u", on_event=events.append)
    assert _nudges(events) == []


def test_progress_resets_stall_so_no_premature_nudge() -> None:
    belief = BeliefState()
    # each round records a distinct piece of evidence: information keeps arriving.
    rounds = [
        [{"name": "update_belief", "args": {"evidence": [f"finding {i}"]}, "id": str(i)}]
        for i in range(3)
    ]
    model = _FakeModel([*rounds, "Concluded."])
    events: list[ReasoningEvent] = []
    run_reasoning_loop(
        model, [_echo()], system="s", user="u", max_steps=8, belief=belief, on_event=events.append
    )
    assert "stall" not in _nudges(events)
