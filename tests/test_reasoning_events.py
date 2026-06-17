"""The reasoning-loop streaming seam — events fire as the agent works.

A live surface (SSE, CLI trace) subscribes via ``on_event``; this pins the
event order (tool_call → tool_result → answer) and that a failing sink never
breaks the loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
    def __init__(self, turns: list[Any]) -> None:
        self._turns = turns

    def bind_tools(self, _schemas: list[dict[str, Any]]) -> _FakeModel:
        return self

    def invoke(self, _messages: list[Any]) -> _AIMessage:
        turn = self._turns.pop(0)
        return _AIMessage(content=turn) if isinstance(turn, str) else _AIMessage(tool_calls=turn)

    def stream(self, _messages: list[Any]):
        turn = self._turns[0]
        if isinstance(turn, str):
            self._turns.pop(0)
            words = turn.split(" ")
            for idx, word in enumerate(words):
                suffix = " " if idx < len(words) - 1 else ""
                yield _AIMessage(content=word + suffix)
        else:
            self._turns.pop(0)
            yield _AIMessage(tool_calls=turn)


def _echo_tool() -> ToolSpec:
    return ToolSpec(
        name="echo",
        description="echo",
        parameters={"type": "object", "properties": {}},
        fn=lambda _args: ({"value": 1}, {"value": 1}),
    )


def test_events_stream_in_order() -> None:
    events: list[ReasoningEvent] = []
    model = _FakeModel([[{"name": "echo", "args": {}, "id": "1"}], "Done."])
    run_reasoning_loop(
        model, [_echo_tool()], system="s", user="u", on_event=events.append
    )
    kinds = [e.kind for e in events]
    assert kinds[:2] == ["tool_call", "tool_result"]
    assert "answer_start" in kinds
    assert "answer_delta" in kinds
    streamed = "".join(e.payload["delta"] for e in events if e.kind == "answer_delta")
    assert streamed == "Done."


def test_no_sink_is_fine() -> None:
    model = _FakeModel(["Just an answer."])
    result = run_reasoning_loop(model, [_echo_tool()], system="s", user="u")
    assert result.answer == "Just an answer."


class _CapturingModel:
    """Records the messages it was handed on its (single) invoke."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.seen: list[Any] = []

    def bind_tools(self, _schemas: list[dict[str, Any]]) -> _CapturingModel:
        return self

    def invoke(self, messages: list[Any]) -> _AIMessage:
        self.seen = list(messages)
        return _AIMessage(content=self.answer)

    def stream(self, messages: list[Any]):
        yield self.invoke(messages)


def test_history_is_seeded_before_the_current_turn() -> None:
    model = _CapturingModel("Wednesday looks similar.")
    history = [
        {"role": "user", "content": "why did I spike Tuesday?"},
        {"role": "assistant", "content": "Late dinner bolus."},
    ]
    run_reasoning_loop(
        model, [_echo_tool()], system="s", user="what about Wednesday?", history=history
    )
    roles_contents = [(m.get("role"), m.get("content")) for m in model.seen if isinstance(m, dict)]
    assert roles_contents[0] == ("system", "s")
    assert ("user", "why did I spike Tuesday?") in roles_contents
    assert ("assistant", "Late dinner bolus.") in roles_contents
    assert roles_contents[-1] == ("user", "what about Wednesday?")


def test_no_history_is_unchanged() -> None:
    model = _CapturingModel("Answer.")
    run_reasoning_loop(model, [_echo_tool()], system="s", user="q")
    dicts = [m for m in model.seen if isinstance(m, dict)]
    assert len(dicts) == 2  # system + user only


def test_failing_sink_never_breaks_the_loop() -> None:
    def boom(_event: ReasoningEvent) -> None:
        raise RuntimeError("sink down")

    model = _FakeModel([[{"name": "echo", "args": {}, "id": "1"}], "Done."])
    result = run_reasoning_loop(
        model, [_echo_tool()], system="s", user="u", on_event=boom
    )
    assert result.answer == "Done."  # loop completes despite the sink raising
