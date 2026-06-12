"""Tests for the reasoning loop and the chat agent.

A fake tool-calling model emulates native function calling: each scripted
turn is either a list of tool calls (the model decides to act) or a final
string answer (the model decides it's done). This exercises the real
multi-step loop — model -> tool -> model -> answer — without an API key.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.agents.chat import ChatAgent
from dexta_intelligence.agents.reason import ToolSpec, run_reasoning_loop
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.models import GlucoseEvent
from dexta_intelligence.store import SQLiteStore

_END = datetime(2026, 6, 1, tzinfo=UTC)
_START = _END - timedelta(days=40)


# ── fake native-tool-calling model ───────────────────────────────────────────


@dataclass
class _AIMessage:
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class _FakeToolModel:
    """Replays scripted turns. Each turn is either tool calls or a final answer."""

    def __init__(self, turns: list[Any]) -> None:
        self._turns = turns
        self.seen_tools: list[str] = []
        self.invocations = 0

    def bind_tools(self, schemas: list[dict[str, Any]]) -> _FakeToolModel:
        self.seen_tools = [s["function"]["name"] for s in schemas]
        return self

    def invoke(self, messages: list[Any]) -> _AIMessage:
        self.invocations += 1
        turn = self._turns.pop(0) if self._turns else "I have no more to say."
        if isinstance(turn, str):
            return _AIMessage(content=turn)
        return _AIMessage(tool_calls=list(turn))


def _store() -> SQLiteStore:
    store = SQLiteStore(":memory:")
    store.migrate()
    rng = random.Random(7)
    glucose: list[GlucoseEvent] = []
    for day in range(40):
        base = _START + timedelta(days=day)
        for hour, mg in ((3, 185), (4, 188), (12, 120), (13, 122)):
            for minute in (0, 15, 30, 45):
                ts = base.replace(hour=hour, minute=minute)
                glucose.append(GlucoseEvent(ts=ts, mg_dl=mg + rng.randint(-8, 8)))
    store.insert_glucose(glucose)
    return store


def _ctx(store: SQLiteStore) -> AgentContext:
    coverage = store.coverage()
    return AgentContext(
        store=store,
        window=(_START.date(), _END.date()),
        gates=ColdStartReport.from_coverage(coverage),
        run_id="chat-test",
    )


# ── the reasoning loop itself ─────────────────────────────────────────────────


def test_loop_executes_tool_then_answers() -> None:
    calls: list[dict[str, Any]] = []

    def echo(args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        calls.append(args)
        return {"value": 42}, {"value": 42}

    tool = ToolSpec(
        name="echo",
        description="echo a value",
        parameters={"type": "object", "properties": {"x": {"type": "integer"}}},
        fn=echo,
    )
    model = _FakeToolModel(
        [
            [{"name": "echo", "args": {"x": 1}, "id": "c1"}],  # model decides to act
            "The value is 42.",  # model decides it's done
        ]
    )
    result = run_reasoning_loop(model, [tool], system="s", user="u")  # type: ignore[arg-type]

    assert result.answer == "The value is 42."
    assert [s.name for s in result.steps] == ["echo"]
    assert calls == [{"x": 1}]
    assert result.evidence  # tool numbers accumulated for the guard


def test_loop_can_answer_without_any_tool() -> None:
    model = _FakeToolModel(["I don't need data for that."])
    result = run_reasoning_loop(model, [], system="s", user="hi")  # type: ignore[arg-type]
    assert result.answer == "I don't need data for that."
    assert result.steps == []


def test_loop_respects_max_steps() -> None:
    tool = ToolSpec(
        name="spin",
        description="spins",
        parameters={"type": "object", "properties": {}},
        fn=lambda _a: ({"ok": True}, {}),
    )
    # Always asks for another tool call — never answers.
    model = _FakeToolModel([[{"name": "spin", "args": {}, "id": "c"}]] * 20)
    result = run_reasoning_loop(model, [tool], system="s", user="u", max_steps=3)  # type: ignore[arg-type]
    assert result.stopped_reason == "max_steps"
    assert len(result.steps) == 3


# ── the chat agent ────────────────────────────────────────────────────────────


def test_chat_reasons_over_tool_and_answers_faithfully() -> None:
    store = _store()
    model = _FakeToolModel(
        [
            [{"name": "tod_compare", "args": {"hours_a": [3, 5], "hours_b": [12, 14]}, "id": "c1"}],
            "Your 03-05h window runs much higher than 12-14h.",  # no fabricated numbers
        ]
    )
    answer = ChatAgent(model=model).ask(_ctx(store), "are my mornings high?")  # type: ignore[arg-type]

    assert answer.faithful
    assert "tod_compare" in answer.tools_used
    # the model was actually offered the recall + stats tools
    assert "recall" in model.seen_tools
    assert "tod_compare" in model.seen_tools


def test_chat_flags_fabricated_number() -> None:
    store = _store()
    model = _FakeToolModel(
        [
            [{"name": "tod_compare", "args": {"hours_a": [3, 5], "hours_b": [12, 14]}, "id": "c1"}],
            "Your glucose is exactly 999 mg/dL every morning.",  # untraceable
        ]
    )
    answer = ChatAgent(model=model).ask(_ctx(store), "how high are my mornings?")  # type: ignore[arg-type]
    assert not answer.faithful
    assert "caution" in answer.text.lower()


def test_chat_recall_reads_memory() -> None:
    store = _store()
    captured: dict[str, Any] = {}
    model = _FakeToolModel(
        [
            [{"name": "recall", "args": {"query": "overnight"}, "id": "c1"}],
            "I have not recorded any overnight findings yet.",
        ]
    )
    answer = ChatAgent(model=model).ask(_ctx(store), "what do you know about my nights?")  # type: ignore[arg-type]
    captured["used"] = answer.tools_used
    assert "recall" in answer.tools_used
    assert answer.faithful


def test_parallel_tool_calls_in_one_turn() -> None:
    calls: list[dict[str, Any]] = []

    def echo_a(args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        calls.append({"tool": "echo_a", "args": args})
        return {"value": 1}, {"value": 1}

    def echo_b(args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        calls.append({"tool": "echo_b", "args": args})
        return {"value": 2}, {"value": 2}

    tool_a = ToolSpec(
        name="echo_a",
        description="echo a",
        parameters={"type": "object", "properties": {"x": {"type": "integer"}}},
        fn=echo_a,
    )
    tool_b = ToolSpec(
        name="echo_b",
        description="echo b",
        parameters={"type": "object", "properties": {"y": {"type": "integer"}}},
        fn=echo_b,
    )
    model = _FakeToolModel(
        [
            [
                {"name": "echo_a", "args": {"x": 1}, "id": "c1"},
                {"name": "echo_b", "args": {"y": 2}, "id": "c2"},
            ],
            "Got both results.",
        ]
    )
    result = run_reasoning_loop(model, [tool_a, tool_b], system="s", user="u")  # type: ignore[arg-type]

    assert result.answer == "Got both results."
    assert set(s.name for s in result.steps) == {"echo_a", "echo_b"}
    assert len(calls) == 2


def test_chat_continues_after_tool_fault() -> None:
    """A failing tool call mid-conversation does not abort ChatAgent.

    The loop unit covers a raising tool; this asserts the same resilience end
    to end through ChatAgent — a bad first call is recorded as not-ok, then a
    real probe runs and the model still produces a faithful answer.
    """
    store = _store()
    model = _FakeToolModel(
        [
            [{"name": "tod_compare", "args": {"bogus": True}, "id": "c1"}],  # bad args → fault
            [{"name": "tod_compare", "args": {"hours_a": [3, 5], "hours_b": [12, 14]}, "id": "c2"}],
            "After the retry, your 03-05h window runs higher than 12-14h.",
        ]
    )
    answer = ChatAgent(model=model).ask(_ctx(store), "are my mornings high?")  # type: ignore[arg-type]

    assert model.invocations == 3  # fault did not short-circuit the loop
    assert answer.tools_used == ("tod_compare", "tod_compare")
    assert answer.faithful
    assert "higher" in answer.text


def test_chat_continues_after_unknown_tool() -> None:
    store = _store()
    model = _FakeToolModel(
        [
            [{"name": "does_not_exist", "args": {}, "id": "c1"}],  # unknown tool → not-ok
            "I could not run that, but your data is available to ask about.",
        ]
    )
    answer = ChatAgent(model=model).ask(_ctx(store), "run a made-up analysis")  # type: ignore[arg-type]

    assert model.invocations == 2
    assert answer.tools_used == ("does_not_exist",)
    assert answer.faithful  # no fabricated numbers in the recovery answer


def test_tool_raising_continues_loop() -> None:
    def bad_tool(args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        raise ValueError("intentional failure")

    def good_tool(args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        return {"ok": True}, {}

    tool_bad = ToolSpec(
        name="bad",
        description="fails",
        parameters={"type": "object", "properties": {}},
        fn=bad_tool,
    )
    tool_good = ToolSpec(
        name="good",
        description="works",
        parameters={"type": "object", "properties": {}},
        fn=good_tool,
    )
    model = _FakeToolModel(
        [
            [{"name": "bad", "args": {}, "id": "c1"}],
            [{"name": "good", "args": {}, "id": "c2"}],
            "Despite the failure, here's the answer.",
        ]
    )
    result = run_reasoning_loop(model, [tool_bad, tool_good], system="s", user="u")  # type: ignore[arg-type]

    assert len(result.steps) == 2
    assert result.steps[0].name == "bad"
    assert not result.steps[0].ok
    assert result.steps[1].name == "good"
    assert result.steps[1].ok
    assert result.answer == "Despite the failure, here's the answer."
