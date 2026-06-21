"""Tests for the router / supervisor agent.

The router classifies a question into a tool family, exposes only that family's
tools to the reasoning loop, and still finishes through the faithfulness guard.
We reuse the scripted ``_FakeToolModel`` pattern (tool calls vs final answer)
and its ``bind_tools`` capture of ``seen_tools`` to prove the focused subset.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.agents.router import FAMILY_TOOLS, Route, RouterAgent
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
    """Replays scripted turns. Each turn is tool calls or a final answer string.

    The first ``invoke`` is the router's classification call (no tools bound yet);
    later calls drive the reasoning loop. ``seen_tools`` captures the focused
    subset the loop bound.
    """

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
        run_id="router-test",
    )


# ── route() classification ────────────────────────────────────────────────────


def test_route_to_two_group_for_comparison() -> None:
    store = _store()
    model = _FakeToolModel(['{"family": "two_group"}'])
    route = RouterAgent(model=model).route(
        _ctx(store), "do my weekends compare worse than weekdays?"
    )
    assert route.name == "two_group"
    # recall + coverage are always present; family instruments included.
    assert "recall" in route.tool_names
    assert "coverage" in route.tool_names
    assert "tod_compare" in route.tool_names
    assert "set_window" not in route.tool_names


def test_route_to_time_traversal_exposes_set_window_not_meal_response() -> None:
    store = _store()
    model = _FakeToolModel(
        [
            '{"family": "time_traversal"}',
            [{"name": "set_window", "args": {"start": "2026-03-01", "end": "2026-03-31"},
              "id": "c1"}],
            "March and April look similar.",
        ]
    )
    answer = RouterAgent(model=model).ask(
        _ctx(store), "what changed in March vs April?"
    )
    assert answer.faithful
    # The loop was bound to the FOCUSED subset: a time tool in, a two-group tool out.
    assert "set_window" in model.seen_tools
    assert "meal_response" not in model.seen_tools
    assert "recall" in model.seen_tools


# ── keyword fallback (no model) ───────────────────────────────────────────────


def test_keyword_fallback_when_model_none() -> None:
    store = _store()
    agent = RouterAgent(model=None)
    assert agent.route(_ctx(store), "weekend vs weekday glucose").name == "two_group"
    assert agent.route(_ctx(store), "what changed this month over time?").name == "time_traversal"
    assert agent.route(_ctx(store), "what do you already know about my nights?").name == "memory"
    assert agent.route(_ctx(store), "is there clinical evidence for this?").name == "evidence"


# ── empty / invalid route falls back to the full belt ─────────────────────────


def test_invalid_family_falls_back_to_keyword_route() -> None:
    store = _store()
    model = _FakeToolModel(['{"family": "not_a_real_family"}'])
    route = RouterAgent(model=model).route(
        _ctx(store), "weekend vs weekday?"
    )
    assert route.name in FAMILY_TOOLS
    assert route.name == "two_group"


def test_empty_route_exposes_full_belt() -> None:
    store = _store()
    # A route whose names match nothing in the belt must fall back to the full belt.
    model = _FakeToolModel(["Nothing to add."])
    agent = RouterAgent(model=model)

    bogus = Route(name="bogus", system="s", tool_names=("does_not_exist",))
    agent.route = lambda _ctx, _q: bogus  # type: ignore[method-assign,assignment]
    agent.ask(_ctx(store), "anything")
    # The full belt was bound because the focused subset was empty. The belt is
    # capability-filtered: this store has no meal/insulin data, so treatment
    # and meal tools are hidden even from the full belt.
    assert "set_window" in model.seen_tools
    assert "recall" in model.seen_tools
    assert "tod_compare" in model.seen_tools
    assert "meal_response" not in model.seen_tools
    assert "get_boluses" not in model.seen_tools


# ── the guard still runs per-route ────────────────────────────────────────────


def test_guard_runs_and_flags_fabricated_number() -> None:
    store = _store()
    model = _FakeToolModel(
        [
            '{"family": "two_group"}',
            [{"name": "tod_compare", "args": {"hours_a": [3, 5], "hours_b": [12, 14]},
              "id": "c1"}],
            "Your glucose is exactly 999 mg/dL every morning.",
        ]
    )
    answer = RouterAgent(model=model).ask(
        _ctx(store), "how high are my mornings vs middays?"
    )
    assert not answer.faithful
    assert "caution" in answer.text.lower()
