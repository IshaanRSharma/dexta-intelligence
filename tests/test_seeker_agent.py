"""Tests for the goal-seeking replanning loop (``GoalSeekingAgent``).

The scripted model serves BOTH reasoning-loop plan turns and reflect turns
from one queue, in the order the seeker calls ``model.invoke``: each round is
[plan turn(s) → answer turn] (the reasoning loop) followed by one reflect turn
(a JSON Reflection). This exercises the real multi-round replan without an API
key, extending the ``_FakeToolModel`` pattern from ``tests/test_chat_agent.py``.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.agents.seeker import GoalSeekingAgent
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.models import GlucoseEvent
from dexta_intelligence.store import SQLiteStore

_END = datetime(2026, 6, 1, tzinfo=UTC)
_START = _END - timedelta(days=40)


# ── fake native-tool-calling model (plan + reflect turns) ─────────────────────


@dataclass
class _AIMessage:
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class _FakeToolModel:
    """Replays scripted turns. A turn is either tool calls, a final answer
    string, or a reflect JSON string - served from one queue in call order."""

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


def _reflect(satisfied: bool, missing: str = "", next_hint: str = "") -> str:
    return json.dumps({"satisfied": satisfied, "missing": missing, "next_hint": next_hint})


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
        run_id="seeker-test",
    )


# ── the goal-seeking agent ────────────────────────────────────────────────────


def test_under_answer_then_replan_runs_two_rounds() -> None:
    """Round 1 compares months but never zooms; reflection says missing the
    spike; round 2 zooms then answers. Exactly two rounds run; final faithful."""
    store = _store()
    model = _FakeToolModel(
        [
            [{"name": "list_segments", "args": {}, "id": "c1"}],
            "I compared the months but did not look inside any spike.",
            _reflect(False, missing="zoom the April spike", next_hint="zoom_event on the spike"),
            [{"name": "zoom_event", "args": {"timestamp": "2026-05-15T13:00:00"}, "id": "c2"}],
            "Zooming the spike, your midday window peaks then settles.",
            _reflect(True),
        ]
    )
    agent = GoalSeekingAgent(model=model)

    answer = agent.pursue(_ctx(store), "what happened around my April spike?")

    assert answer.faithful
    assert answer.tools_used == ("list_segments", "zoom_event")
    # 2 plan turns + 2 answer turns + 2 reflect turns == 6 invocations (2 rounds)
    assert model.invocations == 6


def test_full_answer_round_one_stops_after_one_round() -> None:
    """A satisfying reflection after round 1 halts the loop immediately."""
    store = _store()
    model = _FakeToolModel(
        [
            [{"name": "tod_compare", "args": {"hours_a": [3, 5], "hours_b": [12, 14]}, "id": "c1"}],
            "Your 03-05h window runs much higher than 12-14h.",
            _reflect(True),
        ]
    )
    agent = GoalSeekingAgent(model=model)

    answer = agent.pursue(_ctx(store), "are my mornings high?")

    assert answer.faithful
    assert answer.tools_used == ("tod_compare",)
    # one plan turn + one answer turn + one reflect turn == 3
    assert model.invocations == 3


def test_goal_pursuit_can_compose_investigations() -> None:
    """Goals now get the orchestrator belt: a goal can call an investigation
    shortcut, not just read a metric."""
    store = _store()
    model = _FakeToolModel(
        [
            [{"name": "investigate_spike", "args": {"when": "2026-05-20"}, "id": "c1"}],
            "Composed a spike investigation toward the goal.",
            _reflect(True),
        ]
    )
    agent = GoalSeekingAgent(model=model)

    answer = agent.pursue(_ctx(store), "reduce my dinner spikes")

    assert "investigate_spike" in model.seen_tools
    assert "investigate_spike" in answer.tools_used


def test_model_none_single_round_no_crash() -> None:
    """No model → reflection falls back to satisfied; a single round, no loop."""
    store = _store()
    agent = GoalSeekingAgent(model=None)

    answer = agent.pursue(_ctx(store), "what do you know?")

    # model is None: the reasoning loop returns a model_error answer, which
    # _finish renders as a safe fallback marked faithful. No exception.
    assert answer.faithful
    assert answer.tools_used == ()


def test_round_one_number_stays_faithful_in_final_answer() -> None:
    """A number computed in round 1 and cited in the round-2 answer must stay
    faithful: proves evidence accumulation across rounds feeds the guard."""
    store = _store()
    # tod_compare on the 03-05h vs 12-14h windows yields a real mean_a near 185.
    # The merged pool must carry the round-1 number so the guard does not false-reject.
    model = _FakeToolModel(
        [
            [{"name": "tod_compare", "args": {"hours_a": [3, 5], "hours_b": [12, 14]}, "id": "c1"}],
            "Mornings look elevated; I should confirm the daily trend.",
            _reflect(False, missing="check the daily trend", next_hint="daily_series mean_glucose"),
            [{"name": "daily_series", "args": {"metric": "mean_glucose"}, "id": "c2"}],
            "Your mornings average about 185 mg/dL, and the daily trend confirms it.",
            _reflect(True),
        ]
    )
    agent = GoalSeekingAgent(model=model)

    answer = agent.pursue(_ctx(store), "are my mornings high?")

    assert answer.faithful
    assert "185" in answer.text
    assert answer.tools_used == ("tod_compare", "daily_series")
