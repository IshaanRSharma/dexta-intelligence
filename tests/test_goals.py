"""Tests for goal workflows — composition, measurement, and background ticks."""

from __future__ import annotations

import json
import os
import random
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.models import (
    GlucoseEvent,
    GoalMetric,
    GoalStatus,
    InsulinEvent,
    InsulinKind,
)
from dexta_intelligence.store import SQLiteStore
from dexta_intelligence.store.sqlite import SCHEMA_VERSION
from dexta_intelligence.workflows.goals import (
    compose_goal,
    goal_due,
    measure_metric,
    tick_goal,
)

_END = datetime(2026, 6, 1, tzinfo=UTC)
_START = _END - timedelta(days=30)
_NOW = _END


@dataclass
class _Reply:
    content: str


@dataclass
class _AIMessage:
    content: str = ""
    tool_calls: list[dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.tool_calls is None:
            object.__setattr__(self, "tool_calls", [])


class _FakeModel:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = json.dumps(payload)

    def invoke(self, _messages: list[Any]) -> _Reply:
        return _Reply(self._payload)


class _FakeToolModel:
    """Tool-calling model for reasoning loop tests."""

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


def _store_with_nocturnal_lows() -> SQLiteStore:
    store = SQLiteStore(":memory:")
    store.migrate()
    rng = random.Random(11)
    glucose: list[GlucoseEvent] = []
    for day in range(30):
        base = _START + timedelta(days=day)
        for hour in range(24):
            mg = 60 if hour < 6 else 130
            for minute in (0, 20, 40):
                ts = base.replace(hour=hour, minute=minute)
                glucose.append(GlucoseEvent(ts=ts, mg_dl=mg + rng.randint(-6, 6)))
    store.insert_glucose(glucose)
    return store


def _ctx(store: SQLiteStore) -> AgentContext:
    coverage = store.coverage()
    return AgentContext(
        store=store,
        window=(_START.date(), _END.date()),
        gates=ColdStartReport.from_coverage(coverage),
        run_id="goal-test",
    )


def test_keyword_compose_maps_lows_to_nocturnal_tbr() -> None:
    goal = compose_goal("reduce my overnight lows", now=_NOW)
    assert goal.metric is GoalMetric.NOCTURNAL_TBR
    assert goal.direction == "decrease"
    assert goal.tools  # a default plan was attached


def test_llm_compose_uses_model_choice() -> None:
    model = _FakeModel(
        {
            "metric": "tir",
            "direction": "increase",
            "cadence_days": 5,
            "tools": [
                {"tool": "groupby_compare", "args": {"group_by": "weekend", "target": "tir_pct"}}
            ],
        }
    )
    goal = compose_goal("get more time in range", model=model, now=_NOW)  # type: ignore[arg-type]
    assert goal.metric is GoalMetric.TIR
    assert goal.direction == "increase"
    assert goal.cadence_days == 5


def test_compose_goal_with_explicit_target() -> None:
    goal = compose_goal("reduce my overnight lows", now=_NOW, target=5.0)
    assert goal.target == 5.0
    assert goal.metric is GoalMetric.NOCTURNAL_TBR


def test_explicit_target_overrides_llm_target() -> None:
    model = _FakeModel(
        {
            "metric": "tir",
            "direction": "increase",
            "cadence_days": 7,
            "target": 70.0,
            "tools": [],
        }
    )
    goal = compose_goal("more time in range", model=model, now=_NOW, target=85.0)  # type: ignore[arg-type]
    assert goal.target == 85.0


def test_llm_compose_supplies_target() -> None:
    model = _FakeModel(
        {
            "metric": "tir",
            "direction": "increase",
            "cadence_days": 7,
            "target": 75.0,
            "tools": [],
        }
    )
    goal = compose_goal("more time in range", model=model, now=_NOW)  # type: ignore[arg-type]
    assert goal.target == 75.0


def test_compose_goal_without_target_leaves_none() -> None:
    goal = compose_goal("reduce my overnight lows", now=_NOW)
    assert goal.target is None


def test_tick_with_explicit_composed_target_marks_achieved() -> None:
    store = _store_with_nocturnal_lows()
    # nocturnal TBR is high (~100); decrease toward a generous target → already met
    goal = compose_goal("reduce my overnight lows", now=_NOW, target=200.0)
    assert goal.target == 200.0
    goal_id = store.insert_goal(goal)
    stored = store.get_goals()[0]

    result = tick_goal(stored, _ctx(store), now=_NOW)
    store.insert_goal_checkpoint(result.checkpoint)
    if result.achieved and stored.id is not None:
        store.set_goal_status(stored.id, GoalStatus.ACHIEVED)

    assert result.achieved
    assert store.get_goals()[0].status is GoalStatus.ACHIEVED
    assert goal_id == stored.id


def test_measure_nocturnal_tbr_detects_planted_lows() -> None:
    store = _store_with_nocturnal_lows()
    value = measure_metric(GoalMetric.NOCTURNAL_TBR, _ctx(store))
    assert value is not None and value > 80  # nights are deep in range-below


def test_tick_records_checkpoint_and_arc() -> None:
    store = _store_with_nocturnal_lows()
    goal = compose_goal("reduce my overnight lows", now=_NOW)
    goal_id = store.insert_goal(goal)
    stored = store.get_goals()[0]

    first = tick_goal(stored, _ctx(store), now=_NOW)
    store.insert_goal_checkpoint(first.checkpoint)
    assert first.checkpoint.metric_value is not None
    assert "Baseline" in first.checkpoint.note

    second = tick_goal(stored, _ctx(store), now=_NOW + timedelta(days=7))
    store.insert_goal_checkpoint(second.checkpoint)
    arc = store.get_goal_checkpoints(goal_id)
    assert len(arc) == 2
    assert "→" in second.checkpoint.note  # arc compares to the prior value


def test_tick_marks_goal_achieved_when_target_met() -> None:
    store = _store_with_nocturnal_lows()
    goal = compose_goal("reduce my overnight lows", now=_NOW)
    # target the metric is already past (lots of lows → high TBR; target high so it's "met")
    achievable = goal.model_copy(update={"target": 200.0})
    goal_id = store.insert_goal(achievable)
    stored = store.get_goals()[0]

    result = tick_goal(stored, _ctx(store), now=_NOW)
    store.insert_goal_checkpoint(result.checkpoint)
    if result.achieved and stored.id is not None:
        store.set_goal_status(stored.id, GoalStatus.ACHIEVED)
    # nocturnal TBR is well below 200, decrease direction → achieved
    assert result.achieved
    assert store.get_goals()[0].status is GoalStatus.ACHIEVED
    assert goal_id == stored.id


def test_tick_with_llm_model_takes_reasoning_loop_path() -> None:
    store = _store_with_nocturnal_lows()
    goal = compose_goal("reduce my overnight lows", now=_NOW)
    store.insert_goal(goal)
    stored = store.get_goals()[0]

    model = _FakeToolModel(["Your overnight lows are around 60 mg/dL."])
    result = tick_goal(stored, _ctx(store), now=_NOW, model=model)  # type: ignore[arg-type]

    assert "60 mg/dL" in result.checkpoint.note
    assert model.invocations > 0


def test_goal_due_cadence_logic() -> None:
    store = _store_with_nocturnal_lows()
    goal = compose_goal("reduce my overnight lows", now=_NOW)
    goal = goal.model_copy(update={"cadence_days": 7})
    goal_id = store.insert_goal(goal)
    stored = store.get_goals()[0]

    first = tick_goal(stored, _ctx(store), now=_NOW)
    store.insert_goal_checkpoint(first.checkpoint)
    checkpoints = store.get_goal_checkpoints(goal_id)

    # checkpoint just made; not yet due
    assert not goal_due(stored, checkpoints, now=_NOW)

    # 6 days later; still not due
    assert not goal_due(stored, checkpoints, now=_NOW + timedelta(days=6))

    # 7 days later; now due
    assert goal_due(stored, checkpoints, now=_NOW + timedelta(days=7))

    # no checkpoints; always due
    assert goal_due(stored, [], now=_NOW)


def test_compose_goal_with_malformed_json_falls_back_to_keyword() -> None:
    class _BadModel:
        def invoke(self, _messages: list[Any]) -> _Reply:
            return _Reply("not json")

    goal = compose_goal("reduce my overnight lows", model=_BadModel(), now=_NOW)  # type: ignore[arg-type]
    assert goal.metric is GoalMetric.NOCTURNAL_TBR
    assert goal.direction == "decrease"


def _store_with_nocturnal_lows_and_bolus() -> SQLiteStore:
    store = SQLiteStore(":memory:")
    store.migrate()
    rng = random.Random(11)
    glucose: list[GlucoseEvent] = []
    insulin: list[InsulinEvent] = []
    for day in range(30):
        base = _START + timedelta(days=day)
        for hour in range(24):
            mg = 60 if hour < 6 else 130
            for minute in (0, 20, 40):
                ts = base.replace(hour=hour, minute=minute)
                glucose.append(GlucoseEvent(ts=ts, mg_dl=mg + rng.randint(-6, 6)))
            # Add bolus insulin events a few times per day, near lows (early morning hours)
            if hour in (0, 2, 4):
                ts = base.replace(hour=hour, minute=rng.randint(0, 59))
                insulin.append(
                    InsulinEvent(ts=ts, kind=InsulinKind.BOLUS, units=5.0)
                )
    store.insert_glucose(glucose)
    store.insert_insulin(insulin)
    return store


def test_moderate_large_plan_effect_banks_hypothesis_once() -> None:
    store = _store_with_nocturnal_lows_and_bolus()
    goal = compose_goal("reduce my overnight lows", now=_NOW)
    goal_id = store.insert_goal(goal)
    stored = store.get_goals()[0]

    # First tick: should bank hypothesis if effect is large/moderate
    first = tick_goal(stored, _ctx(store), now=_NOW)
    store.insert_goal_checkpoint(first.checkpoint)

    # Second tick: hypothesis should not be duplicated
    second = tick_goal(stored, _ctx(store), now=_NOW + timedelta(days=7))
    store.insert_goal_checkpoint(second.checkpoint)

    hypotheses = store.get_hypotheses(status="open")
    goal_statements = [h.statement for h in hypotheses if f"[goal #{goal_id}]" in h.statement]
    # Each unique statement should appear exactly once
    assert len(goal_statements) == len(set(goal_statements))


def test_migrate_upgrades_schema_version() -> None:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        store = SQLiteStore(tmp_path)
        store.migrate()
        assert store._conn.execute(
            "SELECT version FROM schema_version"
        ).fetchone()[0] == SCHEMA_VERSION

        # Simulate old schema by downgrading version
        store._conn.execute("UPDATE schema_version SET version = 1")
        store._conn.commit()

        # Migrate again
        store.migrate()
        new_version = store._conn.execute(
            "SELECT version FROM schema_version"
        ).fetchone()[0]
        assert new_version == SCHEMA_VERSION
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
