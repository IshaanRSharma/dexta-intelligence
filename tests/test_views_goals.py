"""Tests for the Goals page view-model (``views_goals.goal_card_view``).

Deterministic: a real in-memory SQLite store, no model or network calls. A
fixed tz-aware ``now`` makes relative-time strings stable.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from dexta_intelligence.models import (
    Goal,
    GoalCheckpoint,
    GoalMetric,
    InvestigationRun,
)
from dexta_intelligence.server.views_goals import goal_card_view
from dexta_intelligence.store import SQLiteStore

NOW = datetime(2026, 6, 17, 12, 0, tzinfo=UTC)
STATEMENT = "increase time in range"


def _store() -> SQLiteStore:
    store = SQLiteStore(":memory:")
    store.migrate()
    return store


def _run(question: str | None, *, run_id: str, finished: datetime) -> InvestigationRun:
    return InvestigationRun(
        run_id=run_id,
        kind="question",
        status="completed",
        question=question,
        window_start=date(2026, 5, 1),
        window_end=date(2026, 6, 1),
        plan=["observation"],
        trace=["ran"],
        findings=[],
        n_findings=2,
        started_at=finished,
        finished_at=finished,
    )


def _seed() -> tuple[SQLiteStore, int]:
    store = _store()
    goal_id = store.insert_goal(
        Goal(
            statement=STATEMENT,
            metric=GoalMetric.TIR,
            direction="increase",
            target=70.0,
            cadence_days=7,
        )
    )
    for i, value in enumerate((31.6, 40.0, 55.0)):
        store.insert_goal_checkpoint(
            GoalCheckpoint(
                goal_id=goal_id,
                ts=NOW - timedelta(days=14 - i * 7),
                metric_value=value,
                note=f"checkpoint {i}",
            )
        )
    store.insert_investigation_run(
        _run(STATEMENT, run_id="match", finished=NOW - timedelta(hours=2))
    )
    store.insert_investigation_run(
        _run("some other question", run_id="other", finished=NOW - timedelta(hours=1))
    )
    return store, goal_id


def test_progress_numbers() -> None:
    store, goal_id = _seed()
    goal = store.get_goals()[0]
    card = goal_card_view(store, goal, now=NOW)

    assert card["id"] == goal_id
    assert card["statement"] == STATEMENT
    assert card["status"] == "active"
    assert card["metric"] == GoalMetric.TIR.value
    assert card["direction"] == "increase"
    assert card["target"] == 70.0
    assert card["baseline"] == pytest.approx(31.6)
    assert card["current"] == pytest.approx(55.0)
    assert card["delta"] == pytest.approx(23.4)
    assert card["n_checkpoints"] == 3


def test_pct_to_target_and_on_track() -> None:
    store, _ = _seed()
    card = goal_card_view(store, store.get_goals()[0], now=NOW)
    # (55 - 31.6) / (70 - 31.6) = 0.6094 -> 61
    assert card["pct_to_target"] == 61
    assert card["on_track"] is True


def test_spark_and_next_check() -> None:
    store, _ = _seed()
    card = goal_card_view(store, store.get_goals()[0], now=NOW)
    assert isinstance(card["spark"], str)
    assert "svg" in card["spark"]
    assert "polyline" in card["spark"]
    assert isinstance(card["next_check"], str)


def test_checkpoints_newest_first_with_notes() -> None:
    store, _ = _seed()
    card = goal_card_view(store, store.get_goals()[0], now=NOW)
    cps = card["checkpoints"]
    assert len(cps) == 3
    assert [c["value"] for c in cps] == pytest.approx([55.0, 40.0, 31.6])
    assert cps[0]["note"] == "checkpoint 2"
    assert all(isinstance(c["when"], str) for c in cps)


def test_only_matching_question_runs_appear() -> None:
    store, _ = _seed()
    card = goal_card_view(store, store.get_goals()[0], now=NOW)
    assert len(card["runs"]) == 1
    assert card["runs"][0]["n_findings"] == 2
    assert card["runs"][0]["status"] == "completed"
    assert isinstance(card["runs"][0]["when"], str)


def test_no_checkpoints_yields_none_fields() -> None:
    store = _store()
    store.insert_goal(
        Goal(
            statement="empty goal",
            metric=GoalMetric.CV,
            direction="decrease",
            target=30.0,
        )
    )
    card = goal_card_view(store, store.get_goals()[0], now=NOW)
    assert card["baseline"] is None
    assert card["current"] is None
    assert card["delta"] is None
    assert card["on_track"] is None
    assert card["pct_to_target"] is None
    assert card["next_check"] is None
    assert card["n_checkpoints"] == 0
    assert card["checkpoints"] == []


def test_goal_card_degrades_when_runs_table_missing() -> None:
    """A goal card must render even if get_investigation_runs fails (old DB)."""
    store = _store()
    goal_id = store.insert_goal(
        Goal(statement=STATEMENT, metric=GoalMetric.TIR, direction="increase")
    )
    store.insert_goal_checkpoint(
        GoalCheckpoint(goal_id=goal_id, ts=NOW, metric_value=31.6, note="baseline")
    )
    # Simulate an older schema without the investigation_runs table.
    store._conn.execute("DROP TABLE investigation_runs")
    store._conn.commit()

    card = goal_card_view(store, store.get_goals()[0], now=NOW)
    assert card["runs"] == []
    assert card["baseline"] == 31.6  # the rest of the card still builds
    store.close()
