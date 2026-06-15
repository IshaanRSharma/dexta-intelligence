"""Tests for the cadence daemon (run_cycle, run_daemon, cmd_daemon)."""

from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta

import pytest

from dexta_intelligence.cli.daemon import cmd_daemon
from dexta_intelligence.cli.main import main
from dexta_intelligence.config import Config
from dexta_intelligence.models import (
    Finding,
    FindingStatus,
    GlucoseEvent,
    Goal,
    GoalMetric,
    GoalStatus,
)
from dexta_intelligence.store import SQLiteStore
from dexta_intelligence.workflows import daemon as daemon_mod
from dexta_intelligence.workflows.daemon import CycleReport, run_cycle, run_daemon

FIXED_NOW = datetime(2025, 6, 10, 12, 0, tzinfo=UTC)


def _seeded_store() -> SQLiteStore:
    """In-memory store with two weeks of dense glucose, including a severe low."""
    store = SQLiteStore(":memory:")
    store.migrate()
    events: list[GlucoseEvent] = []
    start = FIXED_NOW - timedelta(days=14)
    ts = start
    while ts <= FIXED_NOW:
        events.append(GlucoseEvent(ts=ts, mg_dl=120))
        ts += timedelta(minutes=5)
    events.append(GlucoseEvent(ts=FIXED_NOW - timedelta(minutes=12), mg_dl=45))
    store.insert_glucose(events)
    return store


@pytest.fixture(autouse=True)
def _no_connectors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sync is a no-op: no connectors configured for the daemon under test."""
    monkeypatch.setattr(daemon_mod, "build_connectors", lambda _config: [])


def test_run_cycle_runs_all_steps() -> None:
    store = _seeded_store()
    goal = Goal(
        statement="reduce overnight lows",
        metric=GoalMetric.NOCTURNAL_TBR,
        direction="decrease",
        cadence_days=7,
        status=GoalStatus.ACTIVE,
        created_at=FIXED_NOW - timedelta(days=30),
    )
    store.insert_goal(goal)

    report = run_cycle(Config(), store, now=FIXED_NOW)

    assert isinstance(report, CycleReport)
    assert report.ok
    assert report.anomalies >= 1  # the planted severe low
    assert report.goals_ticked == 1
    assert report.deep_ran is False
    store.close()


def test_failing_step_is_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _seeded_store()

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("monitor exploded")

    monkeypatch.setattr(daemon_mod, "run_monitor", _boom)

    report = run_cycle(Config(), store, now=FIXED_NOW)

    assert not report.ok
    assert any(step == "monitor" and "monitor exploded" in msg for step, msg in report.errors)
    assert report.anomalies == 0
    store.close()


def test_deep_invokes_coordinator(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _seeded_store()
    calls: list[object] = []

    class FakeCoordinator:
        def __init__(self, *, model: object = None, config: object = None) -> None:
            del model, config

        def investigate(self, ctx: object, goal: object = None) -> list[Finding]:
            calls.append(ctx)
            return [
                Finding(
                    agent="coordinator",
                    kind="insight",
                    scope="test",
                    headline="planted finding",
                    body_md="planted finding",
                    confidence=0.9,
                    status=FindingStatus.ACTIVE,
                )
            ]

    monkeypatch.setattr(
        "dexta_intelligence.agents.coordinator.CoordinatorAgent", FakeCoordinator
    )

    report = run_cycle(Config(), store, deep=True, now=FIXED_NOW)

    assert len(calls) == 1
    assert report.deep_ran is True
    assert report.findings_persisted == 1
    store.close()


def test_run_daemon_paces_between_cycles(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _seeded_store()
    sleeps: list[float] = []
    monkeypatch.setattr(daemon_mod.time, "sleep", sleeps.append)
    reports: list[CycleReport] = []

    code = run_daemon(
        Config(),
        lambda: store,
        interval_min=5,
        deep_every=0,
        max_cycles=2,
        on_cycle=reports.append,
    )

    assert code == 0
    assert len(reports) == 2
    # Two cycles → exactly one inter-cycle sleep of interval_min minutes.
    assert sleeps == [5 * 60]


def test_cmd_daemon_once(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _seeded_store()
    monkeypatch.setattr(daemon_mod.time, "sleep", lambda _s: pytest.fail("--once must not sleep"))
    out = io.StringIO()

    code = cmd_daemon(
        config=Config(),
        db_path=None,
        out=out,
        once=True,
        opener=lambda _c, _db: store,
    )

    assert code == 0
    text = out.getvalue()
    assert "anomaly" in text
    assert "not a medical device" in text


def test_main_daemon_once(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    store = _seeded_store()
    monkeypatch.setattr(
        "dexta_intelligence.cli.daemon.open_sqlite_store", lambda _c, _db: store
    )
    monkeypatch.setattr(daemon_mod.time, "sleep", lambda _s: pytest.fail("once must not sleep"))

    code = main(["daemon", "--once"])

    assert code == 0
