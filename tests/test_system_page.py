"""System observability page: the pure view-model + the route."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from dexta_intelligence.config import Config
from dexta_intelligence.models import (
    Finding,
    GlucoseEvent,
    InvestigationRun,
    ManualEvent,
    RunFinding,
)
from dexta_intelligence.server.views_system import system_page_view
from dexta_intelligence.store import SQLiteStore

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dexta_intelligence.store.port import StoragePort

_NOW = datetime(2026, 6, 16, 12, 0, tzinfo=UTC)


def _seeded(store: SQLiteStore) -> SQLiteStore:
    ts = _NOW - timedelta(days=10)
    while ts <= _NOW:
        store.insert_glucose([GlucoseEvent(ts=ts, mg_dl=120)])
        ts += timedelta(minutes=5)
    store.add_manual_event(
        ManualEvent(event_type="meal", event_ts=_NOW, title="dinner", created_at=_NOW)
    )
    store.insert_investigation_run(
        InvestigationRun(
            run_id="r1",
            kind="question",
            status="completed",
            question="why overnight lows?",
            window_start=date(2026, 6, 6),
            window_end=date(2026, 6, 16),
            plan=["observation"],
            trace=["Round 1"],
            findings=[RunFinding(headline="h", kind="pattern", confidence=0.7, status="active")],
            n_findings=1,
            started_at=_NOW,
            finished_at=_NOW,
            coverage_summary={"glucose_coverage_pct": 90.0, "limited": False},
            tool_calls=[{"producer": "observation", "n_findings": 1}],
            evidence_items=[{"finding": "h", "numbers": {"mean": 120}}],
        )
    )
    store.insert_finding(
        Finding(agent="observation", kind="pattern", scope="overnight", headline="active one")
    )
    return store


@pytest.fixture
def store() -> SQLiteStore:
    s = SQLiteStore(":memory:")
    s.migrate()
    return s


def test_view_model_sections(store: SQLiteStore) -> None:
    _seeded(store)
    view = system_page_view(Config(), store, _NOW)

    assert view["pipeline"]["glucose"] > 0
    assert view["pipeline"]["manual_logs"] == 1
    assert view["agent_runs"]["total"] == 1
    assert view["agent_runs"]["by_status"]["completed"] == 1
    assert view["instruments"][0]["name"] == "observation"
    assert view["instruments"][0]["findings"] == 1
    assert view["rigor"]["active_findings"] >= 1
    assert view["model"]["provider"] == "anthropic"


def test_view_model_empty_store_is_safe(store: SQLiteStore) -> None:
    view = system_page_view(Config(), store, _NOW)
    assert view["pipeline"]["raw_events"] == 0
    assert view["agent_runs"]["total"] == 0
    assert view["instruments"] == []


pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from dexta_intelligence.server import create_app  # noqa: E402


def _opener(db_path: Path) -> Callable[[Config, Path | None], StoragePort]:
    def _open(_config: Config, _db: Path | None = None) -> StoragePort:
        s = SQLiteStore(db_path)
        s.migrate()
        return s

    return _open


def test_system_route_renders(tmp_path: Path) -> None:
    db = tmp_path / "sys.db"
    seed = SQLiteStore(db)
    seed.migrate()
    _seeded(seed)
    client = TestClient(create_app(Config(), store_opener=_opener(db)))
    resp = client.get("/system")
    assert resp.status_code == 200
    assert "Pipeline health" in resp.text
    assert "Agent runs" in resp.text
    assert "Instruments called" in resp.text
    assert "Rigor signals" in resp.text
