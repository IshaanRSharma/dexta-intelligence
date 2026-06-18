"""Tests for the live investigation SSE endpoint (/api/investigate/stream).

A scripted CoordinatorAgent emits plan / running / producer_done / step events
through the ``RunTrace.on_event`` sink; we assert the SSE body carries them in
order and ends with the ``done`` evidence cards. The below-floor and error
paths degrade to a terminal ``error`` event.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from dexta_intelligence.config import Config
from dexta_intelligence.models import Finding, GlucoseEvent
from dexta_intelligence.server import create_app
from dexta_intelligence.store import SQLiteStore

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dexta_intelligence.store.port import StoragePort

_NOW = datetime(2026, 6, 16, 12, 0, tzinfo=UTC)


def _opener(db_path: str) -> Callable[[Config, Path | None], StoragePort]:
    def _open(_config: Config, _db: Path | None = None) -> StoragePort:
        store = SQLiteStore(db_path)
        store.migrate()
        return store

    return _open


def _seeded_store(tmp_path: Path) -> SQLiteStore:
    store = SQLiteStore(tmp_path / "inv.db")
    store.migrate()
    ts = _NOW - timedelta(days=12)
    while ts <= _NOW:
        store.insert_glucose([GlucoseEvent(ts=ts, mg_dl=120)])
        ts += timedelta(minutes=5)
    return store


def _client(store: SQLiteStore) -> TestClient:
    return TestClient(create_app(Config(), store_opener=_opener(store._path)))


def _read_sse(text: str) -> list[dict[str, Any]]:
    import json  # noqa: PLC0415

    events: list[dict[str, Any]] = []
    for block in text.strip().split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: ") :]))
    return events


class _ScriptedCoordinator:
    """Emits a fixed investigation script through the RunTrace sink."""

    def __init__(self, *_a: object, **_kw: object) -> None:
        pass

    def investigate(
        self, _ctx: object, goal: str | None = None, *, trace: Any = None
    ) -> list[Finding]:
        if trace is not None:
            trace.set_plan(["observation", "pattern"])
            trace.emit("running", {"producer": "observation"})
            trace.emit("producer_done", {"producer": "observation", "n_findings": 1})
            trace.step("Round 1: ran observation, pattern -> 1 finding(s)")
            trace.status = "completed"
        return [
            Finding(
                agent="observation",
                kind="pattern",
                scope="overnight",
                headline="Overnight lows cluster after late boluses.",
                body_md="**Evidence:** clustered overnight.",
                confidence=0.7,
            )
        ]


def _patch(monkeypatch: pytest.MonkeyPatch, coordinator: type) -> None:
    monkeypatch.setattr("dexta_intelligence.server.app.discovery_model", lambda _cfg: None)
    monkeypatch.setattr("dexta_intelligence.agents.coordinator.CoordinatorAgent", coordinator)


def test_investigate_stream_is_event_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _seeded_store(tmp_path)
    _patch(monkeypatch, _ScriptedCoordinator)
    resp = _client(store).get("/api/investigate/stream?q=worst high yesterday")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    store.close()


def test_investigate_stream_emits_plan_trace_and_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _seeded_store(tmp_path)
    _patch(monkeypatch, _ScriptedCoordinator)
    resp = _client(store).get("/api/investigate/stream?q=worst high")
    events = _read_sse(resp.text)
    kinds = [e["kind"] for e in events]

    assert "plan" in kinds
    assert "running" in kinds
    assert "producer_done" in kinds
    assert kinds[-1] == "done"

    plan = next(e for e in events if e["kind"] == "plan")
    assert plan["payload"]["steps"] == ["observation", "pattern"]

    done = events[-1]
    assert done["payload"]["n_findings"] == 1
    card = done["payload"]["findings"][0]
    assert card["headline"].startswith("Overnight lows")
    assert card["confidence_pct"] == 70
    assert "<strong>" in card["body_html"]

    # The finding was persisted to the store.
    persisted = SQLiteStore(store._path).get_findings(limit=10)
    assert any(f.headline.startswith("Overnight lows") for f in persisted)
    store.close()


def test_investigate_stream_below_floor_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SQLiteStore(tmp_path / "thin.db")
    store.migrate()
    store.insert_glucose([GlucoseEvent(ts=_NOW, mg_dl=120)])  # one reading, below floor
    _patch(monkeypatch, _ScriptedCoordinator)
    resp = _client(store).get("/api/investigate/stream?q=anything")
    events = _read_sse(resp.text)
    assert len(events) == 1
    assert events[0]["kind"] == "error"
    assert "Not enough data" in events[0]["payload"]["text"]
    store.close()


def test_investigate_stream_surfaces_coordinator_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _seeded_store(tmp_path)

    class _Boom:
        def __init__(self, *_a: object, **_kw: object) -> None:
            pass

        def investigate(self, *_a: object, **_kw: object) -> list[Finding]:
            raise RuntimeError("planner exploded")

    _patch(monkeypatch, _Boom)
    resp = _client(store).get("/api/investigate/stream?q=boom")
    events = _read_sse(resp.text)
    assert events[-1]["kind"] == "error"
    assert "planner exploded" in events[-1]["payload"]["text"]
    store.close()
