"""Tests for the live investigation SSE endpoint (/api/investigate/stream).

Two modes: question (default) runs the OrchestratorAgent and streams per-tool
tool_call/tool_result events plus an audited answer (the PRD tool shelf + trace);
deep runs the CoordinatorAgent statistical sweep. Both persist an
InvestigationRun. Below-floor and error paths degrade to a terminal error.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from dexta_intelligence.agents.chat import ChatAnswer
from dexta_intelligence.agents.reason import ReasoningEvent
from dexta_intelligence.agents.trace import TraceLine
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


# ── question mode: the orchestrator drill (PRD tool shelf + trace) ──────────────


class _ScriptedOrchestrator:
    """Emits a per-tool script then returns an audited answer."""

    def __init__(self, *_a: object, **_kw: object) -> None:
        pass

    def ask(
        self,
        _ctx: object,
        _question: str,
        *,
        on_event: Callable[[ReasoningEvent], None] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> ChatAnswer:
        if on_event is not None:
            on_event(ReasoningEvent("tool_call", {"name": "find_spikes", "args": {"day": "x"}}))
            on_event(ReasoningEvent("tool_result", {"name": "find_spikes", "ok": True}))
            on_event(ReasoningEvent("tool_call", {"name": "get_boluses", "args": {}}))
            on_event(ReasoningEvent("tool_result", {"name": "get_boluses", "ok": True}))
        return ChatAnswer(
            text="The worst high followed a late bolus.",
            tools_used=("find_spikes", "get_boluses"),
            faithful=True,
            stopped_reason="answered",
            trace=(
                TraceLine("zoom", "scanned for excursions (1 spike)"),
                TraceLine("treatment", "checked bolus timing (1 bolus; +22 min vs carb entry)"),
            ),
        )


def _patch_orchestrator(monkeypatch: pytest.MonkeyPatch, agent: type) -> None:
    monkeypatch.setattr("dexta_intelligence.server.app.discovery_model", lambda _cfg: object())
    monkeypatch.setattr("dexta_intelligence.agents.orchestrator.OrchestratorAgent", agent)


def test_question_mode_streams_tool_calls_then_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _seeded_store(tmp_path)
    _patch_orchestrator(monkeypatch, _ScriptedOrchestrator)
    resp = _client(store).get("/api/investigate/stream?q=worst high yesterday")
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = _read_sse(resp.text)
    kinds = [e["kind"] for e in events]

    assert kinds[0] == "coverage"
    assert kinds.count("tool_call") == 2
    assert kinds.count("tool_result") == 2
    assert kinds[-1] == "answer"
    assert events[1]["payload"]["name"] == "find_spikes"
    final = events[-1]
    assert "late bolus" in final["payload"]["text"]
    assert "<p>" in final["payload"]["html"]
    assert final["payload"]["faithful"] is True
    store.close()


def test_question_mode_persists_a_run_with_real_tool_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _seeded_store(tmp_path)
    _patch_orchestrator(monkeypatch, _ScriptedOrchestrator)
    _client(store).get("/api/investigate/stream?q=worst high")
    runs = SQLiteStore(store._path).get_investigation_runs(limit=5)
    assert runs
    run = runs[0]
    assert run.kind == "question"
    assert [c["name"] for c in run.tool_calls] == ["find_spikes", "get_boluses"]
    assert all(c["ok"] is True for c in run.tool_calls)
    assert run.answer == "The worst high followed a late bolus."
    assert run.trace  # rendered PRD trace lines persisted
    store.close()


def test_persisted_question_run_renders_answer_and_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _seeded_store(tmp_path)
    _patch_orchestrator(monkeypatch, _ScriptedOrchestrator)
    client = _client(store)
    client.get("/api/investigate/stream?q=worst high")
    body = client.get("/investigations").text
    assert "late bolus" in body  # the answer prose
    assert "find_spikes" in body  # the real tool, not a producer name
    assert "Tools called" in body
    store.close()


def test_question_mode_without_model_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _seeded_store(tmp_path)
    monkeypatch.setattr("dexta_intelligence.server.app.discovery_model", lambda _cfg: None)
    events = _read_sse(_client(store).get("/api/investigate/stream?q=anything").text)
    assert events[-1]["kind"] == "error"
    assert "needs a language model" in events[-1]["payload"]["text"]
    store.close()


# ── deep mode: the coordinator statistical sweep ────────────────────────────────


class _ScriptedCoordinator:
    def __init__(self, *_a: object, **_kw: object) -> None:
        pass

    def investigate(
        self, _ctx: object, goal: str | None = None, *, trace: Any = None
    ) -> list[Finding]:
        if trace is not None:
            trace.set_plan(["observation", "pattern"])
            trace.emit("running", {"producer": "observation"})
            trace.producer_done("observation", 1)
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


def _patch_coordinator(monkeypatch: pytest.MonkeyPatch, coordinator: type) -> None:
    monkeypatch.setattr("dexta_intelligence.server.app.discovery_model", lambda _cfg: None)
    monkeypatch.setattr("dexta_intelligence.agents.coordinator.CoordinatorAgent", coordinator)


def test_deep_mode_streams_plan_trace_and_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _seeded_store(tmp_path)
    _patch_coordinator(monkeypatch, _ScriptedCoordinator)
    resp = _client(store).get("/api/investigate/stream?q=worst high&mode=deep")
    events = _read_sse(resp.text)
    kinds = [e["kind"] for e in events]

    assert "plan" in kinds
    assert "producer_done" in kinds
    assert kinds[-1] == "done"
    done = events[-1]
    assert done["payload"]["n_findings"] == 1
    assert done["payload"]["findings"][0]["headline"].startswith("Overnight lows")
    store.close()


def test_below_floor_errors_in_both_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SQLiteStore(tmp_path / "thin.db")
    store.migrate()
    store.insert_glucose([GlucoseEvent(ts=_NOW, mg_dl=120)])  # one reading, below floor
    _patch_coordinator(monkeypatch, _ScriptedCoordinator)
    events = _read_sse(_client(store).get("/api/investigate/stream?q=x&mode=deep").text)
    assert len(events) == 1
    assert events[0]["kind"] == "error"
    assert "Not enough data" in events[0]["payload"]["text"]
    store.close()


def test_deep_mode_surfaces_coordinator_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _seeded_store(tmp_path)

    class _Boom:
        def __init__(self, *_a: object, **_kw: object) -> None:
            pass

        def investigate(self, *_a: object, **_kw: object) -> list[Finding]:
            raise RuntimeError("planner exploded")

    _patch_coordinator(monkeypatch, _Boom)
    events = _read_sse(_client(store).get("/api/investigate/stream?q=boom&mode=deep").text)
    assert events[-1]["kind"] == "error"
    assert "planner exploded" in events[-1]["payload"]["text"]
    store.close()
