"""Tests for the live-streaming reasoning SSE endpoint.

No live model: a fake OrchestratorAgent emits scripted ReasoningEvents through
the ``on_event`` sink, and we assert the SSE body carries them in order ending
with the audited answer. The no-model path is asserted to degrade gracefully.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from dexta_intelligence.agents.chat import ChatAnswer
from dexta_intelligence.agents.reason import ReasoningEvent
from dexta_intelligence.config import Config
from dexta_intelligence.server import create_app
from dexta_intelligence.server.app import _sse
from dexta_intelligence.store import SQLiteStore

if TYPE_CHECKING:
    from collections.abc import Callable

    from dexta_intelligence.store.port import StoragePort


def _opener(db_path: Path) -> Callable[[Config, Path | None], StoragePort]:
    def _open(_config: Config, _db: Path | None = None) -> StoragePort:
        store = SQLiteStore(db_path)
        store.migrate()
        return store

    return _open


def _store(tmp_path: Path) -> SQLiteStore:
    store = SQLiteStore(tmp_path / "stream.db")
    store.migrate()
    return store


def _client(store: SQLiteStore) -> TestClient:
    app = create_app(Config(), store_opener=_opener(Path(store._path)))
    return TestClient(app)


class _ScriptedAgent:
    """Emits a fixed event script through on_event, then returns an answer."""

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
            on_event(ReasoningEvent("tool_call", {"name": "tir_snapshot", "args": {}}))
            on_event(ReasoningEvent("tool_result", {"name": "tir_snapshot", "ok": True}))
            on_event(ReasoningEvent("tool_call", {"name": "find_spikes", "args": {"day": "x"}}))
            on_event(ReasoningEvent("tool_result", {"name": "find_spikes", "ok": False}))
            # The loop's own raw answer must be suppressed by the endpoint.
            on_event(ReasoningEvent("answer", {"text": "raw loop text"}))
        return ChatAnswer(
            text="Your TIR was 68%.",
            tools_used=("tir_snapshot", "find_spikes"),
            faithful=True,
            stopped_reason="answered",
        )


def _patch_model_and_agent(monkeypatch: pytest.MonkeyPatch, agent: type) -> None:
    monkeypatch.setattr(
        "dexta_intelligence.server.app.discovery_model", lambda _cfg: object()
    )
    monkeypatch.setattr(
        "dexta_intelligence.agents.orchestrator.OrchestratorAgent", agent
    )


def _read_sse(text: str) -> list[dict[str, Any]]:
    import json  # noqa: PLC0415

    events: list[dict[str, Any]] = []
    for block in text.strip().split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: ") :]))
    return events


def test_stream_endpoint_is_event_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    _patch_model_and_agent(monkeypatch, _ScriptedAgent)
    resp = _client(store).get("/api/ask/stream?q=how is my TIR")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    store.close()


def test_stream_emits_events_in_order_ending_with_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    _patch_model_and_agent(monkeypatch, _ScriptedAgent)
    resp = _client(store).get("/api/ask/stream?q=why did I spike")
    events = _read_sse(resp.text)

    kinds = [e["kind"] for e in events]
    assert kinds == ["tool_call", "tool_result", "tool_call", "tool_result", "answer"]

    assert events[0]["payload"]["name"] == "tir_snapshot"
    assert events[1]["payload"]["ok"] is True
    assert events[3]["payload"]["ok"] is False

    final = events[-1]
    assert final["payload"]["text"] == "Your TIR was 68%."
    assert "tir_snapshot" in final["payload"]["tools"]
    assert final["payload"]["faithful"] is True
    # The loop's raw answer never reaches the client.
    assert "raw loop text" not in resp.text
    store.close()


def test_stream_no_model_returns_graceful_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    monkeypatch.setattr(
        "dexta_intelligence.server.app.discovery_model", lambda _cfg: None
    )
    resp = _client(store).get("/api/ask/stream?q=anything")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = _read_sse(resp.text)
    assert len(events) == 1
    assert events[0]["kind"] == "error"
    assert "language model" in events[0]["payload"]["text"]
    store.close()


def test_stream_surfaces_agent_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)

    class _Boom:
        def __init__(self, *_a: object, **_kw: object) -> None:
            pass

        def ask(self, *_a: object, **_kw: object) -> ChatAnswer:
            raise RuntimeError("model exploded")

    _patch_model_and_agent(monkeypatch, _Boom)
    resp = _client(store).get("/api/ask/stream?q=boom")
    assert resp.status_code == 200
    events = _read_sse(resp.text)
    assert events[-1]["kind"] == "error"
    assert "model exploded" in events[-1]["payload"]["text"]
    store.close()


class _HistoryCapturingAgent:
    """Records the history it was handed and echoes a turn-numbered answer."""

    seen_history: ClassVar[list[list[dict[str, Any]] | None]] = []

    def __init__(self, *_a: object, **_kw: object) -> None:
        pass

    def ask(
        self,
        _ctx: object,
        question: str,
        *,
        on_event: Callable[[ReasoningEvent], None] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> ChatAnswer:
        type(self).seen_history.append(history)
        return ChatAnswer(
            text=f"answer to {question}",
            tools_used=(),
            faithful=True,
            stopped_reason="answered",
        )


def test_session_carries_history_across_asks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    _HistoryCapturingAgent.seen_history = []
    _patch_model_and_agent(monkeypatch, _HistoryCapturingAgent)
    client = _client(store)

    client.get("/api/ask/stream?q=why did I spike Tuesday&sid=s1")
    client.get("/api/ask/stream?q=what about Wednesday&sid=s1")

    first, second = _HistoryCapturingAgent.seen_history
    assert first is None or first == []  # nothing before the first turn
    # The second ask sees the first turn threaded in.
    assert {"role": "user", "content": "why did I spike Tuesday"} in second
    assert {"role": "assistant", "content": "answer to why did I spike Tuesday"} in second
    store.close()


def test_no_sid_is_stateless(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store(tmp_path)
    _HistoryCapturingAgent.seen_history = []
    _patch_model_and_agent(monkeypatch, _HistoryCapturingAgent)
    client = _client(store)

    client.get("/api/ask/stream?q=first")
    client.get("/api/ask/stream?q=second")
    assert _HistoryCapturingAgent.seen_history == [None, None]  # no session → no memory
    store.close()


def test_sse_serializer_frames_event() -> None:
    frame = _sse({"kind": "tool_call", "payload": {"name": "find_spikes", "args": {}}})
    assert frame.startswith("data: ")
    assert frame.endswith("\n\n")
    import json  # noqa: PLC0415

    decoded = json.loads(frame[len("data: ") :].strip())
    assert decoded["kind"] == "tool_call"
    assert decoded["payload"]["name"] == "find_spikes"
