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
            on_event(ReasoningEvent("answer_start", {}))
            on_event(ReasoningEvent("answer_delta", {"delta": "Your TIR was "}))
            on_event(ReasoningEvent("answer_delta", {"delta": "68%."}))
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
    assert kinds[:4] == ["tool_call", "tool_result", "tool_call", "tool_result"]
    assert "answer_start" in kinds
    assert kinds.count("answer_delta") == 2
    assert kinds[-1] == "answer"

    assert events[0]["payload"]["name"] == "tir_snapshot"
    assert events[1]["payload"]["ok"] is True
    assert events[3]["payload"]["ok"] is False

    streamed = "".join(
        e["payload"]["delta"] for e in events if e["kind"] == "answer_delta"
    )
    assert streamed == "Your TIR was 68%."

    final = events[-1]
    assert final["payload"]["text"] == "Your TIR was 68%."
    assert "<p>" in final["payload"]["html"]
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
    # detail is logged server-side, never sent to the client
    assert "model exploded" not in events[-1]["payload"]["text"]
    assert "went wrong" in events[-1]["payload"]["text"].lower()
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


def test_history_endpoint_returns_session_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    _HistoryCapturingAgent.seen_history = []
    _patch_model_and_agent(monkeypatch, _HistoryCapturingAgent)
    client = _client(store)

    client.get("/api/ask/stream?q=why did I spike Tuesday&sid=s1")
    data = client.get("/api/history?sid=s1").json()

    assert [t["role"] for t in data["turns"]] == ["user", "assistant"]
    assert data["turns"][0]["content"] == "why did I spike Tuesday"
    assert data["turns"][1]["html"]  # assistant turn carries rendered html
    store.close()


def test_history_endpoint_empty_without_session(tmp_path: Path) -> None:
    store = _store(tmp_path)
    client = _client(store)
    assert client.get("/api/history").json() == {"turns": []}
    assert client.get("/api/history?sid=unknown").json() == {"turns": []}
    store.close()


def test_sessions_endpoint_lists_distinct_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    _HistoryCapturingAgent.seen_history = []
    _patch_model_and_agent(monkeypatch, _HistoryCapturingAgent)
    client = _client(store)

    client.get("/api/ask/stream?q=why did I spike Tuesday&sid=s1")
    client.get("/api/ask/stream?q=what about my sleep&sid=s2")

    sessions = client.get("/api/sessions").json()["sessions"]
    by_id = {s["session_id"]: s for s in sessions}
    assert set(by_id) == {"s1", "s2"}
    assert by_id["s1"]["preview"] == "why did I spike Tuesday"
    assert by_id["s2"]["preview"] == "what about my sleep"
    assert by_id["s1"]["turn_count"] == 2  # one user + one assistant turn
    # Newest-active first: s2 was asked last.
    assert sessions[0]["session_id"] == "s2"
    store.close()


def test_sessions_endpoint_empty_without_turns(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert _client(store).get("/api/sessions").json() == {"sessions": []}
    store.close()


def test_delete_session_removes_turns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store(tmp_path)
    _patch_model_and_agent(monkeypatch, _HistoryCapturingAgent)
    client = _client(store)

    client.get("/api/ask/stream?q=first question&sid=s1")
    client.get("/api/ask/stream?q=second question&sid=s2")
    assert len(client.get("/api/sessions").json()["sessions"]) == 2

    resp = client.delete("/api/sessions/s1")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "deleted": 2}
    sessions = client.get("/api/sessions").json()["sessions"]
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == "s2"
    assert client.get("/api/history?sid=s1").json() == {"turns": []}
    store.close()


def test_delete_missing_session_returns_404(tmp_path: Path) -> None:
    store = _store(tmp_path)
    resp = _client(store).delete("/api/sessions/ghost")
    assert resp.status_code == 404
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
