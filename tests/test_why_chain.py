"""The why-chain surfaced: episode context on ChatAnswer and the episode card.

When an answer traverses an episode node (explain_episode), the structured
node rides home on ``ChatAnswer.episode_context`` and the server renders it as
a card: kind, span, duration, extreme, severity, and the typed context edges
with signed offsets. All deterministic tool output; no model authored any of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import pytest

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.agents.chat import ChatAgent, ChatAnswer
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.models import GlucoseEvent, MealEvent
from dexta_intelligence.server.views_episode import episode_card_view
from dexta_intelligence.store import SQLiteStore

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dexta_intelligence.agents.reason import ReasoningEvent
    from dexta_intelligence.store.port import StoragePort

_END = datetime(2026, 6, 1, tzinfo=UTC)
_START = _END - timedelta(days=7)
_SPIKE_DAY = _START + timedelta(days=1)


@dataclass
class _AIMessage:
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class _FakeToolModel:
    def __init__(self, turns: list[Any]) -> None:
        self._turns = turns

    def bind_tools(self, schemas: list[dict[str, Any]]) -> _FakeToolModel:
        return self

    def invoke(self, messages: list[Any]) -> _AIMessage:
        turn = self._turns.pop(0) if self._turns else "Done."
        if isinstance(turn, str):
            return _AIMessage(content=turn)
        return _AIMessage(tool_calls=list(turn))


def _seeded_store(path: str = ":memory:") -> SQLiteStore:
    store = SQLiteStore(path)
    store.migrate()
    glucose: list[GlucoseEvent] = []
    for day in range(7):
        base = _START + timedelta(days=day)
        for hour in range(0, 24, 2):
            for minute in (0, 20, 40):
                glucose.append(GlucoseEvent(ts=base.replace(hour=hour, minute=minute), mg_dl=120))
    for minute in (0, 10, 20, 30, 40, 50):
        glucose.append(GlucoseEvent(ts=_SPIKE_DAY.replace(hour=3, minute=minute), mg_dl=230))
    store.insert_glucose(glucose)
    store.insert_meals(
        [MealEvent(ts=_SPIKE_DAY.replace(hour=2, minute=40), carbs_g=62.0, note="late snack")]
    )
    return store


def _ctx(store: SQLiteStore) -> AgentContext:
    return AgentContext(
        store=store,
        window=(_START.date(), _END.date()),
        gates=ColdStartReport.from_coverage(store.coverage()),
        run_id="why-chain-test",
    )


def test_chat_answer_carries_traversed_episode() -> None:
    store = _seeded_store()
    when = _SPIKE_DAY.replace(hour=3, minute=20).isoformat()
    model = _FakeToolModel(
        [
            [{"name": "explain_episode", "args": {"timestamp": when}, "id": "c1"}],
            "That high lasted a while; a meal preceded it.",
        ]
    )
    answer = ChatAgent(model=model).ask(_ctx(store), "tell me about that 3am high")  # type: ignore[arg-type]

    episode = answer.episode_context
    assert episode is not None
    assert episode["kind"] == "hyper"
    assert episode["extreme_mg_dl"] == 230.0
    kinds = [link["kind"] for link in episode["links"]]
    assert "meal" in kinds


def test_chat_answer_without_traversal_has_no_episode() -> None:
    store = _seeded_store()
    model = _FakeToolModel(["Nothing to traverse."])
    answer = ChatAgent(model=model).ask(_ctx(store), "tell me about my week")  # type: ignore[arg-type]
    assert answer.episode_context is None


def test_episode_card_view_shapes_edges_with_signed_offsets() -> None:
    episode = {
        "kind": "hyper",
        "start": "2026-05-26T03:00:00+00:00",
        "end": "2026-05-26T03:50:00+00:00",
        "duration_min": 50.0,
        "severe": False,
        "clinically_significant": True,
        "extreme_mg_dl": 230.0,
        "links": [
            {
                "kind": "meal",
                "ts": "2026-05-26T02:40:00+00:00",
                "offset_min": -20.0,
                "detail": {"carbs_g": 62.0, "note": "late snack"},
            },
            {
                "kind": "bolus",
                "ts": "2026-05-26T03:10:00+00:00",
                "offset_min": 10.0,
                "detail": {"units": 4.5, "automatic": True},
            },
        ],
    }
    view = episode_card_view(episode, ZoneInfo("UTC"))

    assert view["title"] == "High episode"
    assert view["duration"] == "50 min"
    assert view["extreme"] == "230 mg/dL peak"
    assert view["clinically_significant"] is True
    meal, bolus = view["edges"]
    assert meal["offset"] == "-20 min"
    assert "62 g carbs" in meal["detail"]
    assert bolus["offset"] == "+10 min"
    assert "4.5 U" in bolus["detail"]
    assert "automatic" in bolus["detail"]


# ── server rendering (gated on the optional gui extra) ───────────────────────

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from dexta_intelligence.config import Config as _Config  # noqa: E402
from dexta_intelligence.server import create_app  # noqa: E402

_EPISODE = {
    "id": "hyper:2026-05-26T03:00:00+00:00",
    "kind": "hyper",
    "start": "2026-05-26T03:00:00+00:00",
    "end": "2026-05-26T03:50:00+00:00",
    "duration_min": 50.0,
    "n_readings": 6,
    "severe": False,
    "clinically_significant": True,
    "extreme_mg_dl": 230.0,
    "extreme_ts": "2026-05-26T03:00:00+00:00",
    "links": [
        {
            "kind": "meal",
            "ts": "2026-05-26T02:40:00+00:00",
            "offset_min": -20.0,
            "detail": {"carbs_g": 62.0, "note": "late snack"},
        }
    ],
}


class _EpisodeAgent:
    def __init__(self, **_kw: Any) -> None:
        pass

    def ask(
        self,
        _ctx: object,
        _question: str,
        *,
        on_event: Callable[[ReasoningEvent], None] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> ChatAnswer:
        return ChatAnswer(
            text="That high followed a 62 g snack.",
            tools_used=("explain_episode",),
            faithful=True,
            stopped_reason="answered",
            episode_context=_EPISODE,
        )


def _opener(db_path: Path) -> Callable[[_Config, Path | None], StoragePort]:
    def _open(_config: _Config, _db: Path | None = None) -> StoragePort:
        store = SQLiteStore(db_path)
        store.migrate()
        return store

    return _open


def _client(tmp_path: Path) -> TestClient:
    db_path = tmp_path / "why.db"
    store = SQLiteStore(db_path)
    store.migrate()
    store.close()
    app = create_app(_Config(), store_opener=_opener(db_path))
    return TestClient(app)


def _patch_model_and_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dexta_intelligence.server.app.discovery_model", lambda _cfg: object())
    monkeypatch.setattr("dexta_intelligence.agents.orchestrator.OrchestratorAgent", _EpisodeAgent)


def test_api_ask_renders_episode_card(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_model_and_agent(monkeypatch)
    resp = _client(tmp_path).post("/api/ask", data={"question": "tell me about that high"})
    assert resp.status_code == 200
    assert "High episode" in resp.text
    assert "62 g carbs" in resp.text
    assert "-20 min" in resp.text


def test_stream_answer_carries_episode_html(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json  # noqa: PLC0415

    _patch_model_and_agent(monkeypatch)
    resp = _client(tmp_path).get("/api/ask/stream?q=tell me about that high")
    events = [
        json.loads(line[len("data: ") :])
        for line in resp.text.splitlines()
        if line.startswith("data: ")
    ]
    final = events[-1]
    assert final["kind"] == "answer"
    assert "High episode" in final["payload"]["episode_html"]
    assert "62 g carbs" in final["payload"]["episode_html"]
