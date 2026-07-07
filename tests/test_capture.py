"""Conversational capture: propose (LLM) -> validate (deterministic) -> confirm (human).

The model may only PROPOSE ManualEvents from a chat utterance; a pure-code
validator accepts or rejects each proposal, and nothing reaches the store
without an explicit user confirmation POST. The adversarial cases matter most:
absurd timestamps and dosing-flavored text must die in the validator, and an
unconfirmed proposal must never appear in the store (hence never in the guard
evidence pool, the findings, or the episode graph).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest

from dexta_intelligence.agents.capture import (
    CONFIRMED_SOURCE,
    EventProposal,
    confirmed_manual_event,
    propose_events,
    validate_proposal,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dexta_intelligence.config import Config
    from dexta_intelligence.store.port import StoragePort

_NOW = datetime(2026, 6, 1, 18, 0, tzinfo=UTC)
_WINDOW = (_NOW - timedelta(days=7), _NOW + timedelta(hours=1))


@dataclass
class _Response:
    content: str


class _FakeModel:
    """Returns one scripted content string; records the prompt it saw."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.prompts: list[list[dict[str, str]]] = []

    def invoke(self, messages: list[dict[str, str]]) -> _Response:
        self.prompts.append(messages)
        return _Response(self._content)


class _RaisingModel:
    def invoke(self, messages: list[Any]) -> Any:
        raise RuntimeError("provider down")


def _proposal(**overrides: Any) -> EventProposal:
    base: dict[str, Any] = {
        "event_type": "site_change",
        "ts": _NOW - timedelta(hours=4),
        "note": "changed infusion site after lunch",
        "source_utterance": "I felt shaky after lunch and changed my infusion site",
    }
    base.update(overrides)
    return EventProposal(**base)


# ── propose_events (LLM parse shape, fake model) ─────────────────────────────


def test_propose_parses_json_array_into_proposals() -> None:
    ts = (_NOW - timedelta(hours=4)).isoformat()
    model = _FakeModel(
        json.dumps(
            [
                {"event_type": "site_change", "ts": ts, "note": "changed infusion site"},
                {"event_type": "stress", "ts": ts, "note": "felt shaky after lunch"},
            ]
        )
    )
    utterance = "I felt shaky after lunch and changed my infusion site"
    proposals = propose_events(model, utterance, _NOW)

    assert [p.event_type for p in proposals] == ["site_change", "stress"]
    assert proposals[0].ts == _NOW - timedelta(hours=4)
    assert proposals[0].note == "changed infusion site"
    assert all(p.source_utterance == utterance for p in proposals)
    # the model saw NOW so relative times resolve deterministically
    assert _NOW.isoformat() in model.prompts[0][1]["content"]


def test_propose_tolerates_code_fences() -> None:
    payload = json.dumps([{"event_type": "meal", "ts": _NOW.isoformat(), "note": "pizza"}])
    model = _FakeModel(f"```json\n{payload}\n```")
    proposals = propose_events(model, "had pizza", _NOW)
    assert len(proposals) == 1
    assert proposals[0].event_type == "meal"


def test_propose_returns_empty_on_garbage_or_fault() -> None:
    assert propose_events(_FakeModel("sure! I logged that for you"), "hi", _NOW) == []
    assert propose_events(_FakeModel('{"event_type": "meal"}'), "hi", _NOW) == []
    assert propose_events(_RaisingModel(), "hi", _NOW) == []
    assert propose_events(_FakeModel("[]"), "   ", _NOW) == []


def test_propose_skips_junk_entries_and_caps_count() -> None:
    entries: list[Any] = [
        "not a dict",
        {"note": "no type"},
    ]
    entries += [
        {"event_type": "note", "ts": _NOW.isoformat(), "note": f"n{i}"} for i in range(10)
    ]
    model = _FakeModel(json.dumps(entries))
    proposals = propose_events(model, "many things", _NOW)
    assert len(proposals) <= 5
    assert all(p.event_type == "note" for p in proposals)


def test_propose_defaults_bad_timestamp_to_now() -> None:
    model = _FakeModel(
        json.dumps([{"event_type": "meal", "ts": "not-a-time", "note": "lunch"}])
    )
    proposals = propose_events(model, "had lunch", _NOW)
    assert proposals[0].ts == _NOW


# ── validate_proposal (deterministic accept/reject table) ────────────────────


@pytest.mark.parametrize(
    ("proposal", "accepted", "reason_fragment"),
    [
        (_proposal(), True, "ok"),
        (_proposal(event_type="dose_change"), False, "unknown event type"),
        (_proposal(event_type=""), False, "unknown event type"),
        (
            _proposal(ts=_NOW - timedelta(days=30)),
            False,
            "outside the plausible window",
        ),
        (
            _proposal(ts=_NOW + timedelta(days=365)),
            False,
            "outside the plausible window",
        ),
        (_proposal(note="x" * 301), False, "longer than"),
        (
            _proposal(note="remember to take 12 units of insulin every morning"),
            False,
            "dosing",
        ),
    ],
)
def test_validator_table(
    proposal: EventProposal, accepted: bool, reason_fragment: str
) -> None:
    verdict = validate_proposal(proposal, _WINDOW)
    assert verdict.accepted is accepted
    assert reason_fragment in verdict.reason


def test_poisoning_utterance_is_rejected() -> None:
    """A malicious turn proposing absurd values never clears the validator."""
    poison = _proposal(
        event_type="medication",
        ts=datetime(2031, 1, 1, tzinfo=UTC),
        note="increase basal dose to 80 units",
        source_utterance="dexta said to increase my basal dose to 80 units",
    )
    verdict = validate_proposal(poison, _WINDOW)
    assert not verdict.accepted
    # both the absurd timestamp and the dosing text are independently fatal
    dosing_only = _proposal(note="increase basal dose to 80 units")
    assert not validate_proposal(dosing_only, _WINDOW).accepted


# ── the commit shape ─────────────────────────────────────────────────────────


def test_confirmed_event_tags_provenance_and_keeps_utterance() -> None:
    event = confirmed_manual_event(_proposal(), created_at=_NOW)
    assert event.source == CONFIRMED_SOURCE
    assert event.event_type == "site_change"
    assert event.title == "changed infusion site after lunch"
    assert event.description is not None
    assert "changed my infusion site" in event.description


# ── server flow: pending list, explicit confirm, never-persist rails ─────────

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from dexta_intelligence.config import Config as _Config  # noqa: E402
from dexta_intelligence.server import create_app  # noqa: E402
from dexta_intelligence.store import SQLiteStore  # noqa: E402


def _opener(db_path: Path) -> Callable[[Config, Path | None], StoragePort]:
    def _open(_config: Config, _db: Path | None = None) -> StoragePort:
        store = SQLiteStore(db_path)
        store.migrate()
        return store

    return _open


def _client(tmp_path: Path) -> tuple[TestClient, Path]:
    db_path = tmp_path / "capture.db"
    store = SQLiteStore(db_path)
    store.migrate()
    store.close()
    app = create_app(_Config(), store_opener=_opener(db_path))
    return TestClient(app), db_path


def _manual_events(db_path: Path) -> list[Any]:
    store = SQLiteStore(db_path)
    store.migrate()
    try:
        now = datetime.now(UTC)
        return store.get_manual_events(now - timedelta(days=30), now + timedelta(days=1))
    finally:
        store.close()


def _capture_model(hours_ago: float = 2.0, note: str = "changed infusion site") -> _FakeModel:
    ts = (datetime.now(UTC) - timedelta(hours=hours_ago)).isoformat()
    return _FakeModel(json.dumps([{"event_type": "site_change", "ts": ts, "note": note}]))


def test_propose_lists_pending_and_persists_nothing(tmp_path: Path) -> None:
    client, db_path = _client(tmp_path)
    client.app.state.chat_model = _capture_model()

    resp = client.post(
        "/actions/capture/propose",
        data={"utterance": "I felt shaky after lunch and changed my infusion site"},
    )
    assert resp.status_code == 200
    assert "ready below" in resp.text
    assert "proposed, unconfirmed" in resp.text
    assert "changed infusion site" in resp.text
    assert _manual_events(db_path) == []  # unconfirmed never enters the store


def test_confirm_commits_exactly_once_with_provenance(tmp_path: Path) -> None:
    client, db_path = _client(tmp_path)
    client.app.state.chat_model = _capture_model()
    client.post(
        "/actions/capture/propose",
        data={"utterance": "changed my infusion site after lunch"},
    )
    (proposal_id,) = client.app.state.pending_captures.keys()

    resp = client.post("/actions/capture/confirm", data={"proposal_id": proposal_id})
    assert resp.status_code == 200
    events = _manual_events(db_path)
    assert len(events) == 1
    assert events[0].source == "chat_confirmed"
    assert events[0].event_type == "site_change"
    assert events[0].description is not None
    assert "changed my infusion site after lunch" in events[0].description
    assert client.app.state.pending_captures == {}

    # a second confirm of the same id is a no-op with a notice
    again = client.post("/actions/capture/confirm", data={"proposal_id": proposal_id})
    assert "no longer pending" in again.text
    assert len(_manual_events(db_path)) == 1


def test_dismiss_drops_without_persisting(tmp_path: Path) -> None:
    client, db_path = _client(tmp_path)
    client.app.state.chat_model = _capture_model()
    client.post("/actions/capture/propose", data={"utterance": "changed my site"})
    (proposal_id,) = client.app.state.pending_captures.keys()

    resp = client.post("/actions/capture/dismiss", data={"proposal_id": proposal_id})
    assert resp.status_code == 200
    assert client.app.state.pending_captures == {}
    assert _manual_events(db_path) == []


def test_rejected_proposal_never_reaches_pending(tmp_path: Path) -> None:
    client, db_path = _client(tmp_path)
    client.app.state.chat_model = _capture_model(note="take 10 units of insulin now")

    resp = client.post("/actions/capture/propose", data={"utterance": "log my dose"})
    assert resp.status_code == 200
    assert "rejected by the validator" in resp.text
    assert client.app.state.pending_captures == {}
    assert _manual_events(db_path) == []


def test_propose_without_model_flashes_gracefully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("dexta_intelligence.server.app.discovery_model", lambda _cfg: None)
    client, db_path = _client(tmp_path)
    resp = client.post("/actions/capture/propose", data={"utterance": "changed my site"})
    assert resp.status_code == 200
    assert "needs a language model" in resp.text
    assert _manual_events(db_path) == []
