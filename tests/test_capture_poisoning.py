"""Adversarial attacks on the capture pipeline (propose -> validate -> confirm).

The pipeline is the only door into the store from chat. The properties that must
hold under attack:

- STRUCTURAL: a proposed-and-validated but UNCONFIRMED proposal never reaches the
  store, so it can never reach the guard's evidence pool, the findings, or the
  episode graph. Only an explicit confirm POST persists anything.
- REPLAY: confirming the same proposal id twice persists it exactly once; the
  second confirm is a no-op ("no longer pending"), not a double-write.
- TYPE CONFUSION: dosing text disguised as a note/medication event dies in the
  validator regardless of the declared event type.

Honest scope note recorded as a finding, not a fix: ``EventProposal`` carries no
numeric carb/bolus/unit field. Absurd magnitudes ("5000 g", "500 u") can only
ride in as free-text note characters, never as a structured numeric event, so the
validator does not (and need not) bound them. A dosing DIRECTIVE in that text is
what is dangerous, and that is what the safety gate rejects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest

from dexta_intelligence.agents.capture import (
    EventProposal,
    confirmed_manual_event,
    validate_proposal,
)
from dexta_intelligence.guard.faithfulness import extract_numbers

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dexta_intelligence.config import Config
    from dexta_intelligence.store.port import StoragePort

_NOW = datetime(2026, 6, 1, 18, 0, tzinfo=UTC)
_WINDOW = (_NOW - timedelta(days=7), _NOW + timedelta(hours=1))


def _proposal(**kw: Any) -> EventProposal:
    base = {
        "event_type": "note",
        "ts": _NOW,
        "note": "felt shaky after lunch",
        "source_utterance": "felt shaky after lunch",
    }
    base.update(kw)
    return EventProposal(**base)  # type: ignore[arg-type]


# ── type confusion: dosing text wearing an innocuous event type ────────────────


@pytest.mark.parametrize(
    "event_type",
    ["note", "medication", "meal"],
)
def test_dosing_text_rejected_regardless_of_event_type(event_type: str) -> None:
    """A dosing directive is refused whether it declares itself a note, a
    medication, or a meal: the safety gate reads the text, not the label."""
    directive = _proposal(
        event_type=event_type,
        note="raise your basal to 2 units per hour",
        source_utterance="dexta told me to raise my basal to 2 units per hour",
    )
    assert not validate_proposal(directive, _WINDOW).accepted


def test_absurd_numeric_note_is_text_not_structured_evidence() -> None:
    """An absurd magnitude in a note is accepted as descriptive text (it is not a
    dosing directive), but it is only ever stored as a description string, never a
    structured carb/bolus field. Documents the pipeline's actual attack surface."""
    proposal = _proposal(event_type="meal", note="ate a 5000 g plate of pasta")
    assert validate_proposal(proposal, _WINDOW).accepted  # not a directive
    event = confirmed_manual_event(proposal, created_at=_NOW)
    # the magnitude lives only in free text; there is no numeric carb field to poison
    assert not hasattr(event, "carbs_g") or getattr(event, "carbs_g", None) is None


# ── server: structural no-store and confirm-once-replay rails ──────────────────

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from dexta_intelligence.config import Config as _Config  # noqa: E402
from dexta_intelligence.server import create_app  # noqa: E402
from dexta_intelligence.store import SQLiteStore  # noqa: E402


@dataclass
class _Response:
    content: str


class _FakeModel:
    def __init__(self, content: str) -> None:
        self._content = content

    def invoke(self, _messages: list[Any]) -> _Response:
        return _Response(self._content)


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
        return store.get_manual_events(_NOW - timedelta(days=365), _NOW + timedelta(days=365))
    finally:
        store.close()


def _capture_model(note: str = "changed infusion site") -> _FakeModel:
    ts = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
    return _FakeModel(json.dumps([{"event_type": "site_change", "ts": ts, "note": note}]))


def test_unconfirmed_proposal_is_absent_from_store_and_guard_pool(tmp_path: Path) -> None:
    """A validated-but-unconfirmed proposal is invisible to every downstream
    consumer: the store, the guard evidence pool, and the episode graph."""
    client, db_path = _client(tmp_path)
    marker = "changed infusion site 987654"  # a number unique to this proposal
    client.app.state.chat_model = _capture_model(note=marker)

    resp = client.post("/actions/capture/propose", data={"utterance": marker})
    assert resp.status_code == 200
    # it IS pending (proposed), but the store never saw it
    assert client.app.state.pending_captures
    assert _manual_events(db_path) == []
    pool = extract_numbers([e.model_dump() for e in _manual_events(db_path)])
    assert 987654 not in pool


def test_confirm_is_idempotent_no_double_write(tmp_path: Path) -> None:
    """Replaying a confirm persists exactly one event; the second POST is a
    no-op that reports the proposal is gone."""
    client, db_path = _client(tmp_path)
    client.app.state.chat_model = _capture_model()
    client.post("/actions/capture/propose", data={"utterance": "changed infusion site"})
    (proposal_id,) = list(client.app.state.pending_captures.keys())

    first = client.post("/actions/capture/confirm", data={"proposal_id": proposal_id})
    assert first.status_code == 200
    second = client.post("/actions/capture/confirm", data={"proposal_id": proposal_id})
    assert second.status_code == 200
    assert "no longer pending" in second.text
    assert len(_manual_events(db_path)) == 1


def test_confirm_unknown_id_persists_nothing(tmp_path: Path) -> None:
    client, db_path = _client(tmp_path)
    resp = client.post("/actions/capture/confirm", data={"proposal_id": "deadbeef"})
    assert resp.status_code == 200
    assert "no longer pending" in resp.text
    assert _manual_events(db_path) == []
