"""Conversational capture: PROPOSE (LLM) -> VALIDATE (deterministic) -> CONFIRM (human).

Free text in chat ("I felt shaky after lunch and changed my infusion site") is
real signal that otherwise evaporates when the conversation ends. Capturing it
silently with an LLM would poison the graph, so the pipeline is split:

- :func:`propose_events` lets the model parse ONE utterance into candidate
  :class:`EventProposal` objects. It proposes; it persists nothing.
- :func:`validate_proposal` is pure code: known event type, timestamp inside
  the plausible window, sane field sizes, and nothing that reads as dosing
  content. Anything else is rejected with a reason.
- The commit gate is the user: the server's /log surface lists accepted
  proposals pre-filled, and only an explicit POST persists one, tagged
  ``source="chat_confirmed"`` with the source utterance retained.

Unconfirmed proposals never touch the store, so they can never reach the
guard's evidence pool, the findings, or the episode graph.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from dexta_intelligence.agents import prompts
from dexta_intelligence.agents.brief import _ADVICE_RE
from dexta_intelligence.models import ManualEvent

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "CONFIRMED_SOURCE",
    "KNOWN_EVENT_TYPES",
    "EventProposal",
    "ValidationResult",
    "confirmed_manual_event",
    "propose_events",
    "validate_proposal",
]

#: The ManualEvent event types the validator accepts (the ManualEvent contract).
KNOWN_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "meal",
        "exercise",
        "sleep",
        "illness",
        "stress",
        "alcohol",
        "site_change",
        "sensor_issue",
        "pump_issue",
        "medication",
        "travel",
        "note",
    }
)

#: Provenance tag on a committed capture: proposed in chat, confirmed by the user.
CONFIRMED_SOURCE = "chat_confirmed"

_MAX_PROPOSALS = 5
_MAX_NOTE_CHARS = 300
_MAX_UTTERANCE_CHARS = 1000

_SYSTEM = prompts.with_safety(prompts.load("capture_system"))


@dataclass(frozen=True, slots=True)
class EventProposal:
    """One candidate ManualEvent parsed from a chat turn. Never persisted as-is."""

    event_type: str
    ts: datetime
    note: str
    source_utterance: str


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """The deterministic verdict on one proposal."""

    accepted: bool
    reason: str


def propose_events(model: Any, utterance: str, now: datetime) -> list[EventProposal]:
    """LLM parse of one chat utterance into candidate proposals.

    Proposing only: nothing here touches the store, and every proposal must
    still clear :func:`validate_proposal` and an explicit user confirmation.
    Returns ``[]`` on an empty utterance, a model fault, or non-JSON output.
    """
    text = utterance.strip()[:_MAX_UTTERANCE_CHARS]
    if not text:
        return []
    user = f"NOW: {now.isoformat()}\n\nMessage:\n{text}"
    try:
        response = model.invoke(
            [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user},
            ]
        )
        raw = _parse_json_array(getattr(response, "content", response))
    except Exception:
        logger.warning("capture: proposal model call failed", exc_info=True)
        return []
    proposals: list[EventProposal] = []
    for entry in raw[:_MAX_PROPOSALS]:
        proposal = _proposal_from(entry, text, now)
        if proposal is not None:
            proposals.append(proposal)
    return proposals


def validate_proposal(
    proposal: EventProposal, window: tuple[datetime, datetime]
) -> ValidationResult:
    """Deterministic gate: schema, plausibility, and safety, or a reason string.

    ``window`` is the plausible span for the event timestamp (e.g. the recent
    conversation window). A proposal outside it, of an unknown type, with an
    oversized note, or whose text reads as dosing/treatment content is rejected.
    """
    if proposal.event_type not in KNOWN_EVENT_TYPES:
        return ValidationResult(False, f"unknown event type {proposal.event_type!r}")
    lo, hi = window
    if not lo <= proposal.ts <= hi:
        return ValidationResult(
            False,
            f"timestamp {proposal.ts.isoformat()} outside the plausible window "
            f"{lo.isoformat()}..{hi.isoformat()}",
        )
    if len(proposal.note) > _MAX_NOTE_CHARS:
        return ValidationResult(False, f"note longer than {_MAX_NOTE_CHARS} characters")
    if _ADVICE_RE.search(f"{proposal.note} {proposal.source_utterance}"):
        return ValidationResult(False, "reads as dosing/treatment content")
    return ValidationResult(True, "ok")


def confirmed_manual_event(proposal: EventProposal, *, created_at: datetime) -> ManualEvent:
    """The ManualEvent a user confirmation commits.

    Provenance is explicit: ``source="chat_confirmed"`` and the source
    utterance retained in the description, so a committed capture is always
    distinguishable from a hand-logged event.
    """
    return ManualEvent(
        event_type=proposal.event_type,
        event_ts=proposal.ts,
        title=proposal.note or None,
        description=f'Confirmed from chat: "{proposal.source_utterance}"',
        source=CONFIRMED_SOURCE,
        created_at=created_at,
    )


def _parse_json_array(content: Any) -> Sequence[Any]:
    if isinstance(content, list):
        content = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    if not isinstance(content, str):
        return []
    text = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        logger.warning("capture: non-JSON proposal response: %s", text[:200])
        return []
    return parsed if isinstance(parsed, list) else []


def _proposal_from(entry: Any, utterance: str, now: datetime) -> EventProposal | None:
    if not isinstance(entry, dict):
        return None
    event_type = str(entry.get("event_type", "")).strip().lower()
    if not event_type:
        return None
    ts = _parse_ts(entry.get("ts"), now)
    note = str(entry.get("note", "")).strip()
    return EventProposal(
        event_type=event_type,
        ts=ts,
        note=note,
        source_utterance=utterance,
    )


def _parse_ts(raw: Any, now: datetime) -> datetime:
    """An entry's timestamp, defaulting to ``now`` on anything unparseable."""
    if not isinstance(raw, str):
        return now
    try:
        ts = datetime.fromisoformat(raw)
    except ValueError:
        return now
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=now.tzinfo)
