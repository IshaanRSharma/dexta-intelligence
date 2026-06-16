"""Durable chat-history tests — the ``chat_turns`` StoragePort surface (SQLite).

Turns are scoped by ``session_id`` and returned oldest→newest, capped to the
most-recent ``limit`` rows for the session.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from dexta_intelligence.models import ChatTurn
from dexta_intelligence.store import SQLiteStore

if TYPE_CHECKING:
    from collections.abc import Iterator

T0 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)


@pytest.fixture
def store() -> Iterator[SQLiteStore]:
    s = SQLiteStore(":memory:")
    s.migrate()
    yield s
    s.close()


def _turn(session_id: str, role: str, content: str, ts: datetime) -> ChatTurn:
    return ChatTurn(session_id=session_id, role=role, content=content, ts=ts)


class TestChatTurns:
    def test_append_returns_id(self, store: SQLiteStore) -> None:
        assert store.append_chat_turn(_turn("s1", "user", "hi", T0)) == 1
        assert store.append_chat_turn(_turn("s1", "assistant", "hello", T0)) == 2

    def test_fresh_session_is_empty(self, store: SQLiteStore) -> None:
        assert store.get_chat_turns("never-used") == []

    def test_round_trip_chronological_and_scoped(self, store: SQLiteStore) -> None:
        roles = ["user", "assistant", "user", "assistant"]
        for i, role in enumerate(roles):
            store.append_chat_turn(
                _turn("s1", role, f"s1-{i}", T0 + timedelta(minutes=i))
            )
        # interleave a second session to prove scoping
        store.append_chat_turn(_turn("s2", "user", "other", T0))

        got = store.get_chat_turns("s1")
        assert [t.role for t in got] == roles
        assert [t.content for t in got] == ["s1-0", "s1-1", "s1-2", "s1-3"]
        assert {t.session_id for t in got} == {"s1"}
        # ids ascending == chronological
        assert [t.id for t in got] == sorted(t.id for t in got if t.id is not None)

        (other,) = store.get_chat_turns("s2")
        assert other.content == "other"

    def test_limit_returns_most_recent_but_chronological(self, store: SQLiteStore) -> None:
        for i in range(5):
            store.append_chat_turn(_turn("s1", "user", f"m{i}", T0 + timedelta(minutes=i)))
        got = store.get_chat_turns("s1", limit=2)
        # most-recent 2 (m3, m4) but returned oldest→newest
        assert [t.content for t in got] == ["m3", "m4"]

    def test_ts_round_trips_as_aware_utc(self, store: SQLiteStore) -> None:
        store.append_chat_turn(_turn("s1", "user", "hi", T0))
        (got,) = store.get_chat_turns("s1")
        assert got.ts == T0
        assert got.ts.tzinfo == UTC

    def test_migrate_is_idempotent(self, store: SQLiteStore) -> None:
        store.append_chat_turn(_turn("s1", "user", "before", T0))
        store.migrate()  # second migrate: existing data untouched, still works
        store.append_chat_turn(_turn("s1", "assistant", "after", T0 + timedelta(minutes=1)))
        got = store.get_chat_turns("s1")
        assert [t.content for t in got] == ["before", "after"]
