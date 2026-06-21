from __future__ import annotations

from datetime import UTC, datetime

from dexta_intelligence.models import OpenInvestigation
from dexta_intelligence.store.sqlite import SQLiteStore


def _store() -> SQLiteStore:
    store = SQLiteStore(":memory:")
    store.migrate()
    return store


def _make(
    *,
    question: str = "Why do my lows cluster?",
    condition_type: str = "event_count",
    subject: str = "nocturnal_low",
    target: float = 5.0,
    current: float = 0.0,
    status: str = "collecting",
) -> OpenInvestigation:
    return OpenInvestigation(
        question=question,
        condition_type=condition_type,
        subject=subject,
        target=target,
        current=current,
        status=status,
        created_at=datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
    )


def test_insert_returns_id() -> None:
    store = _store()
    inv_id = store.insert_open_investigation(_make())
    assert isinstance(inv_id, int)
    assert inv_id == 1


def test_round_trip_preserves_all_fields() -> None:
    store = _store()
    inv = _make(
        question="Do meals before workouts spike me?",
        condition_type="days_elapsed",
        subject="",
        target=14.0,
        current=3.0,
        status="collecting",
    )
    inv_id = store.insert_open_investigation(inv)

    got = store.get_open_investigations()
    assert len(got) == 1
    row = got[0]
    assert row.id == inv_id
    assert row.question == inv.question
    assert row.condition_type == inv.condition_type
    assert row.subject == inv.subject
    assert row.target == inv.target
    assert row.current == inv.current
    assert row.status == inv.status
    assert row.created_at == inv.created_at
    assert row.promoted_run_id is None


def test_get_filters_by_status() -> None:
    store = _store()
    store.insert_open_investigation(_make(status="collecting"))
    store.insert_open_investigation(_make(status="ready"))
    store.insert_open_investigation(_make(status="collecting"))

    collecting = store.get_open_investigations(status="collecting")
    assert len(collecting) == 2
    assert all(r.status == "collecting" for r in collecting)

    ready = store.get_open_investigations(status="ready")
    assert len(ready) == 1
    assert ready[0].status == "ready"


def test_update_changes_fields() -> None:
    store = _store()
    inv_id = store.insert_open_investigation(_make(current=0.0, status="collecting"))

    store.update_open_investigation(
        inv_id, current=5.0, status="promoted", promoted_run_id="run-abc"
    )

    row = store.get_open_investigations()[0]
    assert row.id == inv_id
    assert row.current == 5.0
    assert row.status == "promoted"
    assert row.promoted_run_id == "run-abc"


def test_get_returns_newest_first() -> None:
    store = _store()
    first = store.insert_open_investigation(_make(question="first"))
    second = store.insert_open_investigation(_make(question="second"))

    rows = store.get_open_investigations()
    assert [r.id for r in rows] == [second, first]
    assert rows[0].question == "second"
