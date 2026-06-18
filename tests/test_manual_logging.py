"""Manual context logging (PRD section 19) end to end.

Covers the store roundtrip, the three read-only belt tools
(get_manual_events / search_manual_events / get_context_around_event), their
trace lines, and the user-submitted ``/log`` form. The defining rule: manual
events are created ONLY by the user form, never by an agent tool.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.agents.discovery_tools import DiscoveryToolkit, tool_specs
from dexta_intelligence.agents.reason import ToolCall
from dexta_intelligence.agents.trace import render_trace
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.models import GlucoseEvent, ManualEvent
from dexta_intelligence.store import SQLiteStore

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dexta_intelligence.store.port import StoragePort

_DAY = date(2026, 6, 16)
_DINNER = datetime(2026, 6, 16, 19, 0, tzinfo=UTC)
_STRESS = datetime(2026, 6, 16, 9, 0, tzinfo=UTC)


def _seeded_store() -> SQLiteStore:
    store = SQLiteStore(":memory:")
    store.migrate()
    base = datetime(2026, 6, 16, 12, 0, tzinfo=UTC)
    store.insert_glucose(
        [GlucoseEvent(ts=base + timedelta(minutes=5 * i), mg_dl=120 + i) for i in range(40)]
    )
    store.add_manual_event(
        ManualEvent(
            event_type="meal",
            event_ts=_DINNER,
            title="high-fat dinner",
            description="pizza",
            tags=["fat", "dinner"],
            intensity="high",
            created_at=_DINNER,
        )
    )
    store.add_manual_event(
        ManualEvent(
            event_type="stress",
            event_ts=_STRESS,
            description="stressful workday",
            created_at=_DINNER,
        )
    )
    return store


def _toolkit(store: SQLiteStore) -> DiscoveryToolkit:
    ctx = AgentContext(
        store=store,
        window=(_DAY, _DAY),
        gates=ColdStartReport.from_coverage(store.coverage()),
        run_id="test-run",
        timezone="UTC",
    )
    return DiscoveryToolkit(ctx)


# ── store roundtrip ───────────────────────────────────────────────────────────


def test_add_and_get_manual_event_roundtrip() -> None:
    store = SQLiteStore(":memory:")
    store.migrate()
    new_id = store.add_manual_event(
        ManualEvent(
            event_type="site_change",
            event_ts=_DINNER,
            tags=["forearm"],
            created_at=_DINNER,
        )
    )
    assert new_id == 1
    got = store.get_manual_events(_DINNER - timedelta(hours=1), _DINNER + timedelta(hours=1))
    assert len(got) == 1
    assert got[0].id == 1
    assert got[0].event_type == "site_change"
    assert got[0].tags == ["forearm"]
    assert got[0].confidence == "user_reported"
    assert got[0].source == "manual"


def test_get_manual_events_window_is_half_open() -> None:
    store = _seeded_store()
    # window excludes events at or after end
    out = store.get_manual_events(_STRESS, _DINNER)
    assert [e.event_type for e in out] == ["stress"]
    out2 = store.get_manual_events(_STRESS, _DINNER + timedelta(seconds=1))
    assert [e.event_type for e in out2] == ["stress", "meal"]


# ── tools ───────────────────────────────────────────────────────────────────


def test_tools_are_always_on_the_belt_even_without_manual_data() -> None:
    store = SQLiteStore(":memory:")
    store.migrate()
    store.insert_glucose(
        [GlucoseEvent(ts=datetime(2026, 6, 16, 12, tzinfo=UTC), mg_dl=120)]
    )
    ctx = AgentContext(
        store=store,
        window=(_DAY, _DAY),
        gates=ColdStartReport.from_coverage(store.coverage()),
        run_id="t",
    )
    names = {t.name for t in tool_specs(ctx, DiscoveryToolkit(ctx))}
    assert {"get_manual_events", "search_manual_events", "get_context_around_event"} <= names


def test_get_manual_events_returns_user_reported_rows() -> None:
    result = _toolkit(_seeded_store()).get_manual_events()
    assert result["n_events"] == 2
    assert all(row["provenance"] == "user-reported" for row in result["events"])


def test_get_manual_events_empty_explains_it_is_not_inferred() -> None:
    store = SQLiteStore(":memory:")
    store.migrate()
    result = _toolkit(store).get_manual_events()
    assert result["n_events"] == 0
    assert "never inferred" in result["note"]


def test_search_manual_events_matches_title_and_tags() -> None:
    tk = _toolkit(_seeded_store())
    assert tk.search_manual_events("high-fat")["n_events"] == 1
    assert tk.search_manual_events("fat")["n_events"] == 1  # tag match
    assert tk.search_manual_events("stress")["n_events"] == 1
    assert tk.search_manual_events("nothing-here")["n_events"] == 0


def test_get_context_around_event_pads_in_hours() -> None:
    tk = _toolkit(_seeded_store())
    near = tk.get_context_around_event(_DINNER.isoformat(), 1.0)
    assert near["n_events"] == 1  # only the dinner is within 1h
    wide = tk.get_context_around_event(_DINNER.isoformat(), 12.0)
    assert wide["n_events"] == 2  # the morning stress note is now in range


def test_get_context_around_event_rejects_bad_timestamp() -> None:
    assert "error" in _toolkit(_seeded_store()).get_context_around_event("not-a-date")


# ── trace ─────────────────────────────────────────────────────────────────────


def test_manual_trace_lines() -> None:
    tk = _toolkit(_seeded_store())
    line = render_trace(
        [ToolCall(name="get_manual_events", args={}, result=tk.get_manual_events(), ok=True)]
    )[0]
    assert "manual context" in line.text
    assert "2 user-reported" in line.text

    empty = render_trace(
        [ToolCall(name="get_manual_events", args={}, result={"n_events": 0, "events": []}, ok=True)]
    )[0]
    assert "no user-reported context found" in empty.text


# ── server: the only writer is the user form ───────────────────────────────────

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from dexta_intelligence.config import Config  # noqa: E402
from dexta_intelligence.server import create_app  # noqa: E402


def _opener(db_path: Path) -> Callable[[Config, Path | None], StoragePort]:
    def _open(_config: Config, _db: Path | None = None) -> StoragePort:
        store = SQLiteStore(db_path)
        store.migrate()
        return store

    return _open


def _client(tmp_path: Path) -> TestClient:
    db = tmp_path / "gui.db"
    SQLiteStore(db).migrate()
    return TestClient(create_app(Config(), store_opener=_opener(db)))


def test_log_form_renders(tmp_path: Path) -> None:
    resp = _client(tmp_path).get("/log")
    assert resp.status_code == 200
    assert "Log context" in resp.text


def test_post_log_context_creates_a_user_reported_event(tmp_path: Path) -> None:
    db = tmp_path / "gui.db"
    SQLiteStore(db).migrate()
    client = TestClient(create_app(Config(), store_opener=_opener(db)))
    resp = client.post(
        "/actions/log-context",
        data={
            "event_type": "meal",
            "event_ts": "2026-06-16T19:00",
            "title": "high-fat dinner",
            "description": "pizza",
            "tags": "fat, dinner",
            "intensity": "high",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/log?flash=log_ok"
    store = SQLiteStore(db)
    rows = store.get_manual_events(
        datetime(2026, 6, 16, tzinfo=UTC), datetime(2026, 6, 17, tzinfo=UTC)
    )
    assert len(rows) == 1
    assert rows[0].event_type == "meal"
    assert rows[0].title == "high-fat dinner"
    assert rows[0].tags == ["fat", "dinner"]
    assert rows[0].confidence == "user_reported"
    assert rows[0].source == "manual"


def test_post_log_context_rejects_unknown_type(tmp_path: Path) -> None:
    db = tmp_path / "gui.db"
    SQLiteStore(db).migrate()
    client = TestClient(create_app(Config(), store_opener=_opener(db)))
    resp = client.post(
        "/actions/log-context",
        data={"event_type": "bogus", "event_ts": "2026-06-16T19:00"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert "log_badtype" in resp.headers["location"]
    assert SQLiteStore(db).get_manual_events(
        datetime(2026, 6, 16, tzinfo=UTC), datetime(2026, 6, 17, tzinfo=UTC)
    ) == []
