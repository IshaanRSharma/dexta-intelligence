"""The Active Context Acquisition page (/context).

A planted, unexplained spike with no meal or note nearby must surface a logging
question; an empty store must show the calm empty state. The page never 500s.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from dexta_intelligence.models import GlucoseEvent

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dexta_intelligence.config import Config
    from dexta_intelligence.store.port import StoragePort

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from dexta_intelligence.config import Config
from dexta_intelligence.server import create_app
from dexta_intelligence.store import SQLiteStore

_NOW = datetime(2026, 6, 16, 12, 0, tzinfo=UTC)


def _opener(db_path: Path) -> Callable[[Config, Path | None], StoragePort]:
    def _open(_config: Config, _db: Path | None = None) -> StoragePort:
        s = SQLiteStore(db_path)
        s.migrate()
        return s

    return _open


def _seed_spike(db: Path) -> None:
    """Flat glucose over the last day with a planted >=300 mg/dL spike, no meals."""
    store = SQLiteStore(db)
    store.migrate()
    start = _NOW - timedelta(hours=24)
    glucose = [GlucoseEvent(ts=start + timedelta(minutes=5 * i), mg_dl=110) for i in range(288)]
    for i in range(100, 108):
        glucose[i] = GlucoseEvent(ts=glucose[i].ts, mg_dl=300)
    store.insert_glucose(glucose)


def test_context_page_surfaces_question(tmp_path: Path) -> None:
    db = tmp_path / "context.db"
    _seed_spike(db)
    client = TestClient(create_app(Config(), store_opener=_opener(db)))
    resp = client.get("/context")
    assert resp.status_code == 200
    assert "mg/dL" in resp.text
    assert "/log" in resp.text


def test_context_page_empty_store(tmp_path: Path) -> None:
    db = tmp_path / "empty.db"
    SQLiteStore(db).migrate()
    client = TestClient(create_app(Config(), store_opener=_opener(db)))
    resp = client.get("/context")
    assert resp.status_code == 200
    assert "No missing context right now" in resp.text
