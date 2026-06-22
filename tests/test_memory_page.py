"""Memory Inspector route: what the retrieval guard reuses versus withholds.

Seeds an ACTIVE (reusable) finding plus a REJECTED and a CONTRADICTED finding,
then asserts the page surfaces the active headline under "in use" and the two
withheld headlines under "excluded" with their human reasons. Also asserts an
empty store renders 200 with empty states.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dexta_intelligence.models import Finding, FindingStats, FindingStatus

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


def _opener(db_path: Path) -> Callable[[Config, Path | None], StoragePort]:
    def _open(_config: Config, _db: Path | None = None) -> StoragePort:
        s = SQLiteStore(db_path)
        s.migrate()
        return s

    return _open


def _finding(headline: str, *, status: FindingStatus, scope: str) -> Finding:
    return Finding(
        agent="observation",
        kind="pattern",
        scope=scope,
        headline=headline,
        body_md=f"Evidence body for {headline}.",
        stats=FindingStats(n=24, effect_size=0.6),
        confidence=0.8,
        status=status,
    )


def _seeded(db: Path) -> None:
    store = SQLiteStore(db)
    store.migrate()
    store.insert_finding(
        _finding("Overnight lows cluster after evening exercise", status=FindingStatus.ACTIVE,
                 scope="overnight")
    )
    store.insert_finding(
        _finding("Coffee spikes drive morning highs", status=FindingStatus.REJECTED, scope="am")
    )
    store.insert_finding(
        _finding("Lunch boluses run late", status=FindingStatus.CONTRADICTED, scope="lunch")
    )


def test_memory_route_separates_used_from_excluded(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    _seeded(db)
    client = TestClient(create_app(Config(), store_opener=_opener(db)))
    resp = client.get("/memory")
    assert resp.status_code == 200

    body = resp.text
    used_html, excluded_html = body.split("Memory excluded", 1)

    # The ACTIVE finding is reusable: it appears in the "in use" section.
    assert "Overnight lows cluster after evening exercise" in used_html

    # The non-active findings are withheld with their human reasons.
    assert "Coffee spikes drive morning highs" in excluded_html
    assert "Lunch boluses run late" in excluded_html
    assert "rejected" in excluded_html
    assert "contradicted" in excluded_html


def test_memory_empty_store_renders_empty_states(tmp_path: Path) -> None:
    db = tmp_path / "empty.db"
    SQLiteStore(db).migrate()
    client = TestClient(create_app(Config(), store_opener=_opener(db)))
    resp = client.get("/memory")
    assert resp.status_code == 200
    assert "No memory in use." in resp.text
    assert "No memory withheld." in resp.text
