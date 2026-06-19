"""Evaluation / model-card page (PRD section 16): view-model + route."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from dexta_intelligence.config import Config
from dexta_intelligence.models import Finding, FindingStatus, GlucoseEvent
from dexta_intelligence.server.views_evals import evals_page_view
from dexta_intelligence.store import SQLiteStore

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dexta_intelligence.store.port import StoragePort

_NOW = datetime(2026, 6, 16, 12, 0, tzinfo=UTC)


def _store_with_glucose() -> SQLiteStore:
    s = SQLiteStore(":memory:")
    s.migrate()
    ts = _NOW - timedelta(days=3)
    while ts <= _NOW:
        s.insert_glucose([GlucoseEvent(ts=ts, mg_dl=120)])
        ts += timedelta(minutes=5)
    return s


def test_view_model_reports_methodology_safety_and_metrics() -> None:
    store = _store_with_glucose()
    try:
        view = evals_page_view(store, Config(), now=_NOW)
    finally:
        store.close()
    ids = {e["id"] for e in view["methodology"]}
    assert {"E1", "E2", "E3", "E4", "E5", "E6", "Ec"} <= ids
    assert view["safety"]["clean"] is True  # nothing logged -> no violations
    assert view["glucose"]["tir_pct"] == 100.0  # all readings at 120 are in range
    assert view["glucose"]["tir_on_target"] is True
    assert view["model"]["provider"] == "anthropic"


def test_safety_scan_flags_a_dosing_violation() -> None:
    store = _store_with_glucose()
    store.insert_finding(
        Finding(
            agent="observation",
            kind="pattern",
            scope="x",
            headline="Spike pattern",
            body_md="You should increase your basal by 2 units to fix this.",
            status=FindingStatus.ACTIVE,
        )
    )
    try:
        view = evals_page_view(store, Config(), now=_NOW)
    finally:
        store.close()
    assert view["safety"]["violations"] >= 1
    assert view["safety"]["clean"] is False


def test_metrics_none_without_enough_glucose() -> None:
    store = SQLiteStore(":memory:")
    store.migrate()
    try:
        view = evals_page_view(store, Config(), now=_NOW)
    finally:
        store.close()
    assert view["glucose"] is None


pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from dexta_intelligence.server import create_app  # noqa: E402


def _opener(db_path: Path) -> Callable[[Config, Path | None], StoragePort]:
    def _open(_config: Config, _db: Path | None = None) -> StoragePort:
        s = SQLiteStore(db_path)
        s.migrate()
        return s

    return _open


def test_evals_route_renders(tmp_path: Path) -> None:
    db = tmp_path / "evals.db"
    seed = SQLiteStore(db)
    seed.migrate()
    ts = _NOW - timedelta(days=2)
    while ts <= _NOW:
        seed.insert_glucose([GlucoseEvent(ts=ts, mg_dl=120)])
        ts += timedelta(minutes=5)
    client = TestClient(create_app(Config(), store_opener=_opener(db)))
    resp = client.get("/evals")
    assert resp.status_code == 200
    assert "Evaluation suite" in resp.text
    assert "no dosing advice" in resp.text
    assert "Numeric faithfulness" in resp.text
