"""Prediction-reconciliation page: view-model + route.

Uses the demo's planted forecast miss (carb underestimate) copied into a
file-backed store so the FastAPI opener can reopen it per request.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.agents.reconciliation import PredictionReconciliationAgent
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.demo import build_demo_store
from dexta_intelligence.server.views_reconciliation import reconciliation_page_view
from dexta_intelligence.store import SQLiteStore

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dexta_intelligence.config import Config
    from dexta_intelligence.store.port import StoragePort

_WIDE = (datetime(2000, 1, 1, tzinfo=UTC), datetime(2100, 1, 1, tzinfo=UTC))


def _file_demo(tmp_path: Path) -> SQLiteStore:
    """The demo timeline copied into a file-backed store (reopenable per request)."""
    src = build_demo_store()
    dst = SQLiteStore(tmp_path / "recon.db")
    dst.migrate()
    dst.insert_glucose(src.get_glucose(*_WIDE))
    dst.insert_insulin(src.get_insulin(*_WIDE))
    dst.insert_meals(src.get_meals(*_WIDE))
    dst.insert_predictions(src.get_predictions(*_WIDE))
    src.close()
    return dst


def _ctx(store: SQLiteStore) -> AgentContext:
    cov = store.coverage()
    return AgentContext(
        store=store,
        window=(cov.first_ts.date(), cov.last_ts.date()),  # type: ignore[union-attr]
        gates=ColdStartReport.from_coverage(cov),
        run_id=str(uuid.uuid4()),
    )


def test_view_model_shapes_a_carb_underestimate_card(tmp_path: Path) -> None:
    store = _file_demo(tmp_path)
    try:
        findings = PredictionReconciliationAgent().run(_ctx(store))
        view = reconciliation_page_view(store, findings, now=datetime.now(UTC))
    finally:
        store.close()
    assert view["any"]
    card = view["cards"][0]
    assert card["contributor"] == "Carbs underestimated"
    assert "COB" in card["curve"]
    assert "mg/dL" in card["max_error"]
    # Tier A reconstructs both traces for the sparklines.
    assert card["expected_spark"] and card["actual_spark"]
    assert card["limited"] is False


def test_view_model_empty_when_nothing_to_reconcile(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "empty.db")
    store.migrate()
    try:
        view = reconciliation_page_view(store, [], now=datetime.now(UTC))
    finally:
        store.close()
    assert view["any"] is False
    assert view["cards"] == []


pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from dexta_intelligence.config import Config  # noqa: E402
from dexta_intelligence.server import create_app  # noqa: E402


def _opener(db_path: Path) -> Callable[[Config, Path | None], StoragePort]:
    def _open(_config: Config, _db: Path | None = None) -> StoragePort:
        s = SQLiteStore(db_path)
        s.migrate()
        return s

    return _open


def test_reconciliation_route_renders_the_miss(tmp_path: Path) -> None:
    store = _file_demo(tmp_path)
    path = store._path
    store.close()
    client = TestClient(create_app(Config(), store_opener=_opener(path)))
    resp = client.get("/reconciliation")
    assert resp.status_code == 200
    assert "Prediction reconciliation" in resp.text
    assert "Carbs underestimated" in resp.text


def test_reconciliation_route_empty_store(tmp_path: Path) -> None:
    db = tmp_path / "empty.db"
    SQLiteStore(db).migrate()
    client = TestClient(create_app(Config(), store_opener=_opener(db)))
    resp = client.get("/reconciliation")
    assert resp.status_code == 200
    assert "No forecast misses" in resp.text
