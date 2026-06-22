"""Dashboard hero chart view-model."""

from __future__ import annotations

import pytest
from tests.golden import make_store

from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.config import Config
from dexta_intelligence.server.views_hero import hero_chart_view
from dexta_intelligence.store import SQLiteStore


@pytest.fixture(scope="module")
def late_bolus_store():
    store = make_store("late_bolus")
    yield store
    store.close()


def test_hero_chart_on_late_bolus_golden(late_bolus_store) -> None:
    config = Config()
    gates = ColdStartReport.from_coverage(late_bolus_store.coverage())
    view = hero_chart_view(late_bolus_store, config, gates)
    assert view["has_chart"] is True
    assert "chart-glucose" in view["svg"]
    assert "246" in view["subtitle"] or "Peak" in view["subtitle"]
    assert view.get("annotation") == "late bolus, +22 min"


def test_hero_chart_absent_below_floor(tmp_path) -> None:
    store = SQLiteStore(tmp_path / "empty.db")
    store.migrate()
    try:
        gates = ColdStartReport.from_coverage(store.coverage())
        view = hero_chart_view(store, Config(), gates)
    finally:
        store.close()
    assert view["has_chart"] is False
