"""Tests for the treatment-aware anomaly detectors in the monitoring pipeline.

Each of ``rapid_rise``, ``correction_not_working`` and ``low_after_correction``
is exercised directly with crafted glucose/insulin/meal sequences: one positive
case that must fire (asserting name, key and numbers) and one negative case on
clean data that must return ``[]``. A final end-to-end test seeds a real store
with glucose + insulin that trigger ``correction_not_working`` and proves the
``run_monitor`` wiring surfaces it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.analytics.rollups import (
    TARGET_HIGH_MG_DL,
    TARGET_LOW_MG_DL,
)
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.models import (
    GlucoseEvent,
    InsulinEvent,
    InsulinKind,
    MealEvent,
)
from dexta_intelligence.store import SQLiteStore
from dexta_intelligence.workflows.monitor import (
    _correction_not_working,
    _low_after_correction,
    _rapid_rise,
    run_monitor,
)

_END = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
_WINDOW = (_END - timedelta(hours=24), _END)


def _series(start: datetime, values: list[int]) -> list[GlucoseEvent]:
    """A reading every 5 minutes from ``start`` at the given mg/dL values."""
    return [
        GlucoseEvent(ts=start + timedelta(minutes=5 * i), mg_dl=v) for i, v in enumerate(values)
    ]


# ── rapid_rise ────────────────────────────────────────────────────────────────


def test_rapid_rise_fires_unannounced() -> None:
    start = _END - timedelta(hours=2)
    # +90 mg/dL over 25 min, no carbs logged.
    glucose = _series(start, [100, 120, 150, 175, 190, 195])
    anomalies = _rapid_rise(glucose, [], _WINDOW)

    assert len(anomalies) == 1
    a = anomalies[0]
    assert a.name == "rapid_rise"
    assert a.severity == "warning"
    assert a.numbers["rise_mg_dl"] >= 70
    assert a.numbers["from_mg_dl"] == 100
    assert a.numbers["carb_logged"] == 0
    assert a.key == f"rapid_rise:{int(glucose[0].ts.timestamp())}"


def test_rapid_rise_flags_logged_carbs() -> None:
    start = _END - timedelta(hours=2)
    glucose = _series(start, [100, 120, 150, 175, 190, 195])
    meals = [MealEvent(ts=start - timedelta(minutes=20), carbs_g=45.0)]
    anomalies = _rapid_rise(glucose, meals, _WINDOW)

    assert len(anomalies) == 1
    assert anomalies[0].numbers["carb_logged"] == 1


def test_rapid_rise_clean_returns_empty() -> None:
    start = _END - timedelta(hours=2)
    # Gentle drift, never +70 in 30 min.
    glucose = _series(start, [110, 115, 120, 125, 130, 135])
    assert _rapid_rise(glucose, [], _WINDOW) == []
    assert _rapid_rise([], [], _WINDOW) == []


# ── correction_not_working ──────────────────────────────────────────────────────


def test_correction_not_working_fires() -> None:
    start = _END - timedelta(hours=3)
    # High throughout, with readings spanning > 90 min after the bolus (20 x 5 min).
    glucose = _series(start, [245] * 20)
    bolus_ts = start + timedelta(minutes=5)
    insulin = [InsulinEvent(ts=bolus_ts, kind=InsulinKind.BOLUS, units=4.0)]
    anomalies = _correction_not_working(glucose, insulin, _WINDOW)

    assert len(anomalies) == 1
    a = anomalies[0]
    assert a.name == "correction_not_working"
    assert a.severity == "warning"
    assert a.numbers["bolus_units"] == 4.0
    assert a.numbers["glucose_at_bolus"] > TARGET_HIGH_MG_DL
    assert a.numbers["minutes_high_after"] >= 90
    assert a.key == f"correction_not_working:{int(bolus_ts.timestamp())}"


def test_correction_working_returns_empty() -> None:
    start = _END - timedelta(hours=3)
    # Bolus brings glucose below target within the wait window.
    glucose = _series(start, [240, 230, 210, 190, 175, 160, 150, 140, 135, 130, 128, 125])
    bolus_ts = start + timedelta(minutes=5)
    insulin = [InsulinEvent(ts=bolus_ts, kind=InsulinKind.BOLUS, units=4.0)]
    assert _correction_not_working(glucose, insulin, _WINDOW) == []
    assert _correction_not_working(glucose, [], _WINDOW) == []


# ── low_after_correction ─────────────────────────────────────────────────────────


def test_low_after_correction_fires() -> None:
    start = _END - timedelta(hours=4)
    # Bolus at high, then drops below target low within 4 hours.
    glucose = _series(start, [200, 180, 150, 120, 90, 70, 60, 65, 80, 95])
    bolus_ts = start
    insulin = [InsulinEvent(ts=bolus_ts, kind=InsulinKind.BOLUS, units=5.0)]
    anomalies = _low_after_correction(glucose, insulin, _WINDOW)

    assert len(anomalies) == 1
    a = anomalies[0]
    assert a.name == "low_after_correction"
    assert a.severity == "warning"
    assert a.numbers["bolus_units"] == 5.0
    assert a.numbers["nadir_mg_dl"] == 60
    assert a.numbers["nadir_mg_dl"] < TARGET_LOW_MG_DL
    assert a.numbers["minutes_to_low"] > 0
    assert a.key == f"low_after_correction:{int(bolus_ts.timestamp())}"


def test_low_after_correction_clean_returns_empty() -> None:
    start = _END - timedelta(hours=4)
    # No reading ever dips below target low.
    glucose = _series(start, [200, 180, 160, 140, 120, 110, 105, 100, 95, 90])
    insulin = [InsulinEvent(ts=start, kind=InsulinKind.BOLUS, units=5.0)]
    assert _low_after_correction(glucose, insulin, _WINDOW) == []
    assert _low_after_correction(glucose, [], _WINDOW) == []


# ── end-to-end wiring through a seeded store ─────────────────────────────────────


def _ctx(store: SQLiteStore) -> AgentContext:
    coverage = store.coverage()
    return AgentContext(
        store=store,
        window=(_END.date() - timedelta(days=30), _END.date()),
        gates=ColdStartReport.from_coverage(coverage),
        run_id="test-run",
    )


def test_run_monitor_surfaces_correction_not_working() -> None:
    start = _END - timedelta(hours=3)
    glucose = _series(start, [245] * 20)
    bolus_ts = start + timedelta(minutes=5)
    insulin = [InsulinEvent(ts=bolus_ts, kind=InsulinKind.BOLUS, units=4.0)]

    store = SQLiteStore(":memory:")
    store.migrate()
    store.insert_glucose(glucose)
    store.insert_insulin(insulin)

    anomalies = run_monitor(_ctx(store), persist=False, now=_END)
    assert any(a.name == "correction_not_working" for a in anomalies)
