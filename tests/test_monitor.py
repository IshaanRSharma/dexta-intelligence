"""Tests for the monitoring pipeline - deterministic anomaly detectors.

Plants severe lows / highs / TIR cliffs / sensor gaps in an in-memory store
and asserts the right anomalies fire with the correct numbers; clean data
produces nothing; persist writes anomaly Findings; a CollectingNotifier
receives every anomaly; thin data degrades to ``[]`` without raising.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.models import FindingStatus, GlucoseEvent
from dexta_intelligence.notifications import CollectingNotifier
from dexta_intelligence.store import SQLiteStore
from dexta_intelligence.workflows.monitor import run_monitor

_END = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


def _ctx(store: SQLiteStore) -> AgentContext:
    coverage = store.coverage()
    return AgentContext(
        store=store,
        window=(_END.date() - timedelta(days=30), _END.date()),
        gates=ColdStartReport.from_coverage(coverage),
        run_id="test-run",
    )


def _store(glucose: list[GlucoseEvent]) -> SQLiteStore:
    store = SQLiteStore(":memory:")
    store.migrate()
    store.insert_glucose(glucose)
    return store


def _flat_window(value: int, *, hours: int = 24, end: datetime = _END) -> list[GlucoseEvent]:
    """A reading every 5 minutes at ``value`` over the ``hours`` before ``end``."""
    start = end - timedelta(hours=hours)
    n = hours * 12
    return [GlucoseEvent(ts=start + timedelta(minutes=5 * i), mg_dl=value) for i in range(n)]


# ── detectors fire correctly ──────────────────────────────────────────────────


def test_severe_low_fires_urgent() -> None:
    glucose = _flat_window(110)
    glucose[100] = GlucoseEvent(ts=glucose[100].ts, mg_dl=48)
    anomalies = run_monitor(_ctx(_store(glucose)), persist=False, now=_END)

    low = [a for a in anomalies if a.name == "severe_low"]
    assert low, "a sub-54 reading must raise severe_low"
    assert low[0].severity == "urgent"
    assert low[0].numbers["nadir_mg_dl"] == 48
    assert low[0].numbers["n_readings"] == 1


def test_severe_high_requires_sustained_run() -> None:
    glucose = _flat_window(110)
    # 8 consecutive readings (~35 min) above 250
    for i in range(50, 58):
        glucose[i] = GlucoseEvent(ts=glucose[i].ts, mg_dl=300)
    anomalies = run_monitor(_ctx(_store(glucose)), persist=False, now=_END)

    high = [a for a in anomalies if a.name == "severe_high"]
    assert high, "a sustained run above 250 must raise severe_high"
    assert high[0].severity == "warning"
    assert high[0].numbers["peak_mg_dl"] == 300
    assert high[0].numbers["longest_run_min"] >= 30


def test_single_high_reading_does_not_fire() -> None:
    glucose = _flat_window(110)
    glucose[50] = GlucoseEvent(ts=glucose[50].ts, mg_dl=300)  # one spike, not sustained
    anomalies = run_monitor(_ctx(_store(glucose)), persist=False, now=_END)
    assert not [a for a in anomalies if a.name == "severe_high"]


def test_sensor_gap_fires() -> None:
    # Two flat half-days with a 3h hole between them.
    first = _flat_window(110, hours=6, end=_END - timedelta(hours=9))
    second = _flat_window(110, hours=6, end=_END)
    anomalies = run_monitor(_ctx(_store(first + second)), persist=False, now=_END)

    gap = [a for a in anomalies if a.name == "sensor_gap"]
    assert gap, "a multi-hour hole must raise sensor_gap"
    assert gap[0].severity == "warning"  # >= 120 min
    assert gap[0].numbers["max_gap_min"] >= 120


def test_time_in_range_cliff_fires() -> None:
    # Baseline 14 days mostly in range; recent 24h mostly high.
    baseline = _flat_window(110, hours=24 * 14, end=_END - timedelta(hours=24))
    recent = _flat_window(220, hours=24, end=_END)
    anomalies = run_monitor(_ctx(_store(baseline + recent)), persist=False, now=_END)

    cliff = [a for a in anomalies if a.name == "time_in_range_cliff"]
    assert cliff, "recent TIR far below baseline must raise a cliff"
    assert cliff[0].numbers["drop_pct"] >= 15.0
    assert cliff[0].numbers["recent_tir_pct"] == 0.0


# ── clean data, persistence, notification, thin data ──────────────────────────


def test_clean_data_no_anomalies() -> None:
    baseline = _flat_window(110, hours=24 * 14, end=_END - timedelta(hours=24))
    recent = _flat_window(110, hours=24, end=_END)
    anomalies = run_monitor(_ctx(_store(baseline + recent)), persist=False, now=_END)
    assert anomalies == []


def test_persist_writes_anomaly_findings() -> None:
    glucose = _flat_window(110)
    glucose[100] = GlucoseEvent(ts=glucose[100].ts, mg_dl=45)
    store = _store(glucose)
    run_monitor(_ctx(store), persist=True, now=_END)

    findings = store.get_findings(kind="anomaly")
    assert findings, "persist=True must write anomaly Findings"
    f = findings[0]
    assert f.kind == "anomaly"
    assert f.agent == "monitor"
    assert f.scope == "severe_low"
    assert f.status == FindingStatus.ACTIVE
    assert f.evidence["severity"] == "urgent"
    assert f.evidence["nadir_mg_dl"] == 45


def test_collecting_notifier_receives_anomalies() -> None:
    glucose = _flat_window(110)
    glucose[100] = GlucoseEvent(ts=glucose[100].ts, mg_dl=45)
    sink = CollectingNotifier()
    anomalies = run_monitor(_ctx(_store(glucose)), persist=False, notify=sink, now=_END)

    assert len(sink.received) == len(anomalies)
    assert any(a.name == "severe_low" for a in sink.received)


def test_rerun_does_not_duplicate_the_same_anomaly() -> None:
    glucose = _flat_window(110)
    glucose[100] = GlucoseEvent(ts=glucose[100].ts, mg_dl=45)
    store = _store(glucose)
    ctx = _ctx(store)

    run_monitor(ctx, persist=True, now=_END)
    run_monitor(ctx, persist=True, now=_END)  # same data, same anomaly

    active = store.get_findings(kind="anomaly", status=FindingStatus.ACTIVE, limit=1000)
    severe_lows = [f for f in active if f.scope == "severe_low"]
    assert len(severe_lows) == 1, "the same ongoing low must not be re-recorded each run"


def test_rerun_does_not_renotify_unchanged_anomaly() -> None:
    glucose = _flat_window(110)
    glucose[100] = GlucoseEvent(ts=glucose[100].ts, mg_dl=45)
    store = _store(glucose)
    ctx = _ctx(store)

    run_monitor(ctx, persist=True, now=_END)  # first run records + would notify
    sink = CollectingNotifier()
    run_monitor(ctx, persist=True, notify=sink, now=_END)  # second run: nothing new
    assert sink.received == [], "an unchanged ongoing anomaly must not re-notify"


def test_worsened_anomaly_supersedes_and_renotifies() -> None:
    glucose = _flat_window(110)
    glucose[100] = GlucoseEvent(ts=glucose[100].ts, mg_dl=48)
    store = _store(glucose)
    ctx = _ctx(store)
    run_monitor(ctx, persist=True, now=_END)  # records severe_low:48

    # A deeper low appears (new key) - re-run supersedes the stale one and notifies.
    # Off-grid ts so it's a fresh reading (the store dedups glucose on timestamp).
    store.insert_glucose([GlucoseEvent(ts=glucose[120].ts + timedelta(minutes=1), mg_dl=40)])
    sink = CollectingNotifier()
    run_monitor(ctx, persist=True, notify=sink, now=_END)

    active = store.get_findings(kind="anomaly", status=FindingStatus.ACTIVE, limit=1000)
    all_lows = store.get_findings(kind="anomaly", status=None, limit=1000)
    assert len([f for f in active if f.scope == "severe_low"]) == 1  # only the latest is active
    assert len([f for f in all_lows if f.scope == "severe_low"]) == 2  # prior preserved as history
    assert any(a.name == "severe_low" for a in sink.received)  # the worse state notified


def test_thin_data_does_not_crash() -> None:
    store = SQLiteStore(":memory:")
    store.migrate()
    # No data at all.
    assert run_monitor(_ctx(store), persist=True, now=_END) == []

    # A single reading: anchors a window but too thin for cliff; no crash.
    store.insert_glucose([GlucoseEvent(ts=_END - timedelta(hours=1), mg_dl=120)])
    assert run_monitor(_ctx(store), persist=True, now=_END) == []
