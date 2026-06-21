"""Tests for the time-traversal tools on DiscoveryToolkit.

These prove the reasoning agent can re-scope the active window itself: set_window
clamps and re-scopes (a tod_compare after narrowing to March only sees March
data), list_segments returns coarse month rows, zoom_event drills a planted spike
to a tight trace, and daily_series returns per-day values over the active window.
Bad args never raise - they come back as error-style dicts.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.agents.tools.toolkit import DiscoveryToolkit
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.models import GlucoseEvent
from dexta_intelligence.store import SQLiteStore

# A 3-month span so list_segments groups by month (>= 60 days).
_START = datetime(2026, 3, 1, tzinfo=UTC)
_END = datetime(2026, 5, 31, tzinfo=UTC)

# Distinct per-month base glucose so a narrowed window has a provable signature.
_MONTH_BASE = {3: 200, 4: 100, 5: 150}


def _store() -> SQLiteStore:
    """Flat, well-covered glucose with a distinct mean per month."""
    store = SQLiteStore(":memory:")
    store.migrate()
    glucose: list[GlucoseEvent] = []
    day = _START
    while day <= _END:
        base = _MONTH_BASE[day.month]
        for hour in range(0, 24, 2):  # 12 readings/day → above the per-day floor
            ts = day.replace(hour=hour)
            glucose.append(GlucoseEvent(ts=ts, mg_dl=base))
        day += timedelta(days=1)
    store.insert_glucose(glucose)
    return store


def _toolkit(store: SQLiteStore) -> DiscoveryToolkit:
    ctx = AgentContext(
        store=store,
        window=(_START.date(), _END.date()),
        gates=ColdStartReport.from_coverage(store.coverage()),
        run_id="time-test",
    )
    return DiscoveryToolkit(ctx)


# ── set_window: clamps + re-scopes the analysis tools ─────────────────────────


def test_set_window_rescopes_tod_compare_to_march_only() -> None:
    tk = _toolkit(_store())
    # Full window: tod_compare over a TOD band sees all three months mixed.
    full = tk._tod_compare((0, 6), (12, 18))
    assert full.ok
    # Narrow to March (base 200) and re-run: every day-mean must be ~200, never
    # April's 100 - proving the analysis tool honors the active sub-window.
    sel = tk.set_window("2026-03-01", "2026-03-31")
    assert sel["active_start"] == "2026-03-01"
    assert sel["active_end"] == "2026-03-31"
    assert sel["n_days"] == 31
    assert sel["n_readings"] == 31 * 12

    march = tk._tod_compare((0, 6), (12, 18))
    assert march.ok
    assert all(abs(v - 200.0) < 1e-6 for v in march.group_a)
    assert all(abs(v - 200.0) < 1e-6 for v in march.group_b)

    # Re-scope to April (base 100) and prove it now sees only April.
    tk.set_window("2026-04-01", "2026-04-30")
    april = tk._tod_compare((0, 6), (12, 18))
    assert april.ok
    assert all(abs(v - 100.0) < 1e-6 for v in april.group_a)


def test_set_window_clamps_out_of_range() -> None:
    tk = _toolkit(_store())
    # Request way outside the loaded span on both ends → clamps to full window.
    out = tk.set_window("2020-01-01", "2030-01-01")
    assert out["active_start"] == "2026-03-01"
    assert out["active_end"] == "2026-05-31"
    assert out.get("note") == "clamped to available data"


def test_set_window_default_active_is_full_window() -> None:
    """Without any set_window call the toolkit behaves exactly as before."""
    tk = _toolkit(_store())
    # daily_series over the untouched (full) window spans all 92 days.
    series = tk.daily_series("mean_glucose")
    assert series["n_days"] == 92  # 31 + 30 + 31


def test_set_window_bad_dates_are_graceful() -> None:
    tk = _toolkit(_store())
    out = tk.set_window("not-a-date", "2026-04-01")
    assert "error" in out
    # A bad call must not have moved the active window.
    assert tk.daily_series("mean_glucose")["n_days"] == 92


# ── list_segments: coarse month rows ──────────────────────────────────────────


def test_list_segments_returns_month_rows() -> None:
    tk = _toolkit(_store())
    out = tk.list_segments()
    assert out["granularity"] == "month"
    periods = [s["period"] for s in out["segments"]]
    assert periods == ["2026-03", "2026-04", "2026-05"]
    by_period = {s["period"]: s for s in out["segments"]}
    assert abs(by_period["2026-03"]["mean_glucose"] - 200.0) < 1e-6
    assert abs(by_period["2026-04"]["mean_glucose"] - 100.0) < 1e-6
    assert by_period["2026-03"]["n_days"] == 31
    # base 200 and 100 are both in-range (70-180? 200 is high, 100 in range);
    # none are lows (< 70), so n_lows is 0 everywhere here.
    assert all(s["n_lows"] == 0 for s in out["segments"])


def test_list_segments_ignores_active_window() -> None:
    """list_segments always describes the whole record, even after a re-scope."""
    tk = _toolkit(_store())
    tk.set_window("2026-04-01", "2026-04-30")
    out = tk.list_segments()
    assert [s["period"] for s in out["segments"]] == ["2026-03", "2026-04", "2026-05"]


def test_list_segments_uses_weeks_for_short_spans() -> None:
    store = SQLiteStore(":memory:")
    store.migrate()
    start = datetime(2026, 3, 1, tzinfo=UTC)
    glucose: list[GlucoseEvent] = []
    for d in range(20):  # < 60 day span → weekly granularity
        day = start + timedelta(days=d)
        for hour in range(0, 24, 2):
            glucose.append(GlucoseEvent(ts=day.replace(hour=hour), mg_dl=120))
    store.insert_glucose(glucose)
    ctx = AgentContext(
        store=store,
        window=(start.date(), (start + timedelta(days=19)).date()),
        gates=ColdStartReport.from_coverage(store.coverage()),
        run_id="wk",
    )
    out = DiscoveryToolkit(ctx).list_segments()
    assert out["granularity"] == "week"
    assert all(s["period"].count("W") == 1 for s in out["segments"])


# ── zoom_event: tight trace around a planted spike ────────────────────────────


def _store_with_spike() -> tuple[SQLiteStore, datetime]:
    """Flat 100 mg/dL with a single 280 mg/dL spike at a known timestamp."""
    store = SQLiteStore(":memory:")
    store.migrate()
    glucose: list[GlucoseEvent] = []
    base = datetime(2026, 3, 10, tzinfo=UTC)
    spike_ts = base.replace(hour=14, minute=0)
    for d in range(5):
        day = base + timedelta(days=d)
        for minute_block in range(0, 24 * 60, 5):  # 5-min CGM cadence
            ts = day + timedelta(minutes=minute_block)
            mg = 280 if ts == spike_ts else 100
            glucose.append(GlucoseEvent(ts=ts, mg_dl=mg))
    store.insert_glucose(glucose)
    return store, spike_ts


def test_zoom_event_returns_tight_trace_with_correct_peak() -> None:
    store, spike_ts = _store_with_spike()
    ctx = AgentContext(
        store=store,
        window=((spike_ts - timedelta(days=2)).date(), (spike_ts + timedelta(days=2)).date()),
        gates=ColdStartReport.from_coverage(store.coverage()),
        run_id="zoom",
    )
    tk = DiscoveryToolkit(ctx)
    out = tk.zoom_event(spike_ts.isoformat(), pad_hours=2)
    assert out["peak"] == 280.0  # the planted spike is the peak
    assert out["nadir"] == 100.0
    # +/- 2h at 5-min cadence → ~49 readings, far fewer than the full record.
    assert 40 <= out["n_readings"] <= 60
    # The active window is now tight: a daily_series sees only the spike day(s).
    assert tk.daily_series("mean_glucose")["n_days"] <= 2
    # First and last readings sit inside the pad around the centre.
    first = datetime.fromisoformat(out["readings"][0]["ts"])
    last = datetime.fromisoformat(out["readings"][-1]["ts"])
    assert spike_ts - timedelta(hours=2) <= first
    assert last <= spike_ts + timedelta(hours=2)


def test_zoom_event_bad_timestamp_is_graceful() -> None:
    store, _ = _store_with_spike()
    ctx = AgentContext(
        store=store,
        window=(datetime(2026, 3, 10, tzinfo=UTC).date(), datetime(2026, 3, 14, tzinfo=UTC).date()),
        gates=ColdStartReport.from_coverage(store.coverage()),
        run_id="zoom-bad",
    )
    out = DiscoveryToolkit(ctx).zoom_event("garbage", pad_hours=3)
    assert "error" in out


# ── daily_series: per-day values over the active window ───────────────────────


def test_daily_series_returns_per_day_values() -> None:
    tk = _toolkit(_store())
    tk.set_window("2026-03-01", "2026-03-05")
    out = tk.daily_series("mean_glucose")
    assert out["metric"] == "mean_glucose"
    assert out["n_days"] == 5
    assert [r["date"] for r in out["series"]] == [
        "2026-03-01",
        "2026-03-02",
        "2026-03-03",
        "2026-03-04",
        "2026-03-05",
    ]
    assert all(abs(r["value"] - 200.0) < 1e-6 for r in out["series"])


def test_daily_series_metrics_compute() -> None:
    tk = _toolkit(_store())
    tk.set_window("2026-04-01", "2026-04-10")  # base 100, fully in range
    tir = tk.daily_series("tir")
    assert all(abs(r["value"] - 100.0) < 1e-6 for r in tir["series"])  # all in 70-180
    tbr = tk.daily_series("tbr")
    assert all(r["value"] == 0.0 for r in tbr["series"])  # nothing < 70
    cv = tk.daily_series("cv")
    assert all(r["value"] == 0.0 for r in cv["series"])  # flat → zero spread


def test_daily_series_bad_metric_is_graceful() -> None:
    tk = _toolkit(_store())
    out = tk.daily_series("nonsense")
    assert "error" in out
