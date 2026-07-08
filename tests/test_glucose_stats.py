"""The glucose_stats tool - descriptive stats over an arbitrary window.

Closes the gap where the agent refused to compute variance/SD for a specific
window because no tool produced it.
"""

from __future__ import annotations

import statistics
from datetime import UTC, date, datetime, timedelta

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.agents.tools.toolkit import DiscoveryToolkit, tool_specs
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.models import GlucoseEvent
from dexta_intelligence.store import SQLiteStore


def _toolkit(events: list[GlucoseEvent], *, tz: str = "UTC") -> DiscoveryToolkit:
    store = SQLiteStore(":memory:")
    store.migrate()
    store.insert_glucose(events)
    ctx = AgentContext(
        store=store,
        window=(date(2026, 6, 14), date(2026, 6, 16)),
        gates=ColdStartReport.from_coverage(store.coverage()),
        run_id="stats-test",
        timezone=tz,
    )
    return DiscoveryToolkit(ctx, target_low=70, target_high=180)


def test_window_stats_match_hand_computation() -> None:
    vals = [80, 120, 200, 100, 90, 160]
    start = datetime(2026, 6, 15, 1, 0, tzinfo=UTC)
    tk = _toolkit([GlucoseEvent(ts=start + timedelta(minutes=15 * i), mg_dl=v)
                   for i, v in enumerate(vals)])
    r = tk.glucose_stats()
    assert r["n"] == len(vals)
    assert r["mean"] == round(statistics.fmean(vals), 1)
    assert r["sd"] == round(statistics.stdev(vals), 1)
    assert r["variance"] == round(statistics.stdev(vals) ** 2, 1)
    assert r["minimum"] == 80
    assert r["maximum"] == 200
    # 4 of 6 in [70,180]; one <70? none; one >180 (200).
    assert r["tir_pct"] == round(100.0 * 5 / 6, 1)
    assert r["tar_pct"] == round(100.0 * 1 / 6, 1)
    assert r["gmi_pct"] == round(3.31 + 0.02392 * statistics.fmean(vals), 1)


def test_day_and_hours_filter_scopes_to_local_evening() -> None:
    # One reading every hour for 2026-06-15 UTC; evening band must select 17:00-23:00.
    base = datetime(2026, 6, 15, tzinfo=UTC)
    events = [GlucoseEvent(ts=base + timedelta(hours=h), mg_dl=100 + h) for h in range(24)]
    tk = _toolkit(events)  # UTC tz
    r = tk.glucose_stats(day="2026-06-15", hours=(17, 24))
    assert r["day"] == "2026-06-15"
    assert r["hours"] == [17, 24]
    assert r["n"] == 7  # hours 17..23
    assert r["mean"] == round(statistics.fmean([100 + h for h in range(17, 24)]), 1)


def test_empty_window_reports_zero_not_error() -> None:
    tk = _toolkit([GlucoseEvent(ts=datetime(2026, 6, 15, tzinfo=UTC), mg_dl=120)])
    r = tk.glucose_stats(day="2026-06-14")  # no readings that day
    assert r["n"] == 0
    assert "error" not in r


def test_bad_day_returns_error_dict() -> None:
    tk = _toolkit([GlucoseEvent(ts=datetime(2026, 6, 15, tzinfo=UTC), mg_dl=120)])
    assert "error" in tk.glucose_stats(day="not-a-date")


def _gri_toolkit(values: list[float]) -> DiscoveryToolkit:
    start = datetime(2026, 6, 15, 1, 0, tzinfo=UTC)
    return _toolkit(
        [GlucoseEvent(ts=start + timedelta(minutes=5 * i), mg_dl=v)
         for i, v in enumerate(values)]
    )


def test_gri_banded_low_high_differ_from_cumulative_tbr_tar() -> None:
    # 20 readings, hand-partitioned with boundary values on every GRI edge:
    #   very-low (<54):   50, 50                       -> 2  (10%)
    #   low (54..<70):    54, 60, 69                   -> 3  (15%)  (54 in, 70 out)
    #   in-range:         70, 180, 120*8               -> 10 (50%)  (70 and 180 in-range)
    #   high (>180..250): 200, 200, 220, 250           -> 4  (20%)  (250 in, 180 out)
    #   very-high (>250): 260                          -> 1  (5%)
    vals = [50, 50, 54, 60, 69, 70, 180, 120, 120, 120, 120, 120, 120, 120, 120,
            200, 200, 220, 250, 260]
    tk = _gri_toolkit([float(v) for v in vals])
    r = tk.glycemia_risk_index()
    assert r["n_readings"] == 20
    # bands
    assert r["pct_very_low"] == 10.0
    assert r["pct_low"] == 15.0
    assert r["pct_high"] == 20.0
    assert r["pct_very_high"] == 5.0
    # components: hypo = 10 + 0.8*15 = 22 ; hyper = 5 + 0.5*20 = 15
    assert r["hypo_component"] == 22.0
    assert r["hyper_component"] == 15.0
    # gri = 3.0*22 + 1.6*15 = 66 + 24 = 90
    assert r["gri"] == 90.0
    # the correctness trap: banded Low/High EXCLUDE the severe bands, so they
    # differ from cumulative TBR (<70) and TAR (>180), which include them.
    stats = tk.glucose_stats()
    assert stats["tbr_pct"] == 25.0 and r["pct_low"] == 15.0  # 25 = very-low + low
    assert stats["tar_pct"] == 25.0 and r["pct_high"] == 20.0  # 25 = high + very-high
    # only the severe bands map directly onto tbr54 / tar250
    assert stats["tbr54_pct"] == r["pct_very_low"]
    assert stats["tar250_pct"] == r["pct_very_high"]


def test_gri_caps_at_100() -> None:
    # 6 very-low (50) + 4 low (60): hypo = 60 + 0.8*40 = 92 ; raw gri = 3*92 = 276.
    tk = _gri_toolkit([50, 50, 50, 50, 50, 50, 60, 60, 60, 60])
    r = tk.glycemia_risk_index()
    assert r["n_readings"] == 10
    assert r["pct_very_low"] == 60.0
    assert r["pct_low"] == 40.0
    assert r["hypo_component"] == 92.0
    assert r["hyper_component"] == 0.0
    assert r["gri"] == 100.0  # capped, not 276


def test_gri_empty_window_reports_zero_not_error() -> None:
    tk = _gri_toolkit([120.0])
    tk.set_window("2026-06-14", "2026-06-14")  # a day with no readings
    r = tk.glycemia_risk_index()
    assert r["n_readings"] == 0
    assert "error" not in r


def test_glycemia_risk_index_is_in_the_belt() -> None:
    store = SQLiteStore(":memory:")
    store.migrate()
    store.insert_glucose([GlucoseEvent(ts=datetime(2026, 6, 15, tzinfo=UTC), mg_dl=120)])
    ctx = AgentContext(
        store=store,
        window=(date(2026, 6, 14), date(2026, 6, 16)),
        gates=ColdStartReport.from_coverage(store.coverage()),
        run_id="belt",
    )
    names = {s.name for s in tool_specs(ctx, DiscoveryToolkit(ctx))}
    assert "glycemia_risk_index" in names


def test_glucose_stats_is_in_the_belt() -> None:
    store = SQLiteStore(":memory:")
    store.migrate()
    store.insert_glucose([GlucoseEvent(ts=datetime(2026, 6, 15, tzinfo=UTC), mg_dl=120)])
    ctx = AgentContext(
        store=store,
        window=(date(2026, 6, 14), date(2026, 6, 16)),
        gates=ColdStartReport.from_coverage(store.coverage()),
        run_id="belt",
    )
    names = {s.name for s in tool_specs(ctx, DiscoveryToolkit(ctx))}
    assert "glucose_stats" in names
