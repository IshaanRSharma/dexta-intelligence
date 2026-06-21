"""Tests for the pure MCP contract layer (no fastmcp)."""

from __future__ import annotations

import csv
import io
import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from dexta_intelligence.mcp_server.contract import (
    ALERT_DISCLAIMER,
    EPISODE_MIN_DURATION_MINUTES,
    STALE_THRESHOLD_MINUTES,
    analyze_time_blocks,
    check_alerts,
    detect_episodes,
    export_data,
    get_agp_report,
    get_current_glucose,
    get_episode_details,
    get_glucose_readings,
    get_statistics,
    get_status_summary,
)
from dexta_intelligence.models import GlucoseEvent
from dexta_intelligence.store import SQLiteStore

if TYPE_CHECKING:
    from collections.abc import Iterator

T0 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)


@pytest.fixture
def store() -> Iterator[SQLiteStore]:
    s = SQLiteStore(":memory:")
    s.migrate()
    yield s
    s.close()


def _insert(store: SQLiteStore, events: list[GlucoseEvent]) -> None:
    store.insert_glucose(events)


def _reading(ts: datetime, mg_dl: int, trend: str | None = None) -> GlucoseEvent:
    return GlucoseEvent(ts=ts, mg_dl=mg_dl, trend=trend)


class _FakeRealtime:
    def __init__(self, event: GlucoseEvent | None) -> None:
        self._event = event

    def current(self) -> GlucoseEvent | None:
        return self._event


class TestDatetimeValidation:
    def test_rejects_naive_start_in_statistics(self, store: SQLiteStore) -> None:
        naive = datetime(2026, 6, 1, 10, 0)
        with pytest.raises(ValueError, match="naive datetime"):
            get_statistics(store, naive, T0 + timedelta(hours=1))

    def test_rejects_invalid_window(self, store: SQLiteStore) -> None:
        with pytest.raises(ValueError, match="invalid window"):
            get_glucose_readings(store, T0 + timedelta(hours=1), T0)


class TestEmptyWindowHonesty:
    def test_statistics_omits_metrics_when_empty(self, store: SQLiteStore) -> None:
        result = get_statistics(store, T0, T0 + timedelta(hours=1))
        assert result["reading_count"] == 0
        assert "mean_mg_dl" not in result
        assert result["coverage_pct"] is None
        assert "tir_pct" not in result

    def test_export_empty_returns_empty_string(self, store: SQLiteStore) -> None:
        result = export_data(store, T0, T0 + timedelta(hours=1), format="json")
        assert result["count"] == 0
        assert result["data"] == ""


class TestGetGlucoseReadings:
    def test_returns_windowed_readings(self, store: SQLiteStore) -> None:
        events = [_reading(T0 + timedelta(minutes=5 * i), 100 + i) for i in range(4)]
        _insert(store, events)
        result = get_glucose_readings(store, T0, T0 + timedelta(minutes=20))
        assert result["count"] == 4
        assert result["readings"][0]["mg_dl"] == 100

    def test_respects_max_count(self, store: SQLiteStore) -> None:
        events = [_reading(T0 + timedelta(minutes=5 * i), 100) for i in range(6)]
        _insert(store, events)
        result = get_glucose_readings(store, T0, T0 + timedelta(hours=1), max_count=2)
        assert result["count"] == 2


class TestStatistics:
    def test_exact_mean_and_tir(self, store: SQLiteStore) -> None:
        # 3 in-range (100), 1 low (60), 1 high (200) → mean=112, TIR=60%
        events = [
            _reading(T0, 100),
            _reading(T0 + timedelta(minutes=5), 100),
            _reading(T0 + timedelta(minutes=10), 100),
            _reading(T0 + timedelta(minutes=15), 60),
            _reading(T0 + timedelta(minutes=20), 200),
        ]
        _insert(store, events)
        result = get_statistics(store, T0, T0 + timedelta(minutes=30))
        assert result["reading_count"] == 5
        assert result["mean_mg_dl"] == 112.0
        assert result["tir_pct"] == 60.0
        assert result["tbr_pct"] == 20.0
        assert result["tar_pct"] == 20.0
        assert result["gmi_pct"] == 6.0


class TestCurrentGlucose:
    def test_realtime_when_present(self, store: SQLiteStore) -> None:
        live = _reading(T0, 142, trend="Flat")
        rt = _FakeRealtime(live)
        result = get_current_glucose(store, rt, now=T0)
        assert result["source"] == "realtime"
        assert result["stale"] is False
        assert result["reading"]["mg_dl"] == 142

    def test_falls_back_to_store(self, store: SQLiteStore) -> None:
        stored = _reading(T0 - timedelta(minutes=5), 118, trend="SingleUp")
        _insert(store, [stored])
        result = get_current_glucose(store, None, now=T0)
        assert result["source"] == "store"
        assert result["reading"]["mg_dl"] == 118

    def test_stale_flag_when_old(self, store: SQLiteStore) -> None:
        stored = _reading(T0 - timedelta(minutes=STALE_THRESHOLD_MINUTES + 1), 110)
        _insert(store, [stored])
        result = get_current_glucose(store, None, now=T0)
        assert result["stale"] is True

    def test_realtime_none_falls_back_even_with_connector(self, store: SQLiteStore) -> None:
        stored = _reading(T0, 95)
        _insert(store, [stored])
        rt = _FakeRealtime(None)
        result = get_current_glucose(store, rt, now=T0 + timedelta(minutes=1))
        assert result["source"] == "store"
        assert result["reading"]["mg_dl"] == 95


class TestEpisodeDetection:
    def _hypo_series(self, start: datetime, minutes: int, value: int = 55) -> list[GlucoseEvent]:
        return [
            _reading(start + timedelta(minutes=5 * i), value)
            for i in range(minutes // 5 + 1)
        ]

    def test_detects_hypo_and_hyper(self, store: SQLiteStore) -> None:
        hypo_start = T0
        hyper_start = T0 + timedelta(hours=2)
        events = [
            *self._hypo_series(hypo_start, 20, 55),
            _reading(hypo_start + timedelta(minutes=25), 120),
            *self._hypo_series(hyper_start, 20, 210),
        ]
        _insert(store, events)
        result = detect_episodes(store, T0, T0 + timedelta(hours=4))
        kinds = {e["kind"] for e in result["episodes"]}
        assert "hypo" in kinds
        assert "hyper" in kinds

    def test_severe_classification(self, store: SQLiteStore) -> None:
        events = self._hypo_series(T0, 20, 50)
        _insert(store, events)
        result = detect_episodes(store, T0, T0 + timedelta(hours=1))
        assert len(result["episodes"]) == 1
        assert result["episodes"][0]["kind"] == "severe_hypo"

    def test_min_duration_boundary_excludes_short(self, store: SQLiteStore) -> None:
        # 10 minutes - below 15-minute minimum
        events = self._hypo_series(T0, 10, 55)
        _insert(store, events)
        result = detect_episodes(store, T0, T0 + timedelta(hours=1))
        assert result["episodes"] == []

    def test_min_duration_boundary_includes_long(self, store: SQLiteStore) -> None:
        events = self._hypo_series(T0, EPISODE_MIN_DURATION_MINUTES, 65)
        _insert(store, events)
        result = detect_episodes(store, T0, T0 + timedelta(hours=1))
        assert len(result["episodes"]) == 1
        assert result["episodes"][0]["duration_minutes"] >= EPISODE_MIN_DURATION_MINUTES


class TestEpisodeDetails:
    def test_returns_episode_with_context(self, store: SQLiteStore) -> None:
        events = [
            _reading(T0 - timedelta(minutes=30), 120),
            *[_reading(T0 + timedelta(minutes=5 * i), 60) for i in range(4)],
            _reading(T0 + timedelta(minutes=25), 120),
            _reading(T0 + timedelta(minutes=35), 115),
        ]
        _insert(store, events)
        detected = detect_episodes(store, T0 - timedelta(hours=1), T0 + timedelta(hours=1))
        episode_id = detected["episodes"][0]["id"]
        details = get_episode_details(store, episode_id)
        assert details["status"] == "ok"
        assert details["episode"]["id"] == episode_id
        assert len(details["context"]["readings"]) >= 4


class TestTimeBlocks:
    def test_assigns_readings_to_utc_blocks(self, store: SQLiteStore) -> None:
        day = datetime(2026, 6, 1, tzinfo=UTC)
        events = [
            _reading(day.replace(hour=2), 90),
            _reading(day.replace(hour=8), 110),
            _reading(day.replace(hour=14), 130),
            _reading(day.replace(hour=20), 150),
        ]
        _insert(store, events)
        result = analyze_time_blocks(store, day, day + timedelta(days=1))
        assert result["blocks"]["overnight"]["reading_count"] == 1
        assert result["blocks"]["morning"]["reading_count"] == 1
        assert result["blocks"]["afternoon"]["reading_count"] == 1
        assert result["blocks"]["evening"]["reading_count"] == 1


class TestCheckAlerts:
    def test_includes_disclaimer_and_threshold_alerts(self, store: SQLiteStore) -> None:
        _insert(store, [_reading(T0, 52, trend="DoubleDown")])
        result = check_alerts(store, None, now=T0)
        assert result["disclaimer"] == ALERT_DISCLAIMER
        kinds = {a["kind"] for a in result["alerts"]}
        assert "severe_hypo" in kinds
        assert result["projections"]["minutes_15"] is not None


class TestExportData:
    def test_json_round_trip(self, store: SQLiteStore) -> None:
        _insert(store, [_reading(T0, 100, trend="Flat")])
        result = export_data(store, T0, T0 + timedelta(hours=1), format="json")
        loaded = json.loads(result["data"])
        assert loaded[0]["mg_dl"] == 100

    def test_csv_parses(self, store: SQLiteStore) -> None:
        _insert(store, [_reading(T0, 100, trend="Flat")])
        result = export_data(store, T0, T0 + timedelta(hours=1), format="csv")
        rows = list(csv.DictReader(io.StringIO(result["data"])))
        assert rows[0]["mg_dl"] == "100"


class TestAgpReport:
    def test_known_percentiles_on_constructed_pattern(self, store: SQLiteStore) -> None:
        # Two days, same 08:00 slot with values 80 and 120 → median 100
        day1 = datetime(2026, 6, 1, 8, 0, tzinfo=UTC)
        day2 = datetime(2026, 6, 2, 8, 0, tzinfo=UTC)
        _insert(store, [_reading(day1, 80), _reading(day2, 120)])
        result = get_agp_report(store, day1, day2 + timedelta(days=1))
        bin_8am = next(b for b in result["bins"] if b["time_utc"] == "08:00")
        assert bin_8am["p50"] == 100.0
        assert bin_8am["p5"] == 82.0
        assert bin_8am["p95"] == 118.0
        assert result["day_count"] == 2


class TestStatusSummary:
    def test_combines_current_stats_episodes_alerts(self, store: SQLiteStore) -> None:
        now = T0 + timedelta(hours=1)
        _insert(
            store,
            [
                _reading(now - timedelta(minutes=5), 200, trend="SingleUp"),
                _reading(now - timedelta(minutes=10), 210),
            ],
        )
        result = get_status_summary(store, None, now=now)
        assert result["current"]["reading"]["mg_dl"] == 200
        assert result["last_24h"]["reading_count"] == 2
        assert "disclaimer" in result["alerts"]
