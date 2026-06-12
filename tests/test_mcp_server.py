"""Tests for the FastMCP server wrapper."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from dexta_intelligence.mcp_server.server import (
    INSULIN_TOOL_NAMES,
    TOOL_NAMES,
    build_server,
)
from dexta_intelligence.models import GlucoseEvent, InsulinEvent, InsulinKind, MealEvent
from dexta_intelligence.store import SQLiteStore

if TYPE_CHECKING:
    from collections.abc import Iterator

T0 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
MEAL_TS = T0
BOLUS_TS = T0 + timedelta(minutes=22)


@pytest.fixture
def store() -> Iterator[SQLiteStore]:
    s = SQLiteStore(":memory:")
    s.migrate()
    yield s
    s.close()


def _seed(store: SQLiteStore) -> None:
    store.insert_glucose(
        [
            GlucoseEvent(ts=T0 + timedelta(minutes=5 * i), mg_dl=100 + i, trend="Flat")
            for i in range(6)
        ]
    )


def _seed_insulin(store: SQLiteStore) -> None:
    store.insert_meals([MealEvent(ts=MEAL_TS, carbs_g=60.0, note="dinner")])
    store.insert_insulin(
        [
            InsulinEvent(ts=BOLUS_TS, kind=InsulinKind.BOLUS, units=5.0, automatic=False),
            InsulinEvent(ts=T0 - timedelta(hours=1), kind=InsulinKind.BASAL, units=0.8),
        ]
    )


def _window_args(hours: int = 2) -> dict[str, str]:
    return {
        "start": (T0 - timedelta(hours=hours)).isoformat(),
        "end": (T0 + timedelta(hours=hours)).isoformat(),
    }


class TestServerRegistration:
    def test_all_ten_tools_registered(self, store: SQLiteStore) -> None:
        server = build_server(store)
        tools = asyncio.run(server.list_tools())
        names = {tool.name for tool in tools}
        assert names == set(TOOL_NAMES)
        assert len(names) == 10


class TestServerInvocation:
    def test_get_statistics_end_to_end(self, store: SQLiteStore) -> None:
        _seed(store)
        server = build_server(store)
        end = T0 + timedelta(minutes=30)
        result = asyncio.run(
            server.call_tool(
                "get_statistics",
                {
                    "start": T0.isoformat(),
                    "end": end.isoformat(),
                },
            )
        )
        payload = result.structured_content
        assert payload is not None
        assert payload["reading_count"] == 6
        assert payload["mean_mg_dl"] == 102.5

    def test_get_glucose_readings_end_to_end(self, store: SQLiteStore) -> None:
        _seed(store)
        server = build_server(store)
        result = asyncio.run(
            server.call_tool(
                "get_glucose_readings",
                {
                    "start": T0.isoformat(),
                    "end": (T0 + timedelta(hours=1)).isoformat(),
                    "max_count": 3,
                },
            )
        )
        payload = result.structured_content
        assert payload is not None
        assert payload["count"] == 3


class TestInsulinToolGating:
    def test_insulin_tools_absent_without_insulin_data(self, store: SQLiteStore) -> None:
        _seed(store)
        server = build_server(store)
        names = {tool.name for tool in asyncio.run(server.list_tools())}
        assert names == set(TOOL_NAMES)
        assert names.isdisjoint(INSULIN_TOOL_NAMES)

    def test_insulin_tools_registered_with_insulin_data(self, store: SQLiteStore) -> None:
        _seed(store)
        _seed_insulin(store)
        server = build_server(store)
        names = {tool.name for tool in asyncio.run(server.list_tools())}
        assert names == set(TOOL_NAMES) | set(INSULIN_TOOL_NAMES)


class TestInsulinToolInvocation:
    def test_get_boluses(self, store: SQLiteStore) -> None:
        _seed_insulin(store)
        server = build_server(store)
        result = asyncio.run(server.call_tool("get_boluses", _window_args()))
        payload = result.structured_content
        assert payload is not None
        assert payload["n_boluses"] == 1
        assert payload["total_units"] == 5.0
        (bolus,) = payload["boluses"]
        assert bolus["ts"] == BOLUS_TS.isoformat()
        assert bolus["units"] == 5.0
        assert bolus["automatic"] is False

    def test_get_carb_entries(self, store: SQLiteStore) -> None:
        _seed_insulin(store)
        server = build_server(store)
        result = asyncio.run(server.call_tool("get_carb_entries", _window_args()))
        payload = result.structured_content
        assert payload is not None
        assert payload["n_entries"] == 1
        assert payload["total_carbs_g"] == 60.0
        (entry,) = payload["entries"]
        assert entry["ts"] == MEAL_TS.isoformat()
        assert entry["carbs_g"] == 60.0
        assert entry["note"] == "dinner"

    def test_get_basal_timeline_stable(self, store: SQLiteStore) -> None:
        _seed_insulin(store)
        server = build_server(store)
        result = asyncio.run(server.call_tool("get_basal_timeline", _window_args()))
        payload = result.structured_content
        assert payload is not None
        assert payload["n_basal"] == 1
        assert payload["n_temp_basal"] == 0
        assert payload["n_suspend"] == 0
        assert payload["basal_stable"] is True
        assert payload["events"][0]["kind"] == "basal"

    def test_basal_stable_flips_on_temp_basal(self, store: SQLiteStore) -> None:
        _seed_insulin(store)
        store.insert_insulin(
            [
                InsulinEvent(
                    ts=T0 + timedelta(minutes=30),
                    kind=InsulinKind.TEMP_BASAL,
                    units=0.0,
                    duration_min=30.0,
                    automatic=True,
                )
            ]
        )
        server = build_server(store)
        result = asyncio.run(server.call_tool("get_basal_timeline", _window_args()))
        payload = result.structured_content
        assert payload is not None
        assert payload["n_temp_basal"] == 1
        assert payload["basal_stable"] is False

    def test_get_iob_positive_after_bolus(self, store: SQLiteStore) -> None:
        _seed_insulin(store)
        server = build_server(store)
        at = BOLUS_TS + timedelta(minutes=15)
        result = asyncio.run(server.call_tool("get_iob", {"timestamp": at.isoformat()}))
        payload = result.structured_content
        assert payload is not None
        assert payload["iob_units"] > 0
        assert payload["iob_units"] <= 5.0
        assert payload["n_recent_boluses"] == 1
        assert payload["tier"] == "B"
        assert "oref0" in payload["method"]
        assert "dosing" in payload["note"].lower()

    def test_get_iob_zero_before_bolus(self, store: SQLiteStore) -> None:
        _seed_insulin(store)
        server = build_server(store)
        result = asyncio.run(server.call_tool("get_iob", {"timestamp": T0.isoformat()}))
        payload = result.structured_content
        assert payload is not None
        assert payload["iob_units"] == 0.0
        assert payload["n_recent_boluses"] == 0

    def test_bolus_list_capped_at_100(self, store: SQLiteStore) -> None:
        store.insert_insulin(
            [
                InsulinEvent(
                    ts=T0 + timedelta(minutes=i), kind=InsulinKind.BOLUS, units=0.5
                )
                for i in range(120)
            ]
        )
        server = build_server(store)
        result = asyncio.run(server.call_tool("get_boluses", _window_args(hours=4)))
        payload = result.structured_content
        assert payload is not None
        assert payload["n_boluses"] == 120
        assert len(payload["boluses"]) == 100
        assert "100 of 120" in payload["truncation_note"]


class TestInsulinToolErrors:
    @pytest.mark.parametrize(
        "name",
        ["get_boluses", "get_carb_entries", "get_basal_timeline"],
    )
    def test_bad_iso_returns_error_dict(self, store: SQLiteStore, name: str) -> None:
        _seed_insulin(store)
        server = build_server(store)
        result = asyncio.run(
            server.call_tool(name, {"start": "not-a-date", "end": T0.isoformat()})
        )
        payload = result.structured_content
        assert payload is not None
        assert "error" in payload

    def test_inverted_window_returns_error_dict(self, store: SQLiteStore) -> None:
        _seed_insulin(store)
        server = build_server(store)
        result = asyncio.run(
            server.call_tool(
                "get_boluses",
                {
                    "start": (T0 + timedelta(hours=1)).isoformat(),
                    "end": T0.isoformat(),
                },
            )
        )
        payload = result.structured_content
        assert payload is not None
        assert "invalid window" in payload["error"]

    def test_get_iob_bad_timestamp_returns_error_dict(self, store: SQLiteStore) -> None:
        _seed_insulin(store)
        server = build_server(store)
        result = asyncio.run(server.call_tool("get_iob", {"timestamp": "yesterday-ish"}))
        payload = result.structured_content
        assert payload is not None
        assert "error" in payload
