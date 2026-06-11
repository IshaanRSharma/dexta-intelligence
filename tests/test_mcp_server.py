"""Tests for the FastMCP server wrapper."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from dexta_intelligence.mcp_server.server import TOOL_NAMES, build_server
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


def _seed(store: SQLiteStore) -> None:
    store.insert_glucose(
        [
            GlucoseEvent(ts=T0 + timedelta(minutes=5 * i), mg_dl=100 + i, trend="Flat")
            for i in range(6)
        ]
    )


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
