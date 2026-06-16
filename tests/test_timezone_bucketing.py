"""Local-timezone bucketing in the discovery toolkit.

Storage is always UTC, but day/hour grouping must follow the patient's local
clock so "overnight"/per-day analysis lands at the right wall-clock time. A
reading at 02:00 UTC belongs to the *previous* local day (and 22:00 local hour)
for an America/New_York patient — these tests pin that.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.agents.discovery_tools import DiscoveryToolkit
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.models import GlucoseEvent
from dexta_intelligence.store import SQLiteStore

_NY = "America/New_York"
# 2026-03-15T02:00:00Z → 2026-03-14 22:00 in New York (EDT, UTC-4).
_NEAR_MIDNIGHT = datetime(2026, 3, 15, 2, 0, tzinfo=UTC)


def _toolkit(tz: str) -> DiscoveryToolkit:
    store = SQLiteStore(":memory:")
    store.migrate()
    # 48h of hourly readings so each local day clears the per-day minimum.
    start = datetime(2026, 3, 13, tzinfo=UTC)
    store.insert_glucose(
        [GlucoseEvent(ts=start + timedelta(hours=i), mg_dl=120) for i in range(48)]
    )
    ctx = AgentContext(
        store=store,
        window=(date(2026, 3, 12), date(2026, 3, 16)),
        gates=ColdStartReport.from_coverage(store.coverage()),
        run_id="tz-test",
        timezone=tz,
    )
    return DiscoveryToolkit(ctx)


def test_tzinfo_reflects_configured_zone() -> None:
    assert _toolkit(_NY).tzinfo == ZoneInfo(_NY)
    assert _toolkit("UTC").tzinfo == ZoneInfo("UTC")


def test_unknown_zone_falls_back_to_utc() -> None:
    assert _toolkit("Not/AZone").tzinfo == ZoneInfo("UTC")


def test_near_midnight_reading_buckets_to_local_day_and_hour() -> None:
    ny = _toolkit(_NY)
    assert ny._ld(_NEAR_MIDNIGHT) == date(2026, 3, 14)
    assert ny._lh(_NEAR_MIDNIGHT) == 22

    utc = _toolkit("UTC")
    assert utc._ld(_NEAR_MIDNIGHT) == date(2026, 3, 15)
    assert utc._lh(_NEAR_MIDNIGHT) == 2


def test_day_bounds_span_a_local_calendar_day_in_utc() -> None:
    ny = _toolkit(_NY)
    start, end = ny._day_bounds(date(2026, 3, 14))
    # Local midnight 2026-03-14 EDT == 04:00 UTC; the day ends a tick before the next.
    assert start == datetime(2026, 3, 14, 4, 0, tzinfo=UTC)
    assert end < datetime(2026, 3, 15, 4, 0, tzinfo=UTC)
    # The 02:00Z reading is 22:00 local on the 14th — inside that local day's bounds.
    assert start < _NEAR_MIDNIGHT < end


def test_daily_count_shifts_with_zone() -> None:
    """The same readings attribute differently to a local day across zones."""
    target = date(2026, 3, 14)
    # UTC: the full 00:00-23:00Z of the 14th (24 readings).
    assert len(_toolkit("UTC")._day_values(target)) == 24
    # NY: the local day starts at 04:00Z, so fewer of the 14th's UTC readings land.
    assert len(_toolkit(_NY)._day_values(target)) < 24
