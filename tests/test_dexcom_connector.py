"""Dexcom connector tests - pure conversion on stub readings, connector on a stub client.

No network and no real pydexcom session: ``_StubReading`` satisfies the
``DexcomReadingLike`` duck type and ``_StubShareClient`` stands in for the
pydexcom ``Dexcom`` object, recording the request window so the ~24h Share
history cap is testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from dexta_intelligence.config import DexcomConfig, load_config
from dexta_intelligence.connectors.base import Connector, RealtimeConnector
from dexta_intelligence.connectors.dexcom import DexcomConnector, reading_to_event

CONFIG = DexcomConfig(username="user@example.com", password="hunter2", ous=False)


@dataclass(frozen=True)
class _StubReading:
    """Duck-typed stand-in for pydexcom's ``GlucoseReading``."""

    value: int
    trend_direction: str
    datetime: datetime


def _reading(
    value: int = 131,
    trend: str = "Flat",
    ts: datetime | None = None,
) -> _StubReading:
    return _StubReading(
        value=value,
        trend_direction=trend,
        datetime=ts if ts is not None else datetime(2026, 6, 10, 16, 10, tzinfo=UTC),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pure conversion
# ─────────────────────────────────────────────────────────────────────────────


class TestReadingToEvent:
    def test_value_and_trend(self) -> None:
        event = reading_to_event(_reading(value=142, trend="FortyFiveDown"))
        assert event is not None
        assert event.mg_dl == 142
        assert event.trend == "FortyFiveDown"

    def test_none_reading_is_none(self) -> None:
        assert reading_to_event(None) is None

    def test_utc_passthrough(self) -> None:
        event = reading_to_event(_reading())
        assert event is not None
        assert event.ts == datetime(2026, 6, 10, 16, 10, tzinfo=UTC)
        assert event.ts.tzinfo == UTC

    def test_offset_timestamp_converted_to_utc(self) -> None:
        # Share hands pydexcom a local UTC offset; conversion must normalize.
        local = datetime(2026, 6, 10, 12, 10, tzinfo=timezone(timedelta(hours=-4)))
        event = reading_to_event(_reading(ts=local))
        assert event is not None
        assert event.ts == datetime(2026, 6, 10, 16, 10, tzinfo=UTC)
        assert event.ts.tzinfo == UTC

    def test_naive_timestamp_rejected(self) -> None:
        with pytest.raises(ValueError, match="naive"):
            reading_to_event(_reading(ts=datetime(2026, 6, 10, 16, 10)))

    @pytest.mark.parametrize(
        "trend",
        [
            "DoubleUp",
            "SingleUp",
            "FortyFiveUp",
            "Flat",
            "FortyFiveDown",
            "SingleDown",
            "DoubleDown",
        ],
    )
    def test_informative_trends_pass_through(self, trend: str) -> None:
        event = reading_to_event(_reading(trend=trend))
        assert event is not None
        assert event.trend == trend

    @pytest.mark.parametrize("trend", ["None", "NotComputable", "RateOutOfRange", ""])
    def test_uninformative_trends_become_none(self, trend: str) -> None:
        event = reading_to_event(_reading(trend=trend))
        assert event is not None
        assert event.trend is None

    def test_out_of_range_value_rejected(self) -> None:
        with pytest.raises(ValueError, match="mg_dl"):
            reading_to_event(_reading(value=900))


# ─────────────────────────────────────────────────────────────────────────────
# Connector - stubbed Share client, no network
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _StubShareClient:
    """Stands in for pydexcom's ``Dexcom``; records the requested window."""

    readings: list[_StubReading] = field(default_factory=list)
    raise_on_call: Exception | None = None
    calls: list[tuple[int, int]] = field(default_factory=list)

    def get_glucose_readings(
        self, minutes: int = 1440, max_count: int = 288
    ) -> list[_StubReading]:
        if self.raise_on_call is not None:
            raise self.raise_on_call
        self.calls.append((minutes, max_count))
        cutoff = datetime.now(tz=UTC) - timedelta(minutes=minutes)
        return [r for r in self.readings if r.datetime >= cutoff][:max_count]

    def get_latest_glucose_reading(self) -> _StubReading | None:
        if self.raise_on_call is not None:
            raise self.raise_on_call
        return max(self.readings, key=lambda r: r.datetime, default=None)

    def get_current_glucose_reading(self) -> _StubReading | None:
        if self.raise_on_call is not None:
            raise self.raise_on_call
        cutoff = datetime.now(tz=UTC) - timedelta(minutes=10)
        recent = [r for r in self.readings if r.datetime >= cutoff]
        return max(recent, key=lambda r: r.datetime, default=None)


def _recent_readings(count: int = 6, *, end: datetime | None = None) -> list[_StubReading]:
    """``count`` readings at 5-minute spacing ending at ``end`` (default: now)."""
    end = end if end is not None else datetime.now(tz=UTC)
    return [
        _reading(value=120 + i, trend="Flat", ts=end - timedelta(minutes=5 * i))
        for i in range(count)
    ]


def _connector(client: _StubShareClient) -> DexcomConnector:
    return DexcomConnector(CONFIG, client=client)


class TestDexcomConnector:
    def test_satisfies_both_protocols(self) -> None:
        connector = _connector(_StubShareClient())
        assert isinstance(connector, Connector)
        assert isinstance(connector, RealtimeConnector)
        assert connector.source == "dexcom"

    # -- check -----------------------------------------------------------------

    def test_check_ok_reports_latest_reading(self) -> None:
        latest_ts = datetime(2026, 6, 10, 16, 10, tzinfo=UTC)
        client = _StubShareClient(readings=_recent_readings(3, end=latest_ts))
        report = _connector(client).check()
        assert report.ok is True
        assert report.source == "dexcom"
        assert "session" in report.detail
        assert report.latest_data_ts == latest_ts

    def test_check_ok_without_data(self) -> None:
        report = _connector(_StubShareClient()).check()
        assert report.ok is True
        assert report.latest_data_ts is None

    def test_check_auth_failure_is_not_ok(self) -> None:
        client = _StubShareClient(raise_on_call=Exception("Invalid password"))
        report = _connector(client).check()
        assert report.ok is False
        assert report.source == "dexcom"
        assert "Invalid password" in report.detail

    # -- current ---------------------------------------------------------------

    def test_current_returns_event(self) -> None:
        client = _StubShareClient(readings=_recent_readings(2))
        event = _connector(client).current()
        assert event is not None
        assert event.mg_dl == 120
        assert event.trend == "Flat"
        assert event.ts.tzinfo == UTC

    def test_current_none_when_no_recent_reading(self) -> None:
        stale = _recent_readings(3, end=datetime.now(tz=UTC) - timedelta(hours=2))
        client = _StubShareClient(readings=stale)
        assert _connector(client).current() is None

    def test_current_propagates_failures(self) -> None:
        client = _StubShareClient(raise_on_call=Exception("boom"))
        with pytest.raises(Exception, match="boom"):
            _connector(client).current()

    # -- pull ------------------------------------------------------------------

    def test_pull_returns_raws_and_glucose(self) -> None:
        client = _StubShareClient(readings=_recent_readings(6))
        batch = _connector(client).pull(datetime.now(tz=UTC) - timedelta(hours=2))
        assert len(batch.glucose) == 6
        assert len(batch.raw) == 6
        assert all(r.source == "dexcom" for r in batch.raw)
        assert all(r.source_id.startswith("share:") for r in batch.raw)
        assert all(isinstance(r.payload, dict) for r in batch.raw)
        # no other event kinds: Share serves glucose only
        assert batch.insulin == [] and batch.meals == [] and batch.predictions == []

    def test_pull_source_ids_unique(self) -> None:
        client = _StubShareClient(readings=_recent_readings(6))
        batch = _connector(client).pull(datetime.now(tz=UTC) - timedelta(hours=2))
        assert len({r.source_id for r in batch.raw}) == 6

    def test_pull_applies_watermark_with_dedupe_margin(self) -> None:
        now = datetime.now(tz=UTC)
        client = _StubShareClient(readings=_recent_readings(12, end=now))
        since = now - timedelta(minutes=21)
        batch = _connector(client).pull(since)
        # margin pulls the window back 5 min: readings at 0..25 min ago qualify
        assert len(batch.glucose) == 6
        assert min(e.ts for e in batch.glucose) >= since - timedelta(minutes=5)

    def test_pull_requests_minutes_matching_window(self) -> None:
        client = _StubShareClient(readings=_recent_readings(3))
        _connector(client).pull(datetime.now(tz=UTC) - timedelta(hours=2))
        ((minutes, max_count),) = client.calls
        # 120 min window + 5 min dedupe margin, small ceil slack
        assert 125 <= minutes <= 127
        assert max_count == 288

    def test_pull_clamps_to_24h_share_cap(self) -> None:
        """The documented Share limit: ancient ``since`` still asks for <= 1440 min."""
        client = _StubShareClient(readings=_recent_readings(6))
        batch = _connector(client).pull(datetime.now(tz=UTC) - timedelta(days=90))
        ((minutes, max_count),) = client.calls
        assert minutes == 1440
        assert max_count == 288
        assert len(batch.glucose) == 6  # only the live day comes back

    def test_pull_timestamps_are_utc(self) -> None:
        client = _StubShareClient(readings=_recent_readings(4))
        batch = _connector(client).pull(datetime.now(tz=UTC) - timedelta(hours=1))
        for event in batch.glucose:
            assert event.ts.tzinfo == UTC
        for raw in batch.raw:
            assert raw.source_ts.tzinfo == UTC


# ─────────────────────────────────────────────────────────────────────────────
# Config - additive DexcomConfig section + env overrides
# ─────────────────────────────────────────────────────────────────────────────


class TestDexcomConfig:
    def test_defaults(self) -> None:
        config = DexcomConfig()
        assert config.username == ""
        assert config.password == ""
        assert config.ous is False

    def test_env_overrides(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("DEXCOM_USERNAME", "env-user@example.com")
        monkeypatch.setenv("DEXCOM_PASSWORD", "env-secret")
        monkeypatch.setenv("DEXCOM_OUS", "true")
        config = load_config(tmp_path / "missing.toml")
        assert config.dexcom.username == "env-user@example.com"
        assert config.dexcom.password == "env-secret"
        assert config.dexcom.ous is True

    def test_env_absent_keeps_defaults(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        for key in ("DEXCOM_USERNAME", "DEXCOM_PASSWORD", "DEXCOM_OUS"):
            monkeypatch.delenv(key, raising=False)
        config = load_config(tmp_path / "missing.toml")
        assert config.dexcom == DexcomConfig()
