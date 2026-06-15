"""CareLink connector tests - pure conversion on payload fixtures, connector on a stub client.

No network and no real carelink_client session: ``parse_recent_data`` runs on
plain dict fixtures shaped like CareLink's recent-data payload, and
``_StubCareLinkClient`` stands in for the carelink_client ``CareLinkClient``,
serving one canned payload so ``pull`` / ``check`` are testable.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from dexta_intelligence.connectors.base import NormalizedBatch

from dexta_intelligence.config import CareLinkConfig
from dexta_intelligence.connectors.base import Connector
from dexta_intelligence.connectors.carelink import (
    CareLinkConnector,
    parse_recent_data,
)
from dexta_intelligence.models import InsulinKind

CONFIG = CareLinkConfig(username="follower", password="hunter2", country="us")

_NOW = datetime(2026, 6, 12, 16, 10, tzinfo=UTC)


def _iso(ts: datetime) -> str:
    return ts.isoformat()


def _millis(ts: datetime) -> int:
    return int(ts.timestamp() * 1000)


def _sg(value: float, ts: datetime, *, trend: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {"sg": value, "datetime": _iso(ts)}
    if trend is not None:
        record["trend"] = trend
    return record


def _payload(
    *,
    sgs: list[dict[str, Any]] | None = None,
    markers: list[dict[str, Any]] | None = None,
    units: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"sgs": sgs or [], "markers": markers or []}
    if units is not None:
        payload["units"] = units
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# Pure conversion
# ─────────────────────────────────────────────────────────────────────────────


class TestParseRecentData:
    def test_sg_to_glucose_event(self) -> None:
        glucose, insulin, meals = parse_recent_data(_payload(sgs=[_sg(142.0, _NOW, trend="Flat")]))
        assert len(glucose) == 1
        assert glucose[0].mg_dl == 142
        assert glucose[0].trend == "Flat"
        assert glucose[0].ts == _NOW
        assert insulin == [] and meals == []

    def test_sg_fractional_rounds(self) -> None:
        glucose, _, _ = parse_recent_data(_payload(sgs=[_sg(142.6, _NOW)]))
        assert glucose[0].mg_dl == 143

    def test_sg_zero_is_gap_dropped(self) -> None:
        glucose, _, _ = parse_recent_data(_payload(sgs=[_sg(0, _NOW)]))
        assert glucose == []

    def test_mmol_converted_to_mg_dl(self) -> None:
        # 7.8 mmol/L x 18 = 140.4 -> 140
        glucose, _, _ = parse_recent_data(_payload(sgs=[_sg(7.8, _NOW)], units="MMOL_L"))
        assert glucose[0].mg_dl == 140

    def test_mmol_unit_variants(self) -> None:
        for token in ("MMOL_L", "mmol/L", "MMOL/L", "mmol"):
            glucose, _, _ = parse_recent_data(_payload(sgs=[_sg(5.0, _NOW)], units=token))
            assert glucose[0].mg_dl == 90

    def test_mg_dl_is_native_default(self) -> None:
        glucose, _, _ = parse_recent_data(_payload(sgs=[_sg(120.0, _NOW)]))
        assert glucose[0].mg_dl == 120

    def test_epoch_millis_timestamp(self) -> None:
        glucose, _, _ = parse_recent_data(_payload(sgs=[{"sg": 120.0, "timestamp": _millis(_NOW)}]))
        assert glucose[0].ts == _NOW
        assert glucose[0].ts.tzinfo == UTC

    def test_offset_timestamp_converted_to_utc(self) -> None:
        local = datetime(2026, 6, 12, 12, 10, tzinfo=timezone(timedelta(hours=-4)))
        glucose, _, _ = parse_recent_data(_payload(sgs=[_sg(120.0, local)]))
        assert glucose[0].ts == _NOW
        assert glucose[0].ts.tzinfo == UTC

    def test_naive_iso_timestamp_rejected(self) -> None:
        with pytest.raises(ValueError, match="naive"):
            parse_recent_data(_payload(sgs=[{"sg": 120.0, "datetime": "2026-06-12T16:10:00"}]))

    def test_bolus_marker_to_insulin(self) -> None:
        markers = [{"type": "BOLUS", "amount": 4.5, "datetime": _iso(_NOW)}]
        _, insulin, _ = parse_recent_data(_payload(markers=markers))
        assert len(insulin) == 1
        assert insulin[0].kind is InsulinKind.BOLUS
        assert insulin[0].units == 4.5
        assert insulin[0].automatic is None

    def test_insulin_marker_aliased_to_bolus(self) -> None:
        markers = [{"type": "INSULIN", "deliveredFastAmount": 2.0, "datetime": _iso(_NOW)}]
        _, insulin, _ = parse_recent_data(_payload(markers=markers))
        assert insulin[0].kind is InsulinKind.BOLUS
        assert insulin[0].units == 2.0

    def test_basal_marker(self) -> None:
        markers = [{"type": "AUTO_BASAL_DELIVERY", "amount": 0.8, "datetime": _iso(_NOW)}]
        _, insulin, _ = parse_recent_data(_payload(markers=markers))
        assert insulin[0].kind is InsulinKind.TEMP_BASAL  # auto_basal token wins
        assert insulin[0].automatic is True

    def test_plain_basal_marker(self) -> None:
        markers = [{"type": "BASAL", "amount": 0.5, "datetime": _iso(_NOW)}]
        _, insulin, _ = parse_recent_data(_payload(markers=markers))
        assert insulin[0].kind is InsulinKind.BASAL
        assert insulin[0].automatic is True

    def test_temp_basal_marker(self) -> None:
        markers = [
            {"type": "TEMP_BASAL", "amount": 1.2, "duration": 30, "datetime": _iso(_NOW)}
        ]
        _, insulin, _ = parse_recent_data(_payload(markers=markers))
        assert insulin[0].kind is InsulinKind.TEMP_BASAL
        assert insulin[0].duration_min == 30
        assert insulin[0].automatic is True

    def test_suspend_marker(self) -> None:
        markers = [{"type": "INSULIN_SUSPEND", "duration": 15, "datetime": _iso(_NOW)}]
        _, insulin, _ = parse_recent_data(_payload(markers=markers))
        assert insulin[0].kind is InsulinKind.SUSPEND
        assert insulin[0].units is None
        assert insulin[0].duration_min == 15

    def test_lgs_suspend_marker(self) -> None:
        markers = [{"type": "LGS_SUSPEND", "datetime": _iso(_NOW)}]
        _, insulin, _ = parse_recent_data(_payload(markers=markers))
        assert insulin[0].kind is InsulinKind.SUSPEND

    def test_carb_marker_to_meal(self) -> None:
        markers = [{"type": "MEAL", "carbInput": 45.0, "datetime": _iso(_NOW)}]
        _, _, meals = parse_recent_data(_payload(markers=markers))
        assert len(meals) == 1
        assert meals[0].carbs_g == 45.0
        assert meals[0].ts == _NOW

    def test_carb_marker_alias(self) -> None:
        markers = [{"type": "CARB", "carbs": 12.0, "datetime": _iso(_NOW)}]
        _, _, meals = parse_recent_data(_payload(markers=markers))
        assert meals[0].carbs_g == 12.0

    def test_irrelevant_marker_dropped(self) -> None:
        markers = [{"type": "CALIBRATION", "datetime": _iso(_NOW)}]
        _, insulin, meals = parse_recent_data(_payload(markers=markers))
        assert insulin == [] and meals == []

    def test_marker_without_ts_dropped(self) -> None:
        _, insulin, _ = parse_recent_data(_payload(markers=[{"type": "BOLUS", "amount": 1.0}]))
        assert insulin == []

    def test_empty_payload(self) -> None:
        glucose, insulin, meals = parse_recent_data({})
        assert glucose == [] and insulin == [] and meals == []

    def test_out_of_range_value_rejected(self) -> None:
        with pytest.raises(ValueError, match="mg_dl"):
            parse_recent_data(_payload(sgs=[_sg(900.0, _NOW)]))


# ─────────────────────────────────────────────────────────────────────────────
# Connector - stubbed CareLink client, no network
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _StubCareLinkClient:
    """Stands in for carelink_client's ``CareLinkClient``; records calls made."""

    payload: dict[str, Any] | None = None
    raise_on_login: Exception | None = None
    raise_on_data: Exception | None = None
    login_calls: int = 0
    data_calls: int = 0

    def login(self) -> bool:
        if self.raise_on_login is not None:
            raise self.raise_on_login
        self.login_calls += 1
        return True

    def recentData(self) -> dict[str, Any] | None:  # noqa: N802 - upstream API name
        if self.raise_on_data is not None:
            raise self.raise_on_data
        self.data_calls += 1
        return self.payload


def _recent_payload(*, end: datetime | None = None, count: int = 4) -> dict[str, Any]:
    end = end if end is not None else datetime.now(tz=UTC)
    sgs = [_sg(120.0 + i, end - timedelta(minutes=5 * i)) for i in range(count)]
    markers = [
        {"type": "BOLUS", "amount": 3.0, "datetime": _iso(end - timedelta(minutes=10))},
        {"type": "MEAL", "carbInput": 30.0, "datetime": _iso(end - timedelta(minutes=10))},
    ]
    return _payload(sgs=sgs, markers=markers)


def _connector(client: _StubCareLinkClient, config: CareLinkConfig = CONFIG) -> CareLinkConnector:
    return CareLinkConnector(config, client=client)


def _all_source_ids(batch: NormalizedBatch) -> list[str]:
    return [r.source_id for r in batch.raw]


class TestCareLinkConnector:
    def test_satisfies_connector_protocol(self) -> None:
        connector = _connector(_StubCareLinkClient())
        assert isinstance(connector, Connector)
        assert connector.source == "carelink"

    def test_not_realtime(self) -> None:
        # CareLink is batch-only: no current() surface.
        assert not hasattr(_connector(_StubCareLinkClient()), "current")

    # -- check -----------------------------------------------------------------

    def test_check_ok_reports_latest_reading(self) -> None:
        latest = datetime(2026, 6, 12, 16, 10, tzinfo=UTC)
        client = _StubCareLinkClient(payload=_payload(sgs=[_sg(120.0, latest)]))
        report = _connector(client).check()
        assert report.ok is True
        assert report.source == "carelink"
        assert "session" in report.detail
        assert report.latest_data_ts == latest
        assert client.login_calls == 1

    def test_check_ok_without_data(self) -> None:
        report = _connector(_StubCareLinkClient(payload=_payload())).check()
        assert report.ok is True
        assert report.latest_data_ts is None

    def test_check_auth_failure_is_not_ok(self) -> None:
        client = _StubCareLinkClient(raise_on_login=Exception("Invalid login credentials"))
        report = _connector(client).check()
        assert report.ok is False
        assert report.source == "carelink"
        assert "Invalid login credentials" in report.detail

    def test_check_data_failure_is_not_ok(self) -> None:
        client = _StubCareLinkClient(raise_on_data=Exception("boom"))
        report = _connector(client).check()
        assert report.ok is False
        assert "boom" in report.detail

    # -- pull ------------------------------------------------------------------

    def test_pull_returns_all_event_kinds(self) -> None:
        now = datetime.now(tz=UTC)
        client = _StubCareLinkClient(payload=_recent_payload(end=now, count=4))
        batch = _connector(client).pull(now - timedelta(hours=2))
        assert len(batch.glucose) == 4
        assert len(batch.insulin) == 1
        assert len(batch.meals) == 1
        assert len(batch.raw) == 6
        assert all(r.source == "carelink" for r in batch.raw)
        assert all(r.source_id.startswith("carelink:") for r in batch.raw)

    def test_pull_source_id_prefixes(self) -> None:
        now = datetime.now(tz=UTC)
        client = _StubCareLinkClient(payload=_recent_payload(end=now))
        ids = _all_source_ids(_connector(client).pull(now - timedelta(hours=2)))
        assert any(sid.startswith("carelink:sg:") for sid in ids)
        assert any(sid.startswith("carelink:bolus:") for sid in ids)
        assert any(sid.startswith("carelink:meal:") for sid in ids)

    def test_pull_source_ids_unique_and_stable(self) -> None:
        now = datetime.now(tz=UTC)
        client = _StubCareLinkClient(payload=_recent_payload(end=now))
        connector = _connector(client)
        since = now - timedelta(hours=2)
        first = connector.pull(since)
        second = connector.pull(since)
        assert len(set(_all_source_ids(first))) == len(first.raw)
        assert _all_source_ids(first) == _all_source_ids(second)

    def test_pull_applies_watermark(self) -> None:
        now = datetime.now(tz=UTC)
        client = _StubCareLinkClient(payload=_recent_payload(end=now, count=12))
        # readings at 0,5,10,...55 min ago; since = 22 min ago -> 0,5,10,15,20
        batch = _connector(client).pull(now - timedelta(minutes=22))
        assert len(batch.glucose) == 5
        assert min(e.ts for e in batch.glucose) >= now - timedelta(minutes=22)

    def test_pull_empty_when_nothing_recent(self) -> None:
        old = _recent_payload(end=datetime.now(tz=UTC) - timedelta(days=3))
        client = _StubCareLinkClient(payload=old)
        batch = _connector(client).pull(datetime.now(tz=UTC) - timedelta(hours=1))
        assert batch.glucose == [] and batch.insulin == [] and batch.meals == []
        assert batch.raw == []

    def test_pull_empty_when_no_payload(self) -> None:
        batch = _connector(_StubCareLinkClient(payload=None)).pull(datetime.now(tz=UTC))
        assert batch.raw == []

    def test_pull_timestamps_are_utc(self) -> None:
        now = datetime.now(tz=UTC)
        client = _StubCareLinkClient(payload=_recent_payload(end=now))
        batch = _connector(client).pull(now - timedelta(hours=2))
        for event in batch.glucose:
            assert event.ts.tzinfo == UTC
        for raw in batch.raw:
            assert raw.source_ts.tzinfo == UTC

    # -- lazy import -------------------------------------------------------------

    @pytest.mark.skipif(
        importlib.util.find_spec("carelink_client") is not None,
        reason="carelink_client is installed; the install-hint path is unreachable",
    )
    def test_missing_dependency_raises_install_hint(self) -> None:
        with pytest.raises(RuntimeError, match=r"dexta-intelligence\[carelink\]"):
            CareLinkConnector(CONFIG).check()
