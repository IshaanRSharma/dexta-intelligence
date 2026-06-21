"""Libre connector tests: pure conversion on stub measurements, connector on a stub client.

No network and no real pylibrelinkup session. ``_StubMeasurement`` satisfies the
``LibreMeasurementLike`` duck type and ``_StubLinkUpClient`` stands in for the
pylibrelinkup ``PyLibreLinkUp`` object, serving separate graph and logbook lists
so the merge/dedupe behaviour of ``pull`` is testable.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from dexta_intelligence.config import LibreConfig, LibreRegion, load_config
from dexta_intelligence.connectors.base import Connector, RealtimeConnector
from dexta_intelligence.connectors.libre import LibreConnector, measurement_to_event

CONFIG = LibreConfig(email="follower@example.com", password="hunter2", region=LibreRegion.EU)


@dataclass(frozen=True)
class _StubMeasurement:
    """Duck-typed stand-in for pylibrelinkup's ``GlucoseMeasurement``."""

    value_in_mg_per_dl: float
    factory_timestamp: datetime


@dataclass(frozen=True)
class _StubTrendMeasurement(_StubMeasurement):
    """Stand-in for ``GlucoseMeasurementWithTrend`` (only ``latest()`` has one)."""

    trend: int


def _measurement(
    value: float = 131.0,
    ts: datetime | None = None,
    trend: int | None = None,
) -> _StubMeasurement:
    ts = ts if ts is not None else datetime(2026, 6, 10, 16, 10, tzinfo=UTC)
    if trend is None:
        return _StubMeasurement(value_in_mg_per_dl=value, factory_timestamp=ts)
    return _StubTrendMeasurement(value_in_mg_per_dl=value, factory_timestamp=ts, trend=trend)


# ─────────────────────────────────────────────────────────────────────────────
# Pure conversion
# ─────────────────────────────────────────────────────────────────────────────


class TestMeasurementToEvent:
    def test_value_and_trend(self) -> None:
        event = measurement_to_event(_measurement(value=142.0, trend=2))
        assert event is not None
        assert event.mg_dl == 142
        assert event.trend == "FortyFiveDown"

    def test_fractional_mg_dl_rounds(self) -> None:
        event = measurement_to_event(_measurement(value=142.6))
        assert event is not None
        assert event.mg_dl == 143

    def test_none_measurement_is_none(self) -> None:
        assert measurement_to_event(None) is None

    def test_utc_passthrough(self) -> None:
        event = measurement_to_event(_measurement())
        assert event is not None
        assert event.ts == datetime(2026, 6, 10, 16, 10, tzinfo=UTC)
        assert event.ts.tzinfo == UTC

    def test_offset_timestamp_converted_to_utc(self) -> None:
        # factory timestamps are UTC by API contract, but conversion must
        # normalize any aware offset it is handed.
        local = datetime(2026, 6, 10, 12, 10, tzinfo=timezone(timedelta(hours=-4)))
        event = measurement_to_event(_measurement(ts=local))
        assert event is not None
        assert event.ts == datetime(2026, 6, 10, 16, 10, tzinfo=UTC)
        assert event.ts.tzinfo == UTC

    def test_naive_timestamp_rejected(self) -> None:
        with pytest.raises(ValueError, match="naive"):
            measurement_to_event(_measurement(ts=datetime(2026, 6, 10, 16, 10)))

    @pytest.mark.parametrize(
        ("trend", "direction"),
        [
            (1, "SingleDown"),  # Trend.DOWN_FAST
            (2, "FortyFiveDown"),  # Trend.DOWN_SLOW
            (3, "Flat"),  # Trend.STABLE
            (4, "FortyFiveUp"),  # Trend.UP_SLOW
            (5, "SingleUp"),  # Trend.UP_FAST
        ],
    )
    def test_trend_subset_maps_to_directions(self, trend: int, direction: str) -> None:
        """Every member of Libre's five-arrow cloud enum, clamped onto the vocabulary."""
        event = measurement_to_event(_measurement(trend=trend))
        assert event is not None
        assert event.trend == direction

    @pytest.mark.parametrize("trend", [0, 6, 7, -1, 99])
    def test_unknown_trend_values_become_none(self, trend: int) -> None:
        event = measurement_to_event(_measurement(trend=trend))
        assert event is not None
        assert event.trend is None

    def test_double_arrows_unreachable(self) -> None:
        """Libre's cloud enum has no Double* members - no input maps to them."""
        directions = {
            measurement_to_event(_measurement(trend=t)).trend  # type: ignore[union-attr]
            for t in range(-1, 10)
        }
        assert "DoubleUp" not in directions
        assert "DoubleDown" not in directions

    def test_trendless_measurement_has_no_trend(self) -> None:
        """graph()/logbook() measurements carry no trend attribute at all."""
        event = measurement_to_event(_measurement())
        assert event is not None
        assert event.trend is None

    def test_out_of_range_value_rejected(self) -> None:
        with pytest.raises(ValueError, match="mg_dl"):
            measurement_to_event(_measurement(value=900.0))


# ─────────────────────────────────────────────────────────────────────────────
# Connector - stubbed LinkUp client, no network
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _StubPatient:
    patient_id: str


@dataclass
class _StubLinkUpClient:
    """Stands in for pylibrelinkup's ``PyLibreLinkUp``; records calls made."""

    graph_data: list[_StubMeasurement] = field(default_factory=list)
    logbook_data: list[_StubMeasurement] = field(default_factory=list)
    latest_data: _StubMeasurement | None = None
    patients: list[_StubPatient] = field(default_factory=lambda: [_StubPatient("patient-1")])
    raise_on_call: Exception | None = None
    authenticate_calls: int = 0
    requested_patients: list[str] = field(default_factory=list)

    def _maybe_raise(self) -> None:
        if self.raise_on_call is not None:
            raise self.raise_on_call

    def authenticate(self) -> None:
        self._maybe_raise()
        self.authenticate_calls += 1

    def get_patients(self) -> list[_StubPatient]:
        self._maybe_raise()
        return self.patients

    def latest(self, patient_identifier: str) -> _StubMeasurement | None:
        self._maybe_raise()
        self.requested_patients.append(patient_identifier)
        return self.latest_data

    def graph(self, patient_identifier: str) -> list[_StubMeasurement]:
        self._maybe_raise()
        self.requested_patients.append(patient_identifier)
        return self.graph_data

    def logbook(self, patient_identifier: str) -> list[_StubMeasurement]:
        self._maybe_raise()
        self.requested_patients.append(patient_identifier)
        return self.logbook_data


def _recent_measurements(
    count: int = 6, *, end: datetime | None = None, step_min: int = 1
) -> list[_StubMeasurement]:
    """``count`` measurements at ``step_min`` spacing ending at ``end`` (default: now)."""
    end = end if end is not None else datetime.now(tz=UTC)
    return [
        _measurement(value=120.0 + i, ts=end - timedelta(minutes=step_min * i))
        for i in range(count)
    ]


def _connector(client: _StubLinkUpClient, config: LibreConfig = CONFIG) -> LibreConnector:
    return LibreConnector(config, client=client)


class TestLibreConnector:
    def test_satisfies_both_protocols(self) -> None:
        connector = _connector(_StubLinkUpClient())
        assert isinstance(connector, Connector)
        assert isinstance(connector, RealtimeConnector)
        assert connector.source == "libre"

    # -- check -----------------------------------------------------------------

    def test_check_ok_reports_latest_reading(self) -> None:
        latest_ts = datetime(2026, 6, 10, 16, 10, tzinfo=UTC)
        client = _StubLinkUpClient(latest_data=_measurement(ts=latest_ts, trend=3))
        report = _connector(client).check()
        assert report.ok is True
        assert report.source == "libre"
        assert "session" in report.detail
        assert report.latest_data_ts == latest_ts
        assert client.authenticate_calls == 1

    def test_check_ok_without_data(self) -> None:
        report = _connector(_StubLinkUpClient()).check()
        assert report.ok is True
        assert report.latest_data_ts is None

    def test_check_auth_failure_is_not_ok(self) -> None:
        client = _StubLinkUpClient(raise_on_call=Exception("Invalid login credentials"))
        report = _connector(client).check()
        assert report.ok is False
        assert report.source == "libre"
        assert "Invalid login credentials" in report.detail

    def test_check_no_patients_is_not_ok(self) -> None:
        client = _StubLinkUpClient(patients=[])
        report = _connector(client).check()
        assert report.ok is False
        assert "patient" in report.detail

    # -- patient resolution ------------------------------------------------------

    def test_first_patient_used_when_unconfigured(self) -> None:
        client = _StubLinkUpClient(
            patients=[_StubPatient("patient-1"), _StubPatient("patient-2")]
        )
        _connector(client).current()
        assert client.requested_patients == ["patient-1"]

    def test_configured_patient_id_wins(self) -> None:
        client = _StubLinkUpClient(patients=[_StubPatient("patient-1")])
        config = LibreConfig(
            email="follower@example.com", password="hunter2", patient_id="patient-9"
        )
        _connector(client, config).current()
        assert client.requested_patients == ["patient-9"]

    def test_authenticates_once_across_calls(self) -> None:
        client = _StubLinkUpClient(latest_data=_measurement(ts=datetime.now(tz=UTC)))
        connector = _connector(client)
        connector.current()
        connector.pull(datetime.now(tz=UTC) - timedelta(hours=1))
        assert client.authenticate_calls == 1

    # -- current ---------------------------------------------------------------

    def test_current_returns_fresh_event(self) -> None:
        ts = datetime.now(tz=UTC) - timedelta(minutes=1)
        client = _StubLinkUpClient(latest_data=_measurement(value=104.0, ts=ts, trend=5))
        event = _connector(client).current()
        assert event is not None
        assert event.mg_dl == 104
        assert event.trend == "SingleUp"
        assert event.ts.tzinfo == UTC

    def test_current_none_when_reading_is_stale(self) -> None:
        # LinkUp's latest() returns the newest reading no matter how old;
        # the connector must enforce freshness itself.
        stale = _measurement(ts=datetime.now(tz=UTC) - timedelta(hours=2))
        client = _StubLinkUpClient(latest_data=stale)
        assert _connector(client).current() is None

    def test_current_none_when_no_reading(self) -> None:
        assert _connector(_StubLinkUpClient()).current() is None

    def test_current_propagates_failures(self) -> None:
        client = _StubLinkUpClient(raise_on_call=Exception("boom"))
        with pytest.raises(Exception, match="boom"):
            _connector(client).current()

    # -- pull ------------------------------------------------------------------

    def test_pull_returns_raws_and_glucose(self) -> None:
        client = _StubLinkUpClient(graph_data=_recent_measurements(6))
        batch = _connector(client).pull(datetime.now(tz=UTC) - timedelta(hours=2))
        assert len(batch.glucose) == 6
        assert len(batch.raw) == 6
        assert all(r.source == "libre" for r in batch.raw)
        assert all(r.source_id.startswith("linkup:") for r in batch.raw)
        assert all(isinstance(r.payload, dict) for r in batch.raw)
        # no other event kinds: LinkUp serves glucose only
        assert batch.insulin == [] and batch.meals == [] and batch.predictions == []

    def test_pull_source_ids_unique_and_stable(self) -> None:
        client = _StubLinkUpClient(graph_data=_recent_measurements(6))
        connector = _connector(client)
        since = datetime.now(tz=UTC) - timedelta(hours=2)
        first = connector.pull(since)
        second = connector.pull(since)
        assert len({r.source_id for r in first.raw}) == 6
        # same readings -> identical ids, so the store's idempotency key holds
        assert [r.source_id for r in first.raw] == [r.source_id for r in second.raw]

    def test_pull_merges_graph_and_logbook_deduped(self) -> None:
        now = datetime.now(tz=UTC)
        graph = _recent_measurements(6, end=now)
        # logbook overlaps the newest 3 readings and adds 2 older ones
        logbook = graph[:3] + _recent_measurements(2, end=now - timedelta(minutes=30))
        client = _StubLinkUpClient(graph_data=graph, logbook_data=logbook)
        batch = _connector(client).pull(now - timedelta(hours=2))
        assert len(batch.glucose) == 8
        assert len({r.source_id for r in batch.raw}) == 8

    def test_pull_applies_watermark_with_dedupe_margin(self) -> None:
        now = datetime.now(tz=UTC)
        client = _StubLinkUpClient(graph_data=_recent_measurements(12, end=now, step_min=5))
        since = now - timedelta(minutes=21)
        batch = _connector(client).pull(since)
        # margin pulls the window back 5 min: readings at 0..25 min ago qualify
        assert len(batch.glucose) == 6
        assert min(e.ts for e in batch.glucose) >= since - timedelta(minutes=5)

    def test_pull_results_sorted_ascending(self) -> None:
        now = datetime.now(tz=UTC)
        client = _StubLinkUpClient(
            graph_data=_recent_measurements(4, end=now),
            logbook_data=_recent_measurements(3, end=now - timedelta(minutes=20)),
        )
        batch = _connector(client).pull(now - timedelta(hours=2))
        stamps = [e.ts for e in batch.glucose]
        assert stamps == sorted(stamps)
        assert [r.source_ts for r in batch.raw] == stamps

    def test_pull_timestamps_are_utc(self) -> None:
        client = _StubLinkUpClient(graph_data=_recent_measurements(4))
        batch = _connector(client).pull(datetime.now(tz=UTC) - timedelta(hours=1))
        for event in batch.glucose:
            assert event.ts.tzinfo == UTC
        for raw in batch.raw:
            assert raw.source_ts.tzinfo == UTC

    def test_pull_empty_when_nothing_recent(self) -> None:
        old = _recent_measurements(3, end=datetime.now(tz=UTC) - timedelta(days=3))
        client = _StubLinkUpClient(graph_data=old, logbook_data=old)
        batch = _connector(client).pull(datetime.now(tz=UTC) - timedelta(hours=1))
        assert batch.glucose == []
        assert batch.raw == []

    # -- lazy import -------------------------------------------------------------

    @pytest.mark.skipif(
        importlib.util.find_spec("pylibrelinkup") is not None,
        reason="pylibrelinkup is installed; the install-hint path is unreachable",
    )
    def test_missing_dependency_raises_install_hint(self) -> None:
        with pytest.raises(RuntimeError, match=r"dexta-intelligence\[libre\]"):
            LibreConnector(CONFIG).current()


# ─────────────────────────────────────────────────────────────────────────────
# Config - additive LibreConfig section + env overrides
# ─────────────────────────────────────────────────────────────────────────────


class TestLibreConfig:
    def test_defaults(self) -> None:
        config = LibreConfig()
        assert config.email == ""
        assert config.password == ""
        assert config.region is LibreRegion.US
        assert config.patient_id == ""

    def test_regions_mirror_pylibrelinkup_identifiers(self) -> None:
        assert {r.value for r in LibreRegion} == {
            "us", "eu", "eu2", "ae", "ap", "au", "ca", "de", "fr", "jp", "la", "ru",
        }

    def test_invalid_region_rejected(self) -> None:
        with pytest.raises(ValueError, match="region"):
            LibreConfig(region="atlantis")  # type: ignore[arg-type]

    def test_env_overrides(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("LIBRE_EMAIL", "env-follower@example.com")
        monkeypatch.setenv("LIBRE_PASSWORD", "env-secret")
        monkeypatch.setenv("LIBRE_REGION", "EU2")  # case-insensitive
        config = load_config(tmp_path / "missing.toml")
        assert config.libre.email == "env-follower@example.com"
        assert config.libre.password == "env-secret"
        assert config.libre.region is LibreRegion.EU2

    def test_env_absent_keeps_defaults(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        for key in ("LIBRE_EMAIL", "LIBRE_PASSWORD", "LIBRE_REGION"):
            monkeypatch.delenv(key, raising=False)
        config = load_config(tmp_path / "missing.toml")
        assert config.libre == LibreConfig()
