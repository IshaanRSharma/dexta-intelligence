"""Tandem connector tests - pure conversion on stub records, connector on a stub client.

No network and no real tconnectsync session: ``_StubBolus`` satisfies the
``BolusLike`` duck type (all-string fields, like the real ``Bolus`` dataclass),
basals are plain dicts shaped like the ControlIQ parser's output, and
``_StubTConnectClient`` stands in for tconnectsync's ``TConnectApi`` - serving a
canned therapy_timeline so ``pull`` / ``check`` are testable end to end.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest

from dexta_intelligence.config import TandemConfig
from dexta_intelligence.connectors.base import Connector, RealtimeConnector
from dexta_intelligence.connectors.tandem import (
    TandemConnector,
    basal_to_event,
    bolus_to_events,
)
from dexta_intelligence.models import InsulinKind

CONFIG = TandemConfig(email="user@example.com", password="hunter2", region="us")

# tconnectsync formats device-local times with an offset; "-04:00" stands in.
_TZ = timezone(timedelta(hours=-4))


def _iso(ts: datetime) -> str:
    return ts.isoformat()


@dataclass(frozen=True)
class _StubBolus:
    """Duck-typed stand-in for tconnectsync's ``Bolus`` (all-string fields)."""

    insulin: str = ""
    carbs: str = ""
    completion_time: str = ""
    request_time: str = ""


def _bolus(
    *,
    insulin: str = "",
    carbs: str = "",
    completion: datetime | None = None,
    request: datetime | None = None,
) -> _StubBolus:
    return _StubBolus(
        insulin=insulin,
        carbs=carbs,
        completion_time=_iso(completion) if completion else "",
        request_time=_iso(request) if request else "",
    )


def _basal(
    *,
    delivery_type: str,
    time: datetime,
    duration_mins: float | None = None,
    basal_rate: float | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {"delivery_type": delivery_type, "time": _iso(time)}
    if duration_mins is not None:
        record["duration_mins"] = duration_mins
    if basal_rate is not None:
        record["basal_rate"] = basal_rate
    return record


# ─────────────────────────────────────────────────────────────────────────────
# Pure conversion - bolus
# ─────────────────────────────────────────────────────────────────────────────


class TestBolusToEvents:
    def test_bolus_becomes_insulin_event(self) -> None:
        ts = datetime(2026, 6, 10, 12, 30, tzinfo=_TZ)
        events = bolus_to_events(_bolus(insulin="4.5", completion=ts))
        assert len(events) == 1
        event = events[0]
        assert event.kind == InsulinKind.BOLUS  # type: ignore[union-attr]
        assert event.units == 4.5  # type: ignore[union-attr]
        assert event.ts == ts.astimezone(UTC)
        assert event.ts.tzinfo == UTC

    def test_meal_bolus_yields_insulin_and_meal(self) -> None:
        ts = datetime(2026, 6, 10, 12, 30, tzinfo=_TZ)
        events = bolus_to_events(_bolus(insulin="6", carbs="45", completion=ts))
        kinds = {type(e).__name__ for e in events}
        assert kinds == {"InsulinEvent", "MealEvent"}
        meal = next(e for e in events if type(e).__name__ == "MealEvent")
        assert meal.carbs_g == 45.0  # type: ignore[union-attr]
        assert all(e.ts == ts.astimezone(UTC) for e in events)

    def test_carbs_only_yields_meal_only(self) -> None:
        ts = datetime(2026, 6, 10, 12, 30, tzinfo=_TZ)
        events = bolus_to_events(_bolus(carbs="30", completion=ts))
        assert len(events) == 1
        assert type(events[0]).__name__ == "MealEvent"

    def test_zero_bolus_no_carbs_yields_nothing(self) -> None:
        ts = datetime(2026, 6, 10, 12, 30, tzinfo=_TZ)
        assert bolus_to_events(_bolus(insulin="0", carbs="0", completion=ts)) == []

    def test_completion_time_preferred_over_request(self) -> None:
        completion = datetime(2026, 6, 10, 12, 35, tzinfo=_TZ)
        request = datetime(2026, 6, 10, 12, 30, tzinfo=_TZ)
        events = bolus_to_events(_bolus(insulin="2", completion=completion, request=request))
        assert events[0].ts == completion.astimezone(UTC)

    def test_falls_back_to_request_time(self) -> None:
        request = datetime(2026, 6, 10, 12, 30, tzinfo=_TZ)
        events = bolus_to_events(_bolus(insulin="2", request=request))
        assert events[0].ts == request.astimezone(UTC)

    def test_no_timestamp_yields_nothing(self) -> None:
        assert bolus_to_events(_bolus(insulin="2")) == []

    def test_naive_timestamp_rejected(self) -> None:
        naive = _StubBolus(insulin="2", completion_time="2026-06-10T12:30:00")
        with pytest.raises(ValueError, match="naive"):
            bolus_to_events(naive)


# ─────────────────────────────────────────────────────────────────────────────
# Pure conversion - basal
# ─────────────────────────────────────────────────────────────────────────────


class TestBasalToEvent:
    def test_temp_rate_is_temp_basal_with_duration(self) -> None:
        ts = datetime(2026, 6, 10, 13, 0, tzinfo=_TZ)
        event = basal_to_event(
            _basal(delivery_type="TempRate", time=ts, duration_mins=30.0, basal_rate=1.2)
        )
        assert event is not None
        assert event.kind == InsulinKind.TEMP_BASAL
        assert event.duration_min == 30.0
        assert event.units == pytest.approx(1.2 * 30.0 / 60.0)
        assert event.automatic is None
        assert event.ts == ts.astimezone(UTC)

    def test_algorithm_is_automatic_temp_basal(self) -> None:
        ts = datetime(2026, 6, 10, 13, 0, tzinfo=_TZ)
        event = basal_to_event(
            _basal(delivery_type="Algorithm", time=ts, duration_mins=5.0, basal_rate=0.6)
        )
        assert event is not None
        assert event.kind == InsulinKind.TEMP_BASAL
        assert event.automatic is True
        assert event.units == pytest.approx(0.6 * 5.0 / 60.0)

    def test_profile_is_scheduled_basal_no_units(self) -> None:
        ts = datetime(2026, 6, 10, 13, 0, tzinfo=_TZ)
        event = basal_to_event(_basal(delivery_type="Profile", time=ts, basal_rate=0.9))
        assert event is not None
        assert event.kind == InsulinKind.BASAL
        assert event.units is None
        assert event.duration_min is None

    def test_suspension_is_suspend(self) -> None:
        ts = datetime(2026, 6, 10, 13, 0, tzinfo=_TZ)
        event = basal_to_event(_basal(delivery_type="Suspension", time=ts, duration_mins=15.0))
        assert event is not None
        assert event.kind == InsulinKind.SUSPEND
        assert event.units is None
        assert event.duration_min == 15.0

    def test_temp_rate_without_rate_has_no_units(self) -> None:
        ts = datetime(2026, 6, 10, 13, 0, tzinfo=_TZ)
        event = basal_to_event(_basal(delivery_type="TempRate", time=ts, duration_mins=30.0))
        assert event is not None
        assert event.units is None
        assert event.duration_min == 30.0

    def test_unknown_delivery_type_is_none(self) -> None:
        ts = datetime(2026, 6, 10, 13, 0, tzinfo=_TZ)
        assert basal_to_event(_basal(delivery_type="Mystery", time=ts)) is None

    def test_no_timestamp_is_none(self) -> None:
        assert basal_to_event({"delivery_type": "TempRate"}) is None

    def test_naive_timestamp_rejected(self) -> None:
        with pytest.raises(ValueError, match="naive"):
            basal_to_event({"delivery_type": "TempRate", "time": "2026-06-10T13:00:00"})


# ─────────────────────────────────────────────────────────────────────────────
# Connector - stubbed t:connect client, no network
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _StubControlIQ:
    timeline: dict[str, Any]
    raise_on_call: Exception | None = None
    calls: list[tuple[str, str]] | None = None

    def therapy_timeline(self, start_date: str, end_date: str) -> dict[str, Any]:
        if self.calls is not None:
            self.calls.append((start_date, end_date))
        if self.raise_on_call is not None:
            raise self.raise_on_call
        return self.timeline


@dataclass
class _StubTConnectClient:
    """Stands in for tconnectsync's ``TConnectApi``."""

    controliq: _StubControlIQ


def _client(
    *,
    bolus: list[_StubBolus] | None = None,
    basal: list[dict[str, Any]] | None = None,
    raise_on_call: Exception | None = None,
) -> _StubTConnectClient:
    timeline: dict[str, Any] = {"bolus": bolus or [], "basal": basal or []}
    return _StubTConnectClient(
        controliq=_StubControlIQ(timeline=timeline, raise_on_call=raise_on_call, calls=[])
    )


def _connector(client: _StubTConnectClient) -> TandemConnector:
    return TandemConnector(CONFIG, client=client)


class TestTandemConnector:
    def test_satisfies_connector_protocol(self) -> None:
        connector = _connector(_client())
        assert isinstance(connector, Connector)
        assert connector.source == "tandem"

    def test_is_not_realtime(self) -> None:
        # t:connect is not live-fresh; batch-only by design.
        assert not isinstance(_connector(_client()), RealtimeConnector)

    # -- check -----------------------------------------------------------------

    def test_check_ok_reports_latest_event(self) -> None:
        ts = datetime.now(tz=UTC) - timedelta(hours=2)
        client = _client(bolus=[_bolus(insulin="3", completion=ts)])
        report = _connector(client).check()
        assert report.ok is True
        assert report.source == "tandem"
        assert "session" in report.detail
        assert report.latest_data_ts == ts

    def test_check_ok_without_data(self) -> None:
        report = _connector(_client()).check()
        assert report.ok is True
        assert report.latest_data_ts is None

    def test_check_auth_failure_is_not_ok(self) -> None:
        client = _client(raise_on_call=Exception("Invalid login credentials"))
        report = _connector(client).check()
        assert report.ok is False
        assert report.source == "tandem"
        assert "Invalid login credentials" in report.detail

    # -- pull ------------------------------------------------------------------

    def test_pull_returns_insulin_and_meals(self) -> None:
        now = datetime.now(tz=UTC)
        client = _client(
            bolus=[
                _bolus(insulin="5", carbs="40", completion=now - timedelta(hours=1)),
                _bolus(insulin="2", completion=now - timedelta(hours=2)),
            ],
            basal=[
                _basal(
                    delivery_type="TempRate",
                    time=now - timedelta(hours=3),
                    duration_mins=30.0,
                    basal_rate=1.0,
                ),
                _basal(delivery_type="Suspension", time=now - timedelta(hours=4),
                       duration_mins=10.0),
            ],
        )
        batch = _connector(client).pull(now - timedelta(days=1))
        kinds = sorted(e.kind for e in batch.insulin)
        assert kinds == sorted(
            [InsulinKind.BOLUS, InsulinKind.BOLUS, InsulinKind.TEMP_BASAL, InsulinKind.SUSPEND]
        )
        assert len(batch.meals) == 1
        assert batch.meals[0].carbs_g == 40.0
        assert all(r.source == "tandem" for r in batch.raw)
        # one raw per emitted event: 2 bolus units + 1 meal + 2 basal = 5
        assert len(batch.raw) == 5

    def test_pull_source_ids_stable_and_distinct(self) -> None:
        now = datetime.now(tz=UTC)
        client = _client(
            bolus=[_bolus(insulin="5", carbs="40", completion=now - timedelta(hours=1))],
            basal=[
                _basal(delivery_type="TempRate", time=now - timedelta(hours=2),
                       duration_mins=30.0, basal_rate=1.0)
            ],
        )
        batch = _connector(client).pull(now - timedelta(days=1))
        ids = [r.source_id for r in batch.raw]
        assert len(set(ids)) == len(ids)
        assert any(i.startswith("tandem:bolus:") for i in ids)
        assert any(i.startswith("tandem:carbs:") for i in ids)
        assert any(i.startswith("tandem:temp_basal:") for i in ids)

    def test_double_pull_is_idempotent(self) -> None:
        now = datetime.now(tz=UTC)
        client = _client(
            bolus=[_bolus(insulin="5", carbs="40", completion=now - timedelta(hours=1))],
            basal=[_basal(delivery_type="Profile", time=now - timedelta(hours=2),
                          basal_rate=0.8)],
        )
        connector = _connector(client)
        since = now - timedelta(days=1)
        first = {r.source_id for r in connector.pull(since).raw}
        second = {r.source_id for r in connector.pull(since).raw}
        assert first == second

    def test_pull_filters_before_window(self) -> None:
        now = datetime.now(tz=UTC)
        client = _client(
            bolus=[
                _bolus(insulin="3", completion=now - timedelta(minutes=10)),
                _bolus(insulin="9", completion=now - timedelta(days=5)),
            ]
        )
        batch = _connector(client).pull(now - timedelta(hours=1))
        assert len(batch.insulin) == 1
        assert batch.insulin[0].units == 3.0

    def test_pull_accepts_nested_basal_events(self) -> None:
        now = datetime.now(tz=UTC)
        nested = {
            "bolus": [],
            "basal": {
                "events": [
                    _basal(delivery_type="Profile", time=now - timedelta(hours=1), basal_rate=0.7)
                ]
            },
        }
        client = _StubTConnectClient(controliq=_StubControlIQ(timeline=nested, calls=[]))
        batch = _connector(client).pull(now - timedelta(days=1))
        assert len(batch.insulin) == 1
        assert batch.insulin[0].kind == InsulinKind.BASAL

    def test_pull_timestamps_are_utc(self) -> None:
        now = datetime.now(tz=UTC)
        client = _client(bolus=[_bolus(insulin="2", completion=now - timedelta(hours=1))])
        batch = _connector(client).pull(now - timedelta(days=1))
        assert all(e.ts.tzinfo == UTC for e in batch.insulin)
        assert all(r.source_ts.tzinfo == UTC for r in batch.raw)

    def test_pull_empty_timeline(self) -> None:
        batch = _connector(_client()).pull(datetime.now(tz=UTC) - timedelta(days=1))
        assert batch.insulin == []
        assert batch.meals == []
        assert batch.raw == []

    # -- lazy import -----------------------------------------------------------

    @pytest.mark.skipif(
        importlib.util.find_spec("tconnectsync") is not None,
        reason="tconnectsync is installed; the install-hint path is unreachable",
    )
    def test_missing_dependency_raises_install_hint(self) -> None:
        with pytest.raises(RuntimeError, match=r"dexta-intelligence\[tandem\]"):
            TandemConnector(CONFIG).check()
