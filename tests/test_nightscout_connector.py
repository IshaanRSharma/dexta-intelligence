"""Nightscout connector tests: pure parsing against fixtures, client via MockTransport.

The connector tests run against an ``httpx.MockTransport`` emulating the
Nightscout v1 query API (``find[...]`` filters + ``count``), exercising real
pagination behaviour.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from dexta_intelligence.config import NightscoutConfig
from dexta_intelligence.connectors.nightscout import (
    NightscoutConnector,
    parse_devicestatus,
    parse_entry,
    parse_treatment,
)
from dexta_intelligence.models import InsulinEvent, InsulinKind, MealEvent

FIXTURES = Path(__file__).parent / "fixtures"
TOKEN = "test-token-1234"


def _load(name: str) -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = json.loads((FIXTURES / name).read_text())
    return data


ENTRIES = _load("nightscout_entries.json")
TREATMENTS = _load("nightscout_treatments.json")
DEVICESTATUS = _load("nightscout_devicestatus.json")


# ─────────────────────────────────────────────────────────────────────────────
# Pure parsing - entries
# ─────────────────────────────────────────────────────────────────────────────


class TestParseEntry:
    def test_sgv_records_parse(self) -> None:
        events = [e for e in (parse_entry(doc) for doc in ENTRIES) if e is not None]
        assert len(events) == 5  # 6 docs, one is an mbg fingerstick

    def test_values_and_trend(self) -> None:
        newest = parse_entry(ENTRIES[0])
        assert newest is not None
        assert newest.mg_dl == 131
        assert newest.trend == "Flat"
        oldest = parse_entry(ENTRIES[-1])
        assert oldest is not None
        assert oldest.mg_dl == 148
        assert oldest.trend == "FortyFiveDown"

    def test_timestamps_are_utc(self) -> None:
        event = parse_entry(ENTRIES[0])
        assert event is not None
        assert event.ts == datetime(2026, 6, 10, 16, 10, tzinfo=UTC)
        assert event.ts.tzinfo == UTC

    def test_mbg_record_skipped(self) -> None:
        mbg = next(doc for doc in ENTRIES if doc.get("type") == "mbg")
        assert parse_entry(mbg) is None

    def test_missing_sgv_skipped(self) -> None:
        assert parse_entry({"type": "sgv", "date": 1781107200000}) is None

    def test_iso_fallback_when_no_epoch_date(self) -> None:
        event = parse_entry({"type": "sgv", "sgv": 120, "dateString": "2026-06-10T16:00:00.000Z"})
        assert event is not None
        assert event.ts == datetime(2026, 6, 10, 16, 0, tzinfo=UTC)


# ─────────────────────────────────────────────────────────────────────────────
# Pure parsing - treatments
# ─────────────────────────────────────────────────────────────────────────────


def _all_treatment_events() -> list[InsulinEvent | MealEvent]:
    events: list[InsulinEvent | MealEvent] = []
    for doc in TREATMENTS:
        events.extend(parse_treatment(doc))
    return events


class TestParseTreatment:
    def test_counts(self) -> None:
        events = _all_treatment_events()
        insulin = [e for e in events if isinstance(e, InsulinEvent)]
        meals = [e for e in events if isinstance(e, MealEvent)]
        assert len(insulin) == 5  # 3 boluses + 1 temp basal + 1 suspend
        assert len(meals) == 2

    def test_meal_bolus_yields_insulin_and_meal(self) -> None:
        doc = next(d for d in TREATMENTS if d["eventType"] == "Meal Bolus")
        events = parse_treatment(doc)
        assert len(events) == 2
        bolus = next(e for e in events if isinstance(e, InsulinEvent))
        meal = next(e for e in events if isinstance(e, MealEvent))
        assert bolus.kind == InsulinKind.BOLUS
        assert bolus.units == 4.5
        assert bolus.ts == datetime(2026, 6, 10, 12, 2, 11, tzinfo=UTC)
        assert meal.carbs_g == 45
        assert meal.protein_g == 22
        assert meal.fat_g == 14
        assert meal.note == "lunch - rice bowl"

    def test_smb_marked_automatic(self) -> None:
        doc = next(d for d in TREATMENTS if d.get("isSMB") is True)
        (bolus,) = parse_treatment(doc)
        assert isinstance(bolus, InsulinEvent)
        assert bolus.kind == InsulinKind.BOLUS
        assert bolus.units == 0.35
        assert bolus.automatic is True

    def test_manual_bolus_not_marked_automatic(self) -> None:
        doc = next(
            d
            for d in TREATMENTS
            if d["eventType"] == "Correction Bolus" and "isSMB" not in d
        )
        (bolus,) = parse_treatment(doc)
        assert isinstance(bolus, InsulinEvent)
        # enteredBy "loop://..." alone must NOT imply automatic (manual app boluses).
        assert bolus.automatic is None

    def test_temp_basal(self) -> None:
        doc = next(d for d in TREATMENTS if d["eventType"] == "Temp Basal")
        (event,) = parse_treatment(doc)
        assert isinstance(event, InsulinEvent)
        assert event.kind == InsulinKind.TEMP_BASAL
        assert event.duration_min == 30
        assert event.units == pytest.approx(0.925)  # 1.85 U/h x 0.5 h scheduled
        assert event.automatic is True  # enteredBy openaps://

    def test_suspend(self) -> None:
        doc = next(d for d in TREATMENTS if d["eventType"] == "Suspend Pump")
        (event,) = parse_treatment(doc)
        assert isinstance(event, InsulinEvent)
        assert event.kind == InsulinKind.SUSPEND
        assert event.duration_min == 45
        assert event.units is None

    def test_carb_correction(self) -> None:
        doc = next(d for d in TREATMENTS if d["eventType"] == "Carb Correction")
        (meal,) = parse_treatment(doc)
        assert isinstance(meal, MealEvent)
        assert meal.carbs_g == 15
        assert meal.note == "juice box for pre-walk"

    def test_bg_check_yields_nothing(self) -> None:
        doc = next(d for d in TREATMENTS if d["eventType"] == "BG Check")
        assert parse_treatment(doc) == []

    def test_timestamps_are_utc(self) -> None:
        for event in _all_treatment_events():
            assert event.ts.tzinfo == UTC


# ─────────────────────────────────────────────────────────────────────────────
# Pure parsing - devicestatus prediction curves
# ─────────────────────────────────────────────────────────────────────────────


class TestParseDevicestatus:
    def test_openaps_predbgs_all_four_curves(self) -> None:
        doc = DEVICESTATUS[0]
        events = parse_devicestatus(doc)
        assert len(events) == 4
        assert {e.curve_kind for e in events} == {"iob", "cob", "uam", "zt"}
        assert all(e.source == "openaps" for e in events)
        cycle_ts = datetime(2026, 6, 10, 16, 0, 5, tzinfo=UTC)
        assert all(e.ts == cycle_ts for e in events)

    def test_openaps_curve_values(self) -> None:
        events = {e.curve_kind: e for e in parse_devicestatus(DEVICESTATUS[0])}
        assert events["iob"].values_mg_dl == [139.0, 136.0, 132.0, 128.0, 124.0, 121.0, 118.0]
        assert events["uam"].values_mg_dl == [139.0, 142.0, 147.0, 151.0, 153.0, 152.0, 149.0]
        assert events["cob"].values_mg_dl[0] == 139.0
        assert events["zt"].values_mg_dl[-1] == 118.0

    def test_loop_predicted(self) -> None:
        events = parse_devicestatus(DEVICESTATUS[1])
        assert len(events) == 1
        (event,) = events
        assert event.source == "loop"
        assert event.curve_kind == "loop"
        assert event.ts == datetime(2026, 6, 10, 16, 0, tzinfo=UTC)
        assert event.values_mg_dl[:3] == [139.0, 137.5, 136.2]
        assert len(event.values_mg_dl) == 10
        assert all(isinstance(v, float) for v in event.values_mg_dl)

    def test_openaps_doc_without_predbgs_skipped(self) -> None:
        assert parse_devicestatus(DEVICESTATUS[2]) == []

    def test_plain_uploader_doc_skipped(self) -> None:
        assert parse_devicestatus(DEVICESTATUS[3]) == []

    def test_empty_doc_skipped(self) -> None:
        assert parse_devicestatus({}) == []


# ─────────────────────────────────────────────────────────────────────────────
# Connector - mocked transport emulating the Nightscout v1 query API
# ─────────────────────────────────────────────────────────────────────────────


def _filter_by_date(docs: list[dict[str, Any]], params: httpx.QueryParams) -> list[dict[str, Any]]:
    out = sorted(docs, key=lambda d: d["date"], reverse=True)
    if gt := params.get("find[date][$gt]"):
        out = [d for d in out if d["date"] > int(gt)]
    if lt := params.get("find[date][$lt]"):
        out = [d for d in out if d["date"] < int(lt)]
    return out[: int(params.get("count", "10"))]


def _filter_by_created_at(
    docs: list[dict[str, Any]], params: httpx.QueryParams
) -> list[dict[str, Any]]:
    out = sorted(docs, key=lambda d: d["created_at"], reverse=True)
    if gt := params.get("find[created_at][$gt]"):
        out = [d for d in out if d["created_at"] > gt]
    if lt := params.get("find[created_at][$lt]"):
        out = [d for d in out if d["created_at"] < lt]
    return out[: int(params.get("count", "10"))]


def _handler(request: httpx.Request) -> httpx.Response:
    if request.url.params.get("token") != TOKEN:
        return httpx.Response(401, json={"status": 401, "message": "Unauthorized"})
    path = request.url.path
    params = request.url.params
    if path == "/api/v1/status.json":
        return httpx.Response(200, json={"status": "ok", "version": "15.0.3"})
    if path == "/api/v1/entries/sgv.json":
        sgv_only = [d for d in ENTRIES if d.get("type") == "sgv"]
        return httpx.Response(200, json=_filter_by_date(sgv_only, params))
    if path == "/api/v1/treatments.json":
        return httpx.Response(200, json=_filter_by_created_at(TREATMENTS, params))
    if path == "/api/v1/devicestatus.json":
        return httpx.Response(200, json=_filter_by_created_at(DEVICESTATUS, params))
    return httpx.Response(404)


def _connector(page_size: int = 1000, token: str = TOKEN) -> NightscoutConnector:
    config = NightscoutConfig(url="https://ns.example.com/", token=token)
    client = httpx.Client(transport=httpx.MockTransport(_handler))
    return NightscoutConnector(config, client=client, page_size=page_size)


class TestNightscoutConnector:
    def test_check_ok(self) -> None:
        report = _connector().check()
        assert report.ok is True
        assert report.source == "nightscout"
        assert "15.0.3" in report.detail
        assert report.latest_data_ts == datetime(2026, 6, 10, 16, 10, tzinfo=UTC)

    def test_check_bad_token(self) -> None:
        report = _connector(token="wrong").check()
        assert report.ok is False
        assert report.source == "nightscout"

    def test_failed_check_does_not_leak_token(self) -> None:
        # The token rides in the URL query; a failed request must not surface it
        # in the health-check detail (which is shown in the GUI and logs).
        report = _connector(token="super-secret-xyz").check()
        assert report.ok is False
        assert "super-secret-xyz" not in (report.detail or "")
        assert "token=[redacted]" in (report.detail or "")

    def test_pull_full_window(self) -> None:
        since = datetime(2026, 6, 10, 11, 0, tzinfo=UTC)
        batch = _connector().pull(since)
        assert len(batch.glucose) == 5
        assert len(batch.insulin) == 5
        assert len(batch.meals) == 2
        assert len(batch.predictions) == 5  # 4 openaps curves + 1 loop curve
        # one RawEvent per fetched document: 5 sgv + 7 treatments + 4 devicestatus
        assert len(batch.raw) == 16
        assert all(r.source == "nightscout" for r in batch.raw)
        assert all(r.source_id for r in batch.raw)

    def test_pull_applies_watermark(self) -> None:
        since = datetime(2026, 6, 10, 16, 3, tzinfo=UTC)
        batch = _connector().pull(since)
        # dedupe margin pulls the window back to 15:58
        assert len(batch.glucose) == 3  # 16:00, 16:05, 16:10
        assert len(batch.insulin) == 0
        assert len(batch.meals) == 0
        assert len(batch.predictions) == 5  # both prediction docs at ~16:00
        assert min(e.ts for e in batch.glucose) == datetime(
            2026, 6, 10, 16, 0, tzinfo=UTC
        )

    def test_pull_paginates(self) -> None:
        since = datetime(2026, 6, 10, 11, 0, tzinfo=UTC)
        small_pages = _connector(page_size=2).pull(since)
        one_page = _connector().pull(since)
        assert {e.ts for e in small_pages.glucose} == {e.ts for e in one_page.glucose}
        assert len(small_pages.insulin) == len(one_page.insulin)
        assert len(small_pages.meals) == len(one_page.meals)
        assert len(small_pages.predictions) == len(one_page.predictions)
        assert len(small_pages.raw) == len(one_page.raw)

    def test_pull_timestamps_are_utc(self) -> None:
        since = datetime(2026, 6, 10, 11, 0, tzinfo=UTC)
        batch = _connector().pull(since)
        for event in [*batch.glucose, *batch.insulin, *batch.meals, *batch.predictions]:
            assert event.ts.tzinfo == UTC
        for raw in batch.raw:
            assert raw.source_ts.tzinfo == UTC


# ─────────────────────────────────────────────────────────────────────────────
# Connector - mocked transport emulating the secured Nightscout v3 API
# ─────────────────────────────────────────────────────────────────────────────

V3_JWT = "v3-jwt-header-token"


def _to_v3(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """v1 fixture docs as v3 documents: ``_id`` becomes ``identifier`` + srv meta."""
    out: list[dict[str, Any]] = []
    for doc in docs:
        v3 = dict(doc)
        v3["identifier"] = v3.pop("_id")
        v3["srvModified"] = v3.get("date", 0)
        v3["srvCreated"] = v3.get("date", 0)
        out.append(v3)
    return out


ENTRIES_V3 = _to_v3(ENTRIES)
TREATMENTS_V3 = _to_v3(TREATMENTS)
DEVICESTATUS_V3 = _to_v3(DEVICESTATUS)


def _v3_filter(
    field: str, docs: list[dict[str, Any]], params: httpx.QueryParams, *, numeric: bool
) -> list[dict[str, Any]]:
    out = sorted(docs, key=lambda d: d[field], reverse=True)
    if (gt := params.get(f"{field}$gt")) is not None:
        bound = int(gt) if numeric else gt
        out = [d for d in out if d[field] > bound]
    if (lt := params.get(f"{field}$lt")) is not None:
        bound = int(lt) if numeric else lt
        out = [d for d in out if d[field] < bound]
    if (type_eq := params.get("type$eq")) is not None:
        out = [d for d in out if d.get("type") == type_eq]
    return out[: int(params.get("limit", "10"))]


def _v3_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    params = request.url.params
    if path == "/api/v3/version":
        return httpx.Response(200, json={"version": "15.0.3", "apiVersion": "3.0.3"})
    if path.startswith("/api/v2/authorization/request/"):
        if path == f"/api/v2/authorization/request/{TOKEN}":
            return httpx.Response(200, json={"token": V3_JWT})
        return httpx.Response(401, json={"status": 401, "message": "Unauthorized"})
    if request.headers.get("Authorization") != f"Bearer {V3_JWT}":
        return httpx.Response(401, json={"status": 401, "message": "Unauthorized"})
    if path == "/api/v3/entries":
        result = _v3_filter("date", ENTRIES_V3, params, numeric=True)
        return httpx.Response(200, json={"status": 200, "result": result})
    if path == "/api/v3/treatments":
        result = _v3_filter("created_at", TREATMENTS_V3, params, numeric=False)
        return httpx.Response(200, json={"status": 200, "result": result})
    if path == "/api/v3/devicestatus":
        result = _v3_filter("created_at", DEVICESTATUS_V3, params, numeric=False)
        return httpx.Response(200, json={"status": 200, "result": result})
    return httpx.Response(404)


def _v1_only_handler(request: httpx.Request) -> httpx.Response:
    """v3 endpoints 404 (pre-v3 server); v1 query API answers normally."""
    path = request.url.path
    if path == "/api/v3/version" or path.startswith("/api/v2/authorization/request/"):
        return httpx.Response(404)
    return _handler(request)


def _v3_connector(page_size: int = 1000, token: str = TOKEN) -> NightscoutConnector:
    config = NightscoutConfig(url="https://ns.example.com/", token=token)
    client = httpx.Client(transport=httpx.MockTransport(_v3_handler))
    return NightscoutConnector(config, client=client, page_size=page_size)


def _fallback_connector(page_size: int = 1000) -> NightscoutConnector:
    config = NightscoutConfig(url="https://ns.example.com/", token=TOKEN)
    client = httpx.Client(transport=httpx.MockTransport(_v1_only_handler))
    return NightscoutConnector(config, client=client, page_size=page_size)


def _signatures(events: list[Any]) -> list[str]:
    return sorted(str(event) for event in events)


class TestNightscoutV3:
    def test_check_detects_v3(self) -> None:
        connector = _v3_connector()
        report = connector.check()
        assert report.ok is True
        assert connector._api_version == "v3"
        assert "3.0.3" in report.detail
        assert report.latest_data_ts == datetime(2026, 6, 10, 16, 10, tzinfo=UTC)

    def test_pull_v3_happy_path(self) -> None:
        since = datetime(2026, 6, 10, 11, 0, tzinfo=UTC)
        batch = _v3_connector().pull(since)
        assert len(batch.glucose) == 5
        assert len(batch.insulin) == 5
        assert len(batch.meals) == 2
        assert len(batch.predictions) == 5
        assert len(batch.raw) == 16
        assert all(r.source == "nightscout" for r in batch.raw)
        assert all(r.source_id for r in batch.raw)

    def test_v3_normalizes_to_the_same_events_as_v1(self) -> None:
        since = datetime(2026, 6, 10, 11, 0, tzinfo=UTC)
        v3_batch = _v3_connector().pull(since)
        v1_batch = _connector().pull(since)
        assert _signatures(v3_batch.glucose) == _signatures(v1_batch.glucose)
        assert _signatures(v3_batch.insulin) == _signatures(v1_batch.insulin)
        assert _signatures(v3_batch.meals) == _signatures(v1_batch.meals)
        assert _signatures(v3_batch.predictions) == _signatures(v1_batch.predictions)

    def test_v3_source_id_backfilled_from_identifier(self) -> None:
        since = datetime(2026, 6, 10, 11, 0, tzinfo=UTC)
        batch = _v3_connector().pull(since)
        identifiers = {d["identifier"] for d in ENTRIES_V3 + TREATMENTS_V3 + DEVICESTATUS_V3}
        assert all(r.source_id in identifiers for r in batch.raw)
        assert not any(r.source_id.startswith("synthetic:") for r in batch.raw)

    def test_v3_paginates(self) -> None:
        since = datetime(2026, 6, 10, 11, 0, tzinfo=UTC)
        small_pages = _v3_connector(page_size=2).pull(since)
        one_page = _v3_connector().pull(since)
        assert {e.ts for e in small_pages.glucose} == {e.ts for e in one_page.glucose}
        assert len(small_pages.raw) == len(one_page.raw)
        assert len(small_pages.insulin) == len(one_page.insulin)
        assert len(small_pages.predictions) == len(one_page.predictions)

    def test_v3_never_leaks_token_in_authorization_path(self) -> None:
        # The access token rides in the JWT-request path; a failed mint must not
        # surface it in the health-check detail.
        report = _v3_connector(token="super-secret-xyz").check()
        assert "super-secret-xyz" not in (report.detail or "")


class TestNightscoutV1Fallback:
    def test_pull_falls_back_to_v1_when_v3_unavailable(self) -> None:
        since = datetime(2026, 6, 10, 11, 0, tzinfo=UTC)
        connector = _fallback_connector()
        batch = connector.pull(since)
        assert connector._api_version == "v1"
        assert len(batch.glucose) == 5
        assert len(batch.insulin) == 5
        assert len(batch.meals) == 2
        assert len(batch.raw) == 16

    def test_check_falls_back_to_v1_when_v3_unavailable(self) -> None:
        connector = _fallback_connector()
        report = connector.check()
        assert report.ok is True
        assert connector._api_version == "v1"
        assert "15.0.3" in report.detail

    def test_dialect_is_latched_once(self) -> None:
        since = datetime(2026, 6, 10, 11, 0, tzinfo=UTC)
        connector = _v3_connector()
        connector.check()
        assert connector._api_version == "v3"
        connector.pull(since)  # must not re-detect or downgrade
        assert connector._api_version == "v3"
