"""Dexcom official API connector tests - pure conversion + mocked transport.

No live network: the connector tests run against an ``httpx.MockTransport``
emulating the Dexcom developer API v3 (bearer auth, ``startDate``/``endDate``
egvs filtering, and the OAuth refresh-token grant), exercising real auth,
date-window filtering, and token-refresh behaviour without a real account.
"""

from __future__ import annotations

import builtins
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import parse_qs

import httpx
import pytest

from dexta_intelligence.config import DexcomApiConfig
from dexta_intelligence.connectors.base import Connector
from dexta_intelligence.connectors.dexcom_api import (
    PROD_BASE,
    SANDBOX_BASE,
    DexcomApiConnector,
    egv_to_event,
)

GOOD_TOKEN = "good-access-token"
EXPIRED_TOKEN = "expired-access-token"
REFRESHED_TOKEN = "refreshed-access-token"
REFRESH_TOKEN = "valid-refresh-token"
CLIENT_ID = "test-client-id"
CLIENT_SECRET = "test-client-secret"


def _egv(
    *,
    system_time: str = "2026-06-10T16:10:00Z",
    display_time: str = "2026-06-10T12:10:00",
    value: int = 131,
    trend: str = "flat",
) -> dict[str, Any]:
    return {
        "recordType": "EGV",
        "systemTime": system_time,
        "displayTime": display_time,
        "value": value,
        "trend": trend,
        "unit": "mg/dL",
    }


# A small recorded /egvs window: six readings at 5-minute spacing, newest first.
RECORDED_EGVS = [
    _egv(
        system_time=f"2026-06-10T16:{10 - 5 * i:02d}:00Z",
        display_time=f"2026-06-10T12:{10 - 5 * i:02d}:00",
        value=120 + i,
        trend="flat",
    )
    for i in range(3)
] + [
    _egv(
        system_time=f"2026-06-10T15:{55 - 5 * i:02d}:00Z",
        display_time=f"2026-06-10T11:{55 - 5 * i:02d}:00",
        value=130 + i,
        trend="fortyFiveUp",
    )
    for i in range(3)
]


# ─────────────────────────────────────────────────────────────────────────────
# Pure conversion
# ─────────────────────────────────────────────────────────────────────────────


class TestEgvToEvent:
    def test_value_and_systemtime(self) -> None:
        event = egv_to_event(_egv(value=142))
        assert event is not None
        assert event.mg_dl == 142
        assert event.ts == datetime(2026, 6, 10, 16, 10, tzinfo=UTC)
        assert event.ts.tzinfo == UTC

    def test_naive_systemtime_rejected(self) -> None:
        with pytest.raises(ValueError, match="naive"):
            egv_to_event(_egv(system_time="2026-06-10T16:10:00"))

    def test_offset_systemtime_converted_to_utc(self) -> None:
        event = egv_to_event(_egv(system_time="2026-06-10T12:10:00-04:00"))
        assert event is not None
        assert event.ts == datetime(2026, 6, 10, 16, 10, tzinfo=UTC)
        assert event.ts.tzinfo == UTC

    def test_z_suffix_systemtime(self) -> None:
        event = egv_to_event(_egv(system_time="2026-06-10T16:10:00Z"))
        assert event is not None
        assert event.ts == datetime(2026, 6, 10, 16, 10, tzinfo=UTC)

    def test_missing_systemtime_is_none(self) -> None:
        rec = _egv()
        del rec["systemTime"]
        assert egv_to_event(rec) is None

    def test_missing_value_is_none(self) -> None:
        rec = _egv()
        del rec["value"]
        assert egv_to_event(rec) is None

    @pytest.mark.parametrize(
        ("camel", "expected"),
        [
            ("doubleUp", "DoubleUp"),
            ("singleUp", "SingleUp"),
            ("fortyFiveUp", "FortyFiveUp"),
            ("flat", "Flat"),
            ("fortyFiveDown", "FortyFiveDown"),
            ("singleDown", "SingleDown"),
            ("doubleDown", "DoubleDown"),
        ],
    )
    def test_camelcase_trends_normalized(self, camel: str, expected: str) -> None:
        event = egv_to_event(_egv(trend=camel))
        assert event is not None
        assert event.trend == expected

    @pytest.mark.parametrize(
        "trend", ["none", "notComputable", "rateOutOfRange", "bogus", ""]
    )
    def test_noninformative_trends_become_none(self, trend: str) -> None:
        event = egv_to_event(_egv(trend=trend))
        assert event is not None
        assert event.trend is None

    def test_missing_trend_is_none(self) -> None:
        rec = _egv()
        del rec["trend"]
        event = egv_to_event(rec)
        assert event is not None
        assert event.trend is None


# ─────────────────────────────────────────────────────────────────────────────
# Connector - mocked transport emulating the Dexcom official API
# ─────────────────────────────────────────────────────────────────────────────


class _DexcomServer:
    """Stateful mock: bearer auth, egvs date filtering, token refresh."""

    def __init__(self, base: str = PROD_BASE) -> None:
        self.base = base
        self.valid_token = GOOD_TOKEN
        self.refresh_calls = 0
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if str(request.url).split("?")[0] != self.base + request.url.path:
            return httpx.Response(404, json={"error": "wrong host"})
        if request.url.path == "/v2/oauth2/token":
            return self._token_grant(request)
        if request.headers.get("Authorization") != f"Bearer {self.valid_token}":
            return httpx.Response(401, json={"error": "invalid_token"})
        if request.url.path == "/v3/users/self/egvs":
            return self._egvs(request.url.params)
        return httpx.Response(404)

    def _token_grant(self, request: httpx.Request) -> httpx.Response:
        form = parse_qs(request.content.decode())
        if (
            form.get("grant_type") == ["refresh_token"]
            and form.get("refresh_token") == [REFRESH_TOKEN]
            and form.get("client_id") == [CLIENT_ID]
            and form.get("client_secret") == [CLIENT_SECRET]
        ):
            self.refresh_calls += 1
            self.valid_token = REFRESHED_TOKEN
            return httpx.Response(
                200,
                json={
                    "access_token": REFRESHED_TOKEN,
                    "refresh_token": "rotated-refresh-token",
                    "expires_in": 7200,
                    "token_type": "Bearer",
                },
            )
        return httpx.Response(400, json={"error": "invalid_grant"})

    def _egvs(self, params: httpx.QueryParams) -> httpx.Response:
        start = params.get("startDate")
        end = params.get("endDate")
        records = RECORDED_EGVS
        if start is not None:
            records = [r for r in records if r["systemTime"] >= start]
        if end is not None:
            records = [r for r in records if r["systemTime"] <= end]
        return httpx.Response(200, json={"unit": "mg/dL", "records": records})


def _connector(
    server: _DexcomServer,
    *,
    access_token: str = GOOD_TOKEN,
    refresh_token: str = "",
    client_id: str = "",
    client_secret: str = "",
    sandbox: bool = False,
) -> DexcomApiConnector:
    config = DexcomApiConfig(
        access_token=access_token,
        refresh_token=refresh_token,
        client_id=client_id,
        client_secret=client_secret,
        sandbox=sandbox,
    )
    client = httpx.Client(transport=httpx.MockTransport(server.handler))
    return DexcomApiConnector(config, client=client)


# To pull the recorded June-10 window we freeze the connector's feed edge by
# choosing a ``since`` in that window; the mock ignores "now" and only filters
# by the requested dates, so the readings come back regardless of wall clock.
PULL_SINCE = datetime(2026, 6, 10, 15, 0, tzinfo=UTC)


class TestDexcomApiConnector:
    def test_satisfies_connector_protocol(self) -> None:
        connector = _connector(_DexcomServer())
        assert isinstance(connector, Connector)
        assert connector.source == "dexcom_api"

    # -- base URL selection --------------------------------------------------

    def test_production_base_default(self) -> None:
        connector = _connector(_DexcomServer())
        assert connector._base == PROD_BASE
    def test_sandbox_base_selected(self) -> None:
        server = _DexcomServer(base=SANDBOX_BASE)
        connector = _connector(server, sandbox=True)
        assert connector._base == SANDBOX_BASE        # and requests actually go to the sandbox host
        report = connector.check()
        assert report.ok is True
        assert all(str(r.url).startswith(SANDBOX_BASE) for r in server.requests)

    # -- check ---------------------------------------------------------------

    def test_check_ok(self) -> None:
        report = _connector(_DexcomServer()).check()
        assert report.ok is True
        assert report.source == "dexcom_api"

    def test_check_401_reports_auth_failed(self) -> None:
        report = _connector(_DexcomServer(), access_token=EXPIRED_TOKEN).check()
        assert report.ok is False
        assert report.source == "dexcom_api"
        assert report.detail == "auth failed"

    def test_check_refreshes_expired_token(self) -> None:
        server = _DexcomServer()
        connector = _connector(
            server,
            access_token=EXPIRED_TOKEN,
            refresh_token=REFRESH_TOKEN,
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
        )
        report = connector.check()
        assert report.ok is True
        assert server.refresh_calls == 1

    # -- pull ----------------------------------------------------------------

    def test_pull_returns_raws_and_glucose(self) -> None:
        batch = _connector(_DexcomServer()).pull(PULL_SINCE)
        assert len(batch.glucose) == 6
        assert len(batch.raw) == 6
        assert all(r.source == "dexcom_api" for r in batch.raw)
        assert all(r.source_id.startswith("dexcom_api:") for r in batch.raw)
        assert all(isinstance(r.payload, dict) for r in batch.raw)
        # the official feed serves glucose only
        assert batch.insulin == [] and batch.meals == [] and batch.predictions == []

    def test_pull_source_ids_stable_and_unique(self) -> None:
        batch = _connector(_DexcomServer()).pull(PULL_SINCE)
        ids = [r.source_id for r in batch.raw]
        assert len(set(ids)) == 6
        assert "dexcom_api:2026-06-10T16:10:00+00:00" in ids

    def test_pull_idempotent_across_runs(self) -> None:
        first = _connector(_DexcomServer()).pull(PULL_SINCE)
        second = _connector(_DexcomServer()).pull(PULL_SINCE)
        assert {r.source_id for r in first.raw} == {r.source_id for r in second.raw}

    def test_pull_applies_watermark(self) -> None:
        # since at 16:00 minus the 5-min dedupe margin keeps readings >= 15:55,
        # so the three 15:xx readings before 15:55 (15:50, 15:45) fall out.
        since = datetime(2026, 6, 10, 16, 0, tzinfo=UTC)
        batch = _connector(_DexcomServer()).pull(since)
        assert len(batch.glucose) == 4
        assert min(e.ts for e in batch.glucose) == since - timedelta(minutes=5)

    def test_pull_trend_normalized(self) -> None:
        batch = _connector(_DexcomServer()).pull(PULL_SINCE)
        trends = {e.trend for e in batch.glucose}
        assert trends == {"Flat", "FortyFiveUp"}

    def test_pull_timestamps_are_utc(self) -> None:
        batch = _connector(_DexcomServer()).pull(PULL_SINCE)
        for event in batch.glucose:
            assert event.ts.tzinfo == UTC
        for raw in batch.raw:
            assert raw.source_ts.tzinfo == UTC

    def test_pull_auth_failure_raises(self) -> None:
        connector = _connector(_DexcomServer(), access_token=EXPIRED_TOKEN)
        with pytest.raises(httpx.HTTPStatusError):
            connector.pull(PULL_SINCE)


# ─────────────────────────────────────────────────────────────────────────────
# Missing optional dependency
# ─────────────────────────────────────────────────────────────────────────────


def test_missing_httpx_dependency_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without an injected client, a missing httpx must surface the extra hint."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "httpx":
            raise ImportError("No module named 'httpx'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match=r"dexta-intelligence\[dexcom-api\]"):
        DexcomApiConnector(DexcomApiConfig())
