"""Whoop connector tests - pure parsing against fixtures, client via MockTransport.

No live network calls: the connector tests run against an ``httpx.MockTransport``
that emulates the Whoop v2 developer API (bearer auth, ``start`` filtering,
``nextToken`` pagination, and the OAuth refresh-token grant), which lets us
exercise real pagination and token-refresh behaviour.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import httpx
import pytest

from dexta_intelligence.config import WhoopConfig
from dexta_intelligence.connectors.whoop import (
    WhoopConnector,
    parse_recovery,
    parse_sleep,
    parse_workout,
)

FIXTURES = Path(__file__).parent / "fixtures"

GOOD_TOKEN = "good-access-token"
EXPIRED_TOKEN = "expired-access-token"
REFRESHED_TOKEN = "refreshed-access-token"
REFRESH_TOKEN = "valid-refresh-token"
CLIENT_ID = "test-client-id"
CLIENT_SECRET = "test-client-secret"


def _load(name: str) -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = json.loads((FIXTURES / name).read_text())
    return data


SLEEPS = _load("whoop_sleep.json")
RECOVERIES = _load("whoop_recovery.json")
WORKOUTS = _load("whoop_workout.json")

# Cycles are ingested raw-only (no timeline model), so two inline records
# suffice - no fixture file needed.
CYCLES: list[dict[str, Any]] = [
    {
        "id": 93791554,
        "user_id": 451823,
        "created_at": "2026-06-09T12:30:05.000Z",
        "updated_at": "2026-06-10T12:30:05.000Z",
        "start": "2026-06-09T12:30:00.000Z",
        "end": "2026-06-10T12:30:00.000Z",
        "timezone_offset": "-04:00",
        "score_state": "SCORED",
        "score": {
            "strain": 14.2,
            "kilojoule": 9120.4,
            "average_heart_rate": 71,
            "max_heart_rate": 181,
        },
    },
    {
        "id": 93845162,
        "user_id": 451823,
        "created_at": "2026-06-10T12:30:05.000Z",
        "updated_at": "2026-06-10T23:10:00.000Z",
        "start": "2026-06-10T12:30:00.000Z",
        "end": None,
        "timezone_offset": "-04:00",
        "score_state": "SCORED",
        "score": {
            "strain": 11.6,
            "kilojoule": 7204.1,
            "average_heart_rate": 68,
            "max_heart_rate": 174,
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Pure parsing — sleep
# ─────────────────────────────────────────────────────────────────────────────


class TestParseSleep:
    def test_scored_records_parse(self) -> None:
        events = [e for e in (parse_sleep(doc) for doc in SLEEPS) if e is not None]
        assert len(events) == 2  # 3 docs, one PENDING_SCORE

    def test_main_sleep_values(self) -> None:
        event = parse_sleep(SLEEPS[0])
        assert event is not None
        assert event.ts_start == datetime(2026, 6, 10, 5, 0, tzinfo=UTC)
        assert event.ts_end == datetime(2026, 6, 10, 12, 30, tzinfo=UTC)
        assert event.duration_min == 450.0
        assert event.score == 91.0
        assert event.stages == {"light": 225.0, "sws": 90.0, "rem": 95.0, "awake": 40.0}

    def test_nap_parses(self) -> None:
        event = parse_sleep(SLEEPS[1])
        assert event is not None
        assert event.duration_min == 45.0
        assert event.score == 76.0
        assert event.stages is not None
        assert event.stages["light"] == 30.0

    def test_timestamps_are_utc(self) -> None:
        event = parse_sleep(SLEEPS[0])
        assert event is not None
        assert event.ts_start.tzinfo == UTC
        assert event.ts_end.tzinfo == UTC

    def test_pending_score_skipped(self) -> None:
        pending = next(doc for doc in SLEEPS if doc["score_state"] == "PENDING_SCORE")
        assert parse_sleep(pending) is None

    def test_missing_end_skipped(self) -> None:
        doc = {**SLEEPS[0], "end": None}
        assert parse_sleep(doc) is None


# ─────────────────────────────────────────────────────────────────────────────
# Pure parsing — recovery
# ─────────────────────────────────────────────────────────────────────────────


class TestParseRecovery:
    def test_scored_records_parse(self) -> None:
        events = [e for e in (parse_recovery(doc) for doc in RECOVERIES) if e is not None]
        assert len(events) == 2  # 3 docs, one PENDING_SCORE

    def test_values(self) -> None:
        event = parse_recovery(RECOVERIES[0])
        assert event is not None
        assert event.ts == datetime(2026, 6, 10, 12, 35, tzinfo=UTC)
        assert event.score == 67.0
        assert event.hrv_ms == 78.4
        assert event.rhr_bpm == 52.0

    def test_timestamps_are_utc(self) -> None:
        for doc in RECOVERIES:
            event = parse_recovery(doc)
            if event is not None:
                assert event.ts.tzinfo == UTC

    def test_pending_score_skipped(self) -> None:
        pending = next(doc for doc in RECOVERIES if doc["score_state"] == "PENDING_SCORE")
        assert parse_recovery(pending) is None

    def test_missing_score_dict_skipped(self) -> None:
        assert parse_recovery({"cycle_id": 1, "created_at": "2026-06-10T12:00:00.000Z"}) is None


# ─────────────────────────────────────────────────────────────────────────────
# Pure parsing — workouts
# ─────────────────────────────────────────────────────────────────────────────


class TestParseWorkout:
    def test_scored_records_parse(self) -> None:
        events = [e for e in (parse_workout(doc) for doc in WORKOUTS) if e is not None]
        assert len(events) == 2  # 3 docs, one UNSCORABLE

    def test_run_values(self) -> None:
        event = parse_workout(WORKOUTS[0])
        assert event is not None
        assert event.ts == datetime(2026, 6, 10, 15, 0, tzinfo=UTC)
        assert event.ts.tzinfo == UTC
        assert event.kind == "running"
        assert event.duration_min == 45.0
        assert event.strain == 10.3
        assert event.intensity == 152.0  # average_heart_rate, bpm

    def test_weightlifting_values(self) -> None:
        event = parse_workout(WORKOUTS[1])
        assert event is not None
        assert event.kind == "weightlifting"
        assert event.duration_min == 60.0
        assert event.strain == 6.8

    def test_unscorable_skipped(self) -> None:
        unscorable = next(doc for doc in WORKOUTS if doc["score_state"] == "UNSCORABLE")
        assert parse_workout(unscorable) is None

    def test_sport_name_fallbacks(self) -> None:
        by_id = parse_workout({**WORKOUTS[0], "sport_name": None, "sport_id": 17})
        assert by_id is not None
        assert by_id.kind == "sport_17"
        generic = parse_workout({**WORKOUTS[0], "sport_name": None, "sport_id": None})
        assert generic is not None
        assert generic.kind == "workout"


# ─────────────────────────────────────────────────────────────────────────────
# Connector — mocked transport emulating the Whoop v2 developer API
# ─────────────────────────────────────────────────────────────────────────────


class _WhoopServer:
    """Stateful mock of the Whoop API: bearer auth, pagination, token refresh."""

    def __init__(self) -> None:
        self.valid_token = GOOD_TOKEN
        self.refresh_calls = 0
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/oauth/oauth2/token":
            return self._token_grant(request)
        if request.headers.get("Authorization") != f"Bearer {self.valid_token}":
            return httpx.Response(401, json={"error": "invalid_token"})

        path = request.url.path
        params = request.url.params
        if path == "/developer/v2/user/profile/basic":
            return httpx.Response(
                200,
                json={
                    "user_id": 451823,
                    "email": "athlete@example.com",
                    "first_name": "Test",
                    "last_name": "Athlete",
                },
            )
        if path == "/developer/v2/activity/sleep":
            return self._collection(SLEEPS, "start", params)
        if path == "/developer/v2/recovery":
            return self._collection(RECOVERIES, "created_at", params)
        if path == "/developer/v2/activity/workout":
            return self._collection(WORKOUTS, "start", params)
        if path == "/developer/v2/cycle":
            return self._collection(CYCLES, "start", params)
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
                    "expires_in": 3600,
                    "token_type": "bearer",
                },
            )
        return httpx.Response(400, json={"error": "invalid_grant"})

    def _collection(
        self, docs: list[dict[str, Any]], ts_key: str, params: httpx.QueryParams
    ) -> httpx.Response:
        # Whoop returns newest-first; both sides use Z-suffixed millisecond
        # ISO strings, so lexicographic comparison matches chronology.
        out = sorted(docs, key=lambda d: str(d[ts_key]), reverse=True)
        if start := params.get("start"):
            out = [d for d in out if str(d[ts_key]) >= start]
        limit = int(params.get("limit", "10"))
        offset = int(params.get("nextToken", "0"))
        page = out[offset : offset + limit]
        next_token = str(offset + limit) if offset + limit < len(out) else None
        return httpx.Response(200, json={"records": page, "next_token": next_token})


def _connector(
    server: _WhoopServer,
    *,
    access_token: str = GOOD_TOKEN,
    refresh_token: str = "",
    client_id: str = "",
    client_secret: str = "",
    page_size: int = 25,
) -> WhoopConnector:
    config = WhoopConfig(
        access_token=access_token,
        refresh_token=refresh_token,
        client_id=client_id,
        client_secret=client_secret,
    )
    client = httpx.Client(transport=httpx.MockTransport(server.handler))
    return WhoopConnector(config, client=client, page_size=page_size)


class TestWhoopConnector:
    def test_check_ok(self) -> None:
        report = _connector(_WhoopServer()).check()
        assert report.ok is True
        assert report.source == "whoop"
        assert "451823" in report.detail
        assert report.latest_data_ts == datetime(2026, 6, 10, 12, 30, tzinfo=UTC)

    def test_check_bad_token_without_refresh_credentials(self) -> None:
        report = _connector(_WhoopServer(), access_token=EXPIRED_TOKEN).check()
        assert report.ok is False
        assert report.source == "whoop"
        assert "401" in report.detail

    def test_check_refreshes_expired_token(self) -> None:
        server = _WhoopServer()
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

    def test_pull_auth_failure_raises(self) -> None:
        connector = _connector(_WhoopServer(), access_token=EXPIRED_TOKEN)
        since = datetime(2026, 6, 8, 0, 0, tzinfo=UTC)
        with pytest.raises(httpx.HTTPStatusError):
            connector.pull(since)

    def test_pull_full_window(self) -> None:
        since = datetime(2026, 6, 8, 0, 0, tzinfo=UTC)
        batch = _connector(_WhoopServer()).pull(since)
        assert len(batch.sleep) == 2
        assert len(batch.recovery) == 2
        assert len(batch.activity) == 2
        # one RawEvent per fetched record, unscored included:
        # 3 sleeps + 3 recoveries + 3 workouts + 2 cycles
        assert len(batch.raw) == 11
        assert all(r.source == "whoop" for r in batch.raw)
        source_ids = [r.source_id for r in batch.raw]
        assert len(set(source_ids)) == len(source_ids)
        assert "sleep:a1b2c3d4-0001-4000-8000-000000000001" in source_ids
        assert "recovery:93845162" in source_ids
        assert "workout:b2c3d4e5-0001-4000-8000-000000000001" in source_ids
        assert "cycle:93845162" in source_ids

    def test_pull_applies_watermark(self) -> None:
        since = datetime(2026, 6, 10, 0, 0, tzinfo=UTC)
        batch = _connector(_WhoopServer()).pull(since)
        # June 9 records (nap, prior recovery, weightlifting, prior cycle) fall
        # out; the June 10 pending/unscorable records still produce raws.
        assert len(batch.sleep) == 1
        assert batch.sleep[0].ts_start == datetime(2026, 6, 10, 5, 0, tzinfo=UTC)
        assert len(batch.recovery) == 1
        assert batch.recovery[0].score == 67.0
        assert len(batch.activity) == 1
        assert batch.activity[0].kind == "running"
        assert len(batch.raw) == 7  # 2 sleeps + 2 recoveries + 2 workouts + 1 cycle

    def test_pull_paginates_with_next_token(self) -> None:
        since = datetime(2026, 6, 8, 0, 0, tzinfo=UTC)
        server = _WhoopServer()
        small_pages = _connector(server, page_size=1).pull(since)
        one_page = _connector(_WhoopServer()).pull(since)
        assert {r.source_id for r in small_pages.raw} == {r.source_id for r in one_page.raw}
        assert len(small_pages.sleep) == len(one_page.sleep)
        assert len(small_pages.recovery) == len(one_page.recovery)
        assert len(small_pages.activity) == len(one_page.activity)
        assert any("nextToken" in r.url.params for r in server.requests)

    def test_pull_timestamps_are_utc(self) -> None:
        since = datetime(2026, 6, 8, 0, 0, tzinfo=UTC)
        batch = _connector(_WhoopServer()).pull(since)
        for sleep_event in batch.sleep:
            assert sleep_event.ts_start.tzinfo == UTC
            assert sleep_event.ts_end.tzinfo == UTC
        for event in [*batch.recovery, *batch.activity]:
            assert event.ts.tzinfo == UTC
        for raw in batch.raw:
            assert raw.source_ts.tzinfo == UTC
