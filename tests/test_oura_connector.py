"""Oura connector tests — pure parsing against fixtures, client via MockTransport.

No live network calls: the connector tests run against an ``httpx.MockTransport``
that emulates the Oura v2 user-collection API (bearer auth, ``start_date`` /
``end_date`` filtering, and ``next_token`` pagination).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from dexta_intelligence.config import OuraConfig, load_config
from dexta_intelligence.connectors.oura import (
    OuraConnector,
    parse_daily_activity,
    parse_readiness,
    parse_sleep,
    parse_workout,
)

FIXTURES = Path(__file__).parent / "fixtures"

GOOD_TOKEN = "good-access-token"
EXPIRED_TOKEN = "expired-access-token"


def _load(name: str) -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = json.loads((FIXTURES / name).read_text())
    return data


SLEEPS = _load("oura_sleep.json")
DAILY_SLEEPS = _load("oura_daily_sleep.json")
READINESS = _load("oura_daily_readiness.json")
WORKOUTS = _load("oura_workout.json")
DAILY_ACTIVITIES = _load("oura_daily_activity.json")

DAILY_SLEEP_SCORES = {
    doc["day"]: doc["score"]
    for doc in DAILY_SLEEPS
    if isinstance(doc.get("day"), str)
}


# ─────────────────────────────────────────────────────────────────────────────
# Pure parsing — sleep
# ─────────────────────────────────────────────────────────────────────────────


class TestParseSleep:
    def test_scored_records_parse(self) -> None:
        events = [
            parse_sleep(doc, daily_score=DAILY_SLEEP_SCORES.get(doc["day"]))
            for doc in SLEEPS
            if doc.get("type") != "deleted" and doc.get("bedtime_end") is not None
        ]
        events = [e for e in events if e is not None]
        assert len(events) == 2

    def test_main_sleep_values(self) -> None:
        event = parse_sleep(SLEEPS[0], daily_score=91)
        assert event is not None
        assert event.ts_start == datetime(2026, 6, 10, 5, 0, tzinfo=UTC)
        assert event.ts_end == datetime(2026, 6, 10, 12, 30, tzinfo=UTC)
        assert event.duration_min == 450.0
        assert event.score == 91.0
        assert event.stages == {"deep": 90.0, "light": 225.0, "rem": 95.0, "awake": 40.0}

    def test_nap_parses(self) -> None:
        event = parse_sleep(SLEEPS[1], daily_score=76)
        assert event is not None
        assert event.duration_min == 45.0
        assert event.score == 76.0
        assert event.stages is not None
        assert event.stages["light"] == 30.0

    def test_timestamps_are_utc(self) -> None:
        event = parse_sleep(SLEEPS[0], daily_score=91)
        assert event is not None
        assert event.ts_start.tzinfo == UTC
        assert event.ts_end.tzinfo == UTC

    def test_deleted_type_skipped(self) -> None:
        deleted = next(doc for doc in SLEEPS if doc["type"] == "deleted")
        assert parse_sleep(deleted) is None

    def test_missing_end_skipped(self) -> None:
        incomplete = next(doc for doc in SLEEPS if doc.get("bedtime_end") is None)
        assert parse_sleep(incomplete) is None


# ─────────────────────────────────────────────────────────────────────────────
# Pure parsing — readiness
# ─────────────────────────────────────────────────────────────────────────────


class TestParseReadiness:
    def test_scored_records_parse(self) -> None:
        events = [e for e in (parse_readiness(doc) for doc in READINESS) if e is not None]
        assert len(events) == 2

    def test_values_with_sleep_enrichment(self) -> None:
        event = parse_readiness(READINESS[0], sleep_doc=SLEEPS[0])
        assert event is not None
        assert event.ts == datetime(2026, 6, 10, 16, 35, tzinfo=UTC)
        assert event.score == 67.0
        assert event.hrv_ms == 78.4
        assert event.rhr_bpm == 52.0

    def test_day_fallback_timestamp(self) -> None:
        doc = {**READINESS[1], "timestamp": None}
        event = parse_readiness(doc, sleep_doc=SLEEPS[1])
        assert event is not None
        assert event.ts == datetime(2026, 6, 9, 0, 0, tzinfo=UTC)

    def test_timestamps_are_utc(self) -> None:
        for doc in READINESS:
            event = parse_readiness(doc)
            if event is not None:
                assert event.ts.tzinfo == UTC

    def test_null_score_skipped(self) -> None:
        pending = next(doc for doc in READINESS if doc["score"] is None)
        assert parse_readiness(pending) is None


# ─────────────────────────────────────────────────────────────────────────────
# Pure parsing — workouts and daily activity
# ─────────────────────────────────────────────────────────────────────────────


class TestParseWorkout:
    def test_complete_records_parse(self) -> None:
        events = [e for e in (parse_workout(doc) for doc in WORKOUTS) if e is not None]
        assert len(events) == 2

    def test_run_values(self) -> None:
        event = parse_workout(WORKOUTS[0])
        assert event is not None
        assert event.ts == datetime(2026, 6, 10, 15, 0, tzinfo=UTC)
        assert event.ts.tzinfo == UTC
        assert event.kind == "running"
        assert event.duration_min == 45.0
        assert event.intensity == 2.0

    def test_weight_training_values(self) -> None:
        event = parse_workout(WORKOUTS[1])
        assert event is not None
        assert event.kind == "weight_training"
        assert event.duration_min == 60.0
        assert event.intensity == 3.0

    def test_missing_start_skipped(self) -> None:
        incomplete = next(doc for doc in WORKOUTS if doc.get("start_datetime") is None)
        assert parse_workout(incomplete) is None


class TestParseDailyActivity:
    def test_scored_records_parse(self) -> None:
        events = [
            e for e in (parse_daily_activity(doc) for doc in DAILY_ACTIVITIES) if e is not None
        ]
        assert len(events) == 2

    def test_values(self) -> None:
        event = parse_daily_activity(DAILY_ACTIVITIES[0])
        assert event is not None
        assert event.ts == datetime(2026, 6, 11, 3, 59, tzinfo=UTC)
        assert event.kind == "daily_activity"
        assert event.intensity == 88.0
        assert event.duration_min == 210.0

    def test_null_score_skipped(self) -> None:
        pending = next(doc for doc in DAILY_ACTIVITIES if doc["score"] is None)
        assert parse_daily_activity(pending) is None


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


class TestOuraConfig:
    def test_defaults(self) -> None:
        config = OuraConfig()
        assert config.access_token == ""

    def test_env_overrides(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("OURA_ACCESS_TOKEN", "env-oura-token")
        config = load_config(tmp_path / "missing.toml")
        assert config.oura.access_token == "env-oura-token"

    def test_env_absent_keeps_defaults(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("OURA_ACCESS_TOKEN", raising=False)
        config = load_config(tmp_path / "missing.toml")
        assert config.oura == OuraConfig()


# ─────────────────────────────────────────────────────────────────────────────
# Connector — mocked transport emulating the Oura v2 user-collection API
# ─────────────────────────────────────────────────────────────────────────────


class _OuraServer:
    """Stateful mock of the Oura API: bearer auth, date filtering, pagination."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.headers.get("Authorization") != f"Bearer {GOOD_TOKEN}":
            return httpx.Response(401, json={"detail": "Invalid token"})

        path = request.url.path
        params = request.url.params
        if path == "/v2/usercollection/personal_info":
            return httpx.Response(
                200,
                json={
                    "id": "oura-user-451823",
                    "age": 32,
                    "weight": 72.5,
                    "height": 1.78,
                    "biological_sex": "male",
                    "email": "athlete@example.com",
                },
            )
        if path == "/v2/usercollection/sleep":
            return self._collection(SLEEPS, "bedtime_start", params)
        if path == "/v2/usercollection/daily_sleep":
            return self._collection(DAILY_SLEEPS, "timestamp", params)
        if path == "/v2/usercollection/daily_readiness":
            return self._collection(READINESS, "timestamp", params)
        if path == "/v2/usercollection/workout":
            return self._collection(WORKOUTS, "start_datetime", params)
        if path == "/v2/usercollection/daily_activity":
            return self._collection(DAILY_ACTIVITIES, "timestamp", params)
        return httpx.Response(404)

    def _collection(
        self,
        docs: list[dict[str, Any]],
        ts_key: str,
        params: httpx.QueryParams,
    ) -> httpx.Response:
        out = [doc for doc in docs if self._in_date_window(doc, ts_key, params)]
        limit = 1
        offset = int(params.get("next_token", "0"))
        page = out[offset : offset + limit]
        next_token = str(offset + limit) if offset + limit < len(out) else None
        return httpx.Response(200, json={"data": page, "next_token": next_token})

    def _in_date_window(
        self,
        doc: dict[str, Any],
        ts_key: str,
        params: httpx.QueryParams,
    ) -> bool:
        start_date = params.get("start_date")
        end_date = params.get("end_date")
        day = doc.get("day")
        if isinstance(day, str):
            if isinstance(start_date, str) and day < start_date:
                return False
            if isinstance(end_date, str) and day > end_date:
                return False
            return True
        ts_value = doc.get(ts_key)
        if not isinstance(ts_value, str):
            return False
        day_from_ts = datetime.fromisoformat(ts_value.replace("Z", "+00:00")).date().isoformat()
        if isinstance(start_date, str) and day_from_ts < start_date:
            return False
        if isinstance(end_date, str) and day_from_ts > end_date:
            return False
        return True


def _connector(
    server: _OuraServer,
    *,
    access_token: str = GOOD_TOKEN,
) -> OuraConnector:
    config = OuraConfig(access_token=access_token)
    client = httpx.Client(transport=httpx.MockTransport(server.handler))
    return OuraConnector(config, client=client)


class TestOuraConnector:
    def test_check_ok(self) -> None:
        report = _connector(_OuraServer()).check()
        assert report.ok is True
        assert report.source == "oura"
        assert "oura-user-451823" in report.detail
        if report.latest_data_ts is not None:
            assert report.latest_data_ts.tzinfo == UTC

    def test_check_reports_latest_sleep_end(self) -> None:
        fixed_now = datetime(2026, 6, 11, 8, 0, tzinfo=UTC)
        with patch("dexta_intelligence.connectors.oura.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            mock_dt.fromisoformat = datetime.fromisoformat
            mock_dt.strptime = datetime.strptime
            report = _connector(_OuraServer()).check()
        assert report.latest_data_ts == datetime(2026, 6, 10, 12, 30, tzinfo=UTC)

    def test_check_bad_token(self) -> None:
        report = _connector(_OuraServer(), access_token=EXPIRED_TOKEN).check()
        assert report.ok is False
        assert report.source == "oura"
        assert "401" in report.detail

    def test_pull_auth_failure_raises(self) -> None:
        connector = _connector(_OuraServer(), access_token=EXPIRED_TOKEN)
        since = datetime(2026, 6, 8, 0, 0, tzinfo=UTC)
        with pytest.raises(httpx.HTTPStatusError):
            connector.pull(since)

    def test_pull_full_window(self) -> None:
        since = datetime(2026, 6, 8, 0, 0, tzinfo=UTC)
        batch = _connector(_OuraServer()).pull(since)
        assert len(batch.sleep) == 2
        assert len(batch.recovery) == 2
        assert len(batch.activity) == 4
        assert len(batch.raw) == 15
        assert all(r.source == "oura" for r in batch.raw)
        source_ids = [r.source_id for r in batch.raw]
        assert len(set(source_ids)) == len(source_ids)
        assert "a1b2c3d4-0001-4000-8000-000000000001" in source_ids
        assert "r1b2c3d4-0001-4000-8000-000000000001" in source_ids
        assert "w1b2c3d4-0001-4000-8000-000000000001" in source_ids

    def test_pull_applies_watermark(self) -> None:
        since = datetime(2026, 6, 10, 0, 0, tzinfo=UTC)
        batch = _connector(_OuraServer()).pull(since)
        assert len(batch.sleep) == 1
        assert batch.sleep[0].ts_start == datetime(2026, 6, 10, 5, 0, tzinfo=UTC)
        assert len(batch.recovery) == 1
        assert batch.recovery[0].score == 67.0
        assert len([a for a in batch.activity if a.kind == "running"]) == 1
        assert len(batch.activity) == 3
        assert len(batch.raw) == 11

    def test_pull_paginates_with_next_token(self) -> None:
        since = datetime(2026, 6, 8, 0, 0, tzinfo=UTC)
        server = _OuraServer()
        batch = _connector(server).pull(since)
        assert len(batch.sleep) == 2
        assert any("next_token" in r.url.params for r in server.requests)

    def test_pull_timestamps_are_utc(self) -> None:
        since = datetime(2026, 6, 8, 0, 0, tzinfo=UTC)
        batch = _connector(_OuraServer()).pull(since)
        for sleep_event in batch.sleep:
            assert sleep_event.ts_start.tzinfo == UTC
            assert sleep_event.ts_end.tzinfo == UTC
        for event in [*batch.recovery, *batch.activity]:
            assert event.ts.tzinfo == UTC
        for raw in batch.raw:
            assert raw.source_ts.tzinfo == UTC
