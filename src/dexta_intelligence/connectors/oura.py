"""Oura connector - sleep, readiness, workouts, and daily activity to timeline events.

Oura is the template community driver: personal-access-token auth against the
official v2 REST API (``/v2/usercollection/...``), ``next_token`` pagination,
and date-window queries via ``start_date`` / ``end_date``.

The module is split in two layers so parsing stays fixture-testable:

- **Pure parsers** (``parse_sleep``, ``parse_readiness``, ``parse_workout``,
  ``parse_daily_activity``) take one raw Oura v2 JSON dict and return a typed
  event. No I/O, no clock, no config.
- **OuraConnector** owns the thin HTTP layer: bearer auth, explicit timeouts,
  ``next_token`` pagination, and the ``since`` watermark.

Scoring (documented best-effort): daily summaries expose ``score: null`` while
Oura is still computing. Those records are skipped for normalized events but
kept in ``raw`` for provenance. Detailed ``sleep`` periods skip ``deleted``
rows and rows missing a bedtime window.

Day-level timestamp convention: daily summaries (``daily_readiness``,
``daily_sleep``, ``daily_activity``) use the API ``timestamp`` when present
(ISO string with offset, normalized to UTC). When ``timestamp`` is absent, the
canonical event time is ``day`` at 00:00 UTC - the same midnight anchor used
when comparing date-filtered windows.

Field mapping notes: :class:`~dexta_intelligence.models.RecoveryEvent` has no
temperature field, so readiness contributors and temperature deviations stay in
the raw payload only. ``hrv_ms`` and ``rhr_bpm`` are enriched from the longest
scored ``sleep`` period on the same ``day`` (``average_hrv``,
``lowest_heart_rate``). Workout ``intensity`` maps Oura's ``easy`` /
``moderate`` / ``hard`` enum to 1 / 2 / 3 because the model has no dedicated
HR fields.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx

from dexta_intelligence.connectors.base import HealthReport, NormalizedBatch
from dexta_intelligence.models import ActivityEvent, RawEvent, RecoveryEvent, SleepEvent

if TYPE_CHECKING:
    from dexta_intelligence.config import OuraConfig

__all__ = [
    "OuraConnector",
    "parse_daily_activity",
    "parse_readiness",
    "parse_sleep",
    "parse_workout",
]

SOURCE = "oura"

OURA_API_BASE = "https://api.ouraring.com"

_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
_DEDUPE_MARGIN = timedelta(minutes=5)

_SLEEP_STAGE_FIELDS = {
    "deep": "deep_sleep_duration",
    "light": "light_sleep_duration",
    "rem": "rem_sleep_duration",
    "awake": "awake_time",
}

_WORKOUT_INTENSITY = {
    "easy": 1.0,
    "moderate": 2.0,
    "hard": 3.0,
}


# -----------------------------------------------------------------------------
# Pure parsing - raw Oura v2 JSON dicts in, typed events out
# -----------------------------------------------------------------------------


def _parse_iso(value: str) -> datetime:
    """Oura ISO timestamp to aware UTC. Naive strings are assumed UTC."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_day(value: str) -> datetime:
    """``YYYY-MM-DD`` calendar day at 00:00 UTC."""
    parsed = datetime.strptime(value, "%Y-%m-%d")
    return parsed.replace(tzinfo=UTC)


def _day_level_ts(raw: dict[str, Any]) -> datetime | None:
    """Canonical timestamp for day-level Oura summaries."""
    ts_value = raw.get("timestamp")
    if isinstance(ts_value, str):
        return _parse_iso(ts_value)
    day = raw.get("day")
    if isinstance(day, str):
        return _parse_day(day)
    return None


def _as_float(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def _daily_scored(raw: dict[str, Any]) -> bool:
    """Daily summaries with ``score: null`` are still syncing - skip normalization."""
    return raw.get("score") is not None


def parse_sleep(raw: dict[str, Any], *, daily_score: float | None = None) -> SleepEvent | None:
    """One ``/v2/usercollection/sleep`` record to :class:`SleepEvent`.

    Uses ``bedtime_start`` / ``bedtime_end`` for the window and stage durations
    in seconds from Oura's per-period fields. ``daily_score`` comes from the
    matching ``daily_sleep`` row (same ``day``) when available. Returns ``None``
    for deleted periods and rows missing a valid bedtime window.
    """
    if raw.get("type") == "deleted":
        return None
    start_value = raw.get("bedtime_start")
    end_value = raw.get("bedtime_end")
    if not isinstance(start_value, str) or not isinstance(end_value, str):
        return None
    ts_start = _parse_iso(start_value)
    ts_end = _parse_iso(end_value)
    if ts_end < ts_start:
        return None

    stages: dict[str, float] | None = None
    parsed_stages = {
        name: round(seconds / 60.0, 1)
        for name, key in _SLEEP_STAGE_FIELDS.items()
        if (seconds := _as_float(raw.get(key))) is not None
    }
    if parsed_stages:
        stages = parsed_stages

    return SleepEvent(
        ts_start=ts_start,
        ts_end=ts_end,
        duration_min=(ts_end - ts_start).total_seconds() / 60.0,
        score=daily_score,
        stages=stages,
    )


def parse_readiness(
    raw: dict[str, Any],
    *,
    sleep_doc: dict[str, Any] | None = None,
) -> RecoveryEvent | None:
    """One ``/v2/usercollection/daily_readiness`` record to :class:`RecoveryEvent`.

    Unscored rows (``score: null``) yield ``None``. ``hrv_ms`` and ``rhr_bpm``
    are taken from the optional same-day ``sleep`` period (``average_hrv``,
    ``lowest_heart_rate``); readiness contributor scores stay raw-only.
    """
    if not _daily_scored(raw):
        return None
    ts = _day_level_ts(raw)
    if ts is None:
        return None

    hrv_ms: float | None = None
    rhr_bpm: float | None = None
    if sleep_doc is not None:
        hrv_ms = _as_float(sleep_doc.get("average_hrv"))
        rhr_bpm = _as_float(sleep_doc.get("lowest_heart_rate"))

    return RecoveryEvent(
        ts=ts,
        score=_as_float(raw.get("score")),
        hrv_ms=hrv_ms,
        rhr_bpm=rhr_bpm,
    )


def parse_workout(raw: dict[str, Any]) -> ActivityEvent | None:
    """One ``/v2/usercollection/workout`` record to :class:`ActivityEvent`.

    ``kind`` is Oura's ``activity`` string; ``intensity`` maps the
    ``easy`` / ``moderate`` / ``hard`` enum to 1 / 2 / 3. Rows missing
    ``start_datetime`` yield ``None``.
    """
    start_value = raw.get("start_datetime")
    if not isinstance(start_value, str):
        return None
    ts = _parse_iso(start_value)

    duration_min: float | None = None
    end_value = raw.get("end_datetime")
    if isinstance(end_value, str):
        duration = (_parse_iso(end_value) - ts).total_seconds() / 60.0
        if duration >= 0:
            duration_min = duration

    activity = raw.get("activity")
    kind = activity if isinstance(activity, str) and activity else "workout"

    intensity: float | None = None
    raw_intensity = raw.get("intensity")
    if isinstance(raw_intensity, str):
        intensity = _WORKOUT_INTENSITY.get(raw_intensity)

    return ActivityEvent(
        ts=ts,
        kind=kind,
        duration_min=duration_min,
        intensity=intensity,
        strain=None,
    )


def parse_daily_activity(raw: dict[str, Any]) -> ActivityEvent | None:
    """One ``/v2/usercollection/daily_activity`` summary to :class:`ActivityEvent`.

    Unscored rows (``score: null``) yield ``None``. ``kind`` is always
    ``daily_activity``; ``intensity`` carries the 0-100 activity score.
    """
    if not _daily_scored(raw):
        return None
    ts = _day_level_ts(raw)
    if ts is None:
        return None

    duration_min: float | None = None
    active_seconds = sum(
        sec
        for key in ("low_activity_time", "medium_activity_time", "high_activity_time")
        if (sec := _as_float(raw.get(key))) is not None
    )
    if active_seconds > 0:
        duration_min = active_seconds / 60.0

    return ActivityEvent(
        ts=ts,
        kind="daily_activity",
        duration_min=duration_min,
        intensity=_as_float(raw.get("score")),
        strain=None,
    )


# -----------------------------------------------------------------------------
# Connector - thin HTTP layer over the pure parsers
# -----------------------------------------------------------------------------


class OuraConnector:
    """Implements the :class:`~dexta_intelligence.connectors.base.Connector`
    protocol against the Oura v2 user-collection API (personal access token).
    """

    source = SOURCE

    def __init__(
        self,
        config: OuraConfig,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self._access_token = config.access_token
        self._client = client if client is not None else httpx.Client(timeout=_TIMEOUT)

    # -- Connector protocol --------------------------------------------------

    def check(self) -> HealthReport:
        """Probe ``/v2/usercollection/personal_info`` and report the latest sleep end."""
        try:
            profile = self._get_json("/v2/usercollection/personal_info", {})
        except httpx.HTTPError as exc:
            return HealthReport(ok=False, source=self.source, detail=str(exc))

        user_id = profile.get("id", "?") if isinstance(profile, dict) else "?"
        latest_ts: datetime | None = None
        try:
            end_date = datetime.now(tz=UTC).date().isoformat()
            start_date = (datetime.now(tz=UTC).date() - timedelta(days=7)).isoformat()
            payload = self._get_json(
                "/v2/usercollection/sleep",
                {"start_date": start_date, "end_date": end_date},
            )
            data = payload.get("data") if isinstance(payload, dict) else None
            if isinstance(data, list):
                for doc in data:
                    if isinstance(doc, dict):
                        end_value = doc.get("bedtime_end")
                        if isinstance(end_value, str):
                            end_ts = _parse_iso(end_value)
                            if latest_ts is None or end_ts > latest_ts:
                                latest_ts = end_ts
        except httpx.HTTPError:
            pass  # latest-data is decoration; reachability already proven

        return HealthReport(
            ok=True,
            source=self.source,
            detail=f"Oura user {user_id}",
            latest_data_ts=latest_ts,
        )

    def pull(self, since: datetime) -> NormalizedBatch:
        """Fetch everything newer than ``since`` (minus a small dedupe margin).

        Raises ``httpx.HTTPError`` on provider hiccups - the sync workflow
        owns retries. Raw rows are returned for every fetched record
        (including unscored ones, so provenance survives rescoring);
        normalized events only where parsing succeeds.
        """
        window_start = since.astimezone(UTC) - _DEDUPE_MARGIN
        start_date, end_date = _date_window(window_start)

        sleeps = self._page_collection("/v2/usercollection/sleep", start_date, end_date)
        daily_sleeps = self._page_collection(
            "/v2/usercollection/daily_sleep", start_date, end_date
        )
        readiness_rows = self._page_collection(
            "/v2/usercollection/daily_readiness", start_date, end_date
        )
        workouts = self._page_collection("/v2/usercollection/workout", start_date, end_date)
        daily_activities = self._page_collection(
            "/v2/usercollection/daily_activity", start_date, end_date
        )

        daily_sleep_scores = {
            doc["day"]: _as_float(doc.get("score"))
            for doc in daily_sleeps
            if isinstance(doc.get("day"), str)
        }
        sleep_by_day = _longest_sleep_by_day(sleeps)

        raw_events: list[RawEvent] = []
        sleep_events: list[SleepEvent] = []
        recovery_events: list[RecoveryEvent] = []
        activity_events: list[ActivityEvent] = []

        self._ingest_sleeps(
            sleeps,
            daily_sleep_scores,
            window_start,
            raw_events,
            sleep_events,
        )
        self._ingest_daily_sleeps(daily_sleeps, window_start, raw_events)
        self._ingest_readiness(
            readiness_rows, sleep_by_day, window_start, raw_events, recovery_events
        )
        self._ingest_workouts(workouts, window_start, raw_events, activity_events)
        self._ingest_daily_activities(daily_activities, window_start, raw_events, activity_events)

        return NormalizedBatch(
            raw=raw_events,
            activity=activity_events,
            sleep=sleep_events,
            recovery=recovery_events,
        )

    def _ingest_sleeps(
        self,
        sleeps: list[dict[str, Any]],
        daily_sleep_scores: dict[str, float | None],
        window_start: datetime,
        raw_events: list[RawEvent],
        sleep_events: list[SleepEvent],
    ) -> None:
        for doc in sleeps:
            ts = _sleep_record_ts(doc)
            if ts is None or ts < window_start:
                continue
            raw_events.append(self._raw_event(doc.get("id"), doc, ts))
            day = doc.get("day")
            daily_score = daily_sleep_scores.get(day) if isinstance(day, str) else None
            sleep_event = parse_sleep(doc, daily_score=daily_score)
            if sleep_event is not None:
                sleep_events.append(sleep_event)

    def _ingest_daily_sleeps(
        self,
        daily_sleeps: list[dict[str, Any]],
        window_start: datetime,
        raw_events: list[RawEvent],
    ) -> None:
        for doc in daily_sleeps:
            ts = _day_level_ts(doc)
            if ts is None or ts < window_start:
                continue
            raw_events.append(self._raw_event(doc.get("id"), doc, ts))

    def _ingest_readiness(
        self,
        readiness_rows: list[dict[str, Any]],
        sleep_by_day: dict[str, dict[str, Any]],
        window_start: datetime,
        raw_events: list[RawEvent],
        recovery_events: list[RecoveryEvent],
    ) -> None:
        for doc in readiness_rows:
            ts = _day_level_ts(doc)
            if ts is None or ts < window_start:
                continue
            day = doc.get("day")
            sleep_doc = sleep_by_day.get(day) if isinstance(day, str) else None
            raw_events.append(self._raw_event(doc.get("id"), doc, ts))
            recovery_event = parse_readiness(doc, sleep_doc=sleep_doc)
            if recovery_event is not None:
                recovery_events.append(recovery_event)

    def _ingest_workouts(
        self,
        workouts: list[dict[str, Any]],
        window_start: datetime,
        raw_events: list[RawEvent],
        activity_events: list[ActivityEvent],
    ) -> None:
        for doc in workouts:
            ts = _workout_record_ts(doc)
            if ts is None or ts < window_start:
                continue
            raw_events.append(self._raw_event(doc.get("id"), doc, ts))
            activity_event = parse_workout(doc)
            if activity_event is not None:
                activity_events.append(activity_event)

    def _ingest_daily_activities(
        self,
        daily_activities: list[dict[str, Any]],
        window_start: datetime,
        raw_events: list[RawEvent],
        activity_events: list[ActivityEvent],
    ) -> None:
        for doc in daily_activities:
            ts = _day_level_ts(doc)
            if ts is None or ts < window_start:
                continue
            raw_events.append(self._raw_event(doc.get("id"), doc, ts))
            activity_event = parse_daily_activity(doc)
            if activity_event is not None:
                activity_events.append(activity_event)

    # -- HTTP plumbing ---------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._access_token}"}

    def _get_json(self, path: str, params: dict[str, str | int]) -> Any:
        url = f"{OURA_API_BASE}{path}"
        response = self._client.get(url, params=params, headers=self._headers())
        response.raise_for_status()
        return response.json()

    def _raw_event(self, identifier: Any, doc: dict[str, Any], ts: datetime) -> RawEvent:
        if identifier is None or not str(identifier):
            source_id = f"synthetic:{ts.isoformat()}"
        else:
            source_id = str(identifier)
        return RawEvent(source=self.source, source_id=source_id, source_ts=ts, payload=doc)

    def _page_collection(
        self,
        path: str,
        start_date: str,
        end_date: str,
    ) -> list[dict[str, Any]]:
        """Walk an Oura collection endpoint via ``next_token`` pagination."""
        results: list[dict[str, Any]] = []
        next_token: str | None = None
        while True:
            params: dict[str, str | int] = {
                "start_date": start_date,
                "end_date": end_date,
            }
            if next_token is not None:
                params["next_token"] = next_token
            payload = self._get_json(path, params)
            if not isinstance(payload, dict):
                break
            data = payload.get("data")
            if isinstance(data, list):
                results.extend(doc for doc in data if isinstance(doc, dict))
            token = payload.get("next_token")
            if not isinstance(token, str) or not token:
                break
            next_token = token
        return results


def _date_window(window_start: datetime) -> tuple[str, str]:
    start = window_start.date()
    end = max(datetime.now(tz=UTC).date(), start)
    return start.isoformat(), end.isoformat()


def _sleep_record_ts(doc: dict[str, Any]) -> datetime | None:
    start_value = doc.get("bedtime_start")
    return _parse_iso(start_value) if isinstance(start_value, str) else None


def _workout_record_ts(doc: dict[str, Any]) -> datetime | None:
    start_value = doc.get("start_datetime")
    return _parse_iso(start_value) if isinstance(start_value, str) else None


def _longest_sleep_by_day(docs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Pick the longest scored sleep period per calendar day for readiness enrichment."""
    by_day: dict[str, dict[str, Any]] = {}
    best_duration: dict[str, float] = {}
    for doc in docs:
        if doc.get("type") == "deleted":
            continue
        day = doc.get("day")
        if not isinstance(day, str):
            continue
        duration = _as_float(doc.get("total_sleep_duration")) or 0.0
        if duration >= best_duration.get(day, -1.0):
            by_day[day] = doc
            best_duration[day] = duration
    return by_day
