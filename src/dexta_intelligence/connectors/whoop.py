"""Whoop connector - sleeps/recoveries/workouts/cycles to timeline events.

Whoop is the wearable side of the timeline: sleep architecture
(``/v2/activity/sleep``), morning recovery physiology (``/v2/recovery``:
recovery score, HRV, resting heart rate) and workout strain
(``/v2/activity/workout``). Physiological-day cycles (``/v2/cycle``) are
ingested raw-only for provenance and replay - no timeline model maps to a
whole day yet.

The module is split in two layers so parsing stays fixture-testable:

- **Pure parsers** (``parse_sleep``, ``parse_recovery``, ``parse_workout``)
  take one raw Whoop v2 JSON dict and return a typed event. No I/O, no
  clock, no config.
- **WhoopConnector** owns the thin HTTP layer: bearer auth with one-shot
  refresh on 401, explicit timeouts, ``nextToken`` pagination (Whoop caps
  pages at 25 records), and the ``since`` watermark.

Scoring (documented best-effort): Whoop records carry a ``score_state``
(``SCORED`` / ``PENDING_SCORE`` / ``UNSCORABLE``). Only ``SCORED`` records
hold physiology, so the parsers skip everything else; re-pulling an
overlapping window later picks up records once Whoop finishes scoring them.

Field mapping notes: :class:`~dexta_intelligence.models.ActivityEvent` has
no heart-rate fields, so ``score.average_heart_rate`` (bpm) is recorded as
``intensity`` and ``max_heart_rate`` stays available in the raw payload
only. Refreshed OAuth tokens live in connector memory for the process
lifetime; persisting a rotated refresh token is the operator's concern
(config/env), never this module's.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx

from dexta_intelligence.connectors.base import HealthReport, NormalizedBatch
from dexta_intelligence.models import ActivityEvent, RawEvent, RecoveryEvent, SleepEvent

if TYPE_CHECKING:
    from dexta_intelligence.config import WhoopConfig

__all__ = [
    "WhoopConnector",
    "parse_recovery",
    "parse_sleep",
    "parse_workout",
]

SOURCE = "whoop"

WHOOP_API_BASE = "https://api.prod.whoop.com/developer"
WHOOP_TOKEN_URL = "https://api.prod.whoop.com/oauth/oauth2/token"

_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
_DEDUPE_MARGIN = timedelta(minutes=5)
_PAGE_LIMIT_MAX = 25
"""Whoop collection endpoints reject ``limit`` values above 25."""

_SLEEP_STAGE_FIELDS = {
    "light": "total_light_sleep_time_milli",
    "sws": "total_slow_wave_sleep_time_milli",
    "rem": "total_rem_sleep_time_milli",
    "awake": "total_awake_time_milli",
}


# -----------------------------------------------------------------------------
# Pure parsing - raw Whoop v2 JSON dicts in, typed events out
# -----------------------------------------------------------------------------


def _parse_iso(value: str) -> datetime:
    """Whoop ISO timestamp to aware UTC. Naive strings are assumed UTC."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _as_float(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def _scored(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Return the ``score`` dict iff the record is fully scored.

    ``PENDING_SCORE`` and ``UNSCORABLE`` records carry no physiology and
    yield ``None``; a missing ``score_state`` is treated as scored (the
    Whoop API default).
    """
    if raw.get("score_state", "SCORED") != "SCORED":
        return None
    score = raw.get("score")
    return score if isinstance(score, dict) else None


def parse_sleep(raw: dict[str, Any]) -> SleepEvent | None:
    """One ``/v2/activity/sleep`` record to :class:`SleepEvent`.

    Returns ``None`` for unscored records and records missing the start/end
    window. ``duration_min`` is the in-bed window (end minus start);
    ``stages`` holds per-stage minutes converted from Whoop's milliseconds;
    ``score`` is ``sleep_performance_percentage`` (0-100). Naps parse like
    any other sleep.
    """
    score = _scored(raw)
    if score is None:
        return None
    start_value = raw.get("start")
    end_value = raw.get("end")
    if not isinstance(start_value, str) or not isinstance(end_value, str):
        return None
    ts_start = _parse_iso(start_value)
    ts_end = _parse_iso(end_value)
    if ts_end < ts_start:
        return None

    stages: dict[str, float] | None = None
    stage_summary = score.get("stage_summary")
    if isinstance(stage_summary, dict):
        parsed_stages = {
            name: round(milli / 60_000.0, 1)
            for name, key in _SLEEP_STAGE_FIELDS.items()
            if (milli := _as_float(stage_summary.get(key))) is not None
        }
        stages = parsed_stages or None

    return SleepEvent(
        ts_start=ts_start,
        ts_end=ts_end,
        duration_min=(ts_end - ts_start).total_seconds() / 60.0,
        score=_as_float(score.get("sleep_performance_percentage")),
        stages=stages,
    )


def parse_recovery(raw: dict[str, Any]) -> RecoveryEvent | None:
    """One ``/v2/recovery`` record to :class:`RecoveryEvent`.

    Recovery records have no start/end of their own - Whoop computes them on
    wake - so ``created_at`` (falling back to ``updated_at``) is the event
    timestamp. Unscored records yield ``None``.
    """
    score = _scored(raw)
    if score is None:
        return None
    ts_value = raw.get("created_at") or raw.get("updated_at")
    if not isinstance(ts_value, str):
        return None
    return RecoveryEvent(
        ts=_parse_iso(ts_value),
        score=_as_float(score.get("recovery_score")),
        hrv_ms=_as_float(score.get("hrv_rmssd_milli")),
        rhr_bpm=_as_float(score.get("resting_heart_rate")),
    )


def parse_workout(raw: dict[str, Any]) -> ActivityEvent | None:
    """One ``/v2/activity/workout`` record to :class:`ActivityEvent`.

    ``kind`` is ``sport_name`` (falling back to ``sport_<id>``), ``strain``
    is Whoop's 0-21 workout strain, and ``intensity`` carries
    ``average_heart_rate`` in bpm - the model has no dedicated HR fields,
    so max HR remains raw-payload-only. Unscored records yield ``None``.
    """
    score = _scored(raw)
    if score is None:
        return None
    start_value = raw.get("start")
    if not isinstance(start_value, str):
        return None
    ts = _parse_iso(start_value)

    duration_min: float | None = None
    end_value = raw.get("end")
    if isinstance(end_value, str):
        duration = (_parse_iso(end_value) - ts).total_seconds() / 60.0
        if duration >= 0:
            duration_min = duration

    sport = raw.get("sport_name")
    if not isinstance(sport, str) or not sport:
        sport_id = raw.get("sport_id")
        sport = f"sport_{sport_id}" if isinstance(sport_id, int) else "workout"

    return ActivityEvent(
        ts=ts,
        kind=sport,
        duration_min=duration_min,
        intensity=_as_float(score.get("average_heart_rate")),
        strain=_as_float(score.get("strain")),
    )


# -----------------------------------------------------------------------------
# Connector - thin HTTP layer over the pure parsers
# -----------------------------------------------------------------------------


class WhoopConnector:
    """Implements the :class:`~dexta_intelligence.connectors.base.Connector`
    protocol against the Whoop v2 developer API (OAuth bearer tokens).

    A 401 triggers exactly one token refresh and retry when refresh
    credentials (refresh token + client id/secret) are configured;
    otherwise the auth error propagates so ``check()`` can report it.
    """

    source = SOURCE

    def __init__(
        self,
        config: WhoopConfig,
        *,
        client: httpx.Client | None = None,
        page_size: int = _PAGE_LIMIT_MAX,
    ) -> None:
        self._access_token = config.access_token
        self._refresh_token = config.refresh_token
        self._client_id = config.client_id
        self._client_secret = config.client_secret
        self._page_size = min(page_size, _PAGE_LIMIT_MAX)
        self._client = client if client is not None else httpx.Client(timeout=_TIMEOUT)

    # -- Connector protocol --------------------------------------------------

    def check(self) -> HealthReport:
        """Probe ``/v2/user/profile/basic`` and report the latest cycle start."""
        try:
            profile = self._get_json("/v2/user/profile/basic", {})
        except httpx.HTTPError as exc:
            return HealthReport(ok=False, source=self.source, detail=str(exc))

        user_id = profile.get("user_id", "?") if isinstance(profile, dict) else "?"
        latest_ts: datetime | None = None
        try:
            payload = self._get_json("/v2/cycle", {"limit": 1})
            records = payload.get("records") if isinstance(payload, dict) else None
            if isinstance(records, list) and records and isinstance(records[0], dict):
                start = records[0].get("start")
                if isinstance(start, str):
                    latest_ts = _parse_iso(start)
        except httpx.HTTPError:
            pass  # latest-data is decoration; reachability already proven

        return HealthReport(
            ok=True,
            source=self.source,
            detail=f"Whoop user {user_id}",
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

        sleeps = self._page_collection("/v2/activity/sleep", window_start)
        recoveries = self._page_collection("/v2/recovery", window_start)
        workouts = self._page_collection("/v2/activity/workout", window_start)
        cycles = self._page_collection("/v2/cycle", window_start)

        raw_events: list[RawEvent] = []
        sleep_events: list[SleepEvent] = []
        recovery_events: list[RecoveryEvent] = []
        activity_events: list[ActivityEvent] = []

        for doc in sleeps:
            ts = self._record_ts(doc, "start")
            if ts is None or ts < window_start:
                continue
            raw_events.append(self._raw_event("sleep", doc.get("id"), doc, ts))
            sleep_event = parse_sleep(doc)
            if sleep_event is not None:
                sleep_events.append(sleep_event)

        for doc in recoveries:
            ts = self._record_ts(doc, "created_at")
            if ts is None or ts < window_start:
                continue
            raw_events.append(self._raw_event("recovery", doc.get("cycle_id"), doc, ts))
            recovery_event = parse_recovery(doc)
            if recovery_event is not None:
                recovery_events.append(recovery_event)

        for doc in workouts:
            ts = self._record_ts(doc, "start")
            if ts is None or ts < window_start:
                continue
            raw_events.append(self._raw_event("workout", doc.get("id"), doc, ts))
            activity_event = parse_workout(doc)
            if activity_event is not None:
                activity_events.append(activity_event)

        for doc in cycles:
            ts = self._record_ts(doc, "start")
            if ts is None or ts < window_start:
                continue
            raw_events.append(self._raw_event("cycle", doc.get("id"), doc, ts))

        return NormalizedBatch(
            raw=raw_events,
            activity=activity_events,
            sleep=sleep_events,
            recovery=recovery_events,
        )

    # -- HTTP plumbing ---------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._access_token}"}

    def _can_refresh(self) -> bool:
        return bool(self._refresh_token and self._client_id and self._client_secret)

    def _refresh_access_token(self) -> None:
        """Exchange the refresh token for a new access (and refresh) token."""
        response = self._client.post(
            WHOOP_TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": self._refresh_token,
                "client_id": self._client_id,
                "client_secret": self._client_secret,
            },
        )
        response.raise_for_status()
        tokens = response.json()
        access = tokens.get("access_token") if isinstance(tokens, dict) else None
        if not isinstance(access, str) or not access:
            msg = "Whoop token refresh response missing access_token"
            raise httpx.HTTPError(msg)
        self._access_token = access
        new_refresh = tokens.get("refresh_token")
        if isinstance(new_refresh, str) and new_refresh:
            self._refresh_token = new_refresh

    def _get_json(self, path: str, params: dict[str, str | int]) -> Any:
        url = f"{WHOOP_API_BASE}{path}"
        response = self._client.get(url, params=params, headers=self._headers())
        if response.status_code == 401 and self._can_refresh():
            self._refresh_access_token()
            response = self._client.get(url, params=params, headers=self._headers())
        response.raise_for_status()
        return response.json()

    def _record_ts(self, doc: dict[str, Any], key: str) -> datetime | None:
        value = doc.get(key)
        return _parse_iso(value) if isinstance(value, str) else None

    def _raw_event(self, kind: str, identifier: Any, doc: dict[str, Any], ts: datetime) -> RawEvent:
        """Build a RawEvent with a collection-prefixed idempotency key.

        Prefixing keeps ids unique across collections - recovery records in
        particular have no ``id`` of their own, only a ``cycle_id``.
        """
        if identifier is None or not str(identifier):
            source_id = f"{kind}:synthetic:{ts.isoformat()}"
        else:
            source_id = f"{kind}:{identifier}"
        return RawEvent(source=self.source, source_id=source_id, source_ts=ts, payload=doc)

    def _page_collection(self, path: str, window_start: datetime) -> list[dict[str, Any]]:
        """Walk a Whoop collection endpoint via ``nextToken`` pagination.

        The request parameter is ``nextToken`` while the response field is
        ``next_token`` - that asymmetry is the v2 API's, not ours.
        """
        start_iso = window_start.strftime("%Y-%m-%dT%H:%M:%S.") + (
            f"{window_start.microsecond // 1000:03d}Z"
        )
        results: list[dict[str, Any]] = []
        next_token: str | None = None
        while True:
            params: dict[str, str | int] = {"limit": self._page_size, "start": start_iso}
            if next_token is not None:
                params["nextToken"] = next_token
            payload = self._get_json(path, params)
            if not isinstance(payload, dict):
                break
            records = payload.get("records")
            if isinstance(records, list):
                results.extend(doc for doc in records if isinstance(doc, dict))
            token = payload.get("next_token")
            if not isinstance(token, str) or not token:
                break
            next_token = token
        return results
