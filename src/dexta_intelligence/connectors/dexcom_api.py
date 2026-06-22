"""Dexcom **official** API connector - OAuth2 ``/egvs``, ToS-clean.

This is the sanctioned complement to the reverse-engineered Share path
(:mod:`dexta_intelligence.connectors.dexcom`). Where Share scrapes the
follower API (no ToS blessing, ~24h live cap, glucose *now*), this connector
speaks Dexcom's documented developer API v3 with OAuth2 bearer tokens.

Two consequences fall out of using the official surface:

- **History, not just live.** ``/v3/users/self/egvs`` serves a date range,
  so ``pull(since)`` can reach arbitrarily far back (paginated by the
  caller's window), unlike Share's hard 24h / 288-reading clamp.
- **~1-3h delayed.** The official feed is *not* realtime: EGVs land roughly
  one to three hours after the sensor reads them. For a true "glucose now"
  surface use the Share connector; this one is the durable, sanctioned
  ingest path. (It therefore implements plain ``Connector``, not
  ``RealtimeConnector``.)

A **sandbox** host is available for development without a real account; pick
it via ``DexcomApiConfig.sandbox``.

The module follows the house connector split:

- **Pure conversion** (:func:`egv_to_event`) takes one EGV record dict and
  returns a :class:`GlucoseEvent`. No I/O, no clock - fixture-testable. The
  official API reports trend in *camelCase* (``fortyFiveUp``); conversion
  normalizes it to the capitalized Nightscout/Share vocabulary stored in
  ``GlucoseEvent.trend`` (``FortyFiveUp``).
- **DexcomApiConnector** owns the thin HTTP layer: lazy ``httpx`` import
  (optional ``[dexcom-api]`` extra), bearer auth with one-shot refresh on
  401 (mirroring the Whoop connector), date-window paging, and the ``since``
  watermark.

Trend mapping (official camelCase -> stored vocabulary)::

    doubleUp        -> DoubleUp        rising quickly
    singleUp        -> SingleUp        rising
    fortyFiveUp     -> FortyFiveUp     rising slightly
    flat            -> Flat            steady
    fortyFiveDown   -> FortyFiveDown   falling slightly
    singleDown      -> SingleDown      falling
    doubleDown      -> DoubleDown      falling quickly

The non-informative trends (``none``, ``notComputable``, ``rateOutOfRange``,
and anything unknown) normalize to ``trend=None``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from dexta_intelligence.connectors.base import HealthReport, NormalizedBatch
from dexta_intelligence.models import GlucoseEvent, RawEvent

if TYPE_CHECKING:
    import httpx

    from dexta_intelligence.config import DexcomApiConfig

__all__ = ["DexcomApiConnector", "egv_to_event"]

SOURCE = "dexcom_api"

PROD_BASE = "https://api.dexcom.com"
SANDBOX_BASE = "https://sandbox-api.dexcom.com"
# v3 token endpoint, matching the v3 /egvs data path used below.
TOKEN_PATH = "/v3/oauth2/token"

_DEDUPE_MARGIN = timedelta(minutes=5)
#: Dexcom serves at most ~30 days per egvs call; page in 30-day spans.
_PAGE_SPAN = timedelta(days=30)
#: The official feed lags the sensor; never request past this edge.
_FEED_DELAY = timedelta(hours=3)
#: Dexcom's egvs endpoint wants second-precision naive-UTC timestamps.
_API_TS_FMT = "%Y-%m-%dT%H:%M:%S"

#: Official camelCase trend names -> the capitalized vocabulary GlucoseEvent
#: stores (shared with Nightscout/Share). Everything else -> None.
_TREND_MAP = {
    "doubleUp": "DoubleUp",
    "singleUp": "SingleUp",
    "fortyFiveUp": "FortyFiveUp",
    "flat": "Flat",
    "fortyFiveDown": "FortyFiveDown",
    "singleDown": "SingleDown",
    "doubleDown": "DoubleDown",
}


# -----------------------------------------------------------------------------
# Pure conversion - one EGV record dict in, a typed event out
# -----------------------------------------------------------------------------


def _parse_ts(value: str) -> datetime:
    """Dexcom ``systemTime`` to aware UTC, rejecting naive values loudly.

    The official API documents ``systemTime`` as UTC; callers must hand it to
    us with an explicit offset (``...Z`` or ``+00:00``). A naive string is
    rejected - silently assuming a zone is exactly the class of CGM time bug
    the models refuse to inherit (the same house rule the Share path
    enforces). Normalize ``Z`` to an offset before parsing.
    """
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        msg = "naive Dexcom systemTime rejected: expected an explicit UTC offset"
        raise ValueError(msg)
    return parsed.astimezone(UTC)


def egv_to_event(record: dict[str, Any]) -> GlucoseEvent | None:
    """One ``/egvs`` record -> :class:`GlucoseEvent` (``None`` if unusable).

    Uses ``systemTime`` (the device UTC clock) as the canonical timestamp;
    ``displayTime`` is the user-local presentation clock and is left to the
    raw payload. A record missing ``systemTime`` or ``value`` yields ``None``.

    The official trend vocabulary is camelCase and is normalized to the
    capitalized keywords stored in ``GlucoseEvent.trend``; non-informative
    trends (``none``/``notComputable``/``rateOutOfRange``/unknown) -> ``None``.
    """
    system_time = record.get("systemTime")
    value = record.get("value")
    if not isinstance(system_time, str) or not isinstance(value, int | float):
        return None
    trend = record.get("trend")
    return GlucoseEvent(
        ts=_parse_ts(system_time),
        mg_dl=int(value),
        trend=_TREND_MAP.get(trend) if isinstance(trend, str) else None,
    )


# -----------------------------------------------------------------------------
# Connector - thin HTTP layer over the pure conversion
# -----------------------------------------------------------------------------


class DexcomApiConnector:
    """Implements :class:`~dexta_intelligence.connectors.base.Connector`
    against Dexcom's official developer API v3 (OAuth2 bearer tokens).

    Batch-only by design: the official feed is ~1-3h delayed, so there is no
    ``current()`` - for live readings use the Share connector. A 401 triggers
    exactly one token refresh and retry when refresh credentials (refresh
    token + client id/secret) are configured; otherwise the auth error
    propagates so ``check()`` can report it.
    """

    source = SOURCE

    def __init__(
        self,
        config: DexcomApiConfig,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self._access_token = config.access_token
        self._refresh_token = config.refresh_token
        self._client_id = config.client_id
        self._client_secret = config.client_secret
        self._base = SANDBOX_BASE if config.sandbox else PROD_BASE
        self._client = client if client is not None else self._build_client()

    # -- Connector protocol --------------------------------------------------

    def check(self) -> HealthReport:
        """Probe a tiny recent egvs window; report the latest reading.

        A 401 (auth) is the expected failure and reports
        ``detail="auth failed"``; any other transport error reports its
        message. Reachability is proven by a successful fetch.
        """
        import httpx  # noqa: PLC0415 - lazy: optional [dexcom-api] extra

        end = datetime.now(tz=UTC) - _FEED_DELAY
        start = end - timedelta(hours=6)
        try:
            payload = self._get_egvs(start, end)
        except httpx.HTTPStatusError as exc:
            detail = "auth failed" if exc.response.status_code == 401 else str(exc)
            return HealthReport(ok=False, source=self.source, detail=detail)
        except httpx.HTTPError as exc:
            return HealthReport(ok=False, source=self.source, detail=str(exc))

        latest_ts: datetime | None = None
        # v3 /egvs wraps readings under "egvs"; "records" was the v2 envelope.
        records = payload.get("egvs") or payload.get("records")
        if isinstance(records, list):
            for rec in records:
                event = egv_to_event(rec) if isinstance(rec, dict) else None
                if event is not None and (latest_ts is None or event.ts > latest_ts):
                    latest_ts = event.ts

        return HealthReport(
            ok=True,
            source=self.source,
            detail="Dexcom official API reachable",
            latest_data_ts=latest_ts,
        )

    def pull(self, since: datetime) -> NormalizedBatch:
        """Fetch EGVs newer than ``since`` (minus a small dedupe margin).

        Unlike Share, the official API serves history, so the window runs
        from ``since`` up to the feed edge (now minus the ~3h delay), paged
        in 30-day spans. EGVs carry no provider id; the reading timestamp is
        the idempotency key (``source_id = "dexcom_api:<iso-ts>"``), safe
        because Dexcom emits at most one EGV per 5-minute slot.
        """
        window_start = since.astimezone(UTC) - _DEDUPE_MARGIN
        feed_edge = datetime.now(tz=UTC) - _FEED_DELAY

        raw_events: list[RawEvent] = []
        glucose: list[GlucoseEvent] = []
        seen: set[str] = set()

        span_start = window_start
        while span_start < feed_edge:
            span_end = min(span_start + _PAGE_SPAN, feed_edge)
            payload = self._get_egvs(span_start, span_end)
            records = payload.get("egvs") or payload.get("records")
            if isinstance(records, list):
                for rec in records:
                    if not isinstance(rec, dict):
                        continue
                    event = egv_to_event(rec)
                    if event is None or event.ts < window_start:
                        continue
                    source_id = f"{self.source}:{event.ts.isoformat()}"
                    if source_id in seen:
                        continue
                    seen.add(source_id)
                    raw_events.append(
                        RawEvent(
                            source=self.source,
                            source_id=source_id,
                            source_ts=event.ts,
                            payload=rec,
                        )
                    )
                    glucose.append(event)
            span_start = span_end

        return NormalizedBatch(raw=raw_events, glucose=glucose)

    # -- HTTP plumbing -------------------------------------------------------

    def _build_client(self) -> httpx.Client:
        try:
            import httpx  # noqa: PLC0415 - lazy: optional [dexcom-api] extra
        except ImportError as exc:  # pragma: no cover - import-path guard
            msg = (
                "Dexcom official API support is not installed. "
                "Install it with: pip install 'dexta-intelligence[dexcom-api]'"
            )
            raise RuntimeError(msg) from exc
        return httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0))

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._access_token}"}

    def _can_refresh(self) -> bool:
        return bool(self._refresh_token and self._client_id and self._client_secret)

    def _refresh_access_token(self) -> None:
        """Exchange the refresh token for a new access (and refresh) token."""
        import httpx  # noqa: PLC0415 - lazy: optional [dexcom-api] extra

        response = self._client.post(
            f"{self._base}{TOKEN_PATH}",
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
            msg = "Dexcom token refresh response missing access_token"
            raise httpx.HTTPError(msg)
        self._access_token = access
        new_refresh = tokens.get("refresh_token")
        if isinstance(new_refresh, str) and new_refresh:
            self._refresh_token = new_refresh

    def _get_egvs(self, start: datetime, end: datetime) -> dict[str, Any]:
        """GET ``/v3/users/self/egvs`` for ``[start, end]`` with 401 refresh."""
        url = f"{self._base}/v3/users/self/egvs"
        params = {
            "startDate": start.astimezone(UTC).strftime(_API_TS_FMT),
            "endDate": end.astimezone(UTC).strftime(_API_TS_FMT),
        }
        response = self._client.get(url, params=params, headers=self._headers())
        if response.status_code == 401 and self._can_refresh():
            self._refresh_access_token()
            response = self._client.get(url, params=params, headers=self._headers())
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        return payload
