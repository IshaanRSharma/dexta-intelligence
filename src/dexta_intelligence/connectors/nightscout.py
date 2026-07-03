"""Nightscout connector - entries/treatments/devicestatus to timeline events.

Nightscout is the OSS CGM remote-monitoring server used by the looping
community, and the richest single source we support: glucose (``entries``),
real pump data (``treatments``: boluses, carbs, temp basals, suspends) and -
for looping users - the dosing algorithm's own forecast curves
(``devicestatus``: ``openaps.suggested.predBGs`` / ``loop.predicted``), which
feed the Prediction Reconciliation agent.

The module is split in two layers so parsing stays fixture-testable:

- **Pure parsers** (``parse_entry``, ``parse_treatment``,
  ``parse_devicestatus``) take one raw Nightscout JSON dict and return typed
  events. No I/O, no clock, no config.
- **NightscoutConnector** owns the thin HTTP layer: token auth, explicit
  timeouts, descending-cursor pagination, and the ``since`` watermark. It
  speaks the secured v3 API (``/api/v3/*`` with a JWT bearer token minted from
  the configured access token) when the server offers it, falling back to the
  legacy v1 query API (``/api/v1/*`` with the token in the query string) on
  older servers. Both dialects normalize to the exact same events.

Temp-basal handling (documented best-effort): Nightscout logs temp basals as
a rate (U/h) plus a duration, not delivered units. We record
``kind=temp_basal`` with ``duration_min`` and, when an absolute rate is
present, ``units = rate x duration/60`` - the *scheduled* delivery, which may
overstate reality if the temp was cancelled early by a later record.
"""

from __future__ import annotations

import contextlib
import re
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

import httpx

from dexta_intelligence.connectors.base import HealthReport, NormalizedBatch
from dexta_intelligence.models import (
    GlucoseEvent,
    InsulinEvent,
    InsulinKind,
    MealEvent,
    PredictionEvent,
    RawEvent,
)

if TYPE_CHECKING:
    from dexta_intelligence.config import NightscoutConfig

__all__ = [
    "NightscoutConnector",
    "parse_devicestatus",
    "parse_entry",
    "parse_treatment",
]

SOURCE = "nightscout"

_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
_DEDUPE_MARGIN = timedelta(minutes=5)

_OPENAPS_CURVES: dict[str, Literal["iob", "cob", "uam", "zt"]] = {
    "IOB": "iob",
    "COB": "cob",
    "UAM": "uam",
    "ZT": "zt",
}

#: The access token rides in the URL query (v1) or the authorization-request
#: path (v3), so httpx error strings (which embed the full URL) would otherwise
#: leak it into health-check details and logs.
_TOKEN_QUERY_RE = re.compile(r"token=[^&\s'\"]+")
_TOKEN_PATH_RE = re.compile(r"(authorization/request/)[^/?\s'\"]+")


def _redact_token(text: str) -> str:
    """Strip the access token out of any URL-bearing string before it is shown."""
    return _TOKEN_PATH_RE.sub(r"\1[redacted]", _TOKEN_QUERY_RE.sub("token=[redacted]", text))


# ─────────────────────────────────────────────────────────────────────────────
# Pure parsing - raw Nightscout JSON dicts in, typed events out
# ─────────────────────────────────────────────────────────────────────────────


def _parse_iso(value: str) -> datetime:
    """Nightscout ISO timestamp to aware UTC. Naive strings are assumed UTC."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _entry_ts(raw: dict[str, Any]) -> datetime | None:
    """Entry timestamp: prefer epoch-ms ``date`` (always UTC), fall back to ISO."""
    date_ms = raw.get("date")
    if isinstance(date_ms, int | float):
        return datetime.fromtimestamp(date_ms / 1000.0, tz=UTC)
    date_string = raw.get("dateString")
    if isinstance(date_string, str):
        return _parse_iso(date_string)
    return None


def _treatment_ts(raw: dict[str, Any]) -> datetime | None:
    for key in ("created_at", "timestamp"):
        value = raw.get(key)
        if isinstance(value, str):
            return _parse_iso(value)
    return None


def _as_float(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def _created_at_floor(window_start: datetime) -> str:
    """Z-suffixed millisecond ISO string for ``created_at`` range queries.

    Nightscout stores ``created_at`` as a Z-suffixed ISO string, so both Mongo
    (v1) and the v3 query layer compare it lexicographically, which matches
    chronological order.
    """
    return window_start.strftime("%Y-%m-%dT%H:%M:%S.") + f"{window_start.microsecond // 1000:03d}Z"


def _normalize_v3_doc(doc: dict[str, Any]) -> dict[str, Any]:
    """Make a v3 document indistinguishable from a v1 one to the parsers.

    v3 keys the same fields the parsers read (``date``, ``sgv``, ``created_at``,
    ``eventType``, ...) but identifies documents by ``identifier`` rather than
    Mongo's ``_id``. Backfill ``_id`` from ``identifier`` so the raw-event
    source id (and thus store dedupe) behaves exactly as on the v1 path.
    """
    if "_id" not in doc:
        identifier = doc.get("identifier")
        if isinstance(identifier, str) and identifier:
            return {"_id": identifier, **doc}
    return doc


def parse_entry(raw: dict[str, Any]) -> GlucoseEvent | None:
    """One ``entries`` record to :class:`GlucoseEvent`.

    Returns ``None`` for non-sgv records (``mbg`` fingersticks, ``cal``
    calibrations) and for records missing a glucose value or timestamp.
    """
    if raw.get("type", "sgv") != "sgv":
        return None
    sgv = _as_float(raw.get("sgv"))
    ts = _entry_ts(raw)
    if sgv is None or ts is None:
        return None
    trend = raw.get("direction")
    return GlucoseEvent(ts=ts, mg_dl=int(sgv), trend=trend if isinstance(trend, str) else None)


def _bolus_automatic(raw: dict[str, Any]) -> bool | None:
    """Explicit algorithm markers only - AAPS ``isSMB``, Loop ``automatic``.

    ``enteredBy`` is deliberately NOT used for boluses: manual boluses issued
    through the Loop app are also uploaded with ``enteredBy: "loop://..."``.
    """
    if raw.get("isSMB") is True or raw.get("automatic") is True:
        return True
    if raw.get("isSMB") is False or raw.get("automatic") is False:
        return False
    return None


def _basal_automatic(raw: dict[str, Any]) -> bool | None:
    """Temp basals from a looping uploader are algorithm-issued by definition."""
    entered_by = str(raw.get("enteredBy", "")).lower()
    if any(marker in entered_by for marker in ("openaps", "loop", "androidaps", "trio")):
        return True
    return None


def parse_treatment(raw: dict[str, Any]) -> list[InsulinEvent | MealEvent]:
    """One ``treatments`` record to zero or more insulin/meal events.

    A single Nightscout treatment can carry both insulin and carbs (e.g.
    ``Meal Bolus``), so the return is a list. Records that are neither
    dosing nor carbs (BG checks, notes, site changes) yield ``[]``.
    """
    ts = _treatment_ts(raw)
    if ts is None:
        return []

    events: list[InsulinEvent | MealEvent] = []
    event_type = str(raw.get("eventType", "")).strip().lower()
    insulin = _as_float(raw.get("insulin"))
    carbs = _as_float(raw.get("carbs"))
    duration_min = _as_float(raw.get("duration"))

    if insulin is not None and insulin > 0:
        events.append(
            InsulinEvent(
                ts=ts,
                kind=InsulinKind.BOLUS,
                units=insulin,
                automatic=_bolus_automatic(raw),
            )
        )

    if "temp basal" in event_type or event_type == "temporary basal":
        rate = _as_float(raw.get("absolute"))
        if rate is None:
            rate = _as_float(raw.get("rate"))
        scheduled_units = (
            rate * duration_min / 60.0 if rate is not None and duration_min is not None else None
        )
        events.append(
            InsulinEvent(
                ts=ts,
                kind=InsulinKind.TEMP_BASAL,
                units=scheduled_units,
                duration_min=duration_min,
                automatic=_basal_automatic(raw),
            )
        )
    elif "suspend" in event_type:
        events.append(
            InsulinEvent(
                ts=ts,
                kind=InsulinKind.SUSPEND,
                duration_min=duration_min,
                automatic=_basal_automatic(raw),
            )
        )

    if carbs is not None and carbs > 0:
        note = raw.get("notes")
        events.append(
            MealEvent(
                ts=ts,
                carbs_g=carbs,
                protein_g=_as_float(raw.get("protein")),
                fat_g=_as_float(raw.get("fat")),
                note=note if isinstance(note, str) else None,
            )
        )

    return events


def _curve_values(values: Any) -> list[float] | None:
    if not isinstance(values, list) or not values:
        return None
    if not all(isinstance(v, int | float) for v in values):
        return None
    return [float(v) for v in values]


def parse_devicestatus(raw: dict[str, Any]) -> list[PredictionEvent]:
    """One ``devicestatus`` record to algorithm forecast curves, if any.

    - oref0/AAPS: ``openaps.suggested.predBGs`` with IOB/COB/UAM/ZT keys,
      each a list of mg/dL at 5-minute spacing from the cycle time
      (``deliverAt``/``timestamp``).
    - Loop: ``loop.predicted`` with ``startDate`` + ``values``.

    Most uploaders (xDrip, pump-only rigs) carry no predictions at all; such
    docs simply yield ``[]``.
    """
    events: list[PredictionEvent] = []

    openaps = raw.get("openaps")
    suggested = openaps.get("suggested") if isinstance(openaps, dict) else None
    if isinstance(suggested, dict):
        pred_bgs = suggested.get("predBGs")
        ts_value = suggested.get("deliverAt") or suggested.get("timestamp") or raw.get("created_at")
        if isinstance(pred_bgs, dict) and isinstance(ts_value, str):
            ts = _parse_iso(ts_value)
            for ns_key, curve_kind in _OPENAPS_CURVES.items():
                values = _curve_values(pred_bgs.get(ns_key))
                if values is not None:
                    events.append(
                        PredictionEvent(
                            ts=ts, source="openaps", curve_kind=curve_kind, values_mg_dl=values
                        )
                    )

    loop = raw.get("loop")
    predicted = loop.get("predicted") if isinstance(loop, dict) else None
    if isinstance(predicted, dict):
        start = predicted.get("startDate")
        values = _curve_values(predicted.get("values"))
        if isinstance(start, str) and values is not None:
            events.append(
                PredictionEvent(
                    ts=_parse_iso(start), source="loop", curve_kind="loop", values_mg_dl=values
                )
            )

    return events


# ─────────────────────────────────────────────────────────────────────────────
# Connector - thin HTTP layer over the pure parsers
# ─────────────────────────────────────────────────────────────────────────────


class NightscoutConnector:
    """Implements the :class:`~dexta_intelligence.connectors.base.Connector`
    protocol against the Nightscout REST API, preferring the secured v3
    interface and falling back to the legacy v1 query API.

    The dialect is detected once per connector instance (see
    :meth:`_detect_version`) and remembered for its lifetime. Pagination walks
    descending through time in both dialects: each page is bounded above by the
    oldest timestamp of the previous page and below by the watermark, so no
    records are skipped regardless of Nightscout's fixed newest-first sort.
    """

    source = SOURCE

    def __init__(
        self,
        config: NightscoutConfig,
        *,
        client: httpx.Client | None = None,
        page_size: int = 1000,
    ) -> None:
        self._base_url = config.url.rstrip("/")
        self._token = config.token
        self._page_size = page_size
        self._client = client if client is not None else httpx.Client(timeout=_TIMEOUT)
        self._api_version: Literal["v1", "v3"] | None = None
        self._jwt: str | None = None

    # -- Connector protocol --------------------------------------------------

    def check(self) -> HealthReport:
        """Report server version + latest sgv over whichever dialect is live."""
        try:
            self._detect_version()
            version = self._server_version()
        except httpx.HTTPError as exc:
            return HealthReport(ok=False, source=self.source, detail=str(exc))

        latest_ts: datetime | None = None
        with contextlib.suppress(httpx.HTTPError):
            latest_ts = self._latest_sgv_ts()  # decoration; reachability already proven

        return HealthReport(
            ok=True,
            source=self.source,
            detail=f"Nightscout {version}",
            latest_data_ts=latest_ts,
        )

    def pull(self, since: datetime) -> NormalizedBatch:
        """Fetch everything newer than ``since`` (minus a small dedupe margin).

        Raises ``httpx.HTTPError`` on provider hiccups - the sync workflow
        owns retries. Raw rows are returned for every fetched document;
        normalized events only where parsing succeeds.
        """
        window_start = since.astimezone(UTC) - _DEDUPE_MARGIN
        self._detect_version()

        if self._api_version == "v3":
            since_ms = int(window_start.timestamp() * 1000)
            since_iso = _created_at_floor(window_start)
            entries = self._page_v3("entries", "date", since_ms, extra={"type$eq": "sgv"})
            treatments = self._page_v3("treatments", "created_at", since_iso)
            devicestatus = self._page_v3("devicestatus", "created_at", since_iso)
        else:
            entries = self._page_entries(window_start)
            treatments = self._page_by_created_at("/api/v1/treatments.json", window_start)
            devicestatus = self._page_by_created_at("/api/v1/devicestatus.json", window_start)

        raw_events: list[RawEvent] = []
        glucose: list[GlucoseEvent] = []
        insulin: list[InsulinEvent] = []
        meals: list[MealEvent] = []
        predictions: list[PredictionEvent] = []

        for doc in entries:
            ts = _entry_ts(doc)
            if ts is None or ts < window_start:
                continue
            raw_events.append(self._raw_event(doc, ts))
            event = parse_entry(doc)
            if event is not None:
                glucose.append(event)

        for doc in treatments:
            ts = _treatment_ts(doc)
            if ts is None or ts < window_start:
                continue
            raw_events.append(self._raw_event(doc, ts))
            for treatment_event in parse_treatment(doc):
                if isinstance(treatment_event, InsulinEvent):
                    insulin.append(treatment_event)
                else:
                    meals.append(treatment_event)

        for doc in devicestatus:
            ts = _treatment_ts(doc)
            if ts is None or ts < window_start:
                continue
            raw_events.append(self._raw_event(doc, ts))
            predictions.extend(parse_devicestatus(doc))

        return NormalizedBatch(
            raw=raw_events,
            glucose=glucose,
            insulin=insulin,
            meals=meals,
            predictions=predictions,
        )

    # -- Dialect detection -----------------------------------------------------

    def _detect_version(self) -> None:
        """Latch v3 vs v1 once for the connector's lifetime.

        v3 is probed via the public ``/api/v3/version`` endpoint plus a JWT
        mint from the configured access token. Any HTTP error there (a 404 on
        pre-v3 servers, a 401 when the token cannot mint a JWT) means v3 is not
        usable, so we fall back to the legacy v1 query API.
        """
        if self._api_version is not None:
            return
        try:
            self._get_json_public("/api/v3/version", {})
            jwt = self._request_jwt()
        except httpx.HTTPError:
            jwt = None
        if jwt is None:
            self._api_version = "v1"
            self._jwt = None
        else:
            self._api_version = "v3"
            self._jwt = jwt

    def _request_jwt(self) -> str | None:
        """Mint a v3 bearer JWT from the configured access token, or ``None``."""
        payload = self._get_json_public(f"/api/v2/authorization/request/{self._token}", {})
        token = payload.get("token") if isinstance(payload, dict) else None
        return token if isinstance(token, str) and token else None

    def _server_version(self) -> str:
        if self._api_version == "v3":
            payload = self._get_json_public("/api/v3/version", {})
            if isinstance(payload, dict):
                version = payload.get("apiVersion") or payload.get("version")
                if isinstance(version, str):
                    return version
            return "?"
        payload = self._get_json("/api/v1/status.json", {})
        return payload.get("version", "?") if isinstance(payload, dict) else "?"

    def _latest_sgv_ts(self) -> datetime | None:
        if self._api_version == "v3":
            docs = self._get_v3_collection(
                "entries", {"limit": 1, "sort$desc": "date", "type$eq": "sgv"}
            )
            return _entry_ts(docs[0]) if docs else None
        entries = self._get_json("/api/v1/entries/sgv.json", {"count": 1})
        if isinstance(entries, list) and entries:
            return _entry_ts(entries[0])
        return None

    # -- HTTP plumbing ---------------------------------------------------------

    def _get_json(self, path: str, params: dict[str, str | int]) -> Any:
        # The token rides in the query string, so a raised httpx error would
        # embed it in the URL it prints. Re-raise with the token redacted so it
        # never reaches a health-check detail, the GUI, or a log.
        merged: dict[str, str | int] = {"token": self._token, **params}
        try:
            response = self._client.get(f"{self._base_url}{path}", params=merged)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise httpx.HTTPError(_redact_token(str(exc))) from None
        return response.json()

    def _get_json_public(self, path: str, params: dict[str, str | int]) -> Any:
        """GET without the query token: v3 version probe and JWT mint.

        The access token still rides in the JWT-request path, so errors are
        redacted the same way before they surface.
        """
        try:
            response = self._client.get(f"{self._base_url}{path}", params=params)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise httpx.HTTPError(_redact_token(str(exc))) from None
        return response.json()

    def _get_v3_collection(
        self, collection: str, params: dict[str, str | int]
    ) -> list[dict[str, Any]]:
        """GET one page of a v3 collection, unwrap ``{result: [...]}``, normalize."""
        headers = {"Authorization": f"Bearer {self._jwt}"}
        try:
            response = self._client.get(
                f"{self._base_url}/api/v3/{collection}", params=params, headers=headers
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise httpx.HTTPError(_redact_token(str(exc))) from None
        payload = response.json()
        result = payload.get("result") if isinstance(payload, dict) else payload
        if not isinstance(result, list):
            return []
        return [_normalize_v3_doc(doc) for doc in result if isinstance(doc, dict)]

    def _page_v3(
        self,
        collection: str,
        cursor_field: str,
        since_value: str | int,
        *,
        extra: dict[str, str | int] | None = None,
    ) -> list[dict[str, Any]]:
        """Descending-cursor page over a v3 collection, mirroring the v1 pagers.

        ``cursor_field`` is ``date`` (epoch-ms int) for entries and
        ``created_at`` (ISO string) for treatments/devicestatus; ``$gt``/``$lt``
        range comparisons work identically for both types.
        """
        results: list[dict[str, Any]] = []
        upper: str | int | None = None
        while True:
            params: dict[str, str | int] = {
                "limit": self._page_size,
                "sort$desc": cursor_field,
                f"{cursor_field}$gt": since_value,
            }
            if extra:
                params.update(extra)
            if upper is not None:
                params[f"{cursor_field}$lt"] = upper
            page = self._get_v3_collection(collection, params)
            if not page:
                break
            results.extend(page)
            if len(page) < self._page_size:
                break
            cursors = [doc[cursor_field] for doc in page if cursor_field in doc]
            if not cursors:
                break
            upper = min(cursors)
        return results

    def _raw_event(self, doc: dict[str, Any], ts: datetime) -> RawEvent:
        source_id = doc.get("_id")
        if not isinstance(source_id, str) or not source_id:
            source_id = f"synthetic:{ts.isoformat()}"
        return RawEvent(source=self.source, source_id=source_id, source_ts=ts, payload=doc)

    def _page_entries(self, window_start: datetime) -> list[dict[str, Any]]:
        """Page sgv entries via the epoch-ms ``date`` field."""
        since_ms = int(window_start.timestamp() * 1000)
        results: list[dict[str, Any]] = []
        upper_ms: int | None = None
        while True:
            params: dict[str, str | int] = {
                "count": self._page_size,
                "find[date][$gt]": since_ms,
            }
            if upper_ms is not None:
                params["find[date][$lt]"] = upper_ms
            page = self._get_json("/api/v1/entries/sgv.json", params)
            if not isinstance(page, list) or not page:
                break
            results.extend(doc for doc in page if isinstance(doc, dict))
            if len(page) < self._page_size:
                break
            dates = [doc["date"] for doc in page if isinstance(doc.get("date"), int | float)]
            if not dates:
                break
            upper_ms = int(min(dates))
        return results

    def _page_by_created_at(self, path: str, window_start: datetime) -> list[dict[str, Any]]:
        """Page treatments/devicestatus via the ISO ``created_at`` field.

        Nightscout stores ``created_at`` as a Z-suffixed ISO string, so
        Mongo's lexicographic comparison matches chronological order.
        """
        since_iso = _created_at_floor(window_start)
        results: list[dict[str, Any]] = []
        upper_iso: str | None = None
        while True:
            params: dict[str, str | int] = {
                "count": self._page_size,
                "find[created_at][$gt]": since_iso,
            }
            if upper_iso is not None:
                params["find[created_at][$lt]"] = upper_iso
            page = self._get_json(path, params)
            if not isinstance(page, list) or not page:
                break
            results.extend(doc for doc in page if isinstance(doc, dict))
            if len(page) < self._page_size:
                break
            stamps = [doc["created_at"] for doc in page if isinstance(doc.get("created_at"), str)]
            if not stamps:
                break
            upper_iso = min(stamps)
        return results
