"""Nightscout connector — entries/treatments/devicestatus → timeline events.

Nightscout is the OSS CGM remote-monitoring server used by the looping
community, and the richest single source we support: glucose (``entries``),
real pump data (``treatments``: boluses, carbs, temp basals, suspends) and —
for looping users — the dosing algorithm's own forecast curves
(``devicestatus``: ``openaps.suggested.predBGs`` / ``loop.predicted``), which
feed the Prediction Reconciliation agent.

The module is split in two layers so parsing stays fixture-testable:

- **Pure parsers** (``parse_entry``, ``parse_treatment``,
  ``parse_devicestatus``) take one raw Nightscout JSON dict and return typed
  events. No I/O, no clock, no config.
- **NightscoutConnector** owns the thin HTTP layer: token auth, explicit
  timeouts, descending-cursor pagination, and the ``since`` watermark.

Temp-basal handling (documented best-effort): Nightscout logs temp basals as
a rate (U/h) plus a duration, not delivered units. We record
``kind=temp_basal`` with ``duration_min`` and, when an absolute rate is
present, ``units = rate x duration/60`` — the *scheduled* delivery, which may
overstate reality if the temp was cancelled early by a later record.
"""

from __future__ import annotations

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


# ─────────────────────────────────────────────────────────────────────────────
# Pure parsing — raw Nightscout JSON dicts in, typed events out
# ─────────────────────────────────────────────────────────────────────────────


def _parse_iso(value: str) -> datetime:
    """Nightscout ISO timestamp → aware UTC. Naive strings are assumed UTC."""
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


def parse_entry(raw: dict[str, Any]) -> GlucoseEvent | None:
    """One ``entries`` record → :class:`GlucoseEvent`.

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
    """Explicit algorithm markers only — AAPS ``isSMB``, Loop ``automatic``.

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
    """One ``treatments`` record → zero or more insulin/meal events.

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
    """One ``devicestatus`` record → algorithm forecast curves, if any.

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
# Connector — thin HTTP layer over the pure parsers
# ─────────────────────────────────────────────────────────────────────────────


class NightscoutConnector:
    """Implements the :class:`~dexta_intelligence.connectors.base.Connector`
    protocol against the Nightscout v1 REST API (token auth).

    Pagination walks descending through time: each page is bounded above by
    the oldest timestamp of the previous page and below by the watermark, so
    no records are skipped regardless of Nightscout's fixed newest-first sort.
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

    # -- Connector protocol --------------------------------------------------

    def check(self) -> HealthReport:
        """Probe ``/api/v1/status`` and report server version + latest sgv."""
        try:
            payload = self._get_json("/api/v1/status.json", {})
        except httpx.HTTPError as exc:
            return HealthReport(ok=False, source=self.source, detail=str(exc))

        version = payload.get("version", "?") if isinstance(payload, dict) else "?"
        latest_ts: datetime | None = None
        try:
            entries = self._get_json("/api/v1/entries/sgv.json", {"count": 1})
            if isinstance(entries, list) and entries:
                latest_ts = _entry_ts(entries[0])
        except httpx.HTTPError:
            pass  # latest-data is decoration; reachability already proven

        return HealthReport(
            ok=True,
            source=self.source,
            detail=f"Nightscout {version}",
            latest_data_ts=latest_ts,
        )

    def pull(self, since: datetime) -> NormalizedBatch:
        """Fetch everything newer than ``since`` (minus a small dedupe margin).

        Raises ``httpx.HTTPError`` on provider hiccups — the sync workflow
        owns retries. Raw rows are returned for every fetched document;
        normalized events only where parsing succeeds.
        """
        window_start = since.astimezone(UTC) - _DEDUPE_MARGIN

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

    # -- HTTP plumbing ---------------------------------------------------------

    def _get_json(self, path: str, params: dict[str, str | int]) -> Any:
        merged: dict[str, str | int] = {"token": self._token, **params}
        response = self._client.get(f"{self._base_url}{path}", params=merged)
        response.raise_for_status()
        return response.json()

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
        since_iso = window_start.strftime("%Y-%m-%dT%H:%M:%S.") + (
            f"{window_start.microsecond // 1000:03d}Z"
        )
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
