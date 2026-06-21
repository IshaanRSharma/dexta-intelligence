"""Tidepool JSON export connector - offline import from tidepool.org exports.

Tidepool's open data model stores CGM readings as ``cbg`` and pump boluses as
``bolus`` records. Users download a JSON export from the Tidepool web app
(Upload then Export device data). This connector reads that file once per sync,
normalizes glucose and insulin, and dedupes via ``(source, source_id)``.

Live Tidepool Platform API sync is out of scope here - it requires OAuth client
registration. JSON export is the zero-friction OSS path.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from dexta_intelligence.connectors.base import HealthReport, NormalizedBatch
from dexta_intelligence.models import GlucoseEvent, InsulinEvent, InsulinKind, RawEvent

if TYPE_CHECKING:

    from dexta_intelligence.config import TidepoolConfig

__all__ = ["TidepoolConnector"]

_DEDUPE_MARGIN = timedelta(minutes=5)
_MMOL_TO_MGDL = 18.016


class TidepoolConnector:
    """Degenerate connector that reads one local Tidepool JSON export."""

    source = "tidepool"

    def __init__(self, config: TidepoolConfig) -> None:
        self._path = config.export_path.expanduser()
        self.skipped: int = 0
        self._row_count: int = 0
        self._insulin_count: int = 0
        self._min_ts: datetime | None = None
        self._parsed: list[tuple[str, datetime, dict[str, Any]]] | None = None

    def check(self) -> HealthReport:
        if not self._path.is_file():
            return HealthReport(
                ok=False,
                source=self.source,
                detail=f"export file not found: {self._path}",
            )
        try:
            self._ensure_parsed()
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return HealthReport(ok=False, source=self.source, detail=str(exc))

        latest = max((ts for _, ts, _ in self._parsed or []), default=None)
        detail = (
            f"Tidepool JSON, {self._row_count} glucose + {self._insulin_count} bolus rows"
            f" (skipped={self.skipped})"
        )
        return HealthReport(
            ok=True,
            source=self.source,
            detail=detail,
            latest_data_ts=latest,
        )

    def pull(self, since: datetime) -> NormalizedBatch:
        self._ensure_parsed()
        since_utc = since.astimezone(UTC)
        if self._min_ts is not None:
            window_start = min(since_utc, self._min_ts) - _DEDUPE_MARGIN
        else:
            window_start = since_utc - _DEDUPE_MARGIN

        raw_events: list[RawEvent] = []
        glucose: list[GlucoseEvent] = []
        insulin: list[InsulinEvent] = []

        assert self._parsed is not None

        for kind, ts, payload in self._parsed:
            if ts < window_start:
                continue
            source_id = _source_id(kind, ts, payload)
            raw_events.append(
                RawEvent(source=self.source, source_id=source_id, source_ts=ts, payload=payload)
            )
            if kind == "glucose":
                mg_dl = _glucose_mg_dl(payload)
                if mg_dl is not None:
                    glucose.append(GlucoseEvent(ts=ts, mg_dl=mg_dl))
            elif kind == "insulin":
                units = _bolus_units(payload)
                if units is not None and units > 0:
                    insulin.append(
                        InsulinEvent(ts=ts, kind=InsulinKind.BOLUS, units=units, automatic=False)
                    )

        return NormalizedBatch(raw=raw_events, glucose=glucose, insulin=insulin)

    def _ensure_parsed(self) -> None:
        if self._parsed is not None:
            return
        self.skipped = 0
        self._row_count = 0
        self._insulin_count = 0
        self._parsed = []

        text = self._path.read_text(encoding="utf-8-sig")
        doc = json.loads(text)
        records = _records_from_export(doc)
        if not records:
            msg = "Tidepool export contains no device data records"
            raise ValueError(msg)

        for record in records:
            if not isinstance(record, dict):
                self.skipped += 1
                continue
            parsed = _parse_record(record)
            if parsed is None:
                if record.get("type"):
                    self.skipped += 1
                continue
            kind, ts, payload = parsed
            self._parsed.append((kind, ts, payload))
            if kind == "glucose":
                self._row_count += 1
            else:
                self._insulin_count += 1
            if self._min_ts is None or ts < self._min_ts:
                self._min_ts = ts


def _records_from_export(doc: Any) -> list[Any]:
    if isinstance(doc, list):
        return doc
    if isinstance(doc, dict):
        for key in ("data", "records", "dataset"):
            value = doc.get(key)
            if isinstance(value, list):
                return value
    return []


def _parse_record(  # noqa: PLR0911 - one return per Tidepool record type
    record: dict[str, Any],
) -> tuple[str, datetime, dict[str, Any]] | None:
    dtype = record.get("type")
    ts = _parse_time(record.get("time") or record.get("createdTime"))
    if ts is None:
        return None

    if dtype == "cbg":
        mg_dl = _glucose_mg_dl(record)
        if mg_dl is None or mg_dl < 10 or mg_dl > 600:
            return None
        return ("glucose", ts, record)

    if dtype == "smbg":
        mg_dl = _glucose_mg_dl(record)
        if mg_dl is None or mg_dl < 10 or mg_dl > 600:
            return None
        return ("glucose", ts, record)

    if dtype == "bolus":
        units = _bolus_units(record)
        if units is None or units <= 0:
            return None
        return ("insulin", ts, record)

    return None


def _parse_time(raw: Any) -> datetime | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _glucose_mg_dl(record: dict[str, Any]) -> int | None:
    value = record.get("value")
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    units = str(record.get("units", "mg/dL")).lower()
    if "mmol" in units:
        return round(numeric * _MMOL_TO_MGDL)
    return round(numeric)


def _bolus_units(record: dict[str, Any]) -> float | None:
    total = 0.0
    for key in ("normal", "extended"):
        raw = record.get(key)
        if raw is None:
            continue
        try:
            total += float(raw)
        except (TypeError, ValueError):
            continue
    return total if total > 0 else None


def _source_id(kind: str, ts: datetime, payload: dict[str, Any]) -> str:
    record_id = payload.get("id") or payload.get("guid") or ""
    digest = hashlib.sha256(f"{kind}:{record_id}:{ts.isoformat()}".encode()).hexdigest()
    return digest[:32]
