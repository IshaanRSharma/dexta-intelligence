"""CSV file-upload connector — Dexcom Clarity and LibreView exports.

A degenerate :class:`~dexta_intelligence.connectors.base.Connector`: ``pull``
reads the file once, normalizes glucose rows, and returns the same
:class:`~dexta_intelligence.models.RawEvent` + :class:`~dexta_intelligence.models.GlucoseEvent`
batch shape as live sources. Re-uploading the same file is idempotent via
``(source, source_id)`` dedupe.

Supported formats
-----------------
**Dexcom Clarity** (``source = "csv:clarity"``):

- ``Timestamp (YYYY-MM-DDThh:mm:ss)`` — device-local, no timezone
- ``Event Type`` — only ``EGV`` rows are ingested (calibrations skipped)
- ``Glucose Value (mg/dL)``

Extra columns are ignored. Minor header spelling variants are tolerated
(e.g. trailing spaces, alternate parenthetical date hints).

**LibreView** (``source = "csv:libreview"``):

- ``Device Timestamp`` — device-local, no timezone
- ``Historic Glucose mg/dL`` and/or ``Scan Glucose mg/dL``
- ``Record Type`` — ``0`` (historic) and ``1`` (scan) only

LibreView exports often prepend 1-2 metadata rows before the real header;
those are skipped automatically. A UTF-8 BOM is stripped. When headers
say ``mmol/L`` instead of ``mg/dL``, values are converted (x18.016, rounded).

Timezone caveat
---------------
Export timestamps are **device-local without a zone**. Pass an explicit IANA
``tz`` (default ``"UTC"``) so rows become aware UTC via :mod:`zoneinfo`.
Wrong zone shifts every reading — there is no offset embedded in the file.

Watermark / ``since``
---------------------
``pull(since)`` filters to events newer than an effective window start of
``min(since, earliest_file_ts)`` minus a small dedupe margin. That way a
stored sync watermark from a prior upload never excludes rows present in a
new file upload; over-pulling is safe because storage dedupes.
"""

from __future__ import annotations

import csv
import hashlib
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal
from zoneinfo import ZoneInfo

from dexta_intelligence.connectors.base import HealthReport, NormalizedBatch
from dexta_intelligence.models import GlucoseEvent, RawEvent

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

__all__ = ["CSVUploadConnector", "detect_csv_format"]

CsvFormat = Literal["clarity", "libreview"]
FormatHint = Literal["auto", "clarity", "libreview"]

_DEDUPE_MARGIN = timedelta(minutes=5)
_MMOL_TO_MGDL = 18.016

_CLARITY_TS = "timestamp (yyyy-mm-ddthh:mm:ss)"
_CLARITY_EVENT = "event type"
_CLARITY_GLUCOSE = "glucose value (mg/dl)"

_LIBRE_TS = "device timestamp"
_LIBRE_RECORD = "record type"


def detect_csv_format(fieldnames: list[str]) -> CsvFormat:
    """Detect export format from a CSV header row."""
    lower = {name.strip().lower(): name for name in fieldnames if name.strip()}

    if _LIBRE_TS in lower and _LIBRE_RECORD in lower:
        return "libreview"

    if _CLARITY_TS in lower or (
        any("timestamp" in key for key in lower) and _CLARITY_EVENT in lower
    ):
        return "clarity"

    msg = (
        "unrecognized CSV header — expected Dexcom Clarity "
        "(Timestamp / Event Type / Glucose Value) or LibreView "
        "(Device Timestamp / Record Type / Historic or Scan Glucose)"
    )
    raise ValueError(msg)


def _norm_key(name: str) -> str:
    return name.strip().lower()


def _find_column(fieldnames: list[str], *candidates: str) -> str | None:
    lower_map = {_norm_key(name): name for name in fieldnames}
    for candidate in candidates:
        key = candidate.lower()
        if key in lower_map:
            return lower_map[key]
    for candidate in candidates:
        key = candidate.lower()
        for norm, original in lower_map.items():
            if key in norm:
                return original
    return None


def _parse_clarity_ts(raw: str) -> datetime | None:
    text = raw.strip()
    if not text:
        return None
    for fmt in (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is not None:
            parsed = parsed.replace(tzinfo=None)
        return parsed
    except ValueError:
        return None


def _parse_libreview_ts(raw: str) -> datetime | None:
    text = raw.strip()
    if not text:
        return None
    for fmt in (
        "%m/%d/%Y %I:%M:%S %p",
        "%m/%d/%Y %I:%M %p",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _local_to_utc(naive: datetime, tz: ZoneInfo) -> datetime:
    return naive.replace(tzinfo=tz).astimezone(UTC)


def _mg_dl_from_text(raw: str, *, mmol: bool) -> int | None:
    text = raw.strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if mmol:
        return round(value * _MMOL_TO_MGDL)
    return round(value)


def _source_id(fmt: CsvFormat, ts: datetime, mg_dl: int) -> str:
    digest = hashlib.sha256(f"{fmt}:{ts.isoformat()}:{mg_dl}".encode()).hexdigest()
    return digest[:32]


class CSVUploadConnector:
    """Degenerate connector that reads one local CSV export file."""

    def __init__(
        self,
        path: Path,
        *,
        format: FormatHint = "auto",
        tz: str = "UTC",
    ) -> None:
        self._path = path.expanduser()
        self._format_hint: FormatHint = format
        self._tz_name = tz
        self._tz = ZoneInfo(tz)
        self.source = "csv"
        self.skipped: int = 0
        self._detected_format: CsvFormat | None = None
        self._row_count: int = 0
        self._min_ts: datetime | None = None
        self._parsed_rows: list[tuple[datetime, int, dict[str, str], CsvFormat]] | None = None

    def check(self) -> HealthReport:
        """Verify the file exists and parses; report format + glucose row count."""
        if not self._path.is_file():
            return HealthReport(
                ok=False,
                source="csv",
                detail=f"file not found: {self._path}",
            )
        try:
            self._ensure_parsed()
        except (OSError, ValueError, csv.Error) as exc:
            return HealthReport(ok=False, source="csv", detail=str(exc))

        assert self._detected_format is not None
        latest = max((ts for ts, _, _, _ in self._parsed_rows or []), default=None)
        detail = (
            f"{self._detected_format} export, {self._row_count} glucose rows"
            f" (tz={self._tz_name}, skipped={self.skipped})"
        )
        return HealthReport(
            ok=True,
            source=self.source,
            detail=detail,
            latest_data_ts=latest,
        )

    def pull(self, since: datetime) -> NormalizedBatch:
        """Read the file once and return rows newer than the effective window."""
        self._ensure_parsed()
        since_utc = since.astimezone(UTC)
        if self._min_ts is not None:
            window_start = min(since_utc, self._min_ts) - _DEDUPE_MARGIN
        else:
            window_start = since_utc - _DEDUPE_MARGIN

        raw_events: list[RawEvent] = []
        glucose: list[GlucoseEvent] = []

        assert self._parsed_rows is not None

        for ts, mg_dl, payload, row_fmt in self._parsed_rows:
            if ts < window_start:
                continue
            source_id = _source_id(row_fmt, ts, mg_dl)
            raw_events.append(
                RawEvent(source=self.source, source_id=source_id, source_ts=ts, payload=payload)
            )
            glucose.append(GlucoseEvent(ts=ts, mg_dl=mg_dl))

        return NormalizedBatch(raw=raw_events, glucose=glucose)

    def _ensure_parsed(self) -> None:
        if self._parsed_rows is not None:
            return
        self.skipped = 0
        self._parsed_rows = []
        try:
            text = self._path.read_text(encoding="utf-8-sig")
        except OSError:
            raise
        reader = csv.reader(text.splitlines())
        fieldnames, data_rows = _read_data_rows(reader)
        fmt = self._resolve_format(fieldnames)
        self._detected_format = fmt
        self.source = f"csv:{fmt}"

        if fmt == "clarity":
            parsed = _parse_clarity_rows(data_rows, fieldnames, self._tz)
        else:
            parsed = _parse_libreview_rows(data_rows, fieldnames, self._tz)

        for item in parsed:
            if item is None:
                self.skipped += 1
                continue
            ts, mg_dl, payload = item
            self._parsed_rows.append((ts, mg_dl, payload, fmt))
            self._row_count += 1
            if self._min_ts is None or ts < self._min_ts:
                self._min_ts = ts

    def _resolve_format(self, fieldnames: list[str]) -> CsvFormat:
        if self._format_hint == "auto":
            return detect_csv_format(fieldnames)
        if self._format_hint == "clarity":
            return "clarity"
        return "libreview"


def _read_data_rows(reader: Iterable[list[str]]) -> tuple[list[str], list[list[str]]]:
    """Skip preamble rows until a recognizable header, then return data rows."""
    fieldnames: list[str] | None = None
    data_rows: list[list[str]] = []
    for row in reader:
        if not row or not any(cell.strip() for cell in row):
            continue
        if fieldnames is None:
            try:
                detect_csv_format(row)
            except ValueError:
                continue
            fieldnames = row
            continue
        data_rows.append(row)
    if fieldnames is None:
        msg = "no recognizable CSV header row found"
        raise ValueError(msg)
    return fieldnames, data_rows


def _row_dict(fieldnames: list[str], row: list[str]) -> dict[str, str]:
    padded = row + [""] * (len(fieldnames) - len(row))
    return dict(zip(fieldnames, padded[: len(fieldnames)], strict=False))


def _parse_clarity_rows(
    data_rows: list[list[str]],
    fieldnames: list[str],
    tz: ZoneInfo,
) -> list[tuple[datetime, int, dict[str, str]] | None]:
    ts_col = _find_column(fieldnames, _CLARITY_TS, "Timestamp")
    event_col = _find_column(fieldnames, _CLARITY_EVENT, "Event Type")
    glucose_col = _find_column(fieldnames, _CLARITY_GLUCOSE, "Glucose Value")
    if ts_col is None or event_col is None or glucose_col is None:
        msg = "Clarity CSV missing required columns (Timestamp, Event Type, Glucose Value)"
        raise ValueError(msg)

    results: list[tuple[datetime, int, dict[str, str]] | None] = []
    for row in data_rows:
        cells = _row_dict(fieldnames, row)
        if cells.get(event_col, "").strip().upper() != "EGV":
            continue
        naive = _parse_clarity_ts(cells.get(ts_col, ""))
        mg_dl = _mg_dl_from_text(cells.get(glucose_col, ""), mmol=False)
        if naive is None or mg_dl is None or mg_dl < 10 or mg_dl > 600:
            if cells.get(ts_col, "").strip() or cells.get(glucose_col, "").strip():
                results.append(None)
            continue
        ts = _local_to_utc(naive, tz)
        results.append((ts, mg_dl, cells))
    return results


def _libreview_mmol(fieldnames: list[str]) -> bool:
    for name in fieldnames:
        lower = _norm_key(name)
        if "glucose" in lower and "mmol/l" in lower:
            return True
    return False


def _parse_libreview_rows(
    data_rows: list[list[str]],
    fieldnames: list[str],
    tz: ZoneInfo,
) -> list[tuple[datetime, int, dict[str, str]] | None]:
    ts_col = _find_column(fieldnames, _LIBRE_TS, "Device Timestamp")
    record_col = _find_column(fieldnames, _LIBRE_RECORD, "Record Type")
    historic_col = _find_column(
        fieldnames,
        "Historic Glucose mg/dL",
        "Historic Glucose mmol/L",
        "Historic Glucose",
    )
    scan_col = _find_column(
        fieldnames,
        "Scan Glucose mg/dL",
        "Scan Glucose mmol/L",
        "Scan Glucose",
    )
    if ts_col is None or record_col is None:
        msg = "LibreView CSV missing required columns (Device Timestamp, Record Type)"
        raise ValueError(msg)

    mmol = _libreview_mmol(fieldnames)
    results: list[tuple[datetime, int, dict[str, str]] | None] = []
    for row in data_rows:
        cells = _row_dict(fieldnames, row)
        record_raw = cells.get(record_col, "").strip()
        if record_raw not in ("0", "1"):
            continue
        naive = _parse_libreview_ts(cells.get(ts_col, ""))
        glucose_raw = ""
        if record_raw == "0":
            glucose_raw = cells.get(historic_col or "", "")
        else:
            glucose_raw = cells.get(scan_col or "", "") or cells.get(historic_col or "", "")
        mg_dl = _mg_dl_from_text(glucose_raw, mmol=mmol)
        if naive is None or mg_dl is None or mg_dl < 10 or mg_dl > 600:
            if cells.get(ts_col, "").strip():
                results.append(None)
            continue
        ts = _local_to_utc(naive, tz)
        results.append((ts, mg_dl, cells))
    return results
