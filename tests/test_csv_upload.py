"""Tests for the CSV file-upload connector."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from dexta_intelligence.connectors.csv_upload import CSVUploadConnector, detect_csv_format
from dexta_intelligence.models import RawEvent

FIXTURES = Path(__file__).parent / "fixtures"
CLARITY = FIXTURES / "clarity_sample.csv"
LIBREVIEW = FIXTURES / "libreview_sample.csv"
LIBREVIEW_MMOL = FIXTURES / "libreview_mmol_sample.csv"
CLARITY_BOM = FIXTURES / "clarity_bom_sample.csv"

EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
NY = ZoneInfo("America/New_York")


class TestDetectFormat:
    def test_clarity_header(self) -> None:
        header = [
            "Timestamp (YYYY-MM-DDThh:mm:ss)",
            "Event Type",
            "Glucose Value (mg/dL)",
        ]
        assert detect_csv_format(header) == "clarity"

    def test_libreview_header(self) -> None:
        header = [
            "Device Timestamp",
            "Historic Glucose mg/dL",
            "Scan Glucose mg/dL",
            "Record Type",
        ]
        assert detect_csv_format(header) == "libreview"

    def test_unknown_header_raises(self) -> None:
        with pytest.raises(ValueError, match="unrecognized CSV header"):
            detect_csv_format(["foo", "bar"])


class TestClarityConnector:
    def test_round_trip_count_and_values(self) -> None:
        conn = CSVUploadConnector(CLARITY, tz="America/New_York")
        batch = conn.pull(EPOCH)

        assert conn.source == "csv:clarity"
        assert len(batch.glucose) == 13
        assert len(batch.raw) == 13
        assert conn.skipped == 1

        first = batch.glucose[0]
        assert first.mg_dl == 120
        assert first.ts == datetime(2024, 3, 10, 6, 30, tzinfo=UTC)

    def test_dst_boundary_utc_conversion(self) -> None:
        conn = CSVUploadConnector(CLARITY, tz="America/New_York")
        batch = conn.pull(EPOCH)

        by_local = {
            g.ts.astimezone(NY).strftime("%Y-%m-%d %H:%M"): g for g in batch.glucose
        }
        assert by_local["2024-03-10 01:30"].ts == datetime(2024, 3, 10, 6, 30, tzinfo=UTC)
        assert by_local["2024-03-10 03:00"].ts == datetime(2024, 3, 10, 7, 0, tzinfo=UTC)

    def test_auto_detection(self) -> None:
        conn = CSVUploadConnector(CLARITY, format="auto")
        report = conn.check()
        assert report.ok
        assert conn.source == "csv:clarity"
        assert "clarity export" in report.detail

    def test_bom_tolerance(self) -> None:
        conn = CSVUploadConnector(CLARITY_BOM)
        batch = conn.pull(EPOCH)
        assert len(batch.glucose) == 2

    def test_idempotent_source_ids(self) -> None:
        conn = CSVUploadConnector(CLARITY)
        first = conn.pull(EPOCH)
        second = conn.pull(EPOCH)
        assert [r.source_id for r in first.raw] == [r.source_id for r in second.raw]

    def test_check_missing_file(self, tmp_path: Path) -> None:
        conn = CSVUploadConnector(tmp_path / "missing.csv")
        report = conn.check()
        assert not report.ok
        assert "file not found" in report.detail


class TestLibreViewConnector:
    def test_round_trip_with_preamble(self) -> None:
        conn = CSVUploadConnector(LIBREVIEW, tz="America/New_York")
        batch = conn.pull(EPOCH)

        assert conn.source == "csv:libreview"
        assert len(batch.glucose) == 13
        assert conn.skipped == 1

        scan = next(g for g in batch.glucose if g.mg_dl == 149)
        assert scan.ts.astimezone(NY).strftime("%Y-%m-%d %I:%M %p") == "2024-03-12 10:30 AM"

    def test_auto_detection(self) -> None:
        conn = CSVUploadConnector(LIBREVIEW, format="auto")
        report = conn.check()
        assert report.ok
        assert "libreview export" in report.detail

    def test_mmol_conversion(self) -> None:
        conn = CSVUploadConnector(LIBREVIEW_MMOL, tz="UTC")
        batch = conn.pull(EPOCH)
        values = sorted(g.mg_dl for g in batch.glucose)
        assert values == [117, 121, 130]

    def test_watermark_does_not_exclude_file_rows(self) -> None:
        conn = CSVUploadConnector(CLARITY, tz="UTC")
        recent_since = datetime(2025, 1, 1, tzinfo=UTC)
        batch = conn.pull(recent_since)
        assert len(batch.glucose) == 13


class TestPullFiltering:
    def test_full_file_even_when_since_within_range(self) -> None:
        """Upload-safe: min(since, file_min) never hides file rows behind a watermark."""
        conn = CSVUploadConnector(CLARITY, tz="UTC")
        mid = datetime(2024, 3, 12, 0, 0, tzinfo=UTC)
        batch = conn.pull(mid)
        assert len(batch.glucose) == 13

    def test_raw_payload_preserved(self) -> None:
        conn = CSVUploadConnector(CLARITY)
        batch = conn.pull(EPOCH)
        assert all(isinstance(r, RawEvent) for r in batch.raw)
        assert batch.raw[0].payload.get("Event Type") == "EGV"
