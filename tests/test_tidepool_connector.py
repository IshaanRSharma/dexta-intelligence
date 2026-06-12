"""Tidepool JSON export connector tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from dexta_intelligence.config import TidepoolConfig
from dexta_intelligence.connectors.tidepool import TidepoolConnector

FIXTURE = Path(__file__).parent / "fixtures" / "tidepool_sample.json"


def test_check_parses_export() -> None:
    conn = TidepoolConnector(TidepoolConfig(export_path=FIXTURE))
    report = conn.check()
    assert report.ok is True
    assert report.source == "tidepool"
    assert "glucose" in report.detail


def test_pull_glucose_and_insulin() -> None:
    conn = TidepoolConnector(TidepoolConfig(export_path=FIXTURE))
    batch = conn.pull(datetime(2024, 1, 1, tzinfo=UTC))
    assert len(batch.glucose) == 2
    assert len(batch.insulin) == 1
    assert batch.glucose[0].mg_dl == 112
    assert batch.glucose[1].mg_dl == pytest.approx(110, abs=2)
    assert batch.insulin[0].units == 4.5


def test_missing_file() -> None:
    conn = TidepoolConnector(TidepoolConfig(export_path=Path("/no/such/file.json")))
    report = conn.check()
    assert report.ok is False
