"""Data-plane commands: init, doctor, sync, upload."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, TextIO

from dexta_intelligence.cli._common import (
    ConnectorFactory,
    CsvFormatHint,
    StoreOpener,
    _format_ts,
    _maybe_close_store,
    _print_coverage,
    build_connectors,
    open_sqlite_store,
)
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.config import Config, load_config
from dexta_intelligence.connectors.base import Connector, HealthReport
from dexta_intelligence.workflows.sync import SyncReport, sync, sync_all

if TYPE_CHECKING:
    from pathlib import Path

_INIT_TEMPLATE = """\
# dexta-intelligence configuration
# Secrets belong in the environment — see comments per section.

[data]
backend = "sqlite"
sqlite_path = "~/.dexta/dexta.db"
# database_url = "postgresql://..."  # or set DATABASE_URL and backend = "postgres"

[nightscout]
url = ""    # your Nightscout base URL (or NIGHTSCOUT_URL)
token = ""  # API token (or NIGHTSCOUT_TOKEN)

[whoop]
access_token = ""  # or WHOOP_ACCESS_TOKEN
# refresh_token = ""  # WHOOP_REFRESH_TOKEN — enables auto-refresh on 401
# client_id = ""      # WHOOP_CLIENT_ID
# client_secret = ""  # WHOOP_CLIENT_SECRET

[dexcom]
username = ""  # or DEXCOM_USERNAME
password = ""  # or DEXCOM_PASSWORD
ous = false    # true for accounts outside the US (DEXCOM_OUS)

[libre]
email = ""     # or LIBRE_EMAIL
password = ""  # or LIBRE_PASSWORD
region = "us"  # us | eu | eu2 | ae | ap | au | ca | de | fr | jp | la | ru
patient_id = ""  # empty = first shared patient

[oura]
access_token = ""  # personal access token (or OURA_ACCESS_TOKEN)

[llm]
provider = "anthropic"
model = "claude-sonnet-4-20250514"
# Per-role overrides live under [llm.roles.<role>] (optional).

[analysis]
target_low = 70
target_high = 180
deep_analysis_window_days = 90
"""


def cmd_init(
    *,
    config_path: Path,
    db_path: Path | None,
    force: bool,
    out: TextIO,
    opener: StoreOpener = open_sqlite_store,
) -> int:
    if config_path.is_file() and not force:
        out.write(f"Refusing to overwrite existing config: {config_path}\n")
        out.write("Re-run with --force to replace it.\n")
        return 1

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(_INIT_TEMPLATE, encoding="utf-8")

    config = load_config(config_path)
    db_location = (db_path or config.data.sqlite_path).expanduser()
    store = opener(config, db_path)
    try:
        out.write(f"Created database at {db_location}\n")
    finally:
        _maybe_close_store(store, opener)

    out.write(f"Wrote config to {config_path}\n")
    out.write("\nNext steps:\n")
    out.write("  1. Edit [nightscout] url + token (or set NIGHTSCOUT_URL / NIGHTSCOUT_TOKEN)\n")
    out.write("  2. dexta doctor   # connectivity + coverage\n")
    out.write("  3. dexta sync     # pull configured sources\n")
    out.write("  4. dexta analyze  # run the agent harness\n")
    return 0


def _check_connector(connector: Connector) -> HealthReport:
    try:
        return connector.check()
    except RuntimeError as exc:
        return HealthReport(ok=False, source=connector.source, detail=str(exc))


def cmd_doctor(
    *,
    config: Config,
    db_path: Path | None,
    out: TextIO,
    connector_factory: ConnectorFactory = build_connectors,
    opener: StoreOpener = open_sqlite_store,
) -> int:
    out.write("dexta doctor\n\n")
    exit_code = 0

    try:
        store = opener(config, db_path)
    except Exception as exc:
        out.write(f"✗ database: {type(exc).__name__}: {exc}\n")
        return 1

    try:
        coverage = store.coverage()
        gates = ColdStartReport.from_coverage(coverage)
        out.write(f"✓ database: reachable ({coverage.n_glucose} glucose events)\n")
        _print_coverage(out, gates)

        connectors = connector_factory(config)
        if not connectors:
            out.write("\nNo data sources configured.\n")
            out.write("Set [nightscout] url + token, or another connector section, then re-run.\n")
            return 0

        out.write("\nSources\n")
        for connector in connectors:
            report = _check_connector(connector)
            mark = "✓" if report.ok else "✗"
            detail = report.detail or ("ok" if report.ok else "check failed")
            out.write(f"  {mark} {report.source}: {detail}")
            if report.latest_data_ts is not None:
                out.write(f" (latest {_format_ts(report.latest_data_ts)})")
            out.write("\n")
            if not report.ok:
                exit_code = 1
    finally:
        _maybe_close_store(store, opener)

    return exit_code


def _format_sync_report(out: TextIO, report: SyncReport) -> None:
    out.write(f"\n{report.source}\n")
    if report.since is not None:
        out.write(f"  window: {_format_ts(report.since)} → {_format_ts(report.until)}\n")
    else:
        out.write(f"  until: {_format_ts(report.until)}\n")
    if report.errors:
        for err in report.errors:
            out.write(f"  ✗ {err}\n")
        return

    out.write(f"  raw new: {report.raw_new}\n")
    if report.inserted:
        parts = ", ".join(f"{k}={v}" for k, v in sorted(report.inserted.items()) if v)
        out.write(f"  inserted: {parts or 'none'}\n")
    out.write(f"  rollup days: {report.rollup_days}\n")
    if report.notes:
        for note in report.notes:
            out.write(f"  note: {note}\n")
    out.write("  ✓ ok\n")


def cmd_upload(
    *,
    path: Path,
    config: Config,
    db_path: Path | None,
    csv_format: CsvFormatHint,
    tz: str,
    out: TextIO,
    opener: StoreOpener = open_sqlite_store,
    now: datetime | None = None,
) -> int:
    if csv_format == "tidepool" or (csv_format == "auto" and path.suffix.lower() == ".json"):
        from dexta_intelligence.config import TidepoolConfig  # noqa: PLC0415
        from dexta_intelligence.connectors.tidepool import TidepoolConnector  # noqa: PLC0415

        connector = TidepoolConnector(TidepoolConfig(export_path=path))
    else:
        from dexta_intelligence.connectors.csv_upload import CSVUploadConnector  # noqa: PLC0415

        connector = CSVUploadConnector(path, format=csv_format, tz=tz)
    health = connector.check()
    if not health.ok:
        out.write(f"✗ upload: {health.detail}\n")
        return 1

    store = opener(config, db_path)
    try:
        report = sync(
            connector,
            store,
            default_lookback=timedelta(days=365 * 20),
            now=now,
        )
    except Exception as exc:
        out.write(f"✗ upload: {type(exc).__name__}: {exc}\n")
        return 1
    finally:
        _maybe_close_store(store, opener)

    out.write("dexta upload\n")
    _format_sync_report(out, report)
    if connector.skipped:
        out.write(f"  note: {connector.skipped} row(s) skipped during parse\n")
    return 0 if report.ok else 1


def cmd_sync(
    *,
    config: Config,
    db_path: Path | None,
    out: TextIO,
    connector_factory: ConnectorFactory = build_connectors,
    opener: StoreOpener = open_sqlite_store,
    now: datetime | None = None,
) -> int:
    connectors = connector_factory(config)
    if not connectors:
        out.write("No data sources configured — nothing to sync.\n")
        return 0

    store = opener(config, db_path)
    try:
        reports = sync_all(connectors, store, now=now)
    finally:
        _maybe_close_store(store, opener)

    out.write("dexta sync\n")
    for report in reports:
        _format_sync_report(out, report)

    successes = sum(1 for report in reports if report.ok)
    if successes == 0:
        out.write("\nAll sources failed.\n")
        return 1
    return 0
