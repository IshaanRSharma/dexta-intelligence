"""dexta CLI — init, doctor, sync, analyze."""

from __future__ import annotations

import argparse
import functools
import sys
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from dexta_intelligence.agents.base import AgentContext, AgentRegistry
from dexta_intelligence.coldstart import CAPABILITY_GATES, HARD_FLOOR_DAYS, ColdStartReport
from dexta_intelligence.config import DEFAULT_CONFIG_PATH, Config, load_config
from dexta_intelligence.connectors.base import Connector, HealthReport
from dexta_intelligence.store import SQLiteStore
from dexta_intelligence.store.port import StoragePort
from dexta_intelligence.workflows.sync import SyncReport, sync_all

if TYPE_CHECKING:
    from dexta_intelligence.models import Finding

__all__ = ["main"]

ConnectorFactory = Callable[[Config], list[Connector]]
StoreOpener = Callable[[Config, Path | None], StoragePort]

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

[llm]
provider = "anthropic"
model = "claude-sonnet-4-20250514"
# Per-role overrides live under [llm.roles.<role>] (optional).

[analysis]
target_low = 70
target_high = 180
deep_analysis_window_days = 90
"""

_DEFAULT_REGISTRY = AgentRegistry()


@functools.lru_cache(maxsize=1)
def get_registry() -> AgentRegistry:
    try:
        from dexta_intelligence.agents.reconciliation import (  # noqa: PLC0415
            register_reconciliation,
        )

        register_reconciliation(_DEFAULT_REGISTRY)
    except (ImportError, Exception):
        pass
    return _DEFAULT_REGISTRY


def resolve_config_path(explicit: Path | None) -> Path:
    """Resolve the config file: explicit flag, then ./dexta.toml, then the default."""
    if explicit is not None:
        return explicit.expanduser()
    local = Path("dexta.toml")
    if local.is_file():
        return local
    return DEFAULT_CONFIG_PATH.expanduser()


def init_config_path(explicit: Path | None) -> Path:
    """Default init target is ./dexta.toml unless --config is given."""
    if explicit is not None:
        return explicit.expanduser()
    return Path("dexta.toml")


def is_nightscout_configured(config: Config) -> bool:
    return bool(config.nightscout.url.strip() and config.nightscout.token.strip())


def is_dexcom_configured(config: Config) -> bool:
    return bool(config.dexcom.username.strip() and config.dexcom.password)


def is_whoop_configured(config: Config) -> bool:
    return bool(config.whoop.access_token.strip())


def is_libre_configured(config: Config) -> bool:
    return bool(config.libre.email.strip() and config.libre.password)


def build_connectors(config: Config) -> list[Connector]:
    """Construct every configured connector (lazy provider imports)."""
    connectors: list[Connector] = []

    if is_nightscout_configured(config):
        from dexta_intelligence.connectors.nightscout import (  # noqa: PLC0415
            NightscoutConnector,
        )

        connectors.append(NightscoutConnector(config.nightscout))

    if is_dexcom_configured(config):
        from dexta_intelligence.connectors.dexcom import DexcomConnector  # noqa: PLC0415

        connectors.append(DexcomConnector(config.dexcom))

    if is_whoop_configured(config):
        from dexta_intelligence.connectors.whoop import WhoopConnector  # noqa: PLC0415

        connectors.append(WhoopConnector(config.whoop))

    if is_libre_configured(config):
        from dexta_intelligence.connectors.libre import LibreConnector  # noqa: PLC0415

        connectors.append(LibreConnector(config.libre))

    return connectors


def open_sqlite_store(config: Config, db_override: Path | None = None) -> SQLiteStore:
    """Open the configured SQLite backend and ensure schema exists."""
    if config.data.backend != "sqlite":
        msg = (
            f"CLI quick-start supports sqlite only (backend={config.data.backend!r}). "
            "Use the library API for postgres deployments."
        )
        raise RuntimeError(msg)
    path = (db_override or config.data.sqlite_path).expanduser()
    store = SQLiteStore(path)
    store.migrate()
    return store


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


def _maybe_close_store(store: StoragePort, opener: StoreOpener) -> None:
    if opener is open_sqlite_store and hasattr(store, "close"):
        store.close()


def _check_connector(connector: Connector) -> HealthReport:
    try:
        return connector.check()
    except RuntimeError as exc:
        return HealthReport(ok=False, source=connector.source, detail=str(exc))


def _format_ts(ts: datetime | None) -> str:
    if ts is None:
        return "—"
    return ts.astimezone(UTC).isoformat()


def _print_coverage(out: TextIO, gates: ColdStartReport) -> None:
    cov = gates.coverage
    out.write("\nCoverage\n")
    out.write(f"  span: {cov.span_days:.1f} days")
    if cov.first_ts is not None and cov.last_ts is not None:
        out.write(f" ({cov.first_ts.date()} → {cov.last_ts.date()})")
    out.write("\n")
    out.write(f"  glucose: {cov.n_glucose} readings ({cov.glucose_coverage_pct:.0f}% coverage)\n")
    out.write(f"  insulin: {cov.n_insulin} events ({cov.days_with_insulin_pct:.0f}% of days)\n")
    out.write(f"  sleep: {cov.n_sleep}  activity: {cov.n_activity}\n")

    out.write("\nCapabilities\n")
    for gate in CAPABILITY_GATES:
        if gate.capability in gates.unlocked:
            out.write(f"  ✓ {gate.capability}: {gate.description}\n")
        else:
            pending = gates.pending.get(gate.capability, gate.description)
            out.write(f"  ○ {gate.capability}: {pending}\n")


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


def _analysis_window(config: Config, coverage_end: date | None) -> tuple[date, date]:
    end = coverage_end or datetime.now(tz=UTC).date()
    start = end - timedelta(days=config.analysis.deep_analysis_window_days)
    return start, end


def _run_agents(
    registry: AgentRegistry,
    ctx: AgentContext,
    out: TextIO,
) -> list[Finding]:
    findings: list[Finding] = []
    for agent in registry:
        reasons = agent.requires.unmet_reasons(ctx.gates)
        if reasons:
            out.write(f"  skipped {agent.name}: {'; '.join(reasons)}\n")
            continue
        try:
            agent_findings = agent.run(ctx)
            findings.extend(agent_findings)
        except Exception as exc:
            out.write(f"  ✗ {agent.name}: {type(exc).__name__}: {exc}\n")
    return findings


def _print_finding(out: TextIO, finding: Finding, *, persisted_id: int | None = None) -> None:
    out.write(f"\n  agent: {finding.agent}\n")
    out.write(f"  kind: {finding.kind}\n")
    out.write(f"  status: {finding.status.value}\n")
    out.write(f"  summary: {finding.headline}\n")
    stats = finding.stats
    stat_bits: list[str] = []
    if stats.n is not None:
        stat_bits.append(f"n={stats.n}")
    if stats.effect_size is not None:
        stat_bits.append(f"effect={stats.effect_size}")
    if stats.p_perm is not None:
        stat_bits.append(f"p={stats.p_perm}")
    if stat_bits:
        out.write(f"  evidence stats: {', '.join(stat_bits)}\n")
    if persisted_id is not None:
        out.write(f"  persisted id: {persisted_id}\n")


def cmd_analyze(
    *,
    config: Config,
    db_path: Path | None,
    out: TextIO,
    opener: StoreOpener = open_sqlite_store,
    registry: AgentRegistry | None = None,
) -> int:
    active_registry = registry if registry is not None else get_registry()
    agents = list(active_registry)
    if not agents:
        out.write("No agents registered — nothing to analyze.\n")
        return 0

    store = opener(config, db_path)
    try:
        coverage = store.coverage()
        gates = ColdStartReport.from_coverage(coverage)
        _print_coverage(out, gates)

        if gates.below_hard_floor:
            out.write(
                f"\nNeed at least {HARD_FLOOR_DAYS:.0f} days of data before analysis "
                f"(have {coverage.span_days:.1f}).\n"
            )
            return 1

        end_date = coverage.last_ts.date() if coverage.last_ts is not None else None
        window = _analysis_window(config, end_date)
        ctx = AgentContext(
            store=store,
            window=window,
            gates=gates,
            run_id=str(uuid.uuid4()),
        )

        out.write(f"\nRunning {len(agents)} agent(s) (run {ctx.run_id})…\n")
        findings = _run_agents(active_registry, ctx, out)

        if not findings:
            out.write("\nNo findings produced.\n")
            return 0

        out.write("\nFindings\n")
        for finding in findings:
            finding_id = store.insert_finding(finding)
            _print_finding(out, finding, persisted_id=finding_id)
    finally:
        _maybe_close_store(store, opener)

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dexta",
        description="Continuous health intelligence for Type 1 diabetes.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to dexta.toml (default: ./dexta.toml if present, else ~/.dexta/dexta.toml)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Override the SQLite database path from config",
    )

    sub = parser.add_subparsers(dest="command", required=False)

    init_p = sub.add_parser("init", help="Write starter dexta.toml and create the SQLite database")
    init_p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing config file",
    )

    sub.add_parser("doctor", help="Connectivity, auth, and coverage checks")
    sub.add_parser("sync", help="Pull from configured data sources")
    sub.add_parser("analyze", help="Run the agent harness on stored data")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command is None:
        parser.print_help()
        return 0

    config_path = resolve_config_path(args.config)
    config = load_config(config_path if config_path.is_file() else args.config)

    if args.command == "init":
        target = init_config_path(args.config)
        return cmd_init(
            config_path=target,
            db_path=args.db,
            force=args.force,
            out=sys.stdout,
        )

    if args.command == "doctor":
        return cmd_doctor(
            config=config,
            db_path=args.db,
            out=sys.stdout,
        )

    if args.command == "sync":
        return cmd_sync(
            config=config,
            db_path=args.db,
            out=sys.stdout,
        )

    if args.command == "analyze":
        return cmd_analyze(
            config=config,
            db_path=args.db,
            out=sys.stdout,
        )

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
