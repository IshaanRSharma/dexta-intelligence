"""Shared CLI helpers: store/connector wiring, config resolution, coverage output."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, TextIO

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.coldstart import CAPABILITY_GATES, ColdStartReport
from dexta_intelligence.config import DEFAULT_CONFIG_PATH, Config
from dexta_intelligence.connectors.base import Connector
from dexta_intelligence.store import SQLiteStore
from dexta_intelligence.store.port import StoragePort

logger = logging.getLogger("dexta_intelligence.cli")

ConnectorFactory = Callable[[Config], list[Connector]]
StoreOpener = Callable[[Config, Path | None], StoragePort]

CsvFormatHint = Literal["auto", "clarity", "libreview", "tidepool"]

#: One-line runtime safety footer for clinical CLI surfaces (see MEDICAL_DISCLAIMER.md).
MEDICAL_DISCLAIMER = (
    "- dexta is observation and discussion support, not a medical device; it never gives "
    "dosing advice. Decisions are for you and your care team."
)


def model_for_role(config: Config, role: str) -> Any:
    """Build the LLM for ``role`` from config, or ``None`` for the deterministic path.

    Returns ``None`` (never raises) when the ``llm`` extra is missing or no
    provider credential is set, so every command always runs - with LLM
    reasoning when a key is present, with the deterministic fallback otherwise.
    Sampling and per-role overrides resolve through ``llm.factory.resolve_spec``.
    """
    try:
        from dexta_intelligence.llm.factory import get_model, resolve_spec  # noqa: PLC0415

        spec = resolve_spec(
            role,
            provider=config.llm.provider,
            model=config.llm.model,
            role_overrides=config.llm.roles,
        )
        return get_model(spec)
    except Exception:
        logger.debug("LLM for role %r unavailable; using deterministic path", role, exc_info=True)
        return None


def discovery_model(config: Config) -> Any:
    """Back-compat wrapper: the discovery-role model (used by analyze/lenses)."""
    return model_for_role(config, "discovery")


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


def is_oura_configured(config: Config) -> bool:
    return bool(config.oura.access_token.strip())


def is_tidepool_configured(config: Config) -> bool:
    path = config.tidepool.export_path.expanduser()
    return bool(str(path)) and path.is_file()


def is_tandem_configured(config: Config) -> bool:
    return bool(config.tandem.email.strip() and config.tandem.password)


def is_carelink_configured(config: Config) -> bool:
    return bool(config.carelink.username.strip() and config.carelink.password)


def is_dexcom_api_configured(config: Config) -> bool:
    return bool(config.dexcom_api.access_token.strip())


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

    if is_oura_configured(config):
        from dexta_intelligence.connectors.oura import OuraConnector  # noqa: PLC0415

        connectors.append(OuraConnector(config.oura))

    if is_tidepool_configured(config):
        from dexta_intelligence.connectors.tidepool import TidepoolConnector  # noqa: PLC0415

        connectors.append(TidepoolConnector(config.tidepool))

    if is_tandem_configured(config):
        from dexta_intelligence.connectors.tandem import TandemConnector  # noqa: PLC0415

        connectors.append(TandemConnector(config.tandem))

    if is_carelink_configured(config):
        from dexta_intelligence.connectors.carelink import CareLinkConnector  # noqa: PLC0415

        connectors.append(CareLinkConnector(config.carelink))

    if is_dexcom_api_configured(config):
        from dexta_intelligence.connectors.dexcom_api import DexcomApiConnector  # noqa: PLC0415

        connectors.append(DexcomApiConnector(config.dexcom_api))

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


def _maybe_close_store(store: StoragePort, opener: StoreOpener) -> None:
    if opener is open_sqlite_store and hasattr(store, "close"):
        store.close()


def _format_ts(ts: datetime | None) -> str:
    if ts is None:
        return "-"
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


def _analysis_window(
    config: Config,
    coverage_end: date | None,
    window_days: int | None = None,
) -> tuple[date, date]:
    end = coverage_end or datetime.now(tz=UTC).date()
    days = window_days if window_days is not None else config.analysis.deep_analysis_window_days
    start = end - timedelta(days=days)
    return start, end


def _ctx_for(config: Config, store: StoragePort) -> AgentContext:
    coverage = store.coverage()
    gates = ColdStartReport.from_coverage(coverage)
    end_date = coverage.last_ts.date() if coverage.last_ts is not None else None
    return AgentContext(
        store=store,
        window=_analysis_window(config, end_date),
        gates=gates,
        timezone=config.analysis.timezone,
        run_id=str(uuid.uuid4()),
    )
