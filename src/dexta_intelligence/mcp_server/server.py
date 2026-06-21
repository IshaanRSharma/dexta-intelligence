"""FastMCP wrapper for the glucose-over-MCP v1 tool contract."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Protocol

from dexta_intelligence.analytics.rollups import (
    TARGET_HIGH_MG_DL,
    TARGET_LOW_MG_DL,
    VERY_HIGH_MG_DL,
    VERY_LOW_MG_DL,
)
from dexta_intelligence.coldstart import CapabilitySet
from dexta_intelligence.config import Config, load_config
from dexta_intelligence.mcp_server import contract

if TYPE_CHECKING:
    from pathlib import Path

    from fastmcp import FastMCP

    from dexta_intelligence.connectors.base import RealtimeConnector
    from dexta_intelligence.store.port import StoragePort

__all__ = [
    "INSULIN_TOOL_NAMES",
    "TOOL_NAMES",
    "build_realtime_connector",
    "build_server",
    "main",
]

TOOL_NAMES: tuple[str, ...] = (
    "get_current_glucose",
    "get_glucose_readings",
    "get_statistics",
    "get_status_summary",
    "detect_episodes",
    "get_episode_details",
    "analyze_time_blocks",
    "check_alerts",
    "export_data",
    "get_agp_report",
)

#: Insulin extension - registered only when the store holds insulin data.
INSULIN_TOOL_NAMES: tuple[str, ...] = (
    "get_boluses",
    "get_carb_entries",
    "get_basal_timeline",
    "get_iob",
)


class _RealtimeConnectorFactory(Protocol):
    def __call__(self, config: Config) -> RealtimeConnector | None: ...


def _import_fastmcp() -> type[FastMCP]:
    try:
        from fastmcp import FastMCP  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - import-path guard
        msg = (
            "MCP support is not installed. "
            "Install it with: pip install 'dexta-intelligence[mcp]'"
        )
        raise RuntimeError(msg) from exc
    return FastMCP


def _parse_dt(value: str) -> datetime:
    return contract._require_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))


def _is_dexcom_configured(config: Config) -> bool:
    return bool(config.dexcom.username.strip() and config.dexcom.password)


def _is_libre_configured(config: Config) -> bool:
    return bool(config.libre.email.strip() and config.libre.password)


def build_realtime_connector(config: Config) -> RealtimeConnector | None:
    """Construct the first configured RealtimeConnector, if any."""
    if _is_dexcom_configured(config):
        from dexta_intelligence.connectors.dexcom import DexcomConnector  # noqa: PLC0415

        return DexcomConnector(config.dexcom)
    if _is_libre_configured(config):
        from dexta_intelligence.connectors.libre import LibreConnector  # noqa: PLC0415

        return LibreConnector(config.libre)
    return None


def _capabilities(store: StoragePort) -> CapabilitySet:
    """Stream presence from store coverage - decides what the server exposes."""
    coverage = store.coverage()
    return CapabilitySet(
        has_insulin=coverage.n_insulin > 0,
        has_meals=coverage.n_meals > 0,
        has_sleep=coverage.n_sleep > 0,
        has_activity=coverage.n_activity > 0,
    )


def _register_insulin_tools(mcp: FastMCP, store: StoragePort) -> None:
    """The 4 read-only insulin-extension tools. Bad args → ``{"error": ...}``."""

    @mcp.tool(name="get_boluses", run_in_thread=False)
    def get_boluses(start: str, end: str) -> dict[str, Any]:
        """Bolus insulin events in a UTC window: timing, units, automatic flag."""
        try:
            return contract.get_boluses(store, _parse_dt(start), _parse_dt(end))
        except ValueError as exc:
            return {"error": str(exc)}

    @mcp.tool(name="get_carb_entries", run_in_thread=False)
    def get_carb_entries(start: str, end: str) -> dict[str, Any]:
        """Meal events with carb counts in a UTC window."""
        try:
            return contract.get_carb_entries(store, _parse_dt(start), _parse_dt(end))
        except ValueError as exc:
            return {"error": str(exc)}

    @mcp.tool(name="get_basal_timeline", run_in_thread=False)
    def get_basal_timeline(start: str, end: str) -> dict[str, Any]:
        """Basal/temp-basal/suspend events in a UTC window, with stability flag."""
        try:
            return contract.get_basal_timeline(store, _parse_dt(start), _parse_dt(end))
        except ValueError as exc:
            return {"error": str(exc)}

    @mcp.tool(name="get_iob", run_in_thread=False)
    def get_iob(timestamp: str) -> dict[str, Any]:
        """Tier B insulin-on-board at an ISO datetime - analysis context, never dosing."""
        try:
            return contract.get_iob(store, _parse_dt(timestamp))
        except ValueError as exc:
            return {"error": str(exc)}


def build_server(
    store: StoragePort,
    realtime: RealtimeConnector | None = None,
) -> FastMCP:
    """Register the 10 core MCP tools, plus the insulin extension when the
    store's coverage shows insulin data (capability gating)."""
    fastmcp_cls = _import_fastmcp()
    mcp = fastmcp_cls(
        "Dexta Intelligence",
        instructions=(
            "Live and historical glucose intelligence from the Dexta harness. "
            "All timestamps are UTC. Tools return computed numbers only."
        ),
    )

    @mcp.tool(name="get_current_glucose", run_in_thread=False)
    def get_current_glucose() -> dict[str, Any]:
        """Latest glucose reading with trend and freshness (stale flag)."""
        return contract.get_current_glucose(store, realtime)

    @mcp.tool(name="get_glucose_readings", run_in_thread=False)
    def get_glucose_readings(
        start: str,
        end: str,
        max_count: int | None = None,
    ) -> dict[str, Any]:
        """Historical readings in a UTC window (half-open: start <= ts < end)."""
        return contract.get_glucose_readings(
            store,
            _parse_dt(start),
            _parse_dt(end),
            max_count=max_count,
        )

    @mcp.tool(name="get_statistics", run_in_thread=False)
    def get_statistics(
        start: str,
        end: str,
        target_low: int = TARGET_LOW_MG_DL,
        target_high: int = TARGET_HIGH_MG_DL,
    ) -> dict[str, Any]:
        """Mean/median/SD/CV/GMI and TIR bands for a UTC window."""
        return contract.get_statistics(
            store,
            _parse_dt(start),
            _parse_dt(end),
            target_low=target_low,
            target_high=target_high,
        )

    @mcp.tool(name="get_status_summary", run_in_thread=False)
    def get_status_summary() -> dict[str, Any]:
        """Current reading, last-24h stats, active episodes, and alerts."""
        return contract.get_status_summary(store, realtime)

    @mcp.tool(name="detect_episodes", run_in_thread=False)
    def detect_episodes(
        start: str,
        end: str,
        target_low: int = TARGET_LOW_MG_DL,
        target_high: int = TARGET_HIGH_MG_DL,
    ) -> dict[str, Any]:
        """Hypo/hyper episodes (min 15 min) in a UTC window."""
        return contract.detect_episodes(
            store,
            _parse_dt(start),
            _parse_dt(end),
            target_low=target_low,
            target_high=target_high,
        )

    @mcp.tool(name="get_episode_details", run_in_thread=False)
    def get_episode_details(
        episode_id: str,
        context_minutes: int = 30,
    ) -> dict[str, Any]:
        """One episode by id plus surrounding context readings."""
        return contract.get_episode_details(
            store,
            episode_id,
            context_minutes=context_minutes,
        )

    @mcp.tool(name="analyze_time_blocks", run_in_thread=False)
    def analyze_time_blocks(
        start: str,
        end: str,
        target_low: int = TARGET_LOW_MG_DL,
        target_high: int = TARGET_HIGH_MG_DL,
    ) -> dict[str, Any]:
        """Statistics per UTC day block (overnight/morning/afternoon/evening)."""
        return contract.analyze_time_blocks(
            store,
            _parse_dt(start),
            _parse_dt(end),
            target_low=target_low,
            target_high=target_high,
        )

    @mcp.tool(name="check_alerts", run_in_thread=False)
    def check_alerts(
        urgent_low: int = VERY_LOW_MG_DL,
        low: int = TARGET_LOW_MG_DL,
        high: int = TARGET_HIGH_MG_DL,
        urgent_high: int = VERY_HIGH_MG_DL,
    ) -> dict[str, Any]:
        """Threshold and trend-projection alerts (informational only)."""
        return contract.check_alerts(
            store,
            realtime,
            urgent_low=urgent_low,
            low=low,
            high=high,
            urgent_high=urgent_high,
        )

    @mcp.tool(name="export_data", run_in_thread=False)
    def export_data(
        start: str,
        end: str,
        format: Literal["json", "csv"] = "json",
    ) -> dict[str, Any]:
        """Export readings as a JSON or CSV string payload."""
        return contract.export_data(store, _parse_dt(start), _parse_dt(end), format=format)

    @mcp.tool(name="get_agp_report", run_in_thread=False)
    def get_agp_report(start: str, end: str) -> dict[str, Any]:
        """AGP percentile profile - 5-minute-of-day bins across days."""
        return contract.get_agp_report(store, _parse_dt(start), _parse_dt(end))

    if _capabilities(store).has_insulin:
        _register_insulin_tools(mcp, store)

    return mcp


def main(
    *,
    config_path: Path | None = None,
    connector_factory: _RealtimeConnectorFactory = build_realtime_connector,
) -> None:
    """Load config, open the store, and run the MCP server over stdio."""
    config = load_config(config_path)
    from dexta_intelligence.store import SQLiteStore  # noqa: PLC0415

    db_path = config.data.sqlite_path.expanduser()
    store = SQLiteStore(db_path)
    store.migrate()
    realtime = connector_factory(config)
    server = build_server(store, realtime)
    try:
        server.run()
    finally:
        store.close()


if __name__ == "__main__":
    main()
