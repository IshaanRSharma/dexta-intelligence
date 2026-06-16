"""``dexta serve`` — run the local web GUI."""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, TextIO

from dexta_intelligence.cli._common import StoreOpener, open_sqlite_store

if TYPE_CHECKING:
    from pathlib import Path

    from dexta_intelligence.config import Config
    from dexta_intelligence.store.port import StoragePort

__all__ = ["cmd_serve"]

logger = logging.getLogger("dexta_intelligence.serve")


def _start_auto_sync(config: Config, opener: StoreOpener, interval_min: int) -> None:
    """Background daemon thread that re-syncs every ``interval_min`` minutes.

    Reuses the same connector + storage seams as ``dexta sync``, so it covers
    whatever the user configured (Nightscout, pumps, a self-hosted Postgres).
    Failures are swallowed — a flaky source must never take the GUI down."""
    from dexta_intelligence.cli._common import build_connectors  # noqa: PLC0415
    from dexta_intelligence.workflows.sync import sync_all  # noqa: PLC0415

    def loop() -> None:
        while True:
            time.sleep(interval_min * 60)
            try:
                connectors = build_connectors(config)
                if not connectors:
                    continue
                store = opener(config, None)
                try:
                    sync_all(connectors, store)
                finally:
                    if hasattr(store, "close"):
                        store.close()
            except Exception:
                logger.debug("auto-sync tick failed", exc_info=True)

    threading.Thread(target=loop, daemon=True, name="dexta-auto-sync").start()


def cmd_serve(
    *,
    config: Config,
    db_path: Path | None,
    out: TextIO,
    host: str = "127.0.0.1",
    port: int = 8787,
    config_path: Path | None = None,
    sync_every: int | None = None,
    opener: StoreOpener = open_sqlite_store,
) -> int:
    """Build the GUI app and serve it with uvicorn.

    Binds localhost by default; pass ``--host 0.0.0.0`` to deliberately expose
    the GUI to your LAN (there is no auth — only do this on a trusted network).
    """
    try:
        import uvicorn  # noqa: PLC0415

        from dexta_intelligence.cli._common import resolve_config_path  # noqa: PLC0415
        from dexta_intelligence.server import create_app  # noqa: PLC0415
    except ModuleNotFoundError:
        out.write(
            "The web GUI needs the optional GUI stack. Install it with:\n"
            "  pip install 'dexta-intelligence[gui]'\n"
        )
        return 1

    base_opener = opener
    if db_path is not None:
        # Pin the override so every request opens the same database.
        def base_opener(cfg: Config, _db: Path | None = None) -> StoragePort:
            return opener(cfg, db_path)

    # Capture the launched config path once so the settings panel reads/writes
    # the file the running server actually loaded — not a per-request re-resolve.
    settings_path = resolve_config_path(config_path)
    app = create_app(config, store_opener=base_opener, config_path=settings_path, host=host)

    interval = sync_every if sync_every is not None else config.server.auto_sync_minutes
    if interval and interval > 0:
        _start_auto_sync(config, base_opener, interval)
        out.write(f"auto-sync every {interval} min (background)\n")

    if host not in ("127.0.0.1", "localhost", "::1"):
        out.write(
            f"WARNING: binding {host} exposes the auth-less PHI GUI to your "
            "network — only do this on a trusted LAN.\n"
        )
    out.write(f"dexta serve · http://{host}:{port}  (Ctrl-C to stop)\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0
