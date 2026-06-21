"""Runtime-managed background data-sync controller for the GUI.

Replaces the fire-and-forget thread in ``cli.serve._start_auto_sync`` with a
controllable object: the GUI can start, stop, and retune the sync interval at
runtime and read the controller's status, instead of being fixed at boot.

A tick mirrors the original loop body: build the configured connectors, open
the store via the opener, run ``sync_all``, then close the store. Per-tick
errors are swallowed so a flaky source never crashes the app.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from dexta_intelligence.cli._common import StoreOpener
    from dexta_intelligence.config import Config

logger = logging.getLogger(__name__)

__all__ = ["AutoSyncController", "AutoSyncStatus"]


@dataclass(frozen=True, slots=True)
class AutoSyncStatus:
    """Immutable snapshot of the controller's runtime state."""

    enabled: bool
    interval_min: int
    last_run: datetime | None
    last_error: str | None


class AutoSyncController:
    """Start, stop, and retune background syncing at runtime.

    The sync work is injectable via ``sync_fn`` so tests can drive ticks
    without touching the network or a database. The default mirrors
    ``cli.serve._start_auto_sync``: build connectors, open the store, run
    ``sync_all``, close the store.
    """

    def __init__(
        self,
        config: Config,
        opener: StoreOpener,
        *,
        sync_fn: Callable[..., object] | None = None,
    ) -> None:
        self._config = config
        self._opener = opener
        self._sync_fn = sync_fn if sync_fn is not None else self._default_sync
        self._lock = threading.Lock()
        self._interval_min = 0
        self._last_run: datetime | None = None
        self._last_error: str | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def configure(self, interval_min: int) -> None:
        """Set the sync interval and (re)start or stop the loop.

        ``interval_min <= 0`` disables syncing and stops any running thread.
        A positive interval sets it and ensures the background loop is running.
        Idempotent and thread-safe.
        """
        if interval_min <= 0:
            self.stop()
            with self._lock:
                self._interval_min = 0
            return

        self.stop()
        with self._lock:
            self._interval_min = interval_min
            self._stop = threading.Event()
            self._thread = threading.Thread(
                target=self._loop,
                args=(self._stop, interval_min),
                daemon=True,
                name="dexta-auto-sync",
            )
            self._thread.start()

    def _loop(self, stop: threading.Event, interval_min: int) -> None:
        while not stop.is_set():
            if stop.wait(interval_min * 60):
                return
            self._tick()

    def _tick(self) -> None:
        """Run one sync attempt. Never raises."""
        try:
            self._sync_fn()
        except Exception as exc:  # a flaky source must never crash the app
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
            logger.debug("auto-sync tick failed", exc_info=True)
            return
        with self._lock:
            self._last_run = datetime.now(tz=UTC)
            self._last_error = None

    def _default_sync(self) -> None:
        from dexta_intelligence.cli._common import build_connectors  # noqa: PLC0415
        from dexta_intelligence.workflows.sync import sync_all  # noqa: PLC0415

        connectors = build_connectors(self._config)
        if not connectors:
            return
        store = self._opener(self._config, None)
        try:
            sync_all(connectors, store)
        finally:
            if hasattr(store, "close"):
                store.close()

    def status(self) -> AutoSyncStatus:
        """Snapshot the current runtime state under the lock."""
        with self._lock:
            return AutoSyncStatus(
                enabled=self._interval_min > 0,
                interval_min=self._interval_min,
                last_run=self._last_run,
                last_error=self._last_error,
            )

    def stop(self) -> None:
        """Signal the loop to stop, join briefly, and clear the thread."""
        with self._lock:
            thread = self._thread
            self._stop.set()
            self._thread = None
        if thread is not None:
            thread.join(timeout=1)
