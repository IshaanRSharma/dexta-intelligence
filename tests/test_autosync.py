"""Tests for the runtime-managed background data-sync controller.

Deterministic: no network, no database, no real sleeping. Ticks are driven
directly via ``_tick``; ``configure`` is asserted via ``status`` snapshots
rather than thread timing.
"""

from __future__ import annotations

from unittest.mock import Mock

from dexta_intelligence.config import Config
from dexta_intelligence.server.autosync import AutoSyncController


def _dummy_opener(config: Config, db: object = None) -> Mock:
    return Mock()


def _make_controller(sync_fn: object) -> AutoSyncController:
    return AutoSyncController(Config(), _dummy_opener, sync_fn=sync_fn)


def test_tick_success_records_last_run() -> None:
    calls: list[int] = []
    controller = _make_controller(lambda: calls.append(1))

    controller._tick()

    status = controller.status()
    assert calls == [1]
    assert status.last_run is not None
    assert status.last_error is None


def test_tick_error_is_isolated() -> None:
    def boom() -> None:
        raise ValueError("flaky source")

    controller = _make_controller(boom)

    controller._tick()  # must not raise

    status = controller.status()
    assert status.last_run is None
    assert status.last_error is not None
    assert "ValueError" in status.last_error


def test_configure_toggles_state() -> None:
    controller = _make_controller(lambda: None)

    controller.configure(0)
    status = controller.status()
    assert status.enabled is False
    assert status.interval_min == 0

    controller.configure(15)
    status = controller.status()
    assert status.enabled is True
    assert status.interval_min == 15

    controller.configure(0)
    status = controller.status()
    assert status.enabled is False
    assert status.interval_min == 0

    controller.stop()  # safe to call after disabling
