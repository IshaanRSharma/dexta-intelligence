"""Daemon command: run the cadence driver continuously or for a single cycle."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TextIO

from dexta_intelligence.cli._common import (
    MEDICAL_DISCLAIMER,
    StoreOpener,
    _maybe_close_store,
    discovery_model,
    open_sqlite_store,
)

if TYPE_CHECKING:
    from pathlib import Path

    from dexta_intelligence.config import Config
    from dexta_intelligence.workflows.daemon import CycleReport


def cmd_daemon(
    *,
    config: Config,
    db_path: Path | None,
    out: TextIO,
    interval: float = 5.0,
    deep_every: int = 12,
    once: bool = False,
    opener: StoreOpener = open_sqlite_store,
    model: Any = None,
) -> int:
    """Run the continuous-intelligence cadence driver.

    ``--once`` runs exactly one cycle (no sleep) and is the testable path;
    otherwise the daemon loops on ``--interval`` minutes, running the deep
    coordinator pass every ``--deep-every`` cycles. Anomalies are logged via the
    default notifier and a per-cycle summary is printed.
    """
    from dexta_intelligence.notifications import LogNotifier  # noqa: PLC0415
    from dexta_intelligence.workflows.daemon import run_cycle, run_daemon  # noqa: PLC0415

    chat_model = model if model is not None else discovery_model(config)
    notify = LogNotifier()

    if once:
        store = opener(config, db_path)
        try:
            report = run_cycle(config, store, model=chat_model, notify=notify, deep=True)
        finally:
            _maybe_close_store(store, opener)
        _print_cycle(out, report)
        out.write(f"\n{MEDICAL_DISCLAIMER}\n")
        return 0

    out.write(
        f"dexta daemon: cycle every {interval:g} min, deep pass every {deep_every} cycle(s). "
        "Ctrl-C to stop.\n"
    )
    out.write(f"{MEDICAL_DISCLAIMER}\n\n")
    return run_daemon(
        config,
        lambda: opener(config, db_path),
        interval_min=interval,
        deep_every=deep_every,
        model=chat_model,
        notify=notify,
        on_cycle=lambda report: _print_cycle(out, report),
    )


def _print_cycle(out: TextIO, report: CycleReport) -> None:
    deep = " deep" if report.deep_ran else ""
    out.write(
        f"[{report.started_at.isoformat()}] synced {report.sources_synced} src · "
        f"{report.anomalies} anomaly(ies) · {report.goals_ticked} goal(s) ticked · "
        f"{report.findings_persisted} finding(s) · {report.stale_pruned} retired{deep}\n"
    )
    for step, msg in report.errors:
        out.write(f"  ! {step}: {msg}\n")
