"""Daemon — the cadence driver behind "continuous health intelligence".

On an interval the daemon runs one :func:`run_cycle`: sync every source, scan
for anomalies, tick due goals, and (less often) run a coordinator deep pass.
Each step is exception-isolated — a step that raises is logged, its error is
recorded on the :class:`CycleReport`, and the cycle continues. ``run_cycle``
never raises.

The loop driver :func:`run_daemon` paces with :func:`time.sleep` between cycles
(never a busy loop) and runs the deep pass every ``deep_every`` cycles. It exits
cleanly on :class:`KeyboardInterrupt`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from dexta_intelligence.cli._common import build_connectors
from dexta_intelligence.models import GoalStatus
from dexta_intelligence.workflows.deep_analysis import persist_findings
from dexta_intelligence.workflows.goals import goal_due, tick_goal
from dexta_intelligence.workflows.monitor import run_monitor
from dexta_intelligence.workflows.sync import sync_all

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.language_models.chat_models import BaseChatModel

    from dexta_intelligence.agents.base import AgentContext
    from dexta_intelligence.config import Config
    from dexta_intelligence.notifications import Notifier
    from dexta_intelligence.store.port import StoragePort

logger = logging.getLogger(__name__)

__all__ = ["CycleReport", "run_cycle", "run_daemon"]


@dataclass(frozen=True, slots=True)
class CycleReport:
    """Outcome of one daemon cycle — per-step counts and any step errors.

    ``errors`` carries ``(step, message)`` pairs; an empty tuple means every
    step that ran completed cleanly.
    """

    started_at: datetime
    sources_synced: int = 0
    anomalies: int = 0
    goals_ticked: int = 0
    findings_persisted: int = 0
    deep_ran: bool = False
    errors: tuple[tuple[str, str], ...] = ()

    @property
    def ok(self) -> bool:
        """True when no step recorded an error."""
        return not self.errors


@dataclass
class _Accumulator:
    sources_synced: int = 0
    anomalies: int = 0
    goals_ticked: int = 0
    findings_persisted: int = 0
    deep_ran: bool = False
    errors: list[tuple[str, str]] = field(default_factory=list)

    def fail(self, step: str, exc: Exception) -> None:
        msg = f"{type(exc).__name__}: {exc}"
        logger.exception("daemon: step %s failed; continuing cycle", step)
        self.errors.append((step, msg))


def run_cycle(
    config: Config,
    store: StoragePort,
    *,
    model: BaseChatModel | None = None,
    notify: Notifier | None = None,
    deep: bool = False,
    now: datetime | None = None,
) -> CycleReport:
    """Run one cadence cycle: sync, monitor, tick due goals, optional deep pass.

    Every step is isolated: a failing step logs, records its error on the
    report, and the cycle proceeds. Returns a :class:`CycleReport` and never
    raises. The deep coordinator pass runs only when ``deep`` is true.
    """
    moment = _resolve_now(now)
    acc = _Accumulator()

    try:
        reports = sync_all(build_connectors(config), store, now=moment)
        acc.sources_synced = sum(1 for r in reports if r.ok)
        for r in reports:
            for err in r.errors:
                acc.errors.append((f"sync:{r.source}", err))
    except Exception as exc:
        acc.fail("sync", exc)

    ctx = _ctx(config, store)

    try:
        anomalies = run_monitor(ctx, notify=notify, persist=True)
        acc.anomalies = len(anomalies)
    except Exception as exc:
        acc.fail("monitor", exc)

    _tick_due_goals(store, ctx, moment, model, acc)

    if deep:
        _deep_pass(store, ctx, config, model, acc)

    return CycleReport(
        started_at=moment,
        sources_synced=acc.sources_synced,
        anomalies=acc.anomalies,
        goals_ticked=acc.goals_ticked,
        findings_persisted=acc.findings_persisted,
        deep_ran=acc.deep_ran,
        errors=tuple(acc.errors),
    )


def run_daemon(
    config: Config,
    store_opener: Callable[[], StoragePort],
    *,
    interval_min: float,
    deep_every: int,
    model: BaseChatModel | None = None,
    notify: Notifier | None = None,
    max_cycles: int | None = None,
    on_cycle: Callable[[CycleReport], None] | None = None,
) -> int:
    """Drive :func:`run_cycle` on a cadence until interrupted.

    Sleeps ``interval_min`` minutes between cycles (no busy loop) and runs the
    deep pass every ``deep_every`` cycles. ``store_opener`` is called once;
    ``max_cycles`` bounds the run for tests. Exits cleanly (returns 0) on
    :class:`KeyboardInterrupt`.
    """
    store = store_opener()
    cycle = 0
    try:
        while max_cycles is None or cycle < max_cycles:
            deep = deep_every > 0 and cycle % deep_every == 0
            report = run_cycle(config, store, model=model, notify=notify, deep=deep)
            if on_cycle is not None:
                on_cycle(report)
            cycle += 1
            if max_cycles is not None and cycle >= max_cycles:
                break
            time.sleep(max(0.0, interval_min) * 60)
    except KeyboardInterrupt:
        logger.info("daemon: interrupted after %d cycle(s); shutting down", cycle)
    finally:
        if hasattr(store, "close"):
            store.close()
    return 0


def _tick_due_goals(
    store: StoragePort,
    ctx: AgentContext,
    now: datetime,
    model: BaseChatModel | None,
    acc: _Accumulator,
) -> None:
    try:
        active = store.get_goals(status=GoalStatus.ACTIVE)
    except Exception as exc:
        acc.fail("goals", exc)
        return
    for goal in active:
        if goal.id is None:
            continue
        try:
            if not goal_due(goal, store.get_goal_checkpoints(goal.id), now=now):
                continue
            result = tick_goal(goal, ctx, now=now, model=model)
            store.insert_goal_checkpoint(result.checkpoint)
            if result.achieved:
                store.set_goal_status(goal.id, GoalStatus.ACHIEVED)
            acc.goals_ticked += 1
        except Exception as exc:
            acc.fail(f"goal:{goal.id}", exc)


def _deep_pass(
    store: StoragePort,
    ctx: AgentContext,
    config: Config,
    model: BaseChatModel | None,
    acc: _Accumulator,
) -> None:
    from dexta_intelligence.agents.coordinator import CoordinatorAgent  # noqa: PLC0415

    try:
        findings = CoordinatorAgent(model=model, config=config).investigate(ctx)
        acc.findings_persisted = len(persist_findings(store, findings))
        acc.deep_ran = True
    except Exception as exc:
        acc.fail("deep", exc)


def _ctx(config: Config, store: StoragePort) -> AgentContext:
    from dexta_intelligence.cli._common import _ctx_for  # noqa: PLC0415

    return _ctx_for(config, store)


def _resolve_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(tz=UTC)
    return now if now.tzinfo is not None else now.replace(tzinfo=UTC)
