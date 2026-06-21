"""View-model logic for the richer Goals observability page.

This module produces pure data (plain dicts), never HTML. The orchestrator
owns the template and route; here we only shape a :class:`Goal` plus its
checkpoints and matching investigation runs into the structure the template
consumes.

Linkage rule: a goal is investigated by the coordinator with
``goal=goal.statement``, and the resulting :class:`InvestigationRun` stores
``question=goal.statement``. So a goal's runs are exactly those whose
``run.question`` equals ``goal.statement``.

All progress numbers (baseline, current, delta, on_track, pct_to_target) are
computed deterministically from checkpoint metric values. Progress is never
LLM-judged.
"""

from __future__ import annotations

import logging
from datetime import UTC, timedelta
from typing import TYPE_CHECKING, Any

from dexta_intelligence.server._format import _relative_time
from dexta_intelligence.server.render import sparkline_svg

if TYPE_CHECKING:
    from datetime import datetime

    from dexta_intelligence.models import Goal, GoalCheckpoint
    from dexta_intelligence.store.port import StoragePort

logger = logging.getLogger(__name__)

__all__ = ["goal_card_view"]

_CHECKPOINT_CAP = 10
_RUN_CAP = 5


def _until(ts: datetime, now: datetime) -> str:
    """Render a future timestamp relative to ``now`` (treat naive ts as UTC)."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    secs = int((ts.astimezone(UTC) - now).total_seconds())
    if secs <= 0:
        return "due now"
    if secs < 86400:
        hours = max(1, secs // 3600)
        return f"in {hours}h"
    return f"in {secs // 86400}d"


def _on_track(
    baseline: float | None,
    current: float | None,
    target: float | None,
    direction: str,
) -> bool | None:
    """Whether ``current`` is progressing toward ``target`` given direction."""
    if target is None or current is None or baseline is None:
        return None
    reached = current >= target if direction == "increase" else current <= target
    closer = abs(target - current) < abs(target - baseline)
    return reached or closer


def _pct_to_target(
    baseline: float | None, current: float | None, target: float | None
) -> int | None:
    """Progress along the baseline->target span, expressed as a 0..100 percent."""
    if target is None or baseline is None or current is None or target == baseline:
        return None
    pct = round(100 * (current - baseline) / (target - baseline))
    return max(0, min(100, pct))


def goal_card_view(store: StoragePort, goal: Goal, *, now: datetime) -> dict[str, Any]:
    """Shape a goal, its checkpoints, and matching runs into a view-model dict."""
    checkpoints: list[GoalCheckpoint] = (
        store.get_goal_checkpoints(goal.id) if goal.id is not None else []
    )
    values = [cp.metric_value for cp in checkpoints]
    spark_values = [v for v in values if v is not None]

    baseline = checkpoints[0].metric_value if checkpoints else None
    current = checkpoints[-1].metric_value if checkpoints else None
    delta = current - baseline if baseline is not None and current is not None else None

    checkpoint_views = [
        {
            "when": _relative_time(cp.ts, now),
            "value": cp.metric_value,
            "note": cp.note,
        }
        for cp in reversed(checkpoints)
    ][:_CHECKPOINT_CAP]

    next_check: str | None = None
    if checkpoints:
        last_ts = checkpoints[-1].ts
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=UTC)
        next_check = _until(last_ts + timedelta(days=goal.cadence_days), now)

    # Linked runs are an enrichment: a failure here (for example an older DB
    # without the investigation_runs table) must never take the goals page down.
    try:
        all_runs = store.get_investigation_runs()
    except Exception:
        logger.debug("goal_card_view: get_investigation_runs failed; degrading", exc_info=True)
        all_runs = []
    runs = [
        {
            "status": run.status,
            "n_findings": run.n_findings,
            "when": _relative_time(run.finished_at, now),
        }
        for run in all_runs
        if run.question == goal.statement
    ][:_RUN_CAP]

    return {
        "id": goal.id,
        "statement": goal.statement,
        "status": goal.status.value,
        "metric": goal.metric.value,
        "direction": goal.direction,
        "cadence_days": goal.cadence_days,
        "target": goal.target,
        "baseline": baseline,
        "current": current,
        "delta": delta,
        "on_track": _on_track(baseline, current, goal.target, goal.direction),
        "pct_to_target": _pct_to_target(baseline, current, goal.target),
        "spark": sparkline_svg(spark_values),
        "n_checkpoints": len(checkpoints),
        "checkpoints": checkpoint_views,
        "next_check": next_check,
        "runs": runs,
    }
