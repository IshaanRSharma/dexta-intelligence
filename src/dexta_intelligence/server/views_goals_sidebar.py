"""Goals summary for the dashboard sidebar."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dexta_intelligence.models import GoalStatus
from dexta_intelligence.workflows.goals import goal_due

if TYPE_CHECKING:
    from datetime import datetime

    from dexta_intelligence.store.port import StoragePort

__all__ = ["goals_sidebar_view"]


def goals_sidebar_view(store: StoragePort, *, now: datetime) -> dict[str, Any]:
    """Active goals count and how many are due for a tick."""
    active = [g for g in store.get_goals() if g.status == GoalStatus.ACTIVE]
    due = 0
    for goal in active:
        if goal.id is None:
            due += 1
            continue
        checkpoints = store.get_goal_checkpoints(goal.id)
        if goal_due(goal, checkpoints, now=now):
            due += 1
    return {"active": len(active), "due": due}
