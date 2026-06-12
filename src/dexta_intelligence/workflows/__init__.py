"""Workflows — orchestration over connectors, the store, and analytics."""

from dexta_intelligence.workflows.deep_analysis import DeepAnalysisReport, run_deep_analysis
from dexta_intelligence.workflows.goals import (
    GoalTick,
    compose_goal,
    goal_due,
    measure_metric,
    tick_goal,
)
from dexta_intelligence.workflows.lenses import BUILTIN_LENSES, PRODUCERS, build_registry
from dexta_intelligence.workflows.sync import SyncReport, sync, sync_all

__all__ = [
    "BUILTIN_LENSES",
    "PRODUCERS",
    "DeepAnalysisReport",
    "GoalTick",
    "SyncReport",
    "build_registry",
    "compose_goal",
    "goal_due",
    "measure_metric",
    "run_deep_analysis",
    "sync",
    "sync_all",
    "tick_goal",
]
