"""Two-group comparison instruments: the hypothesis-testing surface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dexta_intelligence.agents.reason import ToolSpec
from dexta_intelligence.agents.tools.toolkit import _HOURS_PAIR, _numbers

if TYPE_CHECKING:
    from dexta_intelligence.agents.tools.toolkit import DiscoveryToolkit


def compare_specs(toolkit: DiscoveryToolkit) -> list[ToolSpec]:
    """correlate plus the six DiscoveryToolkit.run comparison instruments."""

    def run(tool: str) -> Any:
        def call(args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
            result = toolkit.run(tool, args)
            return result.summary, result.evidence()

        return call

    def correlate(args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        result = toolkit.correlate(str(args.get("x", "")), str(args.get("y", "")))
        return result, _numbers(result, ("n", "pearson_r", "spearman_rho", "p"))

    return [
        ToolSpec(
            name="correlate",
            description=(
                "Correlate two per-day metrics across the active window "
                "(Pearson + Spearman + p). metrics: mean_glucose|tir|tbr|cv|sleep_score."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "x": {
                        "type": "string",
                        "enum": ["mean_glucose", "tir", "tbr", "cv", "sleep_score"],
                    },
                    "y": {
                        "type": "string",
                        "enum": ["mean_glucose", "tir", "tbr", "cv", "sleep_score"],
                    },
                },
                "required": ["x", "y"],
            },
            fn=correlate,
        ),
        ToolSpec(
            name="groupby_compare",
            description=(
                "Compare a daily metric between two groups of days. "
                "group_by: weekend|sleep_bucket|workout_day. target: mean_glucose|tir_pct. "
                "Returns a p-value (p_welch) and effect sizes (cohen_d, cliffs_delta)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "group_by": {
                        "type": "string",
                        "enum": ["weekend", "sleep_bucket", "workout_day"],
                    },
                    "target": {"type": "string", "enum": ["mean_glucose", "tir_pct"]},
                },
                "required": ["group_by", "target"],
            },
            fn=run("groupby_compare"),
        ),
        ToolSpec(
            name="tod_compare",
            description=(
                "Compare mean glucose between two time-of-day windows. "
                "hours_a/hours_b are [start,end) hours 0-24. "
                "Returns a p-value (p_welch) and effect sizes (cohen_d, cliffs_delta)."
            ),
            parameters={
                "type": "object",
                "properties": {"hours_a": _HOURS_PAIR, "hours_b": _HOURS_PAIR},
                "required": ["hours_a", "hours_b"],
            },
            fn=run("tod_compare"),
        ),
        ToolSpec(
            name="event_proximity",
            description=(
                "Average glucose after an event vs the hour before it. "
                "event_type: meal|workout|bolus. window_min: 30-240."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "event_type": {"type": "string", "enum": ["meal", "workout", "bolus"]},
                    "window_min": {"type": "integer", "minimum": 30, "maximum": 240},
                },
                "required": ["event_type"],
            },
            fn=run("event_proximity"),
        ),
        ToolSpec(
            name="basal_overnight",
            description=(
                "Per-night overnight glucose drift, first-half vs second-half nights. "
                "hours is the [start,end) overnight window (default [0,6])."
            ),
            parameters={
                "type": "object",
                "properties": {"hours": _HOURS_PAIR},
            },
            fn=run("basal_overnight"),
        ),
        ToolSpec(
            name="meal_response",
            description=(
                "Per-meal excursion (peak minus pre-meal baseline) for bigger-carb vs "
                "smaller-carb meals, split at median carbs. window_min: 30-240."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "window_min": {"type": "integer", "minimum": 30, "maximum": 240},
                },
            },
            fn=run("meal_response"),
        ),
        ToolSpec(
            name="correction_outcome",
            description=(
                "Per-bolus glucose delta (window-end minus baseline), newer vs older "
                "boluses, plus rebound_low_rate (% with a <70 reading). window_min: 30-240."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "window_min": {"type": "integer", "minimum": 30, "maximum": 240},
                },
            },
            fn=run("correction_outcome"),
        ),
    ]
