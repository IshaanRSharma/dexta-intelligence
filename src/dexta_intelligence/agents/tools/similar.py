"""The find_similar_events tool: the 'N of M similar dinners' recurrence check."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dexta_intelligence.agents.reason import ToolSpec
from dexta_intelligence.agents.tools.toolkit import (
    _SPIKE_THRESHOLD,
    _item_numbers,
    _numbers,
)

if TYPE_CHECKING:
    from dexta_intelligence.agents.tools.toolkit import DiscoveryToolkit


def similar_specs(toolkit: DiscoveryToolkit) -> list[ToolSpec]:
    """Recurrence over the whole record at the same time of day as a timestamp."""

    def find_similar_events(args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        result = toolkit.find_similar_events(
            str(args.get("timestamp", "")), float(args.get("threshold", _SPIKE_THRESHOLD))
        )
        numbers = _numbers(
            result,
            (
                "n_similar",
                "n_spiking",
                "mean_peak_spiking",
                "mean_bolus_delay_spiking_min",
                "mean_bolus_delay_other_min",
            ),
        )
        numbers.update(_item_numbers(result.get("events", []), "similar"))
        return result, numbers

    return [
        ToolSpec(
            name="find_similar_events",
            description=(
                "Recurrence over the WHOLE record: events at the same time of day as the "
                "timestamp (carb entries when logged), each with post-event peak, spiked "
                "flag, bolus_delay_min; plus n_similar, n_spiking, mean spiking vs "
                "non-spiking bolus delays. The 'N of M similar dinners' instrument."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string", "description": "ISO datetime of the event"},
                    "threshold": {"type": "number", "minimum": 140, "maximum": 400},
                },
                "required": ["timestamp"],
            },
            fn=find_similar_events,
        ),
    ]
