"""Manual-context tools. Read-only: these surface what the user logged; they
never create manual events (only the user submits them)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dexta_intelligence.agents.reason import ToolSpec
from dexta_intelligence.agents.tools.toolkit import _numbers

if TYPE_CHECKING:
    from dexta_intelligence.agents.tools.toolkit import DiscoveryToolkit


def manual_specs(toolkit: DiscoveryToolkit) -> list[ToolSpec]:
    """User-reported context: list, search, and pad-around-event lookups."""

    def get_manual_events(_args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        result = toolkit.get_manual_events()
        return result, _numbers(result, ("n_events",))

    def search_manual_events(args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        result = toolkit.search_manual_events(str(args.get("query", "")))
        return result, _numbers(result, ("n_events",))

    def get_context_around_event(args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        result = toolkit.get_context_around_event(
            str(args.get("timestamp", "")), float(args.get("pad_hours", 4.0))
        )
        return result, _numbers(result, ("n_events", "pad_hours"))

    return [
        ToolSpec(
            name="get_manual_events",
            description=(
                "User-reported context in the ACTIVE window (meals, stress, illness, "
                "site changes, notes - {ts, event_type, title, description, tags}). "
                "Provenance is user-reported, never device data. An empty result means "
                "nothing was logged, not that nothing happened."
            ),
            parameters={"type": "object", "properties": {}},
            fn=get_manual_events,
        ),
        ToolSpec(
            name="search_manual_events",
            description=(
                "User-reported context in the ACTIVE window matching a query "
                "(case-insensitive over type/title/description/tags). Use to find a "
                "specific note, e.g. 'high-fat' or 'site change'."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "substring to match"},
                },
                "required": ["query"],
            },
            fn=search_manual_events,
        ),
        ToolSpec(
            name="get_context_around_event",
            description=(
                "User-reported context within pad_hours of a timestamp (whole record). "
                "Pads around a specific glucose event to answer 'what did the user log "
                "near this spike?'."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string", "description": "ISO datetime"},
                    "pad_hours": {"type": "number", "minimum": 0, "maximum": 48},
                },
                "required": ["timestamp"],
            },
            fn=get_context_around_event,
        ),
    ]
