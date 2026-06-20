"""Treatment-inspection tools: read the insulin/carb events, not just aggregates.

Capability filtering (insulin / meals) happens in
:func:`~dexta_intelligence.agents.tools.build_belt`; these are built
unconditionally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dexta_intelligence.agents.reason import ToolSpec
from dexta_intelligence.agents.tools.toolkit import _item_numbers, _numbers

if TYPE_CHECKING:
    from dexta_intelligence.agents.tools.toolkit import DiscoveryToolkit


def treatment_specs(toolkit: DiscoveryToolkit) -> list[ToolSpec]:
    """Carb entries, boluses, basal timeline, IOB, COB, and insulin/therapy profiles."""

    def get_carb_entries(_args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        result = toolkit.get_carb_entries()
        numbers = _numbers(result, ("n_entries", "total_carbs_g"))
        numbers.update(_item_numbers(result.get("entries", []), "entry"))
        return result, numbers

    def get_boluses(_args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        result = toolkit.get_boluses()
        numbers = _numbers(result, ("n_boluses", "total_units"))
        numbers.update(_item_numbers(result.get("boluses", []), "bolus"))
        return result, numbers

    def get_basal_timeline(_args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        result = toolkit.get_basal_timeline()
        numbers = _numbers(result, ("n_basal", "n_temp_basal", "n_suspend"))
        numbers.update(_item_numbers(result.get("events", []), "event"))
        return result, numbers

    def get_iob(args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        result = toolkit.get_iob(str(args.get("timestamp", "")))
        return result, _numbers(result, ("iob_units", "n_recent_boluses"))

    def get_insulin_profile(_args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        result = toolkit.get_insulin_profile()
        active = next(
            (p for p in result.get("profiles") or [] if p.get("active")),
            None,
        )
        numbers = _numbers(result, ("pump_serial",))
        if active:
            numbers["active_dia_hr"] = active.get("dia_hr")
            numbers["n_active_segments"] = len(active.get("segments") or [])
        return result, numbers

    def get_active_profile(args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        result = toolkit.get_active_profile(str(args.get("timestamp", "")))
        return result, _numbers(result, ("pump_serial",))

    def get_cob(args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        result = toolkit.get_cob(str(args.get("timestamp", "")))
        return result, _numbers(result, ("cob_g", "absorbed_g", "n_carb_entries"))

    return [
        ToolSpec(
            name="get_carb_entries",
            description=(
                "Carb entries in the ACTIVE window ({ts, carbs_g, ...}, n_entries, "
                "total_carbs_g). Call when explaining a spike/meal - an empty result "
                "around a spike is itself a signal (possible missing carb entry)."
            ),
            parameters={"type": "object", "properties": {}},
            fn=get_carb_entries,
        ),
        ToolSpec(
            name="get_boluses",
            description=(
                "Boluses in the ACTIVE window ({ts, units, minutes_after_carb_entry}). "
                "minutes_after_carb_entry is the late-bolus signal. Call when explaining "
                "a spike/meal/correction."
            ),
            parameters={"type": "object", "properties": {}},
            fn=get_boluses,
        ),
        ToolSpec(
            name="get_basal_timeline",
            description=(
                "Basal / temp-basal / suspend events in the ACTIVE window plus "
                "basal_stable (no temp-basal/suspend). Rules basal in or out as a "
                "spike contributor."
            ),
            parameters={"type": "object", "properties": {}},
            fn=get_basal_timeline,
        ),
        ToolSpec(
            name="get_iob",
            description=(
                "Insulin-on-board at an ISO datetime, computed from logged boluses "
                "(oref0 curve, tier B - analysis context only, never dosing)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string", "description": "ISO datetime"},
                },
                "required": ["timestamp"],
            },
            fn=get_iob,
        ),
        ToolSpec(
            name="get_insulin_profile",
            description=(
                "Pump-reported basal/ISF/carb-ratio/target segments for the active "
                "profile (and all stored profiles). Synced from Tandem; tier B - "
                "analysis context only, never dosing."
            ),
            parameters={"type": "object", "properties": {}},
            fn=get_insulin_profile,
        ),
        ToolSpec(
            name="get_active_profile",
            description=(
                "The therapy profile VERSION in effect at an ISO datetime - use when "
                "explaining a past event so it reads that period's settings, not "
                "today's. Falls back to the current snapshot if no history exists. "
                "Tier B - analysis context only, never dosing."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string", "description": "ISO datetime of the event"},
                },
                "required": ["timestamp"],
            },
            fn=get_active_profile,
        ),
        ToolSpec(
            name="get_cob",
            description=(
                "Carbs-on-board at an ISO datetime from announced carb entries "
                "(oref0 decay, tier B - analysis context only). Unannounced carbs "
                "do not appear here."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string", "description": "ISO datetime"},
                },
                "required": ["timestamp"],
            },
            fn=get_cob,
        ),
    ]
