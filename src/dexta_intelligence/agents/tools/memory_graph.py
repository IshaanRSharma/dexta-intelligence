"""Findings-graph tools: change detection and contradicted beliefs.

Two instruments over the bitemporal finding edges
(:mod:`dexta_intelligence.memory.changes`), so an agent answers "what changed
recently" and "which beliefs were disproved" from deterministically authored
graph edges instead of re-mining raw data. Anchored to the analysis window's
end (no wall-clock read), so the same store and window always answer the same.
"""

from __future__ import annotations

from datetime import UTC, datetime, time
from typing import TYPE_CHECKING, Any

from dexta_intelligence.agents.reason import ToolSpec
from dexta_intelligence.memory.changes import contradicted_beliefs, what_changed

if TYPE_CHECKING:
    from dexta_intelligence.agents.base import AgentContext

__all__ = ["memory_graph_specs"]

_MAX_WITHIN_DAYS = 365
_DEFAULT_WITHIN_DAYS = 90


def memory_graph_specs(ctx: AgentContext) -> list[ToolSpec]:
    now = datetime.combine(ctx.window[1], time.max, tzinfo=UTC)

    def _what_changed(args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        try:
            within = int(args.get("within_days", _DEFAULT_WITHIN_DAYS))
        except (TypeError, ValueError):
            return {"error": "within_days must be an integer"}, {}
        within = max(1, min(within, _MAX_WITHIN_DAYS))
        changes = what_changed(ctx.store, now=now, within_days=within)
        result: dict[str, Any] = {
            "within_days": within,
            "n_changes": len(changes),
            "changes": changes,
        }
        if not changes:
            result["note"] = (
                "no superseding changes on record in this span - patterns either "
                "held steady or have not been re-derived yet"
            )
        return result, {"n_changes": len(changes)}

    def _contradicted(args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        beliefs = contradicted_beliefs(ctx.store)
        result: dict[str, Any] = {
            "n_contradictions": len(beliefs),
            "contradicted": beliefs,
        }
        if not beliefs:
            result["note"] = "no contradicted beliefs on record"
        return result, {"n_contradictions": len(beliefs)}

    return [
        ToolSpec(
            name="what_changed",
            description=(
                "Recent regime changes from the findings graph (SUPERSEDES edges): "
                "each with when, what_stopped, what_started, the deterministic "
                "evidence string, both finding windows, and hypo/hyper episode "
                "counts for the 14 days either side of the change. Use for 'what "
                "changed recently' / 'did my old pattern stop'."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "within_days": {"type": "integer", "minimum": 1, "maximum": 365},
                },
            },
            fn=_what_changed,
        ),
        ToolSpec(
            name="contradicted_beliefs",
            description=(
                "Beliefs the data later disproved (CONTRADICTS edges): each "
                "contradicted belief with the finding that contradicted it, the "
                "opposing effect evidence, and both evidence windows. Use for "
                "'was I wrong about X' / myth-busting a pattern the user believes."
            ),
            parameters={"type": "object", "properties": {}},
            fn=_contradicted,
        ),
    ]
