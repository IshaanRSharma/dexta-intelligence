"""The recall tool: what dexta already believes (shared cross-agent context)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dexta_intelligence.agents.reason import ToolSpec
from dexta_intelligence.agents.tools.toolkit import _recall

if TYPE_CHECKING:
    from dexta_intelligence.agents.base import AgentContext


def recall_specs(ctx: AgentContext) -> list[ToolSpec]:
    """The structured shared-context channel over prior findings/hypotheses."""

    def recall(args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        return _recall(ctx, str(args.get("query", "")))

    return [
        ToolSpec(
            name="recall",
            description=(
                "What dexta already believes: prior findings (with status, confidence "
                "and the skeptic's confound notes), open questions, and cross-finding "
                "connections. Call FIRST for known patterns - it tells you what was "
                "already doubted so you pick better tools."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "topic, e.g. 'overnight'"}
                },
                "required": ["query"],
            },
            fn=recall,
        ),
    ]
