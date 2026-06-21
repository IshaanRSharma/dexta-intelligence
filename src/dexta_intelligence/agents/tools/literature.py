"""The search_evidence tool: ground a confirmed pattern in published literature."""

from __future__ import annotations

from dexta_intelligence.agents.reason import ToolSpec
from dexta_intelligence.agents.tools.toolkit import _search_evidence


def literature_specs() -> list[ToolSpec]:
    """Clinical-literature search (PubMed by default)."""
    return [
        ToolSpec(
            name="search_evidence",
            description=(
                "Search clinical literature (PubMed). Use to ground a confirmed personal "
                "pattern in published evidence or note contradiction. Cite only returned PMIDs."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "clinical search terms"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 8},
                },
                "required": ["query"],
            },
            fn=_search_evidence,
        ),
    ]
