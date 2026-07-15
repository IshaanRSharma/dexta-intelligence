"""Episode-graph tools: let the reasoning loop query and traverse the temporal
episode graph (:mod:`dexta_intelligence.analytics.episodes`).

Two instruments, an overview and a drill-down, so an agent reasons over segmented
excursions instead of re-deriving them from a raw trace:

- ``episodes``: every hypo/hyper excursion and sensor gap in the active window as
  addressable nodes (each with an ``id``), plus the roll-up counts.
- ``explain_episode``: one node, named by ``id`` or located by ``timestamp``, with
  its typed context edges (the meals, boluses, activity, and sleep around it). The
  traversal behind "why did I go high/low then".
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from dexta_intelligence.agents.reason import ToolSpec
from dexta_intelligence.analytics.episodes import Episode, build_graph

if TYPE_CHECKING:
    from dexta_intelligence.agents.base import AgentContext
    from dexta_intelligence.agents.tools.toolkit import DiscoveryToolkit

_SUMMARY_KEYS = (
    "num_hypo", "num_hyper", "n_sensor_gaps", "n_clinically_significant_hypo",
    "n_severe_hypo", "n_severe_hyper", "longest_hyper_min", "longest_hypo_min",
)


def _node_row(ep: Episode) -> dict[str, Any]:
    return {
        "id": ep.id, "kind": ep.kind, "start": ep.start.isoformat(),
        "duration_min": ep.duration_min, "extreme_mg_dl": ep.extreme_mg_dl,
        "severe": ep.severe, "clinically_significant": ep.clinically_significant,
        "n_links": len(ep.links),
    }


def episode_specs(ctx: AgentContext, toolkit: DiscoveryToolkit) -> list[ToolSpec]:
    low, high = toolkit.target_range()

    def _graph() -> Any:
        start, end = toolkit.active_window()
        return build_graph(ctx.store, start, end, target_low=low, target_high=high)

    def episodes(args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        graph = _graph()
        kind = args.get("kind")
        top_n = max(1, min(int(args.get("top_n", 10)), 50))
        nodes = [e for e in graph.episodes if not kind or e.kind == kind]
        nodes.sort(key=lambda e: -e.duration_min)
        summary = graph.summary()
        result = {"summary": summary, "episodes": [_node_row(e) for e in nodes[:top_n]]}
        return result, {k: summary[k] for k in _SUMMARY_KEYS}

    def explain_episode(args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        graph = _graph()
        ep: Episode | None = None
        episode_id = str(args.get("episode_id", "")).strip()
        if episode_id:
            ep = graph.node(episode_id)
        elif args.get("timestamp"):
            try:
                ep = graph.at(datetime.fromisoformat(str(args["timestamp"])))
            except ValueError:
                return {"error": "timestamp must be ISO-8601"}, {}
        if ep is None:
            return {"error": "no episode matched; call episodes to list valid ids"}, {}
        result = ep.to_dict()
        chain = graph.edges_for(ep.id)
        result["chain"] = {
            "in": [e.to_dict() for e in chain["in"]],
            "out": [e.to_dict() for e in chain["out"]],
        }
        numbers: dict[str, Any] = {"duration_min": ep.duration_min}
        if ep.extreme_mg_dl is not None:
            numbers["extreme_mg_dl"] = ep.extreme_mg_dl
        for link in ep.links:
            for key in ("carbs_g", "units", "score"):
                if isinstance(link.detail.get(key), (int, float)):
                    numbers[f"{link.kind}_{key}"] = link.detail[key]
        return result, numbers

    return [
        ToolSpec(
            name="episodes",
            description=(
                "Every hypo/hyper excursion and sensor gap in the ACTIVE window as "
                "addressable nodes {id, kind, start, duration_min, extreme_mg_dl, "
                "severe, clinically_significant, n_links}, longest first, plus roll-up "
                "counts (num_hypo, num_hyper, longest_hyper_min, sensor gaps). Use to "
                "survey excursions or to get an episode id for explain_episode."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["hypo", "hyper", "sensor_gap"]},
                    "top_n": {"type": "integer", "minimum": 1, "maximum": 50},
                },
            },
            fn=episodes,
        ),
        ToolSpec(
            name="explain_episode",
            description=(
                "One episode with its typed context edges: the meals, boluses, "
                "activity, and sleep around it, each with a signed minute offset from "
                "the episode start. A carb entry and the manual bolus recorded as one "
                "action appear as a single 'treatment' edge (carbs + units). Name it "
                "by episode_id (from episodes) or locate it by timestamp. The "
                "instrument for 'why did I go high/low then' - traverse the episode "
                "to its context instead of guessing from a trace. Includes 'chain': "
                "typed edges to the previous/next episode when they sit within 3 h "
                "(rebound_after_low, low_after_high, follows), each with the "
                "load-bearing bridge event in the gap when one exists."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "episode_id": {"type": "string"},
                    "timestamp": {"type": "string", "description": "ISO-8601 moment"},
                },
            },
            fn=explain_episode,
        ),
    ]
