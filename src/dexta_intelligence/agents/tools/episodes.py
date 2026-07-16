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


_KIND_WORD = {"hypo": "low", "hyper": "high", "sensor_gap": "sensor gap"}
_EXTREME_NOUN = {"hypo": "nadir", "hyper": "peak"}


def _hhmm(iso: Any) -> str:
    if not isinstance(iso, str):
        return ""
    try:
        return datetime.fromisoformat(iso).strftime("%H:%M")
    except ValueError:
        return ""


def _id_parts(episode_id: Any) -> tuple[str, str]:
    """``kind:isotimestamp`` -> (kind word, HH:MM)."""
    if isinstance(episode_id, str) and ":" in episode_id:
        kind, iso = episode_id.split(":", 1)
        return _KIND_WORD.get(kind, "episode"), _hhmm(iso)
    return "episode", ""


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _bridge_phrase(bridge: dict[str, Any]) -> str:
    """The load-bearing gap event as prose for the model, e.g. ``16 g carbs``."""
    kind = str(bridge.get("kind", ""))
    detail = bridge.get("detail") or {}
    carbs, units, note = detail.get("carbs_g"), detail.get("units"), detail.get("note")
    if kind in ("meal", "treatment"):
        halves = []
        if _num(carbs):
            halves.append(f"{carbs:g} g carbs")
        if kind == "treatment" and _num(units):
            halves.append(f"{units:g} U")
        phrase = " + ".join(halves) if halves else "carbs"
        return f"{phrase} ({note})" if note else phrase
    if kind == "bolus":
        return f"{units:g} U insulin" if _num(units) else "insulin"
    return kind


def _edge_clause(edge: dict[str, Any], direction: str) -> str:
    other_id = edge.get("src_id") if direction == "in" else edge.get("dst_id")
    word, at = _id_parts(other_id)
    relation = str(edge.get("relation", "")).replace("_", " ")
    gap = edge.get("gap_min")
    bridge = edge.get("bridge")
    lead = "followed a" if direction == "in" else "was followed by a"
    clause = f"{lead} {word} episode"
    if at:
        clause += f" at {at}"
    clause += f" ({relation}"
    if _num(gap):
        clause += f", {gap:g} min gap"
    clause += ")"
    if isinstance(bridge, dict):
        clause += f", bridged by {_bridge_phrase(bridge)}"
    return clause


def _episode_summary(result: dict[str, Any]) -> str:
    """A deterministic one-to-three-sentence rendering of an episode and its chain,
    handed to the model as the first thing it reads so it reasons over a curated
    causal narrative instead of re-parsing nested JSON. Every value comes from the
    result dict; no model, no re-derivation."""
    kind = str(result.get("kind", ""))
    word = _KIND_WORD.get(kind, "episode")
    facts = []
    ext, dur = result.get("extreme_mg_dl"), result.get("duration_min")
    if _num(ext):
        facts.append(f"{ext:g} mg/dL {_EXTREME_NOUN.get(kind, 'extreme')}")
    if _num(dur):
        facts.append(f"{dur:g} min")
    head = f"A {word} episode"
    if facts:
        head += " (" + ", ".join(facts) + ")"
    sentences = [head + "."]
    chain = result.get("chain") or {}
    clauses = [_edge_clause(e, "in") for e in chain.get("in") or []]
    clauses += [_edge_clause(e, "out") for e in chain.get("out") or []]
    if clauses:
        sentences.append("It " + "; and it ".join(clauses) + ".")
    n_links = len(result.get("links") or [])
    if n_links:
        kinds = sorted({str(link.get("kind")) for link in result["links"]})
        sentences.append(f"Context around it: {n_links} event(s) ({', '.join(kinds)}).")
    return " ".join(sentences)


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
        body = ep.to_dict()
        chain = graph.edges_for(ep.id)
        body["chain"] = {
            "in": [e.to_dict() for e in chain["in"]],
            "out": [e.to_dict() for e in chain["out"]],
        }
        # ``summary`` first: a deterministic causal narrative the model reads before
        # the raw fields, and the one field guaranteed to survive context-budget
        # reduction (see reason._fit_tool_result), so the model always sees the
        # chain it is meant to reason with.
        result = {"summary": _episode_summary(body), **body}
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
