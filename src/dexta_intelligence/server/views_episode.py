"""Shape an episode node (the explain_episode dict) for the why-chain card.

Pure formatting over the deterministic tool output: every value shown comes
straight from the episode dict, no re-derivation and no model. Tolerates
missing keys so a partial dict degrades to a plainer card instead of raising.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from zoneinfo import ZoneInfo

__all__ = ["episode_card_view", "episode_chain_view"]

_KIND_TITLES = {
    "hypo": "Low episode",
    "hyper": "High episode",
    "sensor_gap": "Sensor gap",
}

_KIND_SHORT = {"hypo": "Low", "hyper": "High", "sensor_gap": "Gap"}

_EXTREME_NOUN = {"hypo": "nadir", "hyper": "peak"}


def episode_card_view(episode: dict[str, Any], tz: ZoneInfo) -> dict[str, Any]:
    """One episode dict -> the template view for ``_episode_card.html``."""
    kind = str(episode.get("kind", ""))
    start = _local(episode.get("start"), tz)
    end = _local(episode.get("end"), tz)
    span = ""
    if start is not None and end is not None:
        span = f"{start.strftime('%b %d, %H:%M')} to {end.strftime('%H:%M')}"
    duration = episode.get("duration_min")
    extreme = episode.get("extreme_mg_dl")
    extreme_label = ""
    if isinstance(extreme, (int, float)):
        noun = _EXTREME_NOUN.get(kind, "extreme")
        extreme_label = f"{extreme:g} mg/dL {noun}"
    edges = [
        _edge_view(link, tz)
        for link in episode.get("links") or []
        if isinstance(link, dict)
    ]
    return {
        "title": _KIND_TITLES.get(kind, "Episode"),
        "kind": kind,
        "span": span,
        "duration": f"{duration:g} min" if isinstance(duration, (int, float)) else "",
        "extreme": extreme_label,
        "severe": bool(episode.get("severe")),
        "clinically_significant": bool(episode.get("clinically_significant")),
        "edges": edges,
    }


def _edge_view(link: dict[str, Any], tz: ZoneInfo) -> dict[str, Any]:
    kind = str(link.get("kind", "context"))
    offset = link.get("offset_min")
    offset_label = f"{offset:+g} min" if isinstance(offset, (int, float)) else ""
    ts = _local(link.get("ts"), tz)
    return {
        "kind": kind,
        "offset": offset_label,
        "at": ts.strftime("%H:%M") if ts is not None else "",
        "detail": _detail_line(kind, link.get("detail") or {}),
    }


def _treatment_line(detail: dict[str, Any]) -> str:
    halves: list[str] = []
    carbs = detail.get("carbs_g")
    units = detail.get("units")
    if isinstance(carbs, (int, float)):
        halves.append(f"{carbs:g} g carbs")
    if isinstance(units, (int, float)):
        halves.append(f"{units:g} U")
    parts = [" + ".join(halves)] if halves else []
    if detail.get("note"):
        parts.append(str(detail["note"]))
    return ", ".join(parts)


def _detail_line(kind: str, detail: dict[str, Any]) -> str:
    if kind == "treatment":
        return _treatment_line(detail)
    parts: list[str] = []
    if kind == "meal":
        carbs = detail.get("carbs_g")
        if isinstance(carbs, (int, float)):
            parts.append(f"{carbs:g} g carbs")
        if detail.get("note"):
            parts.append(str(detail["note"]))
    elif kind == "bolus":
        units = detail.get("units")
        if isinstance(units, (int, float)):
            parts.append(f"{units:g} U")
        if detail.get("automatic"):
            parts.append("automatic")
    elif kind == "activity":
        if detail.get("kind"):
            parts.append(str(detail["kind"]))
        intensity = detail.get("intensity")
        if isinstance(intensity, (int, float)):
            parts.append(f"intensity {intensity:g}")
    elif kind == "sleep":
        score = detail.get("score")
        if isinstance(score, (int, float)):
            parts.append(f"score {score:g}")
    return ", ".join(parts)


def episode_chain_view(episode: dict[str, Any], tz: ZoneInfo) -> dict[str, Any] | None:
    """The episode-to-episode chain around this node as a left-to-right sequence.

    Renders the deterministic ``EpisodeEdge`` chain the agent traversed (via
    ``explain_episode``): the predecessor episodes that led into this one and the
    successors that followed, each connector labelled with the descriptive
    relation and the load-bearing bridge event in the gap. Returns ``None`` when
    the node has no chain, so the chat only shows the strip when there is a
    sequence to show. Every value comes from the tool result; no model, no
    re-derivation.
    """
    chain = episode.get("chain")
    if not isinstance(chain, dict):
        return None
    incoming = [
        _chain_step(edge, "in", tz) for edge in chain.get("in") or [] if isinstance(edge, dict)
    ]
    outgoing = [
        _chain_step(edge, "out", tz) for edge in chain.get("out") or [] if isinstance(edge, dict)
    ]
    incoming = [step for step in incoming if step is not None]
    outgoing = [step for step in outgoing if step is not None]
    if not incoming and not outgoing:
        return None
    kind = str(episode.get("kind", ""))
    return {
        "incoming": incoming,
        "outgoing": outgoing,
        "this": {
            "kind": kind,
            "short": _KIND_SHORT.get(kind, "Episode"),
            "at": _id_time(episode.get("id"), tz),
        },
    }


def _chain_step(edge: dict[str, Any], direction: str, tz: ZoneInfo) -> dict[str, Any] | None:
    other_id = edge.get("src_id") if direction == "in" else edge.get("dst_id")
    kind = _id_kind(other_id)
    if not kind:
        return None
    relation = str(edge.get("relation", "")).replace("_", " ")
    gap = edge.get("gap_min")
    bridge = edge.get("bridge")
    return {
        "direction": direction,
        "relation": relation,
        "relation_key": str(edge.get("relation", "")),
        "gap": f"{gap:g} min" if isinstance(gap, (int, float)) else "",
        "bridge": _bridge_label(bridge) if isinstance(bridge, dict) else "",
        "node": {
            "kind": kind,
            "short": _KIND_SHORT.get(kind, "Episode"),
            "at": _id_time(other_id, tz),
        },
    }


def _bridge_label(bridge: dict[str, Any]) -> str:
    """The load-bearing gap event as a short phrase, e.g. ``16 g rescue carbs``."""
    kind = str(bridge.get("kind", ""))
    detail = bridge.get("detail") or {}
    if kind in ("meal", "treatment"):
        line = _treatment_line(detail) if kind == "treatment" else _detail_line("meal", detail)
        return line or "carbs"
    if kind == "bolus":
        units = detail.get("units")
        return f"{units:g} U bolus" if isinstance(units, (int, float)) else "bolus"
    return _detail_line(kind, detail) or kind


def _id_kind(episode_id: Any) -> str:
    """Episode ids are ``kind:isotimestamp``; the prefix is the kind."""
    if isinstance(episode_id, str) and ":" in episode_id:
        return episode_id.split(":", 1)[0]
    return ""


def _id_time(episode_id: Any, tz: ZoneInfo) -> str:
    if not isinstance(episode_id, str) or ":" not in episode_id:
        return ""
    ts = _local(episode_id.split(":", 1)[1], tz)
    return ts.strftime("%b %d, %H:%M") if ts is not None else ""


def _local(value: Any, tz: ZoneInfo) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        ts = datetime.fromisoformat(value)
    except ValueError:
        return None
    return ts.astimezone(tz) if ts.tzinfo is not None else ts
