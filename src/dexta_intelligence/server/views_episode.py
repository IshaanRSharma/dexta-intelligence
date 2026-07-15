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

__all__ = ["episode_card_view"]

_KIND_TITLES = {
    "hypo": "Low episode",
    "hyper": "High episode",
    "sensor_gap": "Sensor gap",
}

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


def _detail_line(kind: str, detail: dict[str, Any]) -> str:
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


def _local(value: Any, tz: ZoneInfo) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        ts = datetime.fromisoformat(value)
    except ValueError:
        return None
    return ts.astimezone(tz) if ts.tzinfo is not None else ts
