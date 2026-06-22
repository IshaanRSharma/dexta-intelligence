"""Trace timeline view-model — shared by persisted investigation runs and templates."""

from __future__ import annotations

__all__ = ["trace_entries_from_lines", "trace_icon_for_text"]

# Keyword hints map persisted trace prose (text-only) to timeline icons.
_ICON_HINTS: tuple[tuple[str, str], ...] = (
    ("zoomed", "zoom"),
    ("excursion", "zoom"),
    ("spike", "zoom"),
    ("narrowed to", "scope"),
    ("scanned", "scan"),
    ("checked what i already know", "recall"),
    ("manual context", "recall"),
    ("literature", "scan"),
    ("compared", "compare"),
    ("similar", "compare"),
    (" vs ", "compare"),
    ("day-by-day", "trend"),
    ("bolus", "treatment"),
    ("carb", "treatment"),
    ("basal", "treatment"),
    ("insulin profile", "treatment"),
    ("anchored 'now'", "time"),
    ("resolved relative date", "time"),
    ("weekday", "time"),
)


def trace_icon_for_text(text: str) -> str:
    """Best-effort icon for a persisted trace line (stored as plain text)."""
    low = text.lower()
    for hint, icon in _ICON_HINTS:
        if hint in low:
            return icon
    return "scope"


_GLYPHS: dict[str, str] = {
    "zoom": "⌖",
    "scope": "◧",
    "compare": "⇔",
    "recall": "◎",
    "scan": "◉",
    "trend": "↗",
    "treatment": "◆",
    "time": "◷",
}


def trace_entries_from_lines(lines: list[str] | None) -> list[dict[str, str]]:
    """Shape plain trace strings into ``{icon, glyph, text}`` for ``_trace_timeline.html``."""
    out: list[dict[str, str]] = []
    for line in lines or []:
        if not line or not line.strip():
            continue
        icon = trace_icon_for_text(line)
        out.append({"icon": icon, "glyph": _GLYPHS.get(icon, "•"), "text": line})
    return out


def answer_faithfulness_flagged(answer: str | None) -> bool:
    """Heuristic when ``InvestigationRun`` has no stored ``faithful`` flag."""
    if not answer:
        return False
    low = answer.lower()
    return "could not be traced" in low or "treat them with caution" in low


def faithfulness_violations_from_answer(answer: str | None) -> list[str]:
    """Extract short violation hints from a flagged answer, if any."""
    if not answer_faithfulness_flagged(answer):
        return []
    return ["figures in the answer could not be fully traced to tool evidence"]
