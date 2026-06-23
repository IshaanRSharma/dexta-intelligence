"""Trace timeline view-model — shared by persisted investigation runs and templates."""

from __future__ import annotations

from dexta_intelligence.agents.trace import trace_icon_for_line

__all__ = [
    "answer_faithfulness_flagged",
    "faithfulness_violations_from_answer",
    "trace_entries_from_lines",
    "trace_icon_for_text",
]


def trace_icon_for_text(text: str) -> str:
    """Best-effort icon for a persisted trace line (stored as plain text)."""
    return trace_icon_for_line(text)


def trace_entries_from_lines(lines: list[str] | None) -> list[dict[str, str]]:
    """Shape plain trace strings into ``{icon, text}`` for ``_trace_timeline.html``."""
    out: list[dict[str, str]] = []
    for line in lines or []:
        if not line or not line.strip():
            continue
        icon = trace_icon_for_line(line)
        out.append({"icon": icon, "text": line})
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
