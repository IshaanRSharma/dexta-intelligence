"""The fade gate — no cause claim without treatment-context inspection.

WAVE5 §4, enforced deterministically: for a cause question (why did I spike /
what caused this high), when insulin data exists, the answer may not claim a
likely contributor unless the tool trace shows the minimum inspection set.
The gate inspects ``ToolCall`` steps — it never asks a model whether the model
did its job.

Fade behavior (the caller implements the retry; this module only judges):
non-compliant → retry once with :attr:`GateReport.retry_hint` injected →
still non-compliant → replace the cause claim with :data:`SAFE_SENTENCE`.

Also enforces "research is downstream of findings": an answer whose only tool
support is ``search_evidence`` is non-compliant for a cause question.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dexta_intelligence.agents.reason import ToolCall
    from dexta_intelligence.coldstart import CapabilitySet

__all__ = [
    "NO_TREATMENT_DISCLAIMER",
    "SAFE_SENTENCE",
    "GateReport",
    "assess_trace",
    "is_cause_question",
]

#: Replaces a persistently non-compliant cause claim (WAVE5 §4, verbatim).
SAFE_SENTENCE = (
    "I can describe the glucose pattern, but I cannot make a strong cause "
    "hypothesis because treatment context was missing or not inspected."
)

#: Appended when insulin capability is absent (WAVE5 §4, verbatim).
NO_TREATMENT_DISCLAIMER = "Insulin/carb data unavailable. This is glucose-shape inference only."

#: Lowercase markers that make a question a *cause* question.
_CAUSE_MARKERS = (
    "why",
    "what caused",
    "what's causing",
    "what is causing",
    "what causes",
    "explain",
    "reason for",
    "what happened",
)

#: Tools that satisfy the "inspected the event itself" requirement.
_ZOOM_TOOLS = frozenset({"zoom_event", "find_spikes"})
#: Insulin-stream inspections required when insulin data exists.
_INSULIN_TOOLS = ("get_boluses", "get_basal_timeline")
#: Meal-stream inspection required when carb entries exist.
_MEAL_TOOLS = ("get_carb_entries",)
#: Tools that count as data work (research must come after one of these).
_DATA_TOOLS_EXEMPT = frozenset({"recall", "coverage", "search_evidence",
                                "get_current_time", "get_weekday", "parse_relative_date"})


@dataclass(frozen=True, slots=True)
class GateReport:
    """The deterministic verdict on one answer's tool path."""

    applies: bool
    compliant: bool
    insulin_available: bool
    missing: tuple[str, ...]
    research_only: bool

    @property
    def retry_hint(self) -> str:
        """The injected hint for the single fade retry — names the gap."""
        if self.research_only:
            return (
                "You cited literature without inspecting the data. Call the data "
                "tools first (zoom_event, get_carb_entries, get_boluses, "
                "get_basal_timeline), then ground the confirmed pattern."
            )
        if self.missing:
            return (
                "Before claiming a likely cause you must inspect treatment "
                f"context. You have not called: {', '.join(self.missing)}. "
                "Call them now, then answer."
            )
        return ""


def is_cause_question(question: str) -> bool:
    """Keyword classification, mirroring the router's fallback style."""
    text = question.lower()
    return any(marker in text for marker in _CAUSE_MARKERS)


def assess_trace(
    question: str,
    steps: Sequence[ToolCall],
    capabilities: CapabilitySet,
) -> GateReport:
    """Judge one answer's tool path against the WAVE5 §4 minimum.

    The required set adapts to capability: streams that do not exist cannot be
    required (their tools are hidden from the belt). When insulin is absent the
    report is *compliant* but ``insulin_available=False`` — the caller must
    carry :data:`NO_TREATMENT_DISCLAIMER` instead of a cause claim.
    """
    applies = is_cause_question(question)
    if not applies:
        return GateReport(
            applies=False, compliant=True, insulin_available=capabilities.has_insulin,
            missing=(), research_only=False,
        )
    called = {step.name for step in steps if step.ok}
    if not capabilities.has_insulin:
        return GateReport(
            applies=True, compliant=True, insulin_available=False,
            missing=(), research_only=False,
        )
    missing: list[str] = []
    if not (called & _ZOOM_TOOLS):
        missing.append("zoom_event")
    if capabilities.has_meals:
        missing.extend(t for t in _MEAL_TOOLS if t not in called)
    missing.extend(t for t in _INSULIN_TOOLS if t not in called)
    research_only = "search_evidence" in called and not (called - _DATA_TOOLS_EXEMPT)
    compliant = not missing and not research_only
    return GateReport(
        applies=True, compliant=compliant, insulin_available=True,
        missing=tuple(missing), research_only=research_only,
    )
