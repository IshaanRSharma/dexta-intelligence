"""The fade gate - no cause claim without treatment-context inspection.

Enforced deterministically: for a cause question (why did I spike /
what caused this high), when insulin data exists, the answer may not claim a
likely contributor unless the tool trace shows the minimum inspection set.
The gate inspects ``ToolCall`` steps - it never asks a model whether the model
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
    "LOW_CARB_CAVEAT",
    "NO_TREATMENT_DISCLAIMER",
    "SAFE_SENTENCE",
    "GateReport",
    "assess_trace",
    "is_cause_question",
]

#: Replaces a persistently non-compliant cause claim (verbatim).
SAFE_SENTENCE = (
    "I can describe the glucose pattern, but I cannot make a strong cause "
    "hypothesis because treatment context was missing or not inspected."
)

#: Appended when insulin capability is absent (verbatim).
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

#: Lowercase markers that make a cause question specifically about LOWS. For a
#: low, a highs finder (find_spikes) is the wrong instrument and insulin-on-board
#: is the factor that matters most, so the required set differs from a spike.
_LOW_MARKERS = (
    "go low", "going low", "goes low", "went low", "low after", "lows",
    "hypo", "crash", "dip below", "below target", "below 70",
)
#: Localizers that count for a lows/pattern question - event_proximity is the
#: lows-equivalent of zooming a spike.
_LOW_LOCALIZE_TOOLS = _ZOOM_TOOLS | frozenset(
    {"event_proximity", "get_context_around_event", "correlate"}
)
#: Insulin-on-board / outcome integrators - the inspection that matters for a low
#: (get_iob integrates basal+bolus into the on-board amount at the event).
_IOB_TOOLS = frozenset({"get_iob", "correction_outcome"})
#: Carb-stream inspections (broadened beyond the single get_carb_entries).
_BROAD_MEAL_TOOLS = frozenset({"get_carb_entries", "get_cob", "meal_response"})

#: Appended to a lows answer whose insulin context WAS inspected but whose carb
#: context was not - a non-fatal gap, surfaced instead of faded to silence.
LOW_CARB_CAVEAT = (
    "I have not inspected carb intake around these events, so treat this as a "
    "partial explanation - under-fueling could also contribute."
)
#: Tools that count as data work (research must come after one of these).
_DATA_TOOLS_EXEMPT = frozenset({"recall", "coverage", "search_evidence",
                                "get_current_time", "get_weekday", "parse_relative_date"})

#: Workflow tools that internally perform the full treatment inspection - a
#: successful call covers the granular requirements (the workflow provably
#: inspects carbs/boluses/basal when the data exists).
_COMPOSITE_COVERS: dict[str, frozenset[str]] = {
    "investigate_spike": frozenset(
        {"zoom_event", "find_spikes", "get_carb_entries", "get_boluses", "get_basal_timeline"}
    ),
}


@dataclass(frozen=True, slots=True)
class GateReport:
    """The deterministic verdict on one answer's tool path."""

    applies: bool
    compliant: bool
    insulin_available: bool
    missing: tuple[str, ...]
    research_only: bool
    #: A non-fatal gap to surface alongside a compliant answer (e.g. insulin was
    #: inspected for a lows question but carbs were not). Empty when none.
    caveat: str = ""

    @property
    def retry_hint(self) -> str:
        """The injected hint for the single fade retry - names the gap."""
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
    """Judge one answer's tool path against the minimum.

    The required set adapts to capability: streams that do not exist cannot be
    required (their tools are hidden from the belt). When insulin is absent the
    report is *compliant* but ``insulin_available=False`` - the caller must
    carry :data:`NO_TREATMENT_DISCLAIMER` instead of a cause claim.
    """
    applies = is_cause_question(question)
    if not applies:
        return GateReport(
            applies=False, compliant=True, insulin_available=capabilities.has_insulin,
            missing=(), research_only=False,
        )
    called = {step.name for step in steps if step.ok}
    for composite, covers in _COMPOSITE_COVERS.items():
        if composite in called:
            called = called | covers
    if not capabilities.has_insulin:
        return GateReport(
            applies=True, compliant=True, insulin_available=False,
            missing=(), research_only=False,
        )
    research_only = "search_evidence" in called and not (called - _DATA_TOOLS_EXEMPT)
    if _is_low_question(question):
        return _assess_low(called, capabilities, research_only)
    missing: list[str] = []
    if not (called & _ZOOM_TOOLS):
        missing.append("zoom_event")
    if capabilities.has_meals:
        missing.extend(t for t in _MEAL_TOOLS if t not in called)
    missing.extend(t for t in _INSULIN_TOOLS if t not in called)
    compliant = not missing and not research_only
    return GateReport(
        applies=True, compliant=compliant, insulin_available=True,
        missing=tuple(missing), research_only=research_only,
    )


def _is_low_question(question: str) -> bool:
    """True when the cause question is specifically about going low/hypo."""
    text = question.lower()
    return any(marker in text for marker in _LOW_MARKERS)


def _assess_low(
    called: set[str], capabilities: CapabilitySet, research_only: bool
) -> GateReport:
    """Compliance for a lows question.

    Insulin-on-board is the factor that matters for a low, so the hard
    requirement is that insulin was inspected (an IOB/outcome integrator, or the
    classic bolus+basal pair) and the event localized (no highs finder required).
    A missing carb inspection is a non-fatal caveat, not a fade: under-fueling is
    a secondary contributor, and silencing a sound insulin-grounded answer over it
    is worse than surfacing the gap.
    """
    missing: list[str] = []
    if not (called & _LOW_LOCALIZE_TOOLS):
        missing.append("zoom_event")
    insulin_ok = bool(called & _IOB_TOOLS) or {"get_boluses", "get_basal_timeline"} <= called
    if not insulin_ok:
        missing.append("get_iob")
    compliant = not missing and not research_only
    caveat = ""
    if compliant and capabilities.has_meals and not (called & _BROAD_MEAL_TOOLS):
        caveat = LOW_CARB_CAVEAT
    return GateReport(
        applies=True, compliant=compliant, insulin_available=True,
        missing=tuple(missing), research_only=research_only, caveat=caveat,
    )
