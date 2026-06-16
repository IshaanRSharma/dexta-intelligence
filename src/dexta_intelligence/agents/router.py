"""Router / supervisor agent — picks a tool family before the reasoning loop.

``tool_specs()`` hands the model one flat belt of ~13 instruments. As the belt
grows a single loop with everything in scope dilutes tool selection and inflates
the prompt. :class:`RouterAgent` runs one cheap classification (a JSON model call,
or a keyword fallback mirroring ``workflows/goals.py:_keyword_compose``), maps the
question to a :class:`Route` (a focused system prompt + a tool-name subset), then
runs the same ``run_reasoning_loop`` over only that subset and finishes through
``agents/chat.py:_finish`` — so the faithfulness guard runs on every route,
regardless of which subset was exposed.

The router never narrows away ``recall`` (memory grounding) or ``coverage``: every
route includes both. Filtering can never produce an empty tool list — an empty or
invalid route falls back to the full belt.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dexta_intelligence.agents.chat import _finish
from dexta_intelligence.agents.discovery_tools import DiscoveryToolkit, tool_specs
from dexta_intelligence.agents.reason import run_reasoning_loop

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

    from dexta_intelligence.agents.base import AgentContext
    from dexta_intelligence.agents.chat import ChatAnswer
    from dexta_intelligence.agents.reason import ToolSpec

logger = logging.getLogger(__name__)

__all__ = ["FAMILY_TOOLS", "Route", "RouterAgent"]

#: Tools every route keeps no matter what — memory grounding, data coverage,
#: and deterministic calendar resolution (LLMs don't understand time).
#: The router NEVER removes these.
_ALWAYS = ("recall", "coverage", "get_current_time", "get_weekday", "parse_relative_date")

#: family → the tool-name subset exposed to the reasoning loop for that family.
#: ``_ALWAYS`` is folded into every entry at build time, so each value here lists
#: only the family-specific instruments.
FAMILY_TOOLS: dict[str, tuple[str, ...]] = {
    # Traverse the record in time: orient, narrow, drill, read trends.
    "time_traversal": (
        "list_segments",
        "set_window",
        "zoom_event",
        "daily_series",
        "tod_compare",
        "groupby_compare",
    ),
    # Explain a spike/high: time traversal + treatment inspection + recurrence.
    "spike_explanation": (
        "list_segments",
        "set_window",
        "zoom_event",
        "daily_series",
        "find_spikes",
        "get_carb_entries",
        "get_boluses",
        "get_basal_timeline",
        "get_iob",
        "get_cob",
        "find_similar_events",
        "event_proximity",
        "search_evidence",
    ),
    # The two-group instruments: compare cohorts of days / events.
    "two_group": (
        "tod_compare",
        "groupby_compare",
        "event_proximity",
        "basal_overnight",
        "meal_response",
        "correction_outcome",
        "get_carb_entries",
        "get_boluses",
        "get_iob",
    ),
    # Pure memory questions — what does dexta already believe?
    "memory": (),
    # Ground a pattern in published literature.
    "evidence": ("search_evidence",),
}

#: Family-specific system prompts layered on the shared safety preamble.
_SAFETY = """You are dexta, a continuous health-intelligence assistant for one \
Type-1 diabetes patient. You reason over their real data using ONLY the tools \
provided — you never compute statistics yourself, you call a tool.

Hard rules:
- Observation and discussion only. NEVER give dosing, insulin, carb-ratio, or \
medication advice. If asked, say that is for their care team and offer to show \
the relevant pattern instead.
- Every number you state must come from a tool result you actually called.
- If the data cannot answer, say so plainly and say what would be needed.
Be concise and specific. Cite the n behind any comparison."""

_FAMILY_SYSTEM: dict[str, str] = {
    "spike_explanation": (
        _SAFETY
        + "\n\nThis question asks WHY a glucose event happened. Follow the "
        "investigation loop: resolve dates (parse_relative_date / get_current_time) "
        "→ list_segments to orient → set_window to the day → find_spikes / "
        "zoom_event to drill → get_carb_entries → get_boluses + get_iob → "
        "get_basal_timeline → find_similar_events for recurrence. NEVER claim a "
        "likely cause before inspecting carb entries, bolus timing, and basal "
        "context; if those tools are not available, say explicitly: "
        '"Insulin/carb data unavailable. This is glucose-shape inference only." '
        "Ground a confirmed pattern with search_evidence AFTER the data work. "
        "Phrase the conclusion as a pattern (e.g. 'more consistent with late "
        "meal insulin context than basal drift'), never as a dosing or timing "
        "directive."
    ),
    "time_traversal": (
        _SAFETY
        + "\n\nThis question is about CHANGE OVER TIME. Orient with list_segments, "
        "narrow with set_window, drill a spike with zoom_event, read the trend with "
        "daily_series, THEN compare windows with tod_compare / groupby_compare."
    ),
    "two_group": (
        _SAFETY
        + "\n\nThis question COMPARES two groups. Pick the instrument that matches "
        "(tod_compare for times of day, groupby_compare for day cohorts, "
        "event_proximity / meal_response / correction_outcome / basal_overnight for "
        "events) and report the delta, effect size, and n."
    ),
    "memory": (
        _SAFETY
        + "\n\nThis question is about what dexta ALREADY KNOWS. Use recall to surface "
        "prior findings and open questions; use coverage to frame how much data exists."
    ),
    "evidence": (
        _SAFETY
        + "\n\nThis question asks for CLINICAL EVIDENCE. Use search_evidence to ground a "
        "pattern in published literature; cite only returned PMIDs. Use recall first to "
        "anchor on the personal pattern being grounded."
    ),
}

#: Keyword → family, in priority order (first match wins). Mirrors the
#: keyword-fallback shape of ``workflows/goals.py:_keyword_compose``.
_KEYWORD_FAMILY: tuple[tuple[frozenset[str], str], ...] = (
    (
        frozenset(
            {
                "evidence",
                "study",
                "studies",
                "research",
                "literature",
                "pubmed",
                "clinical",
                "paper",
            }
        ),
        "evidence",
    ),
    (
        frozenset(
            {
                "remember",
                "already know",
                "known",
                "finding",
                "findings",
                "believe",
                "recall",
                "noticed before",
            }
        ),
        "memory",
    ),
    (
        frozenset(
            {
                "why",
                "cause",
                "caused",
                "what happened",
                "spike",
                "spiked",
                "went high",
                "stubborn high",
            }
        ),
        "spike_explanation",
    ),
    (
        frozenset(
            {
                "change",
                "changed",
                "trend",
                "over time",
                "month",
                "january",
                "february",
                "march",
                "april",
                "since",
                "zoom",
                "lately",
                "recently",
            }
        ),
        "time_traversal",
    ),
    (
        frozenset(
            {
                "compare",
                "versus",
                " vs ",
                "weekend",
                "weekday",
                "sleep",
                "workout",
                "meal",
                "bolus",
                "overnight",
                "correction",
                "after",
                "before",
            }
        ),
        "two_group",
    ),
)

#: Default family when nothing matches — the broadest comparison surface.
_DEFAULT_FAMILY = "two_group"

_ROUTE_PROMPT = """Classify this Type-1 patient's question into ONE tool family:

- spike_explanation: WHY a glucose event happened — explaining a spike, a high, \
a bad day, or a recurring post-meal pattern (e.g. "why did I spike on March 14", \
"what caused last night's high").
- time_traversal: how something CHANGED over time, trends, a specific month/week \
(e.g. "what changed in March vs April", "is my variability trending down").
- two_group: comparing two cohorts of days or events (weekend vs weekday, \
after-meal vs before, sleep, workouts, boluses).
- memory: what dexta ALREADY KNOWS — recalling prior findings or open questions.
- evidence: grounding a pattern in published clinical literature.

Question: "{question}"

Output STRICT JSON, no prose: {{"family": "<one family>"}}"""


@dataclass(frozen=True, slots=True)
class Route:
    name: str
    system: str
    tool_names: tuple[str, ...]


def _route_for(family: str) -> Route:
    """Build the :class:`Route` for a known family (``_ALWAYS`` always folded in)."""
    specific = FAMILY_TOOLS[family]
    names = _ALWAYS + tuple(n for n in specific if n not in _ALWAYS)
    return Route(name=family, system=_FAMILY_SYSTEM[family], tool_names=names)


def _keyword_route(question: str) -> Route:
    """Family from keywords — the fallback when the model is None or fails."""
    text = f" {question.lower()} "
    for keywords, family in _KEYWORD_FAMILY:
        if any(k in text for k in keywords):
            return _route_for(family)
    return _route_for(_DEFAULT_FAMILY)


@dataclass
class RouterAgent:
    model: BaseChatModel | None
    max_steps: int = 6
    target_low: int = 70
    target_high: int = 180

    def route(self, ctx: AgentContext, question: str) -> Route:
        """Pick a tool family. One cheap JSON model call; keyword fallback when
        the model is absent or the call/parse fails."""
        if self.model is None:
            return _keyword_route(question)
        try:
            response = self.model.invoke(
                [
                    {"role": "system", "content": "Respond with ONE JSON object only, no prose."},
                    {"role": "user", "content": _ROUTE_PROMPT.format(question=question)},
                ]
            )
            data = json.loads(_text_of(response))
            family = str(data["family"])
        except Exception:
            logger.warning("router: classification failed; keyword fallback", exc_info=True)
            return _keyword_route(question)
        if family not in FAMILY_TOOLS:
            return _keyword_route(question)
        return _route_for(family)

    def ask(self, ctx: AgentContext, question: str) -> ChatAnswer:
        """Route, expose only that family's tools, run the loop, audit via _finish."""
        route = self.route(ctx, question)
        toolkit = DiscoveryToolkit(ctx, target_low=self.target_low, target_high=self.target_high)
        belt = tool_specs(ctx, toolkit)
        wanted = set(route.tool_names)
        focused: list[ToolSpec] = [t for t in belt if t.name in wanted]
        if not focused:  # an invalid/empty route must never starve the loop
            focused = belt
        result = run_reasoning_loop(
            self.model,
            focused,
            system=route.system,
            user=question,
            max_steps=self.max_steps,
        )

        def rerun(hint: str) -> Any:
            # The fade retry runs over the FULL belt so the gate's named tools
            # exist even when the original route lacked them.
            return run_reasoning_loop(
                self.model,
                belt,
                system=f"{route.system}\n\nGATE: {hint}",
                user=question,
                max_steps=self.max_steps,
            )

        return _finish(
            result, question=question, capabilities=toolkit.capabilities(), rerun=rerun
        )


def _text_of(response: Any) -> str:
    content = getattr(response, "content", response)
    if not isinstance(content, str):
        return str(content)
    stripped = content.strip().removeprefix("```json").removeprefix("```")
    return stripped.removesuffix("```").strip()
