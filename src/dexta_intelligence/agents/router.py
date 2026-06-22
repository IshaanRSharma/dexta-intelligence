"""Router / supervisor agent - picks a tool family before the reasoning loop.

``tool_specs()`` hands the model one flat belt of ~13 instruments. As the belt
grows a single loop with everything in scope dilutes tool selection and inflates
the prompt. :class:`RouterAgent` runs one cheap classification (a JSON model call,
or a keyword fallback mirroring ``workflows/goals.py:_keyword_compose``), maps the
question to a :class:`Route` (a focused system prompt + a tool-name subset), then
runs the same ``run_reasoning_loop`` over only that subset and finishes through
``agents/chat.py:_finish`` - so the faithfulness guard runs on every route,
regardless of which subset was exposed.

The router never narrows away ``recall`` (memory grounding) or ``coverage``: every
route includes both. Filtering can never produce an empty tool list - an empty or
invalid route falls back to the full belt.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dexta_intelligence.agents import prompts
from dexta_intelligence.agents.chat import _finish
from dexta_intelligence.agents.reason import run_reasoning_loop
from dexta_intelligence.agents.tools.toolkit import DiscoveryToolkit, tool_specs

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

    from dexta_intelligence.agents.base import AgentContext
    from dexta_intelligence.agents.chat import ChatAnswer
    from dexta_intelligence.agents.reason import ToolSpec

logger = logging.getLogger(__name__)

__all__ = ["FAMILY_TOOLS", "Route", "RouterAgent"]

#: Tools every route keeps no matter what - memory grounding, data coverage,
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
    # Pure memory questions - what does dexta already believe?
    "memory": (),
    # Ground a pattern in published literature.
    "evidence": ("search_evidence",),
}

#: Family-specific system prompts layered on the shared safety preamble.
_SAFETY = prompts.with_safety(prompts.load("router_safety"))

_FAMILY_SYSTEM: dict[str, str] = {
    k: prompts.with_safety(prompts.load(f"router_family_{k}"))
    for k in ("spike_explanation", "time_traversal", "two_group", "memory", "evidence")
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

#: Default family when nothing matches - the broadest comparison surface.
_DEFAULT_FAMILY = "two_group"

_ROUTE_PROMPT = prompts.load("router_route")


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
    """Family from keywords - the fallback when the model is None or fails."""
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
            data = json.loads(_strip_code_fence(response))
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


def _strip_code_fence(response: Any) -> str:
    content = getattr(response, "content", response)
    if not isinstance(content, str):
        return str(content)
    stripped = content.strip().removeprefix("```json").removeprefix("```")
    return stripped.removesuffix("```").strip()
