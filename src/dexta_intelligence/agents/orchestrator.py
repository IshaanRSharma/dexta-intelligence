"""Top-level orchestrator — the LLM decides the approach.

``run_reasoning_loop`` already lets the model pick granular tools. The
orchestrator widens that authority: its belt includes whole investigation
*workflows* as tools (e.g. ``investigate_spike``), so the model decides whether
to run a full audited investigation in one call, do ad-hoc tool work, or chain
both — and pivot on what it finds. Workflow SELECTION is the model's job, never
a keyword map.

Determinism lives only below the safety line: the instruments the model calls
(deterministic) and the two output rails in :func:`agents.chat._finish`
(faithfulness guard + treatment gate). The agent decides freely; the rails
refuse to emit an unsafe or unfaithful answer.

This replaces the family-classifier ``RouterAgent`` as the default ``dexta ask``
engine. The router remains as the lightweight keyword fallback when no model is
available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dexta_intelligence.agents.chat import _finish
from dexta_intelligence.agents.discovery_tools import DiscoveryToolkit, tool_specs
from dexta_intelligence.agents.reason import ReasoningEvent, ToolSpec, run_reasoning_loop
from dexta_intelligence.agents.trace import render_trace

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.language_models.chat_models import BaseChatModel

    from dexta_intelligence.agents.base import AgentContext
    from dexta_intelligence.agents.chat import ChatAnswer

logger = logging.getLogger(__name__)

__all__ = ["INVESTIGATION_DOCTRINE", "OrchestratorAgent", "workflow_tool_specs"]

#: The shared framing of what an investigation IS — composed, not pre-baked.
#: Both the chat orchestrator and the goal-seeker teach this so either can
#: compose investigations toward a conclusion.
INVESTIGATION_DOCTRINE = """An INVESTIGATION is a line of inquiry you COMPOSE to reach a \
defensible conclusion — never a single tool call. Its shape: orient (list_segments) → locate \
and narrow (set_window, find_spikes, zoom_event) → inspect treatment context (get_carb_entries, \
get_boluses, get_iob, get_cob, get_basal_timeline) → compare against history \
(find_similar_events; tod_compare / groupby_compare / basal_overnight only on windows with \
enough days — never on a single-day set_window) → conclude with the most consistent contributor, \
the evidence behind it, and what you could not check.

There is NO fixed menu of investigations — you BUILD the one the question needs from these \
instruments and pivot as the evidence directs. For a few common cases a certified shortcut \
exists (investigate_spike runs the spike line of inquiry in one audited call and returns a \
working_hypothesis to weigh, not to repeat); use a shortcut when it fits, otherwise compose \
the investigation yourself."""

_HARD_RULES = """Hard rules:
- Observation and discussion only. NEVER give dosing, insulin, carb-ratio, or medication \
advice — that is for their care team; offer to show the pattern instead.
- Every number you state must come from a tool result you actually called.
- If treatment data exists, inspect it (or run a shortcut that does) before naming a likely \
cause; if it does not, say "Insulin/carb data unavailable. This is glucose-shape inference \
only."
- Cite the n behind any comparison. Be concise and specific."""

_SYSTEM = (
    "You are dexta, the reasoning core of a continuous health-intelligence system for one "
    "Type-1 diabetes patient. You DECIDE how to investigate — you are not following a fixed "
    "script.\n\n" + INVESTIGATION_DOCTRINE + "\n\n" + _HARD_RULES
)


def workflow_tool_specs(
    ctx: AgentContext, *, target_low: int, target_high: int
) -> list[ToolSpec]:
    """Whole investigation workflows, exposed as tools the orchestrator can choose.

    Each runs a deterministic, audited investigation and returns its structured
    bundle plus the guard-auditable numbers it produced, so the orchestrator's
    final answer stays traceable.
    """

    def investigate_spike(args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        from dexta_intelligence.investigations.spike import (  # noqa: PLC0415
            gather_spike_evidence,
        )

        when = str(args.get("when", "")).strip()
        if not when:
            return {"error": "investigate_spike needs `when` (an ISO date or datetime)"}, {}
        threshold = float(args.get("threshold", 200.0))
        ev = gather_spike_evidence(
            ctx, when, threshold=threshold, target_low=target_low, target_high=target_high
        )
        result = {
            "working_hypothesis": ev.headline,
            "evidence": ev.evidence,
            "confidence": ev.confidence,
            "limitations": ev.limitations,
            "trace": [line.text for line in render_trace(ev.steps)],
        }
        # Namespace the inner pool so the guard can trace any number the
        # orchestrator cites from this workflow's bundle.
        numbers = {f"investigate_spike.{k}": v for k, v in ev.pool.items()}
        return result, numbers

    return [
        ToolSpec(
            name="investigate_spike",
            description=(
                "Run the full spike investigation for a day or moment and get a "
                "structured evidence bundle: working_hypothesis, evidence lines, "
                "confidence, limitations, trace. Internally inspects carbs, bolus "
                "timing, basal, IOB/COB, and similar past events. Use for 'why did "
                "I spike / go high' questions, then reason over the bundle."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "when": {
                        "type": "string",
                        "description": (
                            "ISO date (auto-locates the day's biggest excursion) "
                            "or ISO datetime of the event"
                        ),
                    },
                    "threshold": {"type": "number", "minimum": 140, "maximum": 400},
                },
                "required": ["when"],
            },
            fn=investigate_spike,
        ),
    ]


@dataclass
class OrchestratorAgent:
    """The top-level decider for ``dexta ask``: full belt + workflows-as-tools.

    No pre-filtering into a fixed family — the model sees every instrument and
    every workflow and chooses. The same guard + treatment gate run via
    :func:`agents.chat._finish`, so the rails are identical to every other surface.
    """

    model: BaseChatModel
    max_steps: int = 20
    target_low: int = 70
    target_high: int = 180

    def ask(
        self,
        ctx: AgentContext,
        question: str,
        *,
        on_event: Callable[[ReasoningEvent], None] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> ChatAnswer:
        toolkit = DiscoveryToolkit(
            ctx, target_low=self.target_low, target_high=self.target_high
        )
        belt = tool_specs(ctx, toolkit) + workflow_tool_specs(
            ctx, target_low=self.target_low, target_high=self.target_high
        )
        result = run_reasoning_loop(
            self.model,
            belt,
            system=_SYSTEM,
            user=question,
            max_steps=self.max_steps,
            on_event=on_event,
            history=history,
        )

        def rerun(hint: str) -> Any:
            return run_reasoning_loop(
                self.model,
                belt,
                system=f"{_SYSTEM}\n\nGATE: {hint}",
                user=question,
                max_steps=self.max_steps,
                history=history,
            )

        return _finish(
            result, question=question, capabilities=toolkit.capabilities(), rerun=rerun
        )
