"""Top-level orchestrator - the LLM decides the approach.

``run_reasoning_loop`` already lets the model pick granular tools. The
orchestrator widens that authority: its belt includes whole investigation
*workflows* as tools (e.g. ``investigate_spike``), so the model decides whether
to run a full audited investigation in one call, do ad-hoc tool work, or chain
both - and pivot on what it finds. Workflow SELECTION is the model's job, never
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

from dexta_intelligence.agents import prompts
from dexta_intelligence.agents.chat import _finish
from dexta_intelligence.agents.investigation import seed_belief_from_store
from dexta_intelligence.agents.reason import ReasoningEvent, ToolSpec, run_reasoning_loop
from dexta_intelligence.agents.tools.toolkit import DiscoveryToolkit, tool_specs
from dexta_intelligence.agents.trace import render_trace

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.language_models.chat_models import BaseChatModel

    from dexta_intelligence.agents.base import AgentContext
    from dexta_intelligence.agents.chat import ChatAnswer
    from dexta_intelligence.agents.investigation import BeliefState

logger = logging.getLogger(__name__)

__all__ = ["INVESTIGATION_DOCTRINE", "OrchestratorAgent", "workflow_tool_specs"]

#: The shared framing of what an investigation IS - composed, not pre-baked.
#: Both the chat orchestrator and the goal-seeker teach this so either can
#: compose investigations toward a conclusion.
INVESTIGATION_DOCTRINE = prompts.load("orchestrator_doctrine")

_SYSTEM = prompts.with_safety(prompts.load("orchestrator_system"))

#: Appended when a working belief state is threaded through the loop, so the model
#: keeps its understanding explicit between probes. It scaffolds; it never decides.
_BELIEF_DIRECTIVE = (
    "Maintain a working belief state with update_belief. After each probe, record "
    "your competing hypotheses and their status, the evidence so far, any gap "
    "still blocking you, and your confidence. Probe to discriminate between live "
    "hypotheses; conclude when one is clearly supported or say what is missing. "
    "update_belief returns suggested_probe: the most discriminating evidence you "
    "have not gathered yet for your open hypotheses. Use it unless you have a "
    "better reason."
)


def _belief_directive(belief: BeliefState) -> str:
    """The belief directive, plus any hypotheses carried in from prior runs.

    Seeded hypotheses must be in the model's view from the first turn to steer
    probing; the update_belief tool only returns the state once the model calls
    it, so they go in the prompt here.
    """
    if not belief.hypotheses:
        return _BELIEF_DIRECTIVE
    carried = "\n".join(f"- [{hid}] {h.statement}" for hid, h in belief.hypotheses.items())
    return (
        f"{_BELIEF_DIRECTIVE}\n\nOpen hypotheses carried from prior analysis "
        f"(discriminate or refute these; reuse the bracketed id in update_belief "
        f"to change a hypothesis's status):\n{carried}"
    )


def workflow_tool_specs(ctx: AgentContext, *, target_low: int, target_high: int) -> list[ToolSpec]:
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

    No pre-filtering into a fixed family - the model sees every instrument and
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
        toolkit = DiscoveryToolkit(ctx, target_low=self.target_low, target_high=self.target_high)
        belt = tool_specs(ctx, toolkit) + workflow_tool_specs(
            ctx, target_low=self.target_low, target_high=self.target_high
        )
        belief = seed_belief_from_store(ctx)
        system = f"{_SYSTEM}\n\n{_belief_directive(belief)}"
        result = run_reasoning_loop(
            self.model,
            belt,
            system=system,
            user=question,
            max_steps=self.max_steps,
            on_event=on_event,
            history=history,
            belief=belief,
        )

        def rerun(hint: str) -> Any:
            return run_reasoning_loop(
                self.model,
                belt,
                system=f"{system}\n\nGATE: {hint}",
                user=question,
                max_steps=self.max_steps,
                history=history,
                belief=belief,
            )

        return _finish(result, question=question, capabilities=toolkit.capabilities(), rerun=rerun)
