"""Chat agent - ``dexta ask "..."``, the conversational reasoning surface.

The model reasons over read-only tools (stats arsenal + recall over its own
memory) and decides per turn whether to compute or answer. Every number in the
answer must trace to a tool result from this conversation; untraceable figures
are flagged, never silently trusted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dexta_intelligence.agents import prompts
from dexta_intelligence.agents.reason import ReasoningResult, run_reasoning_loop
from dexta_intelligence.agents.tools.toolkit import DiscoveryToolkit, tool_specs
from dexta_intelligence.agents.trace import TraceLine, render_trace
from dexta_intelligence.guard.faithfulness import audit
from dexta_intelligence.guard.treatment_gate import (
    NO_TREATMENT_DISCLAIMER,
    SAFE_SENTENCE,
    assess_trace,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.language_models.chat_models import BaseChatModel

    from dexta_intelligence.agents.base import AgentContext
    from dexta_intelligence.coldstart import CapabilitySet

logger = logging.getLogger(__name__)

__all__ = ["ChatAgent", "ChatAnswer", "TraceLine"]

_SYSTEM = prompts.with_safety(prompts.load("chat_system"))


@dataclass(frozen=True, slots=True)
class ChatAnswer:
    text: str
    tools_used: tuple[str, ...]
    faithful: bool
    stopped_reason: str
    trace: tuple[TraceLine, ...] = ()
    violations: tuple[str, ...] = ()


@dataclass
class ChatAgent:
    model: BaseChatModel
    max_steps: int = 6
    target_low: int = 70
    target_high: int = 180

    def ask(self, ctx: AgentContext, question: str) -> ChatAnswer:
        toolkit = DiscoveryToolkit(ctx, target_low=self.target_low, target_high=self.target_high)
        specs = tool_specs(ctx, toolkit)
        result = run_reasoning_loop(
            self.model,
            specs,
            system=_SYSTEM,
            user=question,
            max_steps=self.max_steps,
        )

        def rerun(hint: str) -> ReasoningResult:
            return run_reasoning_loop(
                self.model,
                specs,
                system=f"{_SYSTEM}\n\nGATE: {hint}",
                user=question,
                max_steps=self.max_steps,
            )

        return _finish(
            result, question=question, capabilities=toolkit.capabilities(), rerun=rerun
        )


def _finish(
    result: ReasoningResult,
    *,
    question: str = "",
    capabilities: CapabilitySet | None = None,
    rerun: Callable[[str], ReasoningResult] | None = None,
) -> ChatAnswer:
    if question and capabilities is not None:
        result = _apply_gate(result, question, capabilities, rerun)
    tools_used = tuple(step.name for step in result.steps)
    trace = tuple(render_trace(result.steps))
    if not result.answer:
        if result.stopped_reason == "model_error" and result.error_detail:
            fallback = result.error_detail
        else:
            fallback = {
                "model_error": "The language model is unavailable right now.",
                "max_steps": "I ran out of reasoning steps before reaching a confident answer.",
            }.get(result.stopped_reason, "I could not produce an answer.")
        return ChatAnswer(
            fallback, tools_used, faithful=True, stopped_reason=result.stopped_reason, trace=trace
        )

    report = audit(result.answer, result.evidence)
    if not report.ok:
        logger.warning("chat: %d untraceable number(s) in answer", len(report.violations))
        violations = tuple(str(v) for v in report.violations)
        warned = (
            result.answer
            + "\n\n⚠️ Some figures above could not be traced to your data - "
            "treat them with caution."
        )
        return ChatAnswer(
            warned,
            tools_used,
            faithful=False,
            stopped_reason=result.stopped_reason,
            trace=trace,
            violations=violations,
        )
    return ChatAnswer(
        result.answer, tools_used, faithful=True, stopped_reason=result.stopped_reason, trace=trace
    )


def _apply_gate(  # noqa: PLR0911 - one return per gate outcome
    result: ReasoningResult,
    question: str,
    capabilities: CapabilitySet,
    rerun: Callable[[str], ReasoningResult] | None,
) -> ReasoningResult:
    """The fade gate: retry once with a hint, then the safe sentence.

    Deterministic - inspects the tool trace via ``guard.treatment_gate``; the
    one allowed retry is a fresh loop with the gate's hint injected into the
    system prompt. Evidence and steps from both attempts merge so the final
    audit sees everything."""
    if not result.answer:
        return result
    report = assess_trace(question, result.steps, capabilities)
    if not report.applies:
        return result
    if not report.insulin_available:
        if NO_TREATMENT_DISCLAIMER in result.answer:
            return result
        return ReasoningResult(
            answer=f"{result.answer}\n\n{NO_TREATMENT_DISCLAIMER}",
            steps=result.steps,
            evidence=result.evidence,
            stopped_reason=result.stopped_reason,
        )
    if report.compliant:
        return result
    steps = list(result.steps)
    evidence = {f"try0_{k}": v for k, v in result.evidence.items()}
    if rerun is not None:
        logger.info("treatment gate: retrying with hint (%s)", ", ".join(report.missing))
        retry = rerun(report.retry_hint)
        steps += retry.steps
        evidence.update({f"try1_{k}": v for k, v in retry.evidence.items()})
        if retry.answer and assess_trace(question, steps, capabilities).compliant:
            return ReasoningResult(
                answer=retry.answer,
                steps=steps,
                evidence=evidence,
                stopped_reason=retry.stopped_reason,
            )
    logger.warning("treatment gate: cause claim without treatment inspection - faded")
    return ReasoningResult(
        answer=SAFE_SENTENCE, steps=steps, evidence=evidence, stopped_reason="treatment_gate"
    )
