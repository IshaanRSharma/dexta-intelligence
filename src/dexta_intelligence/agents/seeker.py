"""Goal-seeking replanning loop - the agent reflects on whether it answered.

``run_reasoning_loop`` stops on the first turn with no tool calls or at
``max_steps``; it never asks *"did I actually answer the goal?"*. The
:class:`GoalSeekingAgent` wraps the loop: after each round it runs a
``reflect()`` model step that judges the answer against the original goal and,
if unmet, feeds the gap forward as a hint into the next round.

It sits *on top of* the time-traversal seam. One :class:`DiscoveryToolkit` is
built once and reused across rounds, so the active sub-window - the agent's
working memory of *where it is in time* - carries forward: a round-1 "narrow to
March" persists into round 2 unless the model re-scopes again.

Guard contract: evidence is accumulated across *all* rounds (every round's
``ReasoningResult.evidence`` is merged) and the final answer is audited against
that merged pool via :func:`agents.chat._finish` - so a number computed in
round 1 and cited in the round-3 answer is still traceable, never false-rejected.
The final ``ChatAnswer.tools_used`` reflects the union of tool calls. The guard
is never bypassed; no dosing/treatment content (chat's system prompt forbids it
and is reused as the base here).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dexta_intelligence.agents import prompts
from dexta_intelligence.agents.chat import ChatAnswer, _finish
from dexta_intelligence.agents.orchestrator import workflow_tool_specs
from dexta_intelligence.agents.reason import ReasoningResult, ToolCall, run_reasoning_loop
from dexta_intelligence.agents.tools.toolkit import DiscoveryToolkit, tool_specs

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

    from dexta_intelligence.agents.base import AgentContext

logger = logging.getLogger(__name__)

__all__ = ["GoalSeekingAgent", "Reflection"]

_REFLECT_PROMPT = prompts.load("seeker_reflect")


@dataclass(frozen=True, slots=True)
class Reflection:
    """The replanning verdict on one round's answer against the goal."""

    satisfied: bool
    missing: str
    next_hint: str
    reason: str = ""


@dataclass
class GoalSeekingAgent:
    """Constrained goal-seeking - never an open-ended loop.

    Hard limits, all enforced in code: ``max_rounds`` defaults to 2; the loop
    also stops when a round calls no tool not already called, when the evidence
    pool did not grow, when the round repeats the prior round's tool sequence,
    or when the reflection cannot name a real exposed tool."""

    model: BaseChatModel | None
    max_rounds: int = 2
    max_steps: int = 6
    target_low: int = 70
    target_high: int = 180

    def pursue(self, ctx: AgentContext, goal: str) -> ChatAnswer:
        """Run the goal-seeking loop and return the guard-audited final answer.

        One toolkit + tool spec list is constructed once and reused across
        rounds so the active window carries forward. Each round runs the
        reasoning loop, then a reflection step; the loop stops when reflection
        is satisfied or ``max_rounds`` is exhausted. Evidence and steps from
        every round are merged, and the final answer is audited against the
        merged pool.
        """
        if self.model is None:
            return _finish(ReasoningResult(answer="", stopped_reason="model_error"))

        toolkit = DiscoveryToolkit(ctx, target_low=self.target_low, target_high=self.target_high)
        # Goal pursuit gets the same belt as the orchestrator - instruments PLUS
        # investigation shortcuts - so a goal can compose investigations, not just
        # read a metric.
        specs = tool_specs(ctx, toolkit) + workflow_tool_specs(
            ctx, target_low=self.target_low, target_high=self.target_high
        )
        spec_names = {spec.name for spec in specs}

        merged_evidence: dict[str, Any] = {}
        all_steps: list[ToolCall] = []
        last_result: ReasoningResult | None = None
        hint = ""
        trace_summary = ""
        seen_names: set[str] = set()
        prev_sequence: tuple[str, ...] | None = None

        for round_idx in range(self.max_rounds):
            system = _round_system(trace_summary, hint)
            user = goal if round_idx == 0 else _round_user(goal, hint)
            result = run_reasoning_loop(
                self.model, specs, system=system, user=user, max_steps=self.max_steps
            )
            last_result = result
            evidence_before = len(merged_evidence)
            _merge_round(merged_evidence, all_steps, result, round_idx)

            reflection = self._reflect(goal, result)
            if reflection.satisfied:
                break
            round_names = tuple(step.name for step in result.steps)
            if _must_stop(
                round_idx=round_idx,
                round_names=round_names,
                seen_names=seen_names,
                prev_sequence=prev_sequence,
                evidence_grew=len(merged_evidence) > evidence_before,
                reflection=reflection,
                spec_names=spec_names,
            ):
                break
            seen_names |= set(round_names)
            prev_sequence = round_names
            hint = _next_hint(reflection)
            trace_summary = _trace_summary(result)

        final = ReasoningResult(
            answer=last_result.answer if last_result is not None else "",
            steps=all_steps,
            evidence=merged_evidence,
            stopped_reason=last_result.stopped_reason if last_result is not None else "answered",
        )
        return _finish(final, question=goal, capabilities=toolkit.capabilities())

    # ── reflection ───────────────────────────────────────────────────────────

    def _reflect(self, goal: str, result: ReasoningResult) -> Reflection:
        """Ask the model whether the round answered the goal; fall back to
        ``satisfied=True`` when the model is absent or unparsable so the loop
        never spins forever."""
        if self.model is None or not result.answer:
            return Reflection(satisfied=True, missing="", next_hint="")
        tools = ", ".join(step.name for step in result.steps) or "none"
        prompt = _REFLECT_PROMPT.format(goal=goal, answer=result.answer, tools=tools)
        messages = [
            {"role": "system", "content": "Respond with ONE JSON object only, no prose."},
            {"role": "user", "content": prompt},
        ]
        try:
            response = self.model.invoke(messages)
            data = json.loads(_text_of(response))
        except Exception:
            logger.warning("seeker: reflection failed; treating as satisfied", exc_info=True)
            return Reflection(satisfied=True, missing="", next_hint="")
        if not isinstance(data, dict):
            return Reflection(satisfied=True, missing="", next_hint="")
        return Reflection(
            satisfied=bool(data.get("satisfied", True)),
            missing=str(data.get("missing", "")),
            next_hint=str(data.get("next_tool_hint", data.get("next_hint", ""))),
            reason=str(data.get("reason", "")),
        )


# ── hard stop conditions (bounded agency) ─────────────────────────


def _must_stop(
    *,
    round_idx: int,
    round_names: tuple[str, ...],
    seen_names: set[str],
    prev_sequence: tuple[str, ...] | None,
    evidence_grew: bool,
    reflection: Reflection,
    spec_names: set[str],
) -> bool:
    """True when continuing cannot help. All checks are plain code - the model
    never gets to argue for another round."""
    if round_idx > 0 and not (set(round_names) - seen_names):
        logger.info("seeker: stop - no new tool was called")
        return True
    if round_idx > 0 and not evidence_grew:
        logger.info("seeker: stop - evidence did not increase")
        return True
    if prev_sequence is not None and round_names == prev_sequence:
        logger.info("seeker: stop - same tool sequence repeated")
        return True
    hint = f"{reflection.missing} {reflection.next_hint}"
    if not any(name in hint for name in spec_names):
        logger.info("seeker: stop - reflection names no available tool")
        return True
    return False


# ── round prompt shaping ─────────────────────────────────────────────────────


#: Goal pursuit = composing investigations across rounds toward a conclusion
#: about the goal, layered on the shared chat rails + investigation doctrine.
_GOAL_SYSTEM = prompts.with_safety(prompts.load("seeker_goal_system"))


def _round_system(trace_summary: str, hint: str) -> str:
    if not trace_summary and not hint:
        return _GOAL_SYSTEM
    addendum = ["", "PRIOR ROUND:"]
    if trace_summary:
        addendum.append(trace_summary)
    if hint:
        addendum.append(
            f"You did not fully answer yet. Still missing: {hint} "
            "Compose or extend an investigation to close this gap before answering."
        )
    return _GOAL_SYSTEM + "\n".join(addendum)


def _round_user(goal: str, hint: str) -> str:
    if not hint:
        return goal
    return f'{goal}\n\nFocus this round on what is still missing: {hint}'


def _next_hint(reflection: Reflection) -> str:
    parts = [p for p in (reflection.missing, reflection.next_hint) if p]
    return " - ".join(parts)


def _trace_summary(result: ReasoningResult) -> str:
    if not result.steps:
        return "Last round called no tools."
    names = ", ".join(step.name for step in result.steps)
    return f"Last round called: {names}."


# ── accumulation ─────────────────────────────────────────────────────────────


def _merge_round(
    pool: dict[str, Any],
    steps: list[ToolCall],
    result: ReasoningResult,
    round_idx: int,
) -> None:
    """Fold one round's evidence and steps into the running totals.

    Evidence keys are namespaced by round so a round-2 call cannot clobber a
    round-1 number of the same name - both stay in the merged pool the guard
    audits the final answer against.
    """
    for key, value in result.evidence.items():
        pool[f"r{round_idx}_{key}"] = value
    steps.extend(result.steps)


def _text_of(response: Any) -> str:
    content = getattr(response, "content", response)
    if not isinstance(content, str):
        return str(content)
    stripped = content.strip().removeprefix("```json").removeprefix("```")
    return stripped.removesuffix("```").strip()
