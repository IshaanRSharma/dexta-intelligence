"""Insulin agent - the batch curiosity loop over insulin/meal instruments.

A sibling of the Discovery agent, scoped to questions that need insulin and
meal context: overnight basal drift, meal-size excursions, and correction-bolus
outcomes (including rebound lows). A thin domain configuration over
:class:`Investigator`: it supplies the insulin/meal plan prompt, the
deterministic fallback sweep, the rigor seed, and a domain seed-headline
formatter; the shared machinery does the rest. The model never computes a
statistic, claims are gated by ``stats.rigor.assess`` and
``guard.faithfulness.audit``, and unanswerable questions are banked as open
hypotheses. Without a model it degrades to a deterministic sweep.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dexta_intelligence.agents.base import DataRequirement
from dexta_intelligence.agents.investigator import Investigator

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

    from dexta_intelligence.agents.investigator import _Plan
    from dexta_intelligence.agents.tools.toolkit import ToolResult

__all__ = ["AGENT_NAME", "InsulinAgent", "insulin_agent", "register_insulin"]

AGENT_NAME = "insulin"

#: Rigor permutation seed for the insulin loop (distinct from discovery's 71).
_INSULIN_SEED = 37
#: The deterministic fallback sweep when no model is configured.
_FALLBACK_PLAN: tuple[dict[str, Any], ...] = (
    {"id": "i1", "claim": "Bigger-carb meals drive larger glucose excursions.",
     "tool": "meal_response", "args": {"window_min": 120}},
    {"id": "i2", "claim": "Overnight glucose drift shifts across the run.",
     "tool": "basal_overnight", "args": {"hours": [0, 6]}},
    {"id": "i3", "claim": "Correction boluses move glucose, sometimes too far.",
     "tool": "correction_outcome", "args": {"window_min": 180}},
    {"id": "i4", "claim": "Glucose changes after a bolus.",
     "tool": "event_proximity", "args": {"event_type": "bolus", "window_min": 120}},
)

_PLAN_PROMPT = """You are the insulin & meal-response researcher for one Type-1 patient.
Form 3-5 SPECIFIC, TESTABLE hypotheses about how this patient's insulin and meals
move glucose (overnight basal drift, meal-size excursions, correction outcomes).
Each must be answerable by exactly one tool call.

DATA AVAILABLE
{data_summary}

WHAT YOU ALREADY BELIEVE (do not re-derive; build on or challenge these)
{memory}

QUESTIONS YOU BANKED EARLIER BUT COULD NOT ANSWER (revisit if data now allows)
{open_questions}

{tool_schema}

Prefer the insulin/meal instruments (basal_overnight, meal_response,
correction_outcome, event_proximity with event_type bolus). Output STRICT JSON,
no prose:
{{"hypotheses": [
  {{"id": "h1", "claim": "<=22 words, the suspected pattern",
    "tool": "<tool name>", "args": {{<exact tool args>}},
    "rationale": "<=18 words, why test this for THIS patient"}}
]}}"""


def _seed_headline(plan: _Plan, result: ToolResult) -> str:
    s = result.summary
    a, b = s.get("label_a", "group A"), s.get("label_b", "group B")
    delta = s.get("delta", 0.0)
    if plan.tool == "meal_response":
        return (
            f"{a} meals peak {abs(delta)} mg/dL further above baseline than {b} meals"
            f" (n={s.get('n_a')}/{s.get('n_b')})."
        )
    if plan.tool == "correction_outcome":
        rate = s.get("rebound_low_rate", 0.0)
        return (
            f"Post-bolus glucose moves {abs(delta)} mg/dL {a} vs {b};"
            f" {rate}% of boluses are followed by a low."
        )
    if plan.tool == "basal_overnight":
        direction = "more" if delta > 0 else "less"
        return (
            f"Overnight glucose drifts {abs(delta)} mg/dL {direction} in {a} than {b} nights"
            f" (n={s.get('n_a')}/{s.get('n_b')})."
        )
    if plan.tool == "event_proximity":
        verb = "rises" if delta > 0 else "falls"
        return f"After {plan.args.get('event_type')}, glucose {verb} {abs(delta)} mg/dL on average."
    direction = "higher" if delta > 0 else "lower"
    return f"{a} runs {abs(delta)} {direction} than {b} (n={s.get('n_a')}/{s.get('n_b')})."


@dataclass
class InsulinAgent(Investigator):
    """LLM-reasoning agent over insulin/meal instruments: plan -> probe -> judge."""

    name: str = AGENT_NAME
    requires: DataRequirement = field(
        default_factory=lambda: DataRequirement(min_span_days=14.0, needs_insulin=True)
    )
    rigor_seed: int = _INSULIN_SEED
    fallback_plan: tuple[dict[str, Any], ...] = _FALLBACK_PLAN
    plan_prompt: str = _PLAN_PROMPT
    kind_prefix: str = "insulin"
    scope: str = "insulin"
    seed_headline: Any = field(default=staticmethod(_seed_headline))


#: Default deterministic instance (no model) for registration without an LLM.
insulin_agent = InsulinAgent()


def register_insulin(
    registry: Any,
    *,
    model: BaseChatModel | None = None,
    target_low: int = 70,
    target_high: int = 180,
) -> None:
    """Register an Insulin agent on ``registry``.

    With ``model=None`` the agent runs its deterministic fallback sweep - same
    tools and rigor, no reasoning. Pass a model (built via the BYOM factory) to
    enable the full plan-probe-judge curiosity loop over insulin/meal patterns.
    """
    registry.register(
        InsulinAgent(model=model, target_low=target_low, target_high=target_high)
    )
