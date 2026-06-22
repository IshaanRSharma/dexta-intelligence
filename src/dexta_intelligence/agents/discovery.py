"""Discovery agent - the batch curiosity loop (plan → probe → judge → claim/wonder).

A thin domain configuration over :class:`Investigator`: it supplies the
glucose-pattern plan prompt, the deterministic fallback sweep, the rigor seed,
and a domain seed-headline formatter. All the reasoning machinery (tool probing,
judging, rigor-gated claiming, guard auditing, wonder banking) lives in the
shared :class:`Investigator`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dexta_intelligence.agents import prompts
from dexta_intelligence.agents.base import DataRequirement
from dexta_intelligence.agents.investigator import Investigator

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

    from dexta_intelligence.agents.investigator import _Plan
    from dexta_intelligence.agents.tools.toolkit import ToolResult

__all__ = ["AGENT_NAME", "DiscoveryAgent", "discovery_agent", "register_discovery"]

AGENT_NAME = "discovery"

#: Rigor permutation seed for discovery; the skeptic deliberately re-runs with another.
_DISCOVERY_SEED = 71
#: The deterministic fallback sweep when no model is configured.
_FALLBACK_PLAN: tuple[dict[str, Any], ...] = (
    {"id": "f1", "claim": "Overnight glucose differs from midday.",
     "tool": "tod_compare", "args": {"hours_a": [3, 7], "hours_b": [11, 15]}},
    {"id": "f2", "claim": "Weekend control differs from weekdays.",
     "tool": "groupby_compare", "args": {"group_by": "weekend", "target": "tir_pct"}},
    {"id": "f3", "claim": "Poorer sleep tracks with higher next-day glucose.",
     "tool": "groupby_compare", "args": {"group_by": "sleep_bucket", "target": "mean_glucose"}},
    {"id": "f4", "claim": "Glucose rises after meals.",
     "tool": "event_proximity", "args": {"event_type": "meal", "window_min": 120}},
)

_PLAN_PROMPT = prompts.load("discovery_plan")


def _seed_headline(plan: _Plan, result: ToolResult) -> str:
    s = result.summary
    a, b = s.get("label_a", "group A"), s.get("label_b", "group B")
    delta = s.get("delta", 0.0)
    if plan.tool == "event_proximity":
        verb = "rises" if delta > 0 else "falls"
        return f"After {plan.args.get('event_type')}, glucose {verb} {abs(delta)} mg/dL on average."
    metric = "TIR" if plan.args.get("target") == "tir_pct" else "mean glucose"
    direction = "higher" if delta > 0 else "lower"
    return f"{a} {metric} runs {abs(delta)} {direction} than {b} (n={s.get('n_a')}/{s.get('n_b')})."


@dataclass
class DiscoveryAgent(Investigator):
    """LLM-reasoning agent: plan -> probe -> judge -> (claim | wonder | drop)."""

    name: str = AGENT_NAME
    requires: DataRequirement = field(
        default_factory=lambda: DataRequirement(min_span_days=21.0, min_glucose_coverage_pct=50.0)
    )
    rigor_seed: int = _DISCOVERY_SEED
    fallback_plan: tuple[dict[str, Any], ...] = _FALLBACK_PLAN
    plan_prompt: str = _PLAN_PROMPT
    kind_prefix: str = "discovery"
    scope: str = "discovery"
    seed_headline: Any = field(default=staticmethod(_seed_headline))


#: Default deterministic instance (no model) for registration without an LLM.
discovery_agent = DiscoveryAgent()


def register_discovery(
    registry: Any,
    *,
    model: BaseChatModel | None = None,
    target_low: int = 70,
    target_high: int = 180,
) -> None:
    """Register a Discovery agent on ``registry``.

    With ``model=None`` the agent runs its deterministic fallback sweep - same
    tools and rigor, no reasoning - so a vanilla, LLM-free install still gets
    discovery. Pass a model (built via the BYOM factory) to enable the full
    plan-probe-judge curiosity loop.
    """
    registry.register(
        DiscoveryAgent(model=model, target_low=target_low, target_high=target_high)
    )
