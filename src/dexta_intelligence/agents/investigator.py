"""Investigator - the shared reasoning-investigation machinery.

This is the community-plugin surface. A domain agent is just configuration: an
``Investigator`` holds the generic plan → probe → judge → claim/wonder loop and
parameterizes everything that differs between domains (name, data requirement,
rigor seed, fallback sweep, prompt blurb, finding kind/scope, seed-headline
formatter). Subclass or instantiate it with your own tools and prompts.

The model plans hypotheses, picks tools, and judges results; it never computes a
statistic. Exploration is unguarded (tools are read-only); claims are gated by
``stats.rigor.assess`` and ``guard.faithfulness.audit``. Questions the data
cannot answer yet are banked as open hypotheses. Without a model it degrades to
a deterministic sweep over a fixed hypothesis set.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dexta_intelligence.agents import prompts
from dexta_intelligence.agents.base import DataRequirement
from dexta_intelligence.agents.tools.toolkit import (
    TOOL_SCHEMA_FOR_LLM,
    DiscoveryToolkit,
    ToolResult,
)
from dexta_intelligence.guard.faithfulness import audit
from dexta_intelligence.models import (
    Finding,
    FindingStats,
    Hypothesis,
    HypothesisStatus,
)
from dexta_intelligence.stats.rigor import assess

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.language_models.chat_models import BaseChatModel

    from dexta_intelligence.agents.base import AgentContext

logger = logging.getLogger(__name__)

__all__ = ["Investigator"]

#: Hard cap on tool calls per run - cheap insurance against runaway loops.
_DEFAULT_BUDGET = 8

_REFLECT_PROMPT = prompts.load("investigator_reflect")

_WRITE_PROMPT = prompts.load("investigator_write")


@dataclass
class _Plan:
    id: str
    claim: str
    tool: str
    args: dict[str, Any]


def _default_requires() -> DataRequirement:
    return DataRequirement()


def _generic_seed_headline(plan: _Plan, result: ToolResult) -> str:
    """Domain-agnostic headline: two-group delta with sample sizes."""
    s = result.summary
    a, b = s.get("label_a", "group A"), s.get("label_b", "group B")
    delta = s.get("delta", 0.0)
    direction = "higher" if delta > 0 else "lower"
    return f"{a} runs {abs(delta)} {direction} than {b} (n={s.get('n_a')}/{s.get('n_b')})."


@dataclass
class Investigator:
    """The generic plan → probe → judge → claim/wonder loop.

    A domain agent supplies configuration: ``name``/``requires`` (registry
    identity + gating), ``rigor_seed`` (permutation seed), ``fallback_plan``
    (deterministic sweep when no model), ``plan_prompt`` (domain blurb), the
    finding ``kind_prefix``/``scope``, and a ``seed_headline`` formatter. The
    machinery - tool probing, judging, rigor-gated claiming, guard auditing,
    wonder banking, JSON I/O - is shared.
    """

    name: str = "investigator"
    requires: DataRequirement = field(default_factory=_default_requires)
    rigor_seed: int = 0
    fallback_plan: tuple[dict[str, Any], ...] = ()
    plan_prompt: str = ""
    tool_schema: str = TOOL_SCHEMA_FOR_LLM
    kind_prefix: str = "investigator"
    scope: str = "investigator"
    seed_headline: Callable[[_Plan, ToolResult], str] = field(
        default=staticmethod(_generic_seed_headline)
    )
    model: BaseChatModel | None = None
    budget: int = _DEFAULT_BUDGET
    target_low: int = 70
    target_high: int = 180

    # ── run orchestration ─────────────────────────────────────────────────────

    def run(self, ctx: AgentContext) -> list[Finding]:
        toolkit = DiscoveryToolkit(ctx, target_low=self.target_low, target_high=self.target_high)
        plans = self._plan(ctx, toolkit)[: self.budget]
        rng = random.Random(self.rigor_seed)

        findings: list[Finding] = []
        wonders: list[Hypothesis] = []
        for plan in plans:
            result = toolkit.run(plan.tool, plan.args)
            if not result.ok:
                logger.debug("%s: %s failed: %s", self.name, plan.tool, result.error)
                continue
            verdict = self._judge(plan, result)
            if verdict == "drop":
                continue
            if verdict == "wonder":
                wonders.append(self._wonder(plan, result))
                continue
            finding = self._claim(plan, result, rng, ctx)
            if finding is not None:
                findings.append(finding)
            else:
                wonders.append(self._wonder(plan, result))

        for wonder in wonders:
            ctx.store.insert_hypothesis(wonder)
        return findings

    # ── plan ───────────────────────────────────────────────────────────────────

    def _plan(self, ctx: AgentContext, toolkit: DiscoveryToolkit) -> list[_Plan]:
        if self.model is None:
            return self._fallback_plans()
        prompt = self.plan_prompt.format(
            data_summary=toolkit.data_summary(),
            memory=_memory_digest(ctx),
            open_questions=_open_digest(ctx),
            tool_schema=self.tool_schema,
        )
        data = self._json_call(prompt, max_tokens=1800)
        raw = (data or {}).get("hypotheses") if isinstance(data, dict) else None
        if not isinstance(raw, list):
            logger.info("%s: planner returned nothing usable; using fallback sweep", self.name)
            return self._fallback_plans()
        plans: list[_Plan] = []
        for i, h in enumerate(raw):
            if not isinstance(h, dict) or "tool" not in h:
                continue
            plans.append(
                _Plan(
                    id=str(h.get("id", f"h{i + 1}")),
                    claim=str(h.get("claim", "")),
                    tool=str(h["tool"]),
                    args=dict(h.get("args") or {}),
                )
            )
        return plans

    def _fallback_plans(self) -> list[_Plan]:
        return [
            _Plan(p["id"], p["claim"], p["tool"], dict(p["args"])) for p in self.fallback_plan
        ]

    # ── judge ───────────────────────────────────────────────────────────────────

    def _judge(self, plan: _Plan, result: ToolResult) -> str:
        if self.model is None:
            strong = result.summary.get("interpretation") in ("moderate", "large")
            return "claim" if strong else "drop"
        prompt = _REFLECT_PROMPT.format(
            claim=plan.claim,
            tool=plan.tool,
            args=json.dumps(plan.args),
            result=json.dumps(result.summary, indent=2)[:1500],
        )
        data = self._json_call(prompt, max_tokens=200)
        verdict = (data or {}).get("verdict") if isinstance(data, dict) else None
        return verdict if verdict in ("claim", "wonder", "drop") else "drop"

    # ── claim (rigor-gated) ─────────────────────────────────────────────────────

    def _claim(
        self,
        plan: _Plan,
        result: ToolResult,
        rng: random.Random,
        ctx: AgentContext,
    ) -> Finding | None:
        verdict = assess(result.group_a, result.group_b, rng=rng)
        if verdict.verdict != "pass":
            logger.debug("%s: rigor demoted %s (%s)", self.name, plan.id, verdict.verdict)
            return None

        evidence = result.evidence()
        evidence.update(
            {
                "rigor_verdict": verdict.verdict,
                "rigor_reasons": list(verdict.reasons),
                "p_perm": verdict.p,
                "q_fdr": verdict.q,
            }
        )
        headline = self._write_headline(plan, result, evidence)
        delta = result.summary.get("delta")
        return Finding(
            agent=self.name,
            kind=f"{self.kind_prefix}_{plan.tool}",
            scope=self.scope,
            headline=headline,
            evidence=evidence,
            stats=FindingStats(
                effect_size=float(delta) if delta is not None else None,
                n=int(result.summary.get("n_a", 0)) + int(result.summary.get("n_b", 0)),
                p_perm=verdict.p,
                q_fdr=verdict.q,
                replicated=verdict.replicated,
            ),
            confidence=0.75,
            window_start=_window_dt(ctx, 0),
            window_end=_window_dt(ctx, 1),
        )

    def _write_headline(self, plan: _Plan, result: ToolResult, evidence: dict[str, Any]) -> str:
        deterministic = self.seed_headline(plan, result)
        if self.model is None:
            return deterministic
        prompt = _WRITE_PROMPT.format(
            claim=plan.claim, result=json.dumps(result.summary, indent=2)[:1200]
        )
        data = self._json_call(prompt, max_tokens=300)
        headline = (data or {}).get("headline") if isinstance(data, dict) else None
        if not isinstance(headline, str) or not headline.strip():
            return deterministic
        report = audit(headline, evidence)
        if not report.ok:
            logger.warning(
                "%s: faithfulness guard rejected headline %r (%d violation(s))",
                self.name,
                headline,
                len(report.violations),
            )
            return deterministic
        return headline.strip()

    # ── wonder (the curiosity backlog) ──────────────────────────────────────────

    def _wonder(self, plan: _Plan, result: ToolResult) -> Hypothesis:
        note = result.summary.get("error") or (
            f"{plan.tool}: {result.summary.get('interpretation', 'unclear')} effect"
            f" (delta={result.summary.get('delta')}, n={result.summary.get('n_a')}"
            f"/{result.summary.get('n_b')}) - revisit with more data"
        )
        return Hypothesis(
            statement=f"{plan.claim} [{note}]",
            status=HypothesisStatus.OPEN,
            tests=[{"tool": plan.tool, "args": plan.args, "summary": result.summary}],
        )

    # ── LLM I/O ──────────────────────────────────────────────────────────────────

    def _json_call(self, prompt: str, *, max_tokens: int) -> dict[str, Any] | None:
        if self.model is None:
            return None
        # Dict-form messages are accepted by every LangChain chat model, so the
        # call path never imports langchain_core - the optional ``llm`` extra is
        # only needed to *construct* the model, not to invoke it.
        messages = [
            {"role": "system", "content": "Respond with ONE JSON object only, no prose."},
            {"role": "user", "content": prompt},
        ]
        try:
            response = self.model.invoke(messages)
        except Exception:
            logger.warning("%s: LLM call failed", self.name, exc_info=True)
            return None
        return self._parse_json(response.content)

    def _parse_json(self, content: Any) -> dict[str, Any] | None:
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        else:
            return None
        text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            logger.warning("%s: non-JSON LLM response: %s", self.name, text[:200])
            return None
        return parsed if isinstance(parsed, dict) else None


# ── digests + window helper (shared) ──────────────────────────────────────────


def _memory_digest(ctx: AgentContext, *, limit: int = 8) -> str:
    findings = ctx.store.get_findings(limit=limit)
    if not findings:
        return "(nothing yet - this is an early run)"
    return "\n".join(f"- {f.headline}" for f in findings)


def _open_digest(ctx: AgentContext, *, limit: int = 8) -> str:
    try:
        hypotheses = ctx.store.get_hypotheses(status=HypothesisStatus.OPEN.value)
    except Exception:  # pragma: no cover - defensive over optional backend
        return "(none)"
    if not hypotheses:
        return "(none)"
    return "\n".join(f"- {h.statement}" for h in hypotheses[:limit])


def _window_dt(ctx: AgentContext, idx: int) -> Any:
    from datetime import UTC, datetime, time  # noqa: PLC0415

    edge = time.min if idx == 0 else time.max
    return datetime.combine(ctx.window[idx], edge, tzinfo=UTC)
