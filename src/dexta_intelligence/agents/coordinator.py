"""Coordinator — the LLM-planned counterpart to the fixed deep-analysis fan-out.

``deep_analysis`` runs every producer unconditionally. The coordinator keeps the
same rails but lets the model DECIDE which investigations are worth running for a
given goal (or the whole record), composes a registry from that selection, runs
it under the same gating and exception isolation, then applies the skeptic
post-pass and optional synthesis.

Most producers are deterministic gated instruments; ``discovery`` is the one
open-ended arm (an LLM hypothesis sweep over the tool belt). The coordinator
plans which to run, runs them, then — with a model — may run ONE bounded
follow-up round that pivots on what the first round surfaced (only investigations
not already run; it stops as soon as there is nothing new to add).

The division of labour is strict: the LLM only PLANS (which investigations, and
whether a follow-up is warranted). It never computes or accepts a statistic —
rigor lives inside each producer, the skeptic re-checks every finding, and the
faithfulness guard gates prose. "LLM decides, determinism computes and gates."

Context discipline: the planning prompt never receives raw findings text. It
receives the compact ``recall`` digest (prior-finding headlines + skeptic
confound notes + open questions), so a long-running record cannot blow the
planning context budget no matter how many findings have accumulated.

With ``model=None`` planning degrades to the full producer set — exactly what
``deep_analysis`` would run — so the engine works with no API key.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dexta_intelligence.agents.base import AgentRegistry
from dexta_intelligence.agents.discovery_tools import _recall
from dexta_intelligence.agents.skeptic import skeptic_agent
from dexta_intelligence.config import Config
from dexta_intelligence.models import Finding, FindingStatus
from dexta_intelligence.workflows.lenses import PRODUCERS

if TYPE_CHECKING:
    from collections.abc import Iterable

    from langchain_core.language_models.chat_models import BaseChatModel

    from dexta_intelligence.agents.base import AgentContext

logger = logging.getLogger(__name__)

__all__ = ["CoordinatorAgent"]

_PLAN_PROMPT = """You plan a Type-1 diabetes data investigation. Pick which of the
available investigations to run for the goal below. Each is an independent producer
that returns statistically gated findings; the rigor gate and an independent skeptic
run AFTER you, so prefer running an investigation when in doubt rather than guessing
its result.

GOAL: {goal}

AVAILABLE INVESTIGATIONS:
{producers}

WHAT DEXTA ALREADY BELIEVES (do not re-derive these; pick investigations that
extend, challenge, or fill gaps around them):
{recall}

PAST INVESTIGATIONS (what was already run before — build on these, don't repeat):
{past}

Choose the subset most relevant to the goal. Selecting all is valid when the goal
is broad or the record is thin. Use ONLY names from the list above.

Output STRICT JSON: {{"investigations": ["<name>", ...], "reason": "<one sentence>"}}"""

_REPLAN_PROMPT = """A first round of investigations ran for this goal and produced the findings
below. Decide whether a focused FOLLOW-UP round is warranted — only investigations NOT already
run, chosen to drill into or challenge what the first round surfaced. If the first round already
covers the goal, return an empty list.

GOAL: {goal}
ALREADY RAN: {ran}
FIRST-ROUND FINDINGS:
{findings}
AVAILABLE (not yet run):
{remaining}

Output STRICT JSON (empty list = done):
{{"investigations": ["<name>", ...], "reason": "<one sentence>"}}"""

_PRODUCER_BLURBS: dict[str, str] = {
    "observation": "Descriptive glucose summaries (TIR, variability, time-of-day).",
    "pattern": "Two-group pattern tests (weekday/weekend, sleep, drift) under rigor.",
    "reconciliation": "Where logged/predicted behaviour diverges from outcomes.",
    "discovery": "Open-ended hypothesis sweep over the tool belt.",
    "insulin": "Insulin/bolus/correction-timing investigations (needs insulin data).",
}


@dataclass
class CoordinatorAgent:
    """Plans, composes, and runs deep investigations under the standard rails.

    The model (role ``discovery``) selects which producers run; everything that
    enforces honesty — data-requirement gating, per-agent exception isolation,
    rigor inside each agent, the skeptic re-check, the faithfulness guard — is
    deterministic and non-negotiable.
    """

    model: BaseChatModel | None = None
    config: Config = field(default_factory=Config)
    synthesize_connections: bool = True
    max_rounds: int = 2
    remember: bool = True

    def investigate(self, ctx: AgentContext, goal: str | None = None) -> list[Finding]:
        """Plan, run the selected producers, skeptic-review, optionally synthesize.

        With a model, after the first round it may run bounded follow-up rounds
        that pivot on what was surfaced (only investigations not already run,
        stopping as soon as there is nothing new). When ``remember`` is set, the
        process (which investigations ran, for which goal, and what came back) is
        saved to memory so a later run can recall and build on it. Returns
        reviewed findings; the caller persists. Never raises on thin data or
        planner failure — it degrades to the full producer set, never an exception.
        """
        if self.model is None:
            reviewed = self._run_round(ctx, list(PRODUCERS))
            self._remember(ctx, goal, set(PRODUCERS), reviewed)
            return reviewed

        ran: set[str] = set()
        reviewed_all: list[Finding] = []
        selected = self._plan(ctx, goal)
        for round_idx in range(max(1, self.max_rounds)):
            fresh = [name for name in selected if name not in ran]
            if not fresh:
                break
            ran.update(fresh)
            reviewed_all.extend(self._run_round(ctx, fresh))
            if round_idx + 1 >= self.max_rounds:
                break
            selected = self._replan(ctx, goal, reviewed_all, ran)
            if not selected:
                break

        if self.synthesize_connections and reviewed_all:
            self._synthesize(reviewed_all)
        self._remember(ctx, goal, ran, reviewed_all)
        return reviewed_all

    def _remember(
        self, ctx: AgentContext, goal: str | None, ran: set[str], reviewed: list[Finding]
    ) -> None:
        """Save the investigation process as memory for future runs — which
        investigations ran for this goal and what they returned. A later plan
        recalls these so the coordinator builds on prior work instead of repeating it."""
        if not self.remember or not ran:
            return
        producers = sorted(ran)
        kinds = sorted({f.kind for f in reviewed})
        headline = (
            f"Investigated {goal or 'the whole record'}: ran {', '.join(producers)} "
            f"→ {len(reviewed)} finding(s)"
        )
        try:
            ctx.store.insert_finding(
                Finding(
                    agent="coordinator",
                    kind="investigation",
                    scope=(goal or "whole_record")[:120],
                    headline=headline,
                    body_md=headline,
                    evidence={
                        "goal": goal or "",
                        "producers": producers,
                        "n_findings": len(reviewed),
                        "finding_kinds": kinds,
                    },
                    confidence=1.0,
                    status=FindingStatus.ACTIVE,
                )
            )
        except Exception:
            logger.warning("coordinator: failed to record investigation memory", exc_info=True)

    def _run_round(self, ctx: AgentContext, names: list[str]) -> list[Finding]:
        """Run the named producers under gating + isolation, then skeptic-review."""
        registry = self._build_registry(names)
        raw = registry.run_all(ctx, on_skip=_log_skip)
        return skeptic_agent.review(raw, ctx)

    # ── plan ─────────────────────────────────────────────────────────────────

    def _plan(self, ctx: AgentContext, goal: str | None) -> list[str]:
        """Return the producer names to run (full set when no model / on failure)."""
        full = list(PRODUCERS)
        if self.model is None:
            return full

        prompt = _PLAN_PROMPT.format(
            goal=goal or "Investigate the whole record for anything notable.",
            producers=_producer_catalog(),
            recall=_recall_digest(ctx, goal),
            past=_past_investigations(ctx),
        )
        data = self._json_call(prompt)
        raw = data.get("investigations") if isinstance(data, dict) else None
        if not isinstance(raw, list):
            logger.info("coordinator: planner returned nothing usable; running full set")
            return full

        selected = [name for name in raw if isinstance(name, str) and name in PRODUCERS]
        if not selected:
            logger.info("coordinator: planner selected no known producer; running full set")
            return full
        # Preserve catalog order; drop duplicates.
        return [name for name in full if name in set(selected)]

    def _build_registry(self, selected: list[str]) -> AgentRegistry:
        """Producers only — the skeptic is applied separately via ``review`` so it
        runs over the collected findings rather than re-reading the store."""
        registry = AgentRegistry()
        for name in selected:
            PRODUCERS[name](registry, self.config, self.model)
        return registry

    def _replan(
        self, ctx: AgentContext, goal: str | None, reviewed: list[Finding], ran: set[str]
    ) -> list[str]:
        """Given the findings so far, choose follow-up investigations to drill in.

        Only producers not already run are eligible; an empty result (the planner
        is satisfied, or nothing is left) ends the loop. Never raises."""
        remaining = [name for name in PRODUCERS if name not in ran]
        if not remaining or self.model is None:
            return []
        prompt = _REPLAN_PROMPT.format(
            goal=goal or "the whole record",
            ran=", ".join(sorted(ran)) or "none",
            findings=_findings_digest(reviewed),
            remaining=_producer_catalog(remaining),
        )
        data = self._json_call(prompt)
        raw = data.get("investigations") if isinstance(data, dict) else None
        if not isinstance(raw, list):
            return []
        wanted = {name for name in raw if isinstance(name, str)}
        return [name for name in remaining if name in wanted]

    # ── synthesis ────────────────────────────────────────────────────────────

    def _synthesize(self, reviewed: list[Finding]) -> None:
        from dexta_intelligence.memory.synthesis import synthesize  # noqa: PLC0415

        try:
            synthesize(reviewed, self.model)
        except Exception:
            logger.warning("coordinator: synthesis failed; continuing", exc_info=True)

    # ── LLM I/O ──────────────────────────────────────────────────────────────

    def _json_call(self, prompt: str) -> dict[str, Any] | None:
        if self.model is None:
            return None
        messages = [
            {"role": "system", "content": "Respond with ONE JSON object only, no prose."},
            {"role": "user", "content": prompt},
        ]
        try:
            response = self.model.invoke(messages)
        except Exception:
            logger.warning("coordinator: planning LLM call failed", exc_info=True)
            return None
        return _parse_json(response.content)


def _producer_catalog(names: Iterable[str] = PRODUCERS) -> str:
    return "\n".join(f"- {name}: {_PRODUCER_BLURBS.get(name, 'investigation')}" for name in names)


def _findings_digest(reviewed: list[Finding]) -> str:
    if not reviewed:
        return "(no findings yet)"
    return "\n".join(f"- {f.headline}" for f in reviewed[:20])


def _recall_digest(ctx: AgentContext, goal: str | None) -> str:
    """Compact recall summary for the planner: headlines + skeptic notes + open questions.

    Deliberately the structured ``recall`` payload, never raw finding bodies, so
    planning context stays bounded as findings accumulate.
    """
    try:
        payload, _numbers = _recall(ctx, goal or "")
    except Exception:
        return "(nothing recalled yet)"
    lines: list[str] = []
    for item in payload.get("findings", []):
        head = item.get("headline", "")
        note = item.get("skeptic_notes")
        lines.append(f"- {head}" + (f" [skeptic: {note}]" if note else ""))
    for question in payload.get("open_questions", []):
        lines.append(f"- open question: {question}")
    return "\n".join(lines) if lines else "(nothing yet — early run)"


def _past_investigations(ctx: AgentContext) -> str:
    """Recall prior investigation recipes (the saved process) for the planner."""
    try:
        records = ctx.store.get_findings(
            agent="coordinator", kind="investigation", status=FindingStatus.ACTIVE, limit=8
        )
    except Exception:
        return "(none yet)"
    return "\n".join(f"- {r.headline}" for r in records) if records else "(none yet)"


def _log_skip(name: str, reasons: list[str]) -> None:
    logger.info("coordinator: skipping %s: %s", name, "; ".join(reasons))


def _parse_json(content: Any) -> dict[str, Any] | None:
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
        logger.warning("coordinator: non-JSON planner response: %s", text[:200])
        return None
    return parsed if isinstance(parsed, dict) else None
