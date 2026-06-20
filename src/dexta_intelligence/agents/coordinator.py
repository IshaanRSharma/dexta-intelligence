"""Coordinator - the LLM-planned counterpart to the fixed deep-analysis fan-out.

``deep_analysis`` runs every producer unconditionally. The coordinator keeps the
same rails but lets the model DECIDE which investigations are worth running for a
given goal (or the whole record), composes a registry from that selection, runs
it under the same gating and exception isolation, then applies the skeptic
post-pass and optional synthesis.

Most producers are deterministic gated instruments; ``discovery`` is the one
open-ended arm (an LLM hypothesis sweep over the tool belt). The coordinator
plans which to run, runs them, then - with a model - may run ONE bounded
follow-up round that pivots on what the first round surfaced (only investigations
not already run; it stops as soon as there is nothing new to add).

The division of labour is strict: the LLM only PLANS (which investigations, and
whether a follow-up is warranted). It never computes or accepts a statistic -
rigor lives inside each producer, the skeptic re-checks every finding, and the
faithfulness guard gates prose. "LLM decides, determinism computes and gates."

Context discipline: the planning prompt never receives raw findings text. It
receives the compact ``recall`` digest (prior-finding headlines + skeptic
confound notes + open questions), so a long-running record cannot blow the
planning context budget no matter how many findings have accumulated.

With ``model=None`` planning degrades to the full producer set - exactly what
``deep_analysis`` would run - so the engine works with no API key.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from dexta_intelligence.agents.base import AgentRegistry
from dexta_intelligence.agents.discovery_tools import _recall
from dexta_intelligence.agents.skeptic import skeptic_agent
from dexta_intelligence.config import Config
from dexta_intelligence.models import Finding, FindingStatus, InvestigationRun, RunFinding
from dexta_intelligence.workflows.lenses import PRODUCERS

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from langchain_core.language_models.chat_models import BaseChatModel

    from dexta_intelligence.agents.base import AgentContext

logger = logging.getLogger(__name__)

__all__ = ["CoordinatorAgent", "RunTrace"]


@dataclass
class RunTrace:
    """Mutable recorder the coordinator fills during an investigation.

    Callers pass one in to capture the observable process (planned producers,
    step-by-step trace lines, final status) for persistence as an
    :class:`~dexta_intelligence.models.InvestigationRun`. Passing nothing leaves
    behaviour unchanged.

    Set ``on_event`` to stream the process live (the GUI passes a queue sink so
    the Investigations page can show plan -> trace as it happens). Each event is
    a ``{"kind": str, "payload": dict}`` mapping. The sink must not raise.
    """

    plan: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    status: str = "completed"
    coverage_summary: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    on_event: Callable[[dict[str, Any]], None] | None = None

    def emit(self, kind: str, payload: dict[str, Any]) -> None:
        if self.on_event is not None:
            self.on_event({"kind": kind, "payload": payload})

    def set_plan(self, plan: list[str]) -> None:
        self.plan = list(plan)
        self.emit("plan", {"steps": self.plan})

    def set_coverage(self, summary: dict[str, Any]) -> None:
        self.coverage_summary = summary
        self.emit("coverage", summary)

    def producer_done(self, name: str, n_findings: int) -> None:
        self.tool_calls.append({"producer": name, "n_findings": n_findings})
        self.emit("producer_done", {"producer": name, "n_findings": n_findings})

    def step(self, line: str) -> None:
        self.steps.append(line)
        self.emit("step", {"text": line})

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

PAST INVESTIGATIONS (what was already run before - build on these, don't repeat):
{past}

Choose the subset most relevant to the goal. Selecting all is valid when the goal
is broad or the record is thin. Use ONLY names from the list above.

Output STRICT JSON: {{"investigations": ["<name>", ...], "reason": "<one sentence>"}}"""

_REPLAN_PROMPT = """A first round of investigations ran for this goal and produced the findings
below. Decide whether a focused FOLLOW-UP round is warranted - only investigations NOT already
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
    enforces honesty - data-requirement gating, per-agent exception isolation,
    rigor inside each agent, the skeptic re-check, the faithfulness guard - is
    deterministic and non-negotiable.
    """

    model: BaseChatModel | None = None
    config: Config = field(default_factory=Config)
    synthesize_connections: bool = True
    max_rounds: int = 2
    remember: bool = True

    def investigate(
        self, ctx: AgentContext, goal: str | None = None, *, trace: RunTrace | None = None
    ) -> list[Finding]:
        """Plan, run the selected producers, skeptic-review, optionally synthesize.

        With a model, after the first round it may run bounded follow-up rounds
        that pivot on what was surfaced (only investigations not already run,
        stopping as soon as there is nothing new). When ``remember`` is set, the
        process (which investigations ran, for which goal, and what came back) is
        saved to memory so a later run can recall and build on it. Returns
        reviewed findings; the caller persists. Never raises on thin data or
        planner failure - it degrades to the full producer set, never an exception.
        """
        started_at = datetime.now(UTC)
        rec = trace if trace is not None else RunTrace()
        rec.set_coverage(_coverage_summary(ctx))
        if self.model is None:
            full = list(PRODUCERS)
            rec.set_plan(full)
            rec.step(f"Planned full producer set (no model): {', '.join(full)}")
            reviewed = self._run_round(ctx, full, rec)
            _record_round(rec, 1, full, reviewed)
            rec.status = _final_status(reviewed, rec)
            self._record_run(ctx, goal, rec, reviewed, started_at)
            return reviewed

        ran: set[str] = set()
        reviewed_all: list[Finding] = []
        selected = self._plan(ctx, goal)
        rec.set_plan(selected)
        rec.step(f"Planned: {', '.join(selected) or 'nothing'}")
        for round_idx in range(max(1, self.max_rounds)):
            fresh = [name for name in selected if name not in ran]
            if not fresh:
                break
            ran.update(fresh)
            round_findings = self._run_round(ctx, fresh, rec)
            reviewed_all.extend(round_findings)
            _record_round(rec, round_idx + 1, fresh, round_findings)
            if round_idx + 1 >= self.max_rounds:
                break
            selected = self._replan(ctx, goal, reviewed_all, ran)
            if not selected:
                break

        if self.synthesize_connections and reviewed_all:
            self._synthesize(reviewed_all)
            rec.step(f"Synthesized connections across {len(reviewed_all)} finding(s)")
        rec.status = _final_status(reviewed_all, rec)
        self._record_run(ctx, goal, rec, reviewed_all, started_at)
        return reviewed_all

    def _record_run(
        self,
        ctx: AgentContext,
        goal: str | None,
        rec: RunTrace,
        reviewed: list[Finding],
        started_at: datetime,
    ) -> None:
        """Persist the investigation as an :class:`InvestigationRun` (plan, trace,
        findings snapshot, window). This is the observable record AND the planner's
        memory: ``_past_investigations`` recalls these so a later run builds on
        prior work instead of repeating it."""
        if not self.remember or not rec.plan:
            return
        snapshot = [
            RunFinding(
                headline=f.headline,
                kind=f.kind,
                confidence=f.confidence,
                status=f.status.value,
            )
            for f in reviewed
        ]
        run = InvestigationRun(
            run_id=ctx.run_id,
            kind="question" if goal else "deep_analysis",
            status=rec.status,
            question=goal,
            window_start=ctx.window[0],
            window_end=ctx.window[1],
            plan=rec.plan,
            trace=rec.steps,
            findings=snapshot,
            n_findings=len(reviewed),
            started_at=started_at,
            finished_at=datetime.now(UTC),
            coverage_summary=rec.coverage_summary,
            tool_calls=rec.tool_calls,
            evidence_items=[_evidence_item(f) for f in reviewed],
        )
        try:
            ctx.store.insert_investigation_run(run)
        except Exception:
            logger.warning("coordinator: failed to record investigation run", exc_info=True)

    def _run_round(
        self, ctx: AgentContext, names: list[str], rec: RunTrace | None = None
    ) -> list[Finding]:
        """Run the named producers under gating + isolation, then skeptic-review.

        When ``rec`` streams events, each producer's start and result is emitted
        live so the Investigations page narrates the round as it runs."""
        registry = self._build_registry(names)
        on_start = on_done = None
        if rec is not None:
            on_start = lambda name: rec.emit("running", {"producer": name})  # noqa: E731
            on_done = rec.producer_done
        raw = registry.run_all(
            ctx, on_skip=_log_skip, on_agent_start=on_start, on_agent_done=on_done
        )
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
        """Producers only - the skeptic is applied separately via ``review`` so it
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


#: Below this glucose coverage the run is flagged "limited" (coverage-aware gating).
_LIMITED_COVERAGE_PCT = 70.0


def _coverage_summary(ctx: AgentContext) -> dict[str, Any]:
    """Data-sufficiency snapshot for the run: glucose coverage, span, treatment
    counts, and a ``limited`` flag that drives coverage-aware gating."""
    try:
        cov = ctx.store.coverage()
    except Exception:
        return {"limited": True, "note": "coverage unavailable"}
    pct = float(cov.glucose_coverage_pct)
    return {
        "glucose_coverage_pct": round(pct, 1),
        "span_days": round(float(cov.span_days), 1),
        "n_insulin": cov.n_insulin,
        "n_meals": cov.n_meals,
        "has_treatment": cov.n_insulin > 0 or cov.n_meals > 0,
        "limited": pct < _LIMITED_COVERAGE_PCT,
    }


def _final_status(reviewed: list[Finding], rec: RunTrace) -> str:
    """A run with poor coverage is ``limited`` even when it produced findings."""
    if rec.coverage_summary is not None and rec.coverage_summary.get("limited"):
        return "limited"
    return "completed" if reviewed else "limited"


def _evidence_item(f: Finding) -> dict[str, Any]:
    """The guard-audited numbers behind one finding (evidence-drawer source)."""
    return {
        "finding": f.headline,
        "kind": f.kind,
        "agent": f.agent,
        "numbers": dict(f.evidence),
        "effect_size": f.stats.effect_size,
        "n": f.stats.n,
    }


def _record_round(trace: RunTrace, number: int, names: list[str], findings: list[Finding]) -> None:
    """Append a deterministic trace line summarising one producer round."""
    kept = sum(1 for f in findings if f.status != FindingStatus.REJECTED)
    rejected = len(findings) - kept
    suffix = f" ({rejected} rejected by skeptic)" if rejected else ""
    trace.step(f"Round {number}: ran {', '.join(names)} -> {kept} finding(s){suffix}")


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
    return "\n".join(lines) if lines else "(nothing yet - early run)"


def _past_investigations(ctx: AgentContext) -> str:
    """Recall prior investigation runs (the saved process) for the planner."""
    try:
        runs = ctx.store.get_investigation_runs(limit=8)
    except Exception:
        return "(none yet)"
    if not runs:
        return "(none yet)"
    return "\n".join(
        f"- Investigated {r.question or 'the whole record'}: "
        f"ran {', '.join(r.plan)} -> {r.n_findings} finding(s)"
        for r in runs
    )


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
