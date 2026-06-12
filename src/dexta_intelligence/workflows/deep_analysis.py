"""Deep Analysis workflow — deterministic agents → skeptic → persist.

Spec ordering: producer agents fan out on the blackboard, the skeptic
reviews the collected findings, then callers persist. One crashing producer
never aborts the run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from dexta_intelligence.agents.skeptic import AGENT_NAME as SKEPTIC_NAME
from dexta_intelligence.agents.skeptic import confound_hypotheses, skeptic_agent
from dexta_intelligence.models import FindingStatus, HypothesisStatus

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from dexta_intelligence.agents.base import AgentContext, AgentRegistry
    from dexta_intelligence.models import Finding
    from dexta_intelligence.store.port import StoragePort

logger = logging.getLogger(__name__)

__all__ = ["DeepAnalysisReport", "persist_findings", "run_deep_analysis"]


@dataclass(frozen=True, slots=True)
class DeepAnalysisReport:
    """Outcome of one harness run."""

    findings: tuple[Finding, ...]
    persisted_ids: tuple[int, ...]
    skipped: tuple[tuple[str, tuple[str, ...]], ...] = ()
    errors: tuple[tuple[str, str], ...] = ()
    banked_hypotheses: tuple[int, ...] = ()


@dataclass
class _RunAccumulator:
    findings: list[Finding] = field(default_factory=list)
    skipped: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)


def run_deep_analysis(
    registry: AgentRegistry,
    ctx: AgentContext,
    *,
    skip_skeptic: bool = False,
    persist: bool = True,
    on_skip: Callable[[str, list[str]], None] | None = None,
) -> DeepAnalysisReport:
    """Run producer agents, skeptic review, and optional persistence."""
    acc = _RunAccumulator()

    for agent in registry:
        if agent.name == SKEPTIC_NAME:
            continue
        reasons = agent.requires.unmet_reasons(ctx.gates)
        if reasons:
            acc.skipped.append((agent.name, tuple(reasons)))
            if on_skip is not None:
                on_skip(agent.name, reasons)
            continue
        try:
            acc.findings.extend(agent.run(ctx))
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            logger.exception("agent %s failed; continuing run", agent.name)
            acc.errors.append((agent.name, msg))

    reviewed = acc.findings if skip_skeptic else skeptic_agent.review(acc.findings, ctx)

    persisted: list[int] = []
    banked: list[int] = []
    if persist:
        persisted = persist_findings(ctx.store, reviewed)
        banked = _bank_confounds(reviewed, ctx)

    return DeepAnalysisReport(
        findings=tuple(reviewed),
        persisted_ids=tuple(persisted),
        skipped=tuple(acc.skipped),
        errors=tuple(acc.errors),
        banked_hypotheses=tuple(banked),
    )


def persist_findings(store: StoragePort, findings: Iterable[Finding]) -> list[int]:
    """Insert findings, superseding any prior ACTIVE one with the same key.

    Re-running analysis re-emits the same finding; without dedup each run inserts
    a fresh row, inflating recurrence counts and triple-printing the brief. Keyed
    on ``(agent, kind, scope)``, the prior ACTIVE finding is marked SUPERSEDED and
    linked via ``superseded_by`` to the new row, so exactly one ACTIVE finding
    survives per key and the superseded history stays in the graveyard.
    """
    persisted: list[int] = []
    for finding in findings:
        prior = store.get_findings(
            agent=finding.agent,
            kind=finding.kind,
            status=FindingStatus.ACTIVE,
            limit=100,
        )
        new_id = store.insert_finding(finding)
        persisted.append(new_id)
        for old in prior:
            if old.id is not None and old.scope == finding.scope:
                store.supersede_finding(old.id, new_id)
    return persisted


def _bank_confounds(reviewed: list[Finding], ctx: AgentContext) -> list[int]:
    """Persist skeptic confound flags as open hypotheses, deduped on statement."""
    candidates = confound_hypotheses(reviewed)
    if not candidates:
        return []
    existing = {
        h.statement
        for h in ctx.store.get_hypotheses(status=HypothesisStatus.OPEN.value)
    }
    banked: list[int] = []
    for hypothesis in candidates:
        if hypothesis.statement in existing:
            continue
        banked.append(ctx.store.insert_hypothesis(hypothesis))
        existing.add(hypothesis.statement)
    return banked
