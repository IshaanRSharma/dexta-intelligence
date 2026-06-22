"""Skeptic Agent - independent rigor re-check, memory scan, confound flags.

Consumes candidate findings from the current analysis run (not the timeline
directly). Re-runs :func:`~dexta_intelligence.stats.rigor.assess` with a
*different* random seed than the producing agents, searches persisted memory
for contradicting priors, and flags known confound pairs (e.g. weekday vs
sleep effects competing for the same variance).

Deterministic - no LLM imports in this module.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from dexta_intelligence.agents.base import (
    AgentContext,
    AgentRegistry,
    DataRequirement,
)
from dexta_intelligence.models import Finding, FindingStatus, Hypothesis
from dexta_intelligence.stats.rigor import assess

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "AGENT_NAME",
    "SkepticAgent",
    "confound_hypotheses",
    "register_skeptic",
    "skeptic_agent",
]

#: Marker prefix written into ``skeptic_notes`` by :func:`_confound_note`.
_CONFOUND_FLAG = "confound flag: "

AGENT_NAME = "skeptic"
_SKEPTIC_SEED = 137
_FDR_ALPHA = 0.10
_MIN_PER_GROUP = 8
_MIN_TOTAL = 16
_MIN_PER_HALF = 3

#: When both kinds surface in one run, neither alone establishes causality.
_CONFOUND_KIND_SETS: tuple[frozenset[str], ...] = (
    frozenset({"pattern_weekday_weekend", "pattern_sleep_glucose"}),
    frozenset({"pattern_tod_drift", "pattern_sleep_glucose"}),
)

#: Agents whose findings assert tested effects (observation summaries exempt).
_QUANTITATIVE_AGENTS = frozenset({"pattern", "reconciliation", "discovery"})


class SkepticAgent:
    """Post-producer gate - call :meth:`review`, not :meth:`run`, in pipelines."""

    name: str
    requires: DataRequirement

    def __init__(
        self,
        *,
        name: str = AGENT_NAME,
        requires: DataRequirement | None = None,
    ) -> None:
        self.name = name
        self.requires = requires or DataRequirement()

    def run(self, ctx: AgentContext) -> list[Finding]:
        """Registry hook - reviews active findings already in the store."""
        pending = ctx.store.get_findings(status=FindingStatus.ACTIVE, limit=100)
        producers = [f for f in pending if f.agent != AGENT_NAME]
        return self.review(producers, ctx)

    def review(self, findings: list[Finding], ctx: AgentContext) -> list[Finding]:
        """Return reviewed findings (status/confidence/notes may change)."""
        if not findings:
            return []

        prior = ctx.store.get_findings(status=FindingStatus.ACTIVE, limit=500)
        sibling_pvals = [
            f.stats.p_perm for f in findings if f.stats.p_perm is not None
        ]
        active_kinds = {f.kind for f in findings if f.status == FindingStatus.ACTIVE}

        return [
            self._review_one(finding, prior, sibling_pvals, active_kinds)
            for finding in findings
        ]

    def _review_one(
        self,
        finding: Finding,
        prior: Sequence[Finding],
        sibling_pvals: Sequence[float],
        active_kinds: frozenset[str] | set[str],
    ) -> Finding:
        notes: list[str] = []
        status = finding.status
        confidence = finding.confidence

        if finding.agent == "observation":
            notes.append("observation summary - descriptive, no rigor gate")
            return finding.model_copy(
                update={"skeptic_notes": "; ".join(notes) if notes else finding.skeptic_notes}
            )

        if finding.agent not in _QUANTITATIVE_AGENTS:
            return finding

        groups = _extract_groups(finding.evidence)
        if groups is not None:
            group_a, group_b = groups
            others = [p for p in sibling_pvals if p != finding.stats.p_perm]
            verdict = assess(
                group_a,
                group_b,
                rng=random.Random(_SKEPTIC_SEED),
                min_per_group=min(_MIN_PER_GROUP, len(group_a), len(group_b)),
                min_total=min(_MIN_TOTAL, len(group_a) + len(group_b)),
                min_per_half=min(_MIN_PER_HALF, max(1, len(group_a) // 2)),
                sibling_pvalues=others,
                alpha=_FDR_ALPHA,
            )
            notes.append(f"skeptic rigor re-check (seed={_SKEPTIC_SEED}): {verdict.verdict}")
            for reason in verdict.reasons:
                notes.append(reason)

            if verdict.verdict == "fail":
                status = FindingStatus.REJECTED
                confidence = min(confidence, 0.2)
            elif verdict.verdict == "weak":
                confidence = min(confidence, 0.45)
                notes.append("demoted: significant but not replicated on skeptic re-check")
        else:
            notes.append("no raw groups in evidence - stats-block audit only")
            if finding.stats.p_perm is None and finding.stats.effect_size is not None:
                status = FindingStatus.REJECTED
                notes.append("rejected: quantitative claim without permutation p-value")
                confidence = min(confidence, 0.2)

        prior_note = _contradicts_prior(finding, prior)
        if prior_note is not None:
            notes.append(prior_note)
            confidence = min(confidence, 0.35)

        confound_note = _confound_note(finding.kind, active_kinds)
        if confound_note is not None:
            notes.append(confound_note)
            confidence = min(confidence, 0.5)

        rigor_tag = finding.evidence.get("rigor_verdict")
        if rigor_tag == "fail" and status == FindingStatus.ACTIVE:
            status = FindingStatus.REJECTED
            notes.append("rejected: producer rigor_verdict was fail")

        # Contradiction marks the finding only if it was not already rejected
        # (rejection takes precedence): a live belief the evidence now opposes.
        if prior_note is not None and status == FindingStatus.ACTIVE:
            status = FindingStatus.CONTRADICTED
            notes.append("status: contradicted by a prior finding")

        return finding.model_copy(
            update={
                "status": status,
                "confidence": confidence,
                "skeptic_notes": "; ".join(notes),
            }
        )


skeptic_agent = SkepticAgent()


def register_skeptic(registry: AgentRegistry) -> None:
    """Register :data:`skeptic_agent` on ``registry``."""
    registry.register(skeptic_agent)


def _extract_groups(
    evidence: dict[str, object],
) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    raw_a = evidence.get("skeptic_group_a")
    raw_b = evidence.get("skeptic_group_b")
    if isinstance(raw_a, list) and isinstance(raw_b, list):
        return (tuple(float(x) for x in raw_a), tuple(float(x) for x in raw_b))

    episodes = evidence.get("episodes")
    if isinstance(episodes, list) and episodes:
        errors: list[float] = []
        for ep in episodes:
            if isinstance(ep, dict) and "signed_error_mg_dl" in ep:
                errors.append(float(ep["signed_error_mg_dl"]))
        if errors:
            null_group = tuple([0.0] * max(8, len(errors)))
            return (tuple(errors), null_group[: len(errors)])

    return None


def _contradicts_prior(finding: Finding, prior: Sequence[Finding]) -> str | None:
    effect = finding.stats.effect_size
    if effect is None:
        return None
    for old in prior:
        if old.id is not None and old.id == finding.id:
            continue
        if old.kind != finding.kind or old.agent != finding.agent:
            continue
        prior_effect = old.stats.effect_size
        if prior_effect is None:
            continue
        if effect * prior_effect < 0 and abs(effect) > 1e-6 and abs(prior_effect) > 1e-6:
            return (
                f"contradicts prior finding id={old.id} "
                f"(effect {prior_effect:+.3g} vs current {effect:+.3g})"
            )
    return None


def _confound_note(kind: str, active_kinds: frozenset[str] | set[str]) -> str | None:
    for pair in _CONFOUND_KIND_SETS:
        if kind in pair:
            partners = pair - {kind}
            if partners & active_kinds:
                partner = next(iter(partners & active_kinds))
                return (
                    f"{_CONFOUND_FLAG}{kind} co-occurring with {partner} "
                    f"- shared variance, not isolated causality"
                )
    return None


def confound_hypotheses(findings: Sequence[Finding]) -> list[Hypothesis]:
    """Derive open disentanglement hypotheses from confound flags in a run.

    Each reviewed finding whose ``skeptic_notes`` carries a ``confound flag:``
    marker (written by :func:`_confound_note`) names a co-occurring kind pair
    competing for the same variance. A confound is symmetric - a pair surfaces
    on both findings - so hypotheses are keyed on the unordered ``{kind_a,
    kind_b}`` pair and emitted once each, in first-seen order.

    Returns OPEN :class:`Hypothesis` records with statements of the form::

        Disentangle <kind_a> vs <kind_b>: <note excerpt>
        - stratify when more data allows [skeptic]
    """
    seen: set[frozenset[str]] = set()
    out: list[Hypothesis] = []
    for finding in findings:
        notes = finding.skeptic_notes
        if not notes:
            continue
        for note in notes.split("; "):
            if not note.startswith(_CONFOUND_FLAG):
                continue
            excerpt = note[len(_CONFOUND_FLAG) :]
            partner = _confound_partner(excerpt)
            if partner is None:
                continue
            pair = frozenset({finding.kind, partner})
            if pair in seen:
                continue
            seen.add(pair)
            kind_a, kind_b = sorted(pair)
            out.append(
                Hypothesis(
                    statement=(
                        f"Disentangle {kind_a} vs {kind_b}: {excerpt} "
                        f"- stratify when more data allows [skeptic]"
                    ),
                    source_finding_id=finding.id,
                )
            )
    return out


def _confound_partner(excerpt: str) -> str | None:
    """Pull the co-occurring kind from a ``<kind> co-occurring with <partner> ...`` note."""
    marker = " co-occurring with "
    head = excerpt.find(marker)
    if head < 0:
        return None
    rest = excerpt[head + len(marker) :]
    partner = rest.split(" ", 1)[0]
    return partner or None
