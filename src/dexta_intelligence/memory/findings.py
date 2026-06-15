"""Finding memory helpers — recurrence, similarity, and supersession.

The store owns persistence; this module owns the *semantics* of how findings
relate to each other across analysis runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dexta_intelligence.models import Finding, FindingStatus

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "count_recurrence",
    "find_contradictions",
    "find_similar",
    "recurrence_headline_suffix",
]

#: Minimum |effect| delta to treat two findings as directionally opposed.
_OPPOSITION_EPS = 1e-6


def find_similar(
    finding: Finding,
    prior: Sequence[Finding],
    *,
    agent: str | None = None,
    kind: str | None = None,
    status: FindingStatus | None = FindingStatus.ACTIVE,
) -> list[Finding]:
    """Return prior findings matching agent/kind (defaults to ``finding``'s keys)."""
    target_agent = agent if agent is not None else finding.agent
    target_kind = kind if kind is not None else finding.kind
    return [
        p
        for p in prior
        if p.agent == target_agent
        and p.kind == target_kind
        and (status is None or p.status == status)
        and (finding.id is None or p.id != finding.id)
    ]


def count_recurrence(finding: Finding, prior: Sequence[Finding]) -> int:
    """How many prior active findings share this agent/kind (excluding self)."""
    return len(find_similar(finding, prior))


def recurrence_headline_suffix(recurrence: int) -> str:
    """Human-readable recurrence clause for finding prose."""
    if recurrence <= 0:
        return ""
    total = recurrence + 1
    return f" Similar pattern, {total} occurrence(s) including this run."


def find_contradictions(
    finding: Finding,
    prior: Sequence[Finding],
) -> list[Finding]:
    """Prior findings with the same kind but opposite effect direction."""
    effect = finding.stats.effect_size
    if effect is None:
        return []
    out: list[Finding] = []
    for old in find_similar(finding, prior):
        prior_effect = old.stats.effect_size
        if prior_effect is None:
            continue
        if (
            effect * prior_effect < 0
            and abs(effect) > _OPPOSITION_EPS
            and abs(prior_effect) > _OPPOSITION_EPS
        ):
            out.append(old)
    return out
