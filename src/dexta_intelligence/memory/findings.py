"""Finding memory helpers - recurrence, similarity, and supersession.

The store owns persistence; this module owns the *semantics* of how findings
relate to each other across analysis runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dexta_intelligence.models import EdgeRelation, Finding, FindingEdge, FindingStatus

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

__all__ = [
    "contradicts_edge",
    "count_recurrence",
    "find_contradictions",
    "find_similar",
    "recurrence_headline_suffix",
    "recurrence_line",
    "supersedes_edge",
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


def recurrence_line(finding: Finding) -> str:
    """The recurrence receipt for one finding: "seen 7 times since May 12".

    Built from the finding's own lifecycle fields (``seen_count`` and its
    window bounds), so any surface rendering a finding can attach it without a
    store query. Empty for a first sighting; the since-date is the earliest
    window bound the record retains.
    """
    if finding.seen_count <= 1:
        return ""
    since = finding.window_start or finding.window_end
    if since is None:
        return f"seen {finding.seen_count} times"
    return f"seen {finding.seen_count} times since {since.strftime('%b %d')}"


def _edge_event_time(new: Finding, old: Finding) -> datetime | None:
    """Timeline anchor for an edge: the newest window bound available, in order."""
    for value in (new.window_end, new.window_start, old.window_end, old.window_start):
        if value is not None:
            return value
    return None


def supersedes_edge(
    *, new_id: int, old: Finding, new: Finding, now: datetime
) -> FindingEdge:
    """A SUPERSEDES edge from the new finding to the one it replaced.

    Deterministic: ``event_time`` is the newest finding window bound (when the
    relationship held in the timeline), ``knowledge_time`` is ``now``. ``old.id``
    must be set (the prior is already persisted).
    """
    assert old.id is not None
    return FindingEdge(
        src_id=new_id,
        dst_id=old.id,
        relation=EdgeRelation.SUPERSEDES,
        knowledge_time=now,
        event_time=_edge_event_time(new, old),
        evidence=(
            f"re-derived {new.agent}/{new.kind}/{new.scope}; "
            f"seen_count={new.seen_count}"
        ),
    )


def contradicts_edge(
    *, new_id: int, old: Finding, new: Finding, now: datetime
) -> FindingEdge:
    """A CONTRADICTS edge from the new finding to an opposed prior.

    The reason string records the opposing effect sizes that
    :func:`find_contradictions` matched on. ``old.id`` must be set.
    """
    assert old.id is not None
    prior_effect = old.stats.effect_size
    effect = new.stats.effect_size
    return FindingEdge(
        src_id=new_id,
        dst_id=old.id,
        relation=EdgeRelation.CONTRADICTS,
        knowledge_time=now,
        event_time=_edge_event_time(new, old),
        evidence=f"opposite effect: prior {prior_effect:+.3g} vs current {effect:+.3g}",
    )


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
