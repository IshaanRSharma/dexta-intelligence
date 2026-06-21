"""View-model logic for the Findings page.

Pure data shaping: this module reads findings, hypotheses, and investigation
runs from the store and returns plain dicts the template renders. It produces
DATA, not HTML (apart from finding bodies, which are pre-rendered Markdown).

The small helpers here are reimplemented locally rather than imported from
``server.app``: importing private names from ``app`` would create a circular
dependency.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from dexta_intelligence.models import FindingStatus
from dexta_intelligence.server.render import markdown_to_html

if TYPE_CHECKING:
    from dexta_intelligence.models import Finding, Hypothesis, InvestigationRun
    from dexta_intelligence.store.port import StoragePort

logger = logging.getLogger(__name__)

__all__ = ["evidence_strength", "findings_page_view", "lifecycle_label"]

#: Kinds that are bookkeeping artifacts rather than user-facing findings. They
#: are excluded from both the active and rejected lists on the Findings page.
INTERNAL_FINDING_KINDS = frozenset({"investigation"})

#: Statuses shown in the rejected ("graveyard") section.
_REJECTED_STATUSES = frozenset(
    {FindingStatus.REJECTED, FindingStatus.DISMISSED, FindingStatus.SUPERSEDED}
)


def _relative_time(ts: datetime, now: datetime) -> str:
    """Human relative time of ``ts`` against ``now``. Naive ``ts`` is read as UTC."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    delta = now - ts.astimezone(UTC)
    secs = int(delta.total_seconds())
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{delta.days}d ago"


def _skeptic_survived(finding: Finding) -> bool:
    return finding.skeptic_notes is None or "reject" not in finding.skeptic_notes.lower()


def evidence_strength(finding: Finding) -> str:
    """Qualitative evidence strength (strong / moderate / weak) from the finding's
    confidence, sample size, replication, and whether it survived the skeptic.

    Replaces the misleading raw "confidence 100%" label with an honest band: a
    finding the skeptic flagged is never stronger than weak, and "strong" needs
    both high confidence and either replication or a real sample."""
    if not _skeptic_survived(finding):
        return "weak"
    n = finding.stats.n or 0
    if finding.confidence >= 0.75 and (finding.stats.replicated or n >= 20):
        return "strong"
    if finding.confidence >= 0.5:
        return "moderate"
    return "weak"


def lifecycle_label(finding: Finding) -> str:
    """User-facing lifecycle label for an active finding: a finding re-confirmed
    across runs is "verified", otherwise "supported". (Rejected/superseded
    findings carry their own status and never reach this.)"""
    return "verified" if (finding.seen_count or 1) > 1 else "supported"


def _stats_line(finding: Finding) -> str:
    """Assemble the compact stats summary line for a finding."""
    s = finding.stats
    bits: list[str] = []
    if s.effect_size is not None:
        bits.append(f"effect {s.effect_size:g}")
    if s.n is not None:
        bits.append(f"n={s.n}")
    if s.p_perm is not None:
        bits.append(f"p={s.p_perm:g}")
    if s.q_fdr is not None:
        bits.append(f"q={s.q_fdr:g}")
    if s.replicated is not None:
        bits.append("replicated" if s.replicated else "not replicated")
    return " · ".join(bits)


def _window_label(finding: Finding) -> str:
    """``YYYY-MM-DD to YYYY-MM-DD`` from a finding's window, or "" if either end is None."""
    if finding.window_start is None or finding.window_end is None:
        return ""
    return f"{finding.window_start.date().isoformat()} to {finding.window_end.date().isoformat()}"


def _active_card(finding: Finding, now: datetime) -> dict[str, Any]:
    last_verified_rel = (
        _relative_time(finding.last_verified, now) if finding.last_verified is not None else ""
    )
    return {
        "headline": finding.headline,
        "agent": finding.agent,
        "kind": finding.kind,
        "scope": finding.scope,
        "strength": evidence_strength(finding),
        "lifecycle": lifecycle_label(finding),
        "stats_line": _stats_line(finding),
        "skeptic_survived": _skeptic_survived(finding),
        "skeptic_notes": finding.skeptic_notes,
        "body_html": markdown_to_html(finding.body_md) if finding.body_md else "",
        "seen_count": finding.seen_count,
        "last_verified_rel": last_verified_rel,
        "window_label": _window_label(finding),
    }


def _rejected_row(finding: Finding) -> dict[str, Any]:
    return {
        "headline": finding.headline,
        "status": finding.status.value,
        "agent": finding.agent,
        "skeptic_notes": finding.skeptic_notes,
    }


def _hypothesis_view(hypothesis: Hypothesis) -> dict[str, Any]:
    status = hypothesis.status
    return {
        "statement": hypothesis.statement,
        "status": status.value if hasattr(status, "value") else str(status),
        "source_finding_id": hypothesis.source_finding_id,
    }


def _run_view(run: InvestigationRun, now: datetime) -> dict[str, Any]:
    return {
        "question": run.question or "Whole-record investigation",
        "kind": run.kind,
        "status": run.status,
        "n_findings": run.n_findings,
        "when": _relative_time(run.finished_at, now),
        "plan": run.plan,
    }


def findings_page_view(store: StoragePort, *, now: datetime) -> dict[str, Any]:
    """Shape the Findings page: active cards, hypotheses, rejected rows, and runs.

    Findings are fetched once and partitioned in Python. Internal-kind findings
    (see ``INTERNAL_FINDING_KINDS``) are excluded from both the active and
    rejected lists. STALE findings fall out of the active list naturally because
    they are not ACTIVE.
    """
    findings = store.get_findings(status=None, limit=1_000_000)
    user_facing = [f for f in findings if f.kind not in INTERNAL_FINDING_KINDS]

    active = [f for f in user_facing if f.status == FindingStatus.ACTIVE]
    rejected = [f for f in user_facing if f.status in _REJECTED_STATUSES]

    def order_key(f: Finding) -> int:
        return f.id if f.id is not None else 0

    active.sort(key=order_key, reverse=True)
    rejected.sort(key=order_key, reverse=True)

    hypotheses = store.get_hypotheses(status="open")
    # The investigation log is an enrichment: a failure here (for example an older
    # DB without the investigation_runs table) must never take the page down.
    try:
        runs = store.get_investigation_runs(limit=50)
    except Exception:
        logger.debug("findings_page_view: get_investigation_runs failed; degrading", exc_info=True)
        runs = []

    return {
        "active": [_active_card(f, now) for f in active],
        "hypotheses": [_hypothesis_view(h) for h in hypotheses],
        "rejected": [_rejected_row(f) for f in rejected],
        "runs": [_run_view(r, now) for r in runs],
        "counts": {
            "active": len(active),
            "hypotheses": len(hypotheses),
            "rejected": len(rejected),
            "runs": len(runs),
        },
    }
