"""Finding freshness: retire patterns that stop being re-derived.

A finding that keeps recurring stays fresh, because :func:`persist_findings`
bumps its ``last_verified`` and ``seen_count`` every time it is re-derived. One
that is not re-derived within its TTL is retired to ``STALE``: dropped from agent
recall and the active feed, kept in history. The TTL scales with confidence and
recurrence, so a well-supported, oft-seen pattern lives much longer than a
low-confidence one-off. Decay is driven by non-recurrence, not raw clock age.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dexta_intelligence.models import FindingStatus

if TYPE_CHECKING:
    from datetime import datetime

    from dexta_intelligence.models import Finding
    from dexta_intelligence.store.port import StoragePort

__all__ = [
    "STALE_BASE_DAYS",
    "finding_ttl_days",
    "is_stale",
    "prune_stale_findings",
]

#: Base time-to-live before scaling. A confidence-0.5, seen-once finding lives
#: this long; confidence and recurrence multiply it from here.
STALE_BASE_DAYS = 14.0

#: Recurrence past this many sightings does not further extend the TTL.
_MAX_RECURRENCE_FACTOR = 5


def finding_ttl_days(finding: Finding, *, base_days: float = STALE_BASE_DAYS) -> float:
    """Days a finding stays ACTIVE without re-derivation before it ages out.

    Scaled by confidence (0.5x to 1.5x) and recurrence (1x to 5x), so a
    high-confidence pattern seen many times survives much longer than a
    low-confidence one-off.
    """
    confidence_factor = 0.5 + finding.confidence
    recurrence_factor = min(max(finding.seen_count, 1), _MAX_RECURRENCE_FACTOR)
    return base_days * confidence_factor * recurrence_factor


def is_stale(finding: Finding, now: datetime, *, base_days: float = STALE_BASE_DAYS) -> bool:
    """True when ``finding`` has gone longer than its TTL without re-derivation."""
    anchor = finding.last_verified or finding.window_end
    if anchor is None:
        return False  # no timestamp to age against; leave it alone
    age_days = (now - anchor).total_seconds() / 86_400.0
    return age_days > finding_ttl_days(finding, base_days=base_days)


def prune_stale_findings(
    store: StoragePort, *, now: datetime, base_days: float = STALE_BASE_DAYS
) -> list[int]:
    """Retire ACTIVE findings past their TTL to STALE. Returns the ids retired.

    STALE findings are excluded from agent recall and the active feed but kept in
    history, so the reasoning layer stops building on patterns that no longer hold.
    """
    retired: list[int] = []
    for finding in store.get_findings(status=FindingStatus.ACTIVE, limit=1_000_000):
        if finding.id is not None and is_stale(finding, now, base_days=base_days):
            store.set_finding_status(finding.id, FindingStatus.STALE)
            retired.append(finding.id)
    return retired
