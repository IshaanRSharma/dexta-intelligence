"""Finding-freshness lifecycle: re-derivation keeps a pattern fresh; a pattern
that stops recurring ages out to STALE and drops from recall and the feed."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.agents.tools.toolkit import _recall
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.memory.freshness import (
    finding_ttl_days,
    is_stale,
    prune_stale_findings,
)
from dexta_intelligence.models import Finding, FindingStatus
from dexta_intelligence.store import SQLiteStore
from dexta_intelligence.workflows.deep_analysis import persist_findings

if TYPE_CHECKING:
    from collections.abc import Iterator

NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


@pytest.fixture
def store() -> Iterator[SQLiteStore]:
    s = SQLiteStore(":memory:")
    s.migrate()
    yield s
    s.close()


def _finding(
    *,
    kind: str = "overnight-lows",
    scope: str = "global",
    confidence: float = 0.7,
    seen_count: int = 1,
    last_verified: datetime | None = None,
) -> Finding:
    return Finding(
        agent="pattern",
        kind=kind,
        scope=scope,
        headline=f"{kind} pattern",
        confidence=confidence,
        seen_count=seen_count,
        last_verified=last_verified,
        status=FindingStatus.ACTIVE,
    )


def test_ttl_scales_with_confidence_and_recurrence() -> None:
    weak = _finding(confidence=0.4, seen_count=1)
    strong = _finding(confidence=0.9, seen_count=5)
    assert finding_ttl_days(strong) > finding_ttl_days(weak) * 3


def test_fresh_finding_is_not_stale() -> None:
    f = _finding(last_verified=NOW)
    assert not is_stale(f, NOW + timedelta(days=1))


def test_old_one_off_ages_out() -> None:
    f = _finding(confidence=0.4, seen_count=1, last_verified=NOW - timedelta(days=30))
    assert is_stale(f, NOW)


def test_high_confidence_recurring_survives_longer(store: SQLiteStore) -> None:
    """At 30 days idle a weak one-off is retired but a strong recurring one is kept."""
    idle = NOW - timedelta(days=30)
    weak_id = store.insert_finding(
        _finding(kind="weak", confidence=0.4, seen_count=1, last_verified=idle)
    )
    strong_id = store.insert_finding(
        _finding(kind="strong", confidence=0.9, seen_count=5, last_verified=idle)
    )
    retired = prune_stale_findings(store, now=NOW)
    assert weak_id in retired
    assert strong_id not in retired

    by_id = {f.id: f for f in store.get_findings(status=None, limit=100)}
    assert by_id[weak_id].status == FindingStatus.STALE
    assert by_id[strong_id].status == FindingStatus.ACTIVE


def test_no_anchor_is_never_stale(store: SQLiteStore) -> None:
    """A finding with neither last_verified nor window_end is left alone."""
    f = Finding(agent="pattern", kind="k", scope="s", headline="h", confidence=0.5)
    assert not is_stale(f.model_copy(update={"last_verified": None}), NOW)


def test_persist_bumps_seen_count_and_last_verified(store: SQLiteStore) -> None:
    first = NOW - timedelta(days=10)
    persist_findings(store, [_finding()], now=first)
    persist_findings(store, [_finding()], now=NOW)

    active = store.get_findings(status=FindingStatus.ACTIVE, limit=10)
    assert len(active) == 1  # the prior was superseded
    assert active[0].seen_count == 2
    assert active[0].last_verified == NOW

    superseded = store.get_findings(status=FindingStatus.SUPERSEDED, limit=10)
    assert len(superseded) == 1


def _ctx(store: SQLiteStore) -> AgentContext:
    return AgentContext(
        store=store,
        window=(date(2026, 5, 1), date(2026, 6, 1)),
        gates=ColdStartReport.from_coverage(store.coverage()),
        run_id="freshness-test",
    )


def test_recall_excludes_stale_findings(store: SQLiteStore) -> None:
    store.insert_finding(_finding(kind="fresh-pattern", last_verified=NOW))
    stale_id = store.insert_finding(_finding(kind="stale-pattern", last_verified=NOW))
    store.set_finding_status(stale_id, FindingStatus.STALE)

    payload, _numbers = _recall(_ctx(store), "pattern")
    headlines = {item["headline"] for item in payload["findings"]}
    assert "fresh-pattern pattern" in headlines
    assert "stale-pattern pattern" not in headlines
