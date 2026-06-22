"""Memory-retrieval-guard evals for _recall.

The guard returns ``payload["findings"]`` (only ACTIVE, non-dosing memories,
ranked) and ``payload["excluded"]`` (withheld memories with a reason). These
deterministic cases cover the PRD's memory acceptance criteria: a stale,
rejected, contradicted, superseded, or dosing-like memory is never reusable,
a clean active memory is, and one answer lists both what was used and what was
excluded with reasons.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.agents.tools.toolkit import _recall
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.models import (
    Finding,
    FindingStats,
    FindingStatus,
)
from dexta_intelligence.store import SQLiteStore

_NOW = datetime(2026, 6, 1, tzinfo=UTC)


def _store() -> SQLiteStore:
    store = SQLiteStore(":memory:")
    store.migrate()
    return store


def _ctx(store: SQLiteStore) -> AgentContext:
    return AgentContext(
        store=store,
        window=(_NOW.date(), _NOW.date()),
        gates=ColdStartReport.from_coverage(store.coverage()),
        run_id="test-memory-guard",
    )


def _f(headline: str, status: FindingStatus, scope: str = "overnight") -> Finding:
    return Finding(
        agent="discovery",
        kind="pattern",
        scope=scope,
        headline=headline,
        confidence=0.6,
        status=status,
        stats=FindingStats(effect_size=0.6, n=12),
        window_end=_NOW,
    )


def _used(payload: dict[str, Any]) -> list[str]:
    return [item["headline"] for item in payload["findings"]]


def _excluded(payload: dict[str, Any]) -> dict[str, str]:
    return {e["headline"]: e["reason"] for e in payload.get("excluded", [])}


def test_stale_finding_is_excluded_not_used() -> None:
    store = _store()
    store.insert_finding(_f("Stale overnight pattern.", FindingStatus.STALE))
    payload, _ = _recall(_ctx(store), "")
    assert "Stale overnight pattern." not in _used(payload)
    assert _excluded(payload)["Stale overnight pattern."] == "not_used_stale"


def test_rejected_finding_is_excluded_not_used() -> None:
    store = _store()
    store.insert_finding(_f("Rejected overnight pattern.", FindingStatus.REJECTED))
    payload, _ = _recall(_ctx(store), "")
    assert "Rejected overnight pattern." not in _used(payload)
    assert _excluded(payload)["Rejected overnight pattern."] == "not_used_rejected"


def test_contradicted_finding_is_excluded_not_used() -> None:
    store = _store()
    store.insert_finding(_f("Contradicted overnight pattern.", FindingStatus.CONTRADICTED))
    payload, _ = _recall(_ctx(store), "")
    assert "Contradicted overnight pattern." not in _used(payload)
    assert _excluded(payload)["Contradicted overnight pattern."] == "not_used_contradicted"


def test_superseded_finding_is_excluded_not_used() -> None:
    store = _store()
    store.insert_finding(_f("Superseded overnight pattern.", FindingStatus.SUPERSEDED))
    payload, _ = _recall(_ctx(store), "")
    assert "Superseded overnight pattern." not in _used(payload)
    assert _excluded(payload)["Superseded overnight pattern."] == "not_used_superseded"


def test_active_dosing_headline_is_safety_blocked() -> None:
    store = _store()
    store.insert_finding(_f("Increase basal insulin overnight.", FindingStatus.ACTIVE))
    payload, _ = _recall(_ctx(store), "")
    assert "Increase basal insulin overnight." not in _used(payload)
    assert _excluded(payload)["Increase basal insulin overnight."] == "not_used_safety_blocked"


def test_clean_active_finding_is_used() -> None:
    store = _store()
    store.insert_finding(_f("Overnight glucose drifts up after 3am.", FindingStatus.ACTIVE))
    payload, _ = _recall(_ctx(store), "")
    assert "Overnight glucose drifts up after 3am." in _used(payload)


def test_used_and_excluded_are_both_reported() -> None:
    store = _store()
    store.insert_finding(_f("Active overnight pattern.", FindingStatus.ACTIVE, "overnight"))
    store.insert_finding(_f("Stale dawn pattern.", FindingStatus.STALE, "dawn"))
    store.insert_finding(_f("Rejected dinner pattern.", FindingStatus.REJECTED, "dinner"))
    store.insert_finding(_f("Contradicted lunch pattern.", FindingStatus.CONTRADICTED, "lunch"))
    store.insert_finding(_f("Superseded morning pattern.", FindingStatus.SUPERSEDED, "am"))
    store.insert_finding(_f("Increase bolus before dinner.", FindingStatus.ACTIVE, "dinner"))

    payload, _ = _recall(_ctx(store), "")

    used = _used(payload)
    assert "Active overnight pattern." in used
    for headline in (
        "Stale dawn pattern.",
        "Rejected dinner pattern.",
        "Contradicted lunch pattern.",
        "Superseded morning pattern.",
        "Increase bolus before dinner.",
    ):
        assert headline not in used

    excluded = _excluded(payload)
    assert excluded["Stale dawn pattern."] == "not_used_stale"
    assert excluded["Rejected dinner pattern."] == "not_used_rejected"
    assert excluded["Contradicted lunch pattern."] == "not_used_contradicted"
    assert excluded["Superseded morning pattern."] == "not_used_superseded"
    assert excluded["Increase bolus before dinner."] == "not_used_safety_blocked"


def test_no_exclusions_means_no_excluded_key() -> None:
    store = _store()
    store.insert_finding(_f("Overnight glucose is stable.", FindingStatus.ACTIVE))
    payload, _ = _recall(_ctx(store), "")
    assert "Overnight glucose is stable." in _used(payload)
    assert not payload.get("excluded")
