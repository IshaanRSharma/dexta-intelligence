"""Change detection and myth-busting over the bitemporal findings graph.

``what_changed`` reads SUPERSEDES edges (real changes only; re-verifications
with an unchanged headline are skipped) and ``contradicted_beliefs`` reads
CONTRADICTS edges. Both are pure code over deterministically authored edges,
exposed to the reasoning loop as the ``what_changed`` / ``contradicted_beliefs``
belt tools.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.agents.tools import build_belt
from dexta_intelligence.agents.tools.memory_graph import memory_graph_specs
from dexta_intelligence.agents.tools.toolkit import DiscoveryToolkit
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.memory.changes import contradicted_beliefs, what_changed
from dexta_intelligence.models import (
    EdgeRelation,
    Finding,
    FindingEdge,
    FindingStats,
    FindingStatus,
    GlucoseEvent,
)
from dexta_intelligence.store import SQLiteStore

_NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
_CHANGE_AT = _NOW - timedelta(days=10)


def _store() -> SQLiteStore:
    store = SQLiteStore(":memory:")
    store.migrate()
    return store


def _finding(headline: str, **overrides: object) -> Finding:
    base: dict[str, object] = {
        "agent": "pattern",
        "kind": "post_dinner",
        "scope": "evening",
        "headline": headline,
        "stats": FindingStats(effect_size=0.5, n=12),
        "confidence": 0.7,
        "window_start": _NOW - timedelta(days=30),
        "window_end": _NOW,
    }
    base.update(overrides)
    return Finding.model_validate(base)


def _seed_change(store: SQLiteStore) -> tuple[int, int]:
    old_id = store.insert_finding(
        _finding("Post-dinner spikes recur most evenings", status=FindingStatus.SUPERSEDED)
    )
    new_id = store.insert_finding(_finding("Post-dinner spikes stopped after the site change"))
    store.add_finding_edge(
        FindingEdge(
            src_id=new_id,
            dst_id=old_id,
            relation=EdgeRelation.SUPERSEDES,
            knowledge_time=_NOW,
            event_time=_CHANGE_AT,
            evidence="re-derived pattern/post_dinner/evening; seen_count=2",
        )
    )
    return old_id, new_id


# ── what_changed ─────────────────────────────────────────────────────────────


def test_what_changed_surfaces_stopped_and_started() -> None:
    store = _store()
    _seed_change(store)
    changes = what_changed(store, now=_NOW)

    assert len(changes) == 1
    change = changes[0]
    assert change["when"] == _CHANGE_AT.isoformat()
    assert change["what_stopped"] == "Post-dinner spikes recur most evenings"
    assert change["what_started"] == "Post-dinner spikes stopped after the site change"
    assert "re-derived" in change["evidence"]
    assert "num_hypo" in change["episodes_before"]
    assert "num_hyper" in change["episodes_after"]


def test_what_changed_skips_reverifications() -> None:
    store = _store()
    old_id = store.insert_finding(_finding("Same pattern", status=FindingStatus.SUPERSEDED))
    new_id = store.insert_finding(_finding("Same pattern"))
    store.add_finding_edge(
        FindingEdge(
            src_id=new_id,
            dst_id=old_id,
            relation=EdgeRelation.SUPERSEDES,
            knowledge_time=_NOW,
            event_time=_CHANGE_AT,
            evidence="re-derived; seen_count=3",
        )
    )
    assert what_changed(store, now=_NOW) == []


def test_what_changed_respects_recency_window() -> None:
    store = _store()
    old_id = store.insert_finding(_finding("Old regime", status=FindingStatus.SUPERSEDED))
    new_id = store.insert_finding(_finding("New regime"))
    store.add_finding_edge(
        FindingEdge(
            src_id=new_id,
            dst_id=old_id,
            relation=EdgeRelation.SUPERSEDES,
            knowledge_time=_NOW,
            event_time=_NOW - timedelta(days=200),
            evidence="ancient change",
        )
    )
    assert what_changed(store, now=_NOW, within_days=90) == []
    assert len(what_changed(store, now=_NOW, within_days=365)) == 1


def test_what_changed_episode_counts_reflect_the_record() -> None:
    store = _store()
    _seed_change(store)
    base = _CHANGE_AT - timedelta(days=2)
    store.insert_glucose(
        [
            GlucoseEvent(ts=base + timedelta(minutes=10 * i), mg_dl=220)
            for i in range(4)
        ]
    )
    changes = what_changed(store, now=_NOW)
    assert changes[0]["episodes_before"]["num_hyper"] == 1
    assert changes[0]["episodes_after"]["num_hyper"] == 0


# ── contradicted_beliefs ─────────────────────────────────────────────────────


def test_contradicted_beliefs_pairs_belief_with_disproof() -> None:
    store = _store()
    old_id = store.insert_finding(
        _finding(
            "Rice dinners spike more than pasta dinners",
            status=FindingStatus.CONTRADICTED,
            window_start=_NOW - timedelta(days=60),
            window_end=_NOW - timedelta(days=30),
        )
    )
    new_id = store.insert_finding(_finding("Pasta dinners spike more than rice dinners"))
    store.add_finding_edge(
        FindingEdge(
            src_id=new_id,
            dst_id=old_id,
            relation=EdgeRelation.CONTRADICTS,
            knowledge_time=_NOW,
            event_time=_NOW - timedelta(days=1),
            evidence="opposite effect: prior +0.42 vs current -0.31",
        )
    )
    beliefs = contradicted_beliefs(store)

    assert len(beliefs) == 1
    entry = beliefs[0]
    assert entry["belief"] == "Rice dinners spike more than pasta dinners"
    assert entry["belief_status"] == "contradicted"
    assert entry["contradicted_by"] == "Pasta dinners spike more than rice dinners"
    assert "opposite effect" in entry["evidence"]
    assert entry["belief_window"]["start"] == (_NOW - timedelta(days=60)).isoformat()
    assert entry["contradicting_window"]["end"] == _NOW.isoformat()


def test_contradicted_beliefs_empty_without_edges() -> None:
    store = _store()
    store.insert_finding(_finding("Uncontested pattern"))
    assert contradicted_beliefs(store) == []


# ── belt exposure ────────────────────────────────────────────────────────────


def _ctx(store: SQLiteStore) -> AgentContext:
    return AgentContext(
        store=store,
        window=((_NOW - timedelta(days=30)).date(), _NOW.date()),
        gates=ColdStartReport.from_coverage(store.coverage()),
        run_id="changes-test",
    )


def test_tools_are_on_the_belt() -> None:
    store = _store()
    ctx = _ctx(store)
    belt = build_belt(ctx, DiscoveryToolkit(ctx))
    names = {spec.name for spec in belt}
    assert "what_changed" in names
    assert "contradicted_beliefs" in names


def test_what_changed_tool_returns_changes_and_counts() -> None:
    store = _store()
    _seed_change(store)
    ctx = _ctx(store)
    specs = {spec.name: spec for spec in memory_graph_specs(ctx)}

    result, numbers = specs["what_changed"].fn({})
    assert result["n_changes"] == 1
    assert result["changes"][0]["what_started"].startswith("Post-dinner spikes stopped")
    assert numbers == {"n_changes": 1}

    result, _ = specs["what_changed"].fn({"within_days": "bogus"})
    assert "error" in result


def test_contradicted_beliefs_tool_notes_empty_graph() -> None:
    store = _store()
    ctx = _ctx(store)
    specs = {spec.name: spec for spec in memory_graph_specs(ctx)}
    result, numbers = specs["contradicted_beliefs"].fn({})
    assert result["n_contradictions"] == 0
    assert "note" in result
    assert numbers == {"n_contradictions": 0}
