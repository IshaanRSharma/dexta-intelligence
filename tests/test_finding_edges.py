"""FindingEdge model contract and deterministic edge authoring.

Edges are only ever written by code paths that already compute the relationship:
``persist_findings`` (supersession on re-derive, contradiction per
``find_contradictions``), the monitor's worsened-anomaly supersede, and the
synthesis save path. No LLM authors an edge.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.memory.synthesis import SynthesisResult, save
from dexta_intelligence.models import (
    EdgeRelation,
    Finding,
    FindingEdge,
    FindingStats,
    GlucoseEvent,
)
from dexta_intelligence.store import SQLiteStore
from dexta_intelligence.workflows.deep_analysis import persist_findings
from dexta_intelligence.workflows.monitor import run_monitor

T0 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)


def _store() -> SQLiteStore:
    s = SQLiteStore(":memory:")
    s.migrate()
    return s


def _finding(**overrides: object) -> Finding:
    base: dict[str, object] = {
        "agent": "pattern",
        "kind": "overnight_low",
        "scope": "overnight",
        "headline": "Lows cluster after late boluses",
        "stats": FindingStats(effect_size=0.4, n=14, p_perm=0.01),
        "confidence": 0.7,
        "window_start": T0 - timedelta(days=14),
        "window_end": T0,
    }
    base.update(overrides)
    return Finding.model_validate(base)


# ── model contract ────────────────────────────────────────────────────────────


def test_edge_rejects_naive_knowledge_time() -> None:
    with pytest.raises(ValueError, match="naive datetime"):
        FindingEdge(
            src_id=2, dst_id=1, relation=EdgeRelation.SUPERSEDES,
            knowledge_time=datetime(2026, 6, 1),
        )


def test_edge_rejects_naive_event_time() -> None:
    with pytest.raises(ValueError, match="naive datetime"):
        FindingEdge(
            src_id=2, dst_id=1, relation=EdgeRelation.SUPERSEDES,
            knowledge_time=T0, event_time=datetime(2026, 6, 1),
        )


def test_edge_is_frozen() -> None:
    edge = FindingEdge(
        src_id=2, dst_id=1, relation=EdgeRelation.SUPERSEDES, knowledge_time=T0
    )
    with pytest.raises(Exception, match="frozen"):
        edge.evidence = "mutated"  # type: ignore[misc]


def test_edge_relations() -> None:
    assert {r.value for r in EdgeRelation} == {
        "supersedes", "contradicts", "supports", "co_occurs"
    }


# ── persist_findings authors edges ────────────────────────────────────────────


def test_first_persist_authors_no_edges() -> None:
    store = _store()
    persist_findings(store, [_finding()], now=T0)
    assert store.get_finding_edges() == []


def test_rederive_authors_supersedes_edge() -> None:
    store = _store()
    (old_id,) = persist_findings(store, [_finding(headline="v1")], now=T0)
    moment = T0 + timedelta(days=1)
    new = _finding(headline="v2", window_end=T0 + timedelta(days=1))
    (new_id,) = persist_findings(store, [new], now=moment)

    (edge,) = store.get_finding_edges(relation=EdgeRelation.SUPERSEDES)
    assert (edge.src_id, edge.dst_id) == (new_id, old_id)
    assert edge.knowledge_time == moment
    assert edge.event_time == new.window_end
    assert "pattern/overnight_low/overnight" in edge.evidence
    assert "seen_count=2" in edge.evidence


def test_opposite_effect_authors_contradicts_edge() -> None:
    store = _store()
    (old_id,) = persist_findings(store, [_finding()], now=T0)
    flipped = _finding(stats=FindingStats(effect_size=-0.4, n=14, p_perm=0.01))
    (new_id,) = persist_findings(store, [flipped], now=T0 + timedelta(days=1))

    (edge,) = store.get_finding_edges(relation=EdgeRelation.CONTRADICTS)
    assert (edge.src_id, edge.dst_id) == (new_id, old_id)
    assert "+0.4" in edge.evidence
    assert "-0.4" in edge.evidence


def test_cross_scope_contradiction_without_supersession() -> None:
    """Same agent/kind, different scope: contradiction fires, supersession does not."""
    store = _store()
    (old_id,) = persist_findings(store, [_finding(scope="weekday")], now=T0)
    flipped = _finding(
        scope="weekend", stats=FindingStats(effect_size=-0.4, n=14, p_perm=0.01)
    )
    (new_id,) = persist_findings(store, [flipped], now=T0 + timedelta(days=1))

    assert store.get_finding_edges(relation=EdgeRelation.SUPERSEDES) == []
    (edge,) = store.get_finding_edges(relation=EdgeRelation.CONTRADICTS)
    assert (edge.src_id, edge.dst_id) == (new_id, old_id)


def test_same_direction_rederive_authors_no_contradicts_edge() -> None:
    store = _store()
    persist_findings(store, [_finding()], now=T0)
    persist_findings(store, [_finding(headline="again")], now=T0 + timedelta(days=1))
    assert store.get_finding_edges(relation=EdgeRelation.CONTRADICTS) == []


# ── monitor supersede authors edges ───────────────────────────────────────────


def test_worsened_anomaly_authors_supersedes_edge() -> None:
    end = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    start = end - timedelta(hours=24)
    glucose = [
        GlucoseEvent(ts=start + timedelta(minutes=5 * i), mg_dl=110) for i in range(288)
    ]
    glucose[100] = GlucoseEvent(ts=glucose[100].ts, mg_dl=48)
    store = _store()
    store.insert_glucose(glucose)
    ctx = AgentContext(
        store=store,
        window=(end.date() - timedelta(days=30), end.date()),
        gates=ColdStartReport.from_coverage(store.coverage()),
        run_id="test-run",
    )
    run_monitor(ctx, persist=True, now=end)
    assert store.get_finding_edges() == []

    store.insert_glucose([GlucoseEvent(ts=glucose[120].ts + timedelta(minutes=1), mg_dl=40)])
    run_monitor(ctx, persist=True, now=end)

    lows = store.get_findings(kind="anomaly", status=None, limit=1000)
    ids = {f.id for f in lows if f.scope == "severe_low"}
    edges = [
        e
        for e in store.get_finding_edges(relation=EdgeRelation.SUPERSEDES)
        if e.src_id in ids and e.dst_id in ids
    ]
    assert len(edges) == 1


# ── synthesis save authors edges ──────────────────────────────────────────────


def test_synthesis_resave_authors_supersedes_edge() -> None:
    store = _store()
    save(store, SynthesisResult(topic_paragraphs={"k": "p"}), today=date(2026, 6, 1))
    assert store.get_finding_edges() == []

    save(store, SynthesisResult(topic_paragraphs={"k": "q"}), today=date(2026, 6, 2))
    (edge,) = store.get_finding_edges(relation=EdgeRelation.SUPERSEDES)
    assert edge.event_time is None  # synthesis findings carry no timeline window
    assert edge.knowledge_time == datetime(2026, 6, 2, tzinfo=UTC)
    synth = store.get_findings(agent="synthesis", limit=10)
    ids = {f.id for f in synth}
    assert edge.src_id in ids
    assert edge.dst_id in ids
    assert edge.src_id != edge.dst_id
