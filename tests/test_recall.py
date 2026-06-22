"""recall() as the structured shared-context channel between agents.

Covers what is surfaced (skeptic_notes/status/confidence, open_questions,
connections) and the caps on findings/connections/open_questions, with
backward-compatible keys preserved.
"""

from __future__ import annotations

from datetime import UTC, datetime

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.agents.tools.toolkit import _MAX_RECALL_ITEMS, _recall
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.memory import synthesis
from dexta_intelligence.models import (
    Finding,
    FindingStats,
    FindingStatus,
    Hypothesis,
    HypothesisStatus,
)
from dexta_intelligence.store import SQLiteStore

_NOW = datetime(2026, 6, 1, tzinfo=UTC)


def _ctx(store: SQLiteStore) -> AgentContext:
    return AgentContext(
        store=store,
        window=(_NOW.date(), _NOW.date()),
        gates=ColdStartReport.from_coverage(store.coverage()),
        run_id="test-recall",
    )


def _store() -> SQLiteStore:
    store = SQLiteStore(":memory:")
    store.migrate()
    return store


def test_recall_surfaces_skeptic_notes_status_confidence() -> None:
    store = _store()
    store.insert_finding(
        Finding(
            agent="discovery",
            kind="pattern",
            scope="overnight",
            headline="Overnight glucose drifts up after 3am.",
            confidence=0.7,
            status=FindingStatus.ACTIVE,
            skeptic_notes="Confounded by weekend late meals; sleep score not controlled.",
            stats=FindingStats(effect_size=0.6, n=12),
            window_end=_NOW,
        )
    )
    payload, numbers = _recall(_ctx(store), "overnight")

    assert payload["findings"], "expected the overnight finding back"
    top = payload["findings"][0]
    assert top["status"] == "active"
    assert top["confidence"] == 0.7
    assert "weekend" in top["skeptic_notes"]
    # numbers stay guard-traceable
    assert numbers["finding_1"] == {"effect_size": 0.6, "n": 12, "confidence": 0.7}


def test_recall_omits_skeptic_notes_when_absent() -> None:
    store = _store()
    store.insert_finding(
        Finding(
            agent="discovery",
            kind="pattern",
            scope="overnight",
            headline="Overnight glucose is stable.",
            window_end=_NOW,
        )
    )
    payload, _ = _recall(_ctx(store), "overnight")
    assert payload["findings"]
    assert "skeptic_notes" not in payload["findings"][0]


def test_recall_backward_compatible_keys_and_numbers_tuple() -> None:
    store = _store()
    store.insert_finding(
        Finding(
            agent="discovery",
            kind="pattern",
            scope="overnight",
            headline="Overnight glucose drifts up.",
            stats=FindingStats(effect_size=0.5, n=10),
            window_end=_NOW,
        )
    )
    store.insert_hypothesis(Hypothesis(statement="Is the dawn effect real?"))
    payload, numbers = _recall(_ctx(store), "overnight")

    # shape callers/tests rely on
    assert set(payload) >= {"findings", "open_questions"}
    assert payload["open_questions"] == ["Is the dawn effect real?"]
    assert isinstance(numbers, dict)
    assert "finding_1" in numbers


def test_recall_caps_findings_at_max() -> None:
    store = _store()
    for i in range(_MAX_RECALL_ITEMS + 5):
        store.insert_finding(
            Finding(
                agent="discovery",
                kind="pattern",
                scope="overnight",
                headline=f"Overnight finding number {i}.",
                window_end=_NOW,
            )
        )
    payload, _ = _recall(_ctx(store), "overnight")
    assert len(payload["findings"]) <= _MAX_RECALL_ITEMS


def test_recall_caps_open_questions_with_note() -> None:
    store = _store()
    store.insert_finding(
        Finding(
            agent="discovery",
            kind="pattern",
            scope="overnight",
            headline="Overnight pattern.",
            window_end=_NOW,
        )
    )
    for i in range(_MAX_RECALL_ITEMS + 3):
        store.insert_hypothesis(Hypothesis(statement=f"Open question {i}?"))
    payload, _ = _recall(_ctx(store), "overnight")
    assert len(payload["open_questions"]) == _MAX_RECALL_ITEMS
    assert "open_questions_note" in payload


def test_recall_caps_connections_with_note() -> None:
    store = _store()
    store.insert_finding(
        Finding(
            agent="discovery",
            kind="pattern",
            scope="overnight",
            headline="Overnight pattern.",
            window_end=_NOW,
        )
    )
    conns = [f"Overnight connection observation {i}." for i in range(_MAX_RECALL_ITEMS + 4)]
    synthesis.save(
        store, synthesis.SynthesisResult(connections=conns), today=_NOW.date()
    )

    # query path (ranked) is capped
    payload, _ = _recall(_ctx(store), "overnight")
    assert "connections" in payload
    assert len(payload["connections"]) <= _MAX_RECALL_ITEMS
    assert "connections_note" in payload

    # no-query path is capped too
    payload_nq, _ = _recall(_ctx(store), "")
    assert len(payload_nq["connections"]) <= _MAX_RECALL_ITEMS


def test_recall_open_questions_only_open_status() -> None:
    store = _store()
    store.insert_finding(
        Finding(
            agent="discovery",
            kind="pattern",
            scope="overnight",
            headline="Overnight pattern.",
            window_end=_NOW,
        )
    )
    store.insert_hypothesis(Hypothesis(statement="Still open?"))
    store.insert_hypothesis(
        Hypothesis(statement="Settled.", status=HypothesisStatus.SUPPORTED)
    )
    payload, _ = _recall(_ctx(store), "overnight")
    assert payload["open_questions"] == ["Still open?"]


def _f(scope: str, headline: str, status: FindingStatus) -> Finding:
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


def test_recall_excludes_non_active_and_labels_them() -> None:
    store = _store()
    store.insert_finding(_f("overnight", "Active overnight pattern.", FindingStatus.ACTIVE))
    store.insert_finding(_f("dawn", "Rejected dawn pattern.", FindingStatus.REJECTED))
    store.insert_finding(_f("dinner", "Contradicted dinner pattern.", FindingStatus.CONTRADICTED))
    store.insert_finding(_f("am", "Superseded morning pattern.", FindingStatus.SUPERSEDED))

    payload, _numbers = _recall(_ctx(store), "")

    used = [item["headline"] for item in payload["findings"]]
    assert "Active overnight pattern." in used
    assert "Rejected dawn pattern." not in used
    assert "Contradicted dinner pattern." not in used
    assert "Superseded morning pattern." not in used

    excluded = {e["headline"]: e["reason"] for e in payload.get("excluded", [])}
    assert excluded["Rejected dawn pattern."] == "not_used_rejected"
    assert excluded["Contradicted dinner pattern."] == "not_used_contradicted"
    assert excluded["Superseded morning pattern."] == "not_used_superseded"
