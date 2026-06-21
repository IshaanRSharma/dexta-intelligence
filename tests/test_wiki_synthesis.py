"""Tests for the agentic-wiki synthesis layer (LLM narrative, guard-checked)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.agents.tools.toolkit import _recall
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.memory.synthesis import (
    SynthesisResult,
    load_latest,
    save,
    synthesize,
)
from dexta_intelligence.memory.wiki import generate_wiki
from dexta_intelligence.models import Finding, FindingStats, FindingStatus
from dexta_intelligence.store import SQLiteStore

if TYPE_CHECKING:
    from pathlib import Path

TODAY = date(2026, 6, 11)


@dataclass
class _Response:
    content: str


class _FakeModel:
    """Scripted model: returns a fixed JSON payload, records the messages."""

    def __init__(self, payload: dict[str, Any] | str) -> None:
        self._payload = payload if isinstance(payload, str) else json.dumps(payload)
        self.calls: list[Any] = []

    def invoke(self, messages: Any) -> _Response:
        self.calls.append(messages)
        return _Response(content=self._payload)


class _BoomModel:
    """Model that raises - exercises the graceful-None failure path."""

    def invoke(self, messages: Any) -> _Response:
        msg = "model exploded"
        raise RuntimeError(msg)


def _store() -> SQLiteStore:
    store = SQLiteStore(":memory:")
    store.migrate()
    return store


def _finding(
    *,
    kind: str = "pattern_tod_drift",
    headline: str = "Overnight drift +28 mg/dL on weeknights",
    status: FindingStatus = FindingStatus.ACTIVE,
) -> Finding:
    return Finding(
        agent="pattern",
        kind=kind,
        scope="pattern_analysis",
        headline=headline,
        evidence={"drift_mg_dl": 28.0, "nights": 46},
        stats=FindingStats(effect_size=0.71, n=46, p_perm=0.003, q_fdr=0.04, replicated=True),
        confidence=0.8,
        status=status,
        window_start=datetime(2026, 3, 12, tzinfo=UTC),
        window_end=datetime(2026, 6, 10, tzinfo=UTC),
    )


def test_faithful_synthesis_renders_on_topic_and_index(tmp_path: Path) -> None:
    findings = [_finding()]
    model = _FakeModel(
        {
            "topic_paragraphs": {
                "pattern_tod_drift": "Drift of 28 mg/dL recurred across 46 nights.",
            },
            "connections": ["The drift over 46 nights aligns with the weeknight window."],
        }
    )
    result = synthesize(findings, model)
    assert result.topic_paragraphs["pattern_tod_drift"].startswith("Drift of 28")
    assert len(result.connections) == 1

    store = _store()
    store.insert_finding(findings[0])
    report = generate_wiki(store, root=tmp_path / "wiki", today=TODAY, git=False, synthesis=result)

    topic = (report.root / "topics" / "pattern-tod-drift.md").read_text()
    assert "## Synthesis" in topic
    assert "Drift of 28 mg/dL recurred across 46 nights." in topic

    index = (report.root / "index.md").read_text()
    assert "## Connections" in index
    assert "The drift over 46 nights aligns with the weeknight window." in index
    # Connections section precedes Boards.
    assert index.index("## Connections") < index.index("## Boards")


def test_fabricated_number_paragraph_is_dropped_by_guard(tmp_path: Path) -> None:
    findings = [_finding()]
    model = _FakeModel(
        {
            "topic_paragraphs": {
                # 999 is nowhere in the evidence pool - the guard must reject it.
                "pattern_tod_drift": "An unexplained spike of 999 mg/dL appeared.",
            },
            "connections": ["Another fabricated jump of 777 units was observed."],
        }
    )
    result = synthesize(findings, model)
    assert result.topic_paragraphs == {}
    assert result.connections == []
    assert result.is_empty()

    store = _store()
    store.insert_finding(findings[0])
    report = generate_wiki(store, root=tmp_path / "wiki", today=TODAY, git=False, synthesis=result)
    topic = (report.root / "topics" / "pattern-tod-drift.md").read_text()
    assert "## Synthesis" not in topic
    assert "999" not in topic
    index = (report.root / "index.md").read_text()
    assert "## Connections" not in index


def test_model_failure_yields_empty_result_and_wiki_still_generates(tmp_path: Path) -> None:
    findings = [_finding()]
    result = synthesize(findings, _BoomModel())
    assert result == SynthesisResult()
    assert result.is_empty()

    store = _store()
    store.insert_finding(findings[0])
    report = generate_wiki(store, root=tmp_path / "wiki", today=TODAY, git=False, synthesis=result)
    topic = (report.root / "topics" / "pattern-tod-drift.md").read_text()
    assert "## Synthesis" not in topic
    assert "Overnight drift +28 mg/dL on weeknights" in topic


def test_no_model_or_no_active_findings_is_empty() -> None:
    assert synthesize([_finding()], None) == SynthesisResult()
    inactive = _finding(status=FindingStatus.REJECTED)
    model = _FakeModel({"topic_paragraphs": {}, "connections": []})
    assert synthesize([inactive], model).is_empty()


def test_over_long_connection_is_dropped() -> None:
    long_line = " ".join(["drift"] * 31)  # 31 words, all traceable, still too long
    model = _FakeModel({"topic_paragraphs": {}, "connections": [long_line]})
    result = synthesize([_finding()], model)
    assert result.connections == []


def test_synthesis_none_output_is_byte_identical(tmp_path: Path) -> None:
    store = _store()
    store.insert_finding(_finding())
    baseline = generate_wiki(store, root=tmp_path / "wiki_a", today=TODAY, git=False)
    explicit = generate_wiki(
        store, root=tmp_path / "wiki_b", today=TODAY, git=False, synthesis=None
    )
    for page in baseline.pages:
        rel = page.relative_to(baseline.root)
        assert page.read_bytes() == (explicit.root / rel).read_bytes()


# ── persistence: save / load_latest round-trip + supersession ─────────────────


def test_save_load_latest_round_trip() -> None:
    store = _store()
    result = SynthesisResult(
        topic_paragraphs={"pattern_tod_drift": "Overnight drift recurred."},
        connections=["Overnight drift aligns with the weeknight window."],
    )
    save(store, result, today=TODAY)

    loaded = load_latest(store)
    assert loaded is not None
    assert loaded.topic_paragraphs == {"pattern_tod_drift": "Overnight drift recurred."}
    assert loaded.connections == ["Overnight drift aligns with the weeknight window."]


def test_load_latest_none_when_never_saved() -> None:
    assert load_latest(_store()) is None


def test_save_supersedes_prior_synthesis() -> None:
    store = _store()
    save(store, SynthesisResult(connections=["old connection"]), today=date(2026, 6, 1))
    save(store, SynthesisResult(connections=["new connection"]), today=TODAY)

    active = store.get_findings(agent="synthesis", status=FindingStatus.ACTIVE)
    assert len(active) == 1  # exactly one survives
    assert active[0].evidence["connections"] == ["new connection"]
    superseded = store.get_findings(agent="synthesis", status=FindingStatus.SUPERSEDED)
    assert len(superseded) == 1
    assert superseded[0].evidence["connections"] == ["old connection"]

    loaded = load_latest(store)
    assert loaded is not None
    assert loaded.connections == ["new connection"]


def test_synthesize_excludes_persisted_synthesis_findings() -> None:
    # A persisted synthesis finding must never feed back into synthesize().
    synthesis_finding = Finding(
        agent="synthesis",
        kind="wiki_synthesis",
        scope="memory",
        headline="Wiki synthesis",
        evidence={"connections": ["x"]},
    )
    model = _FakeModel({"topic_paragraphs": {}, "connections": []})
    assert synthesize([synthesis_finding], model).is_empty()


# ── recall reads back connections ─────────────────────────────────────────────


def _ctx(store: SQLiteStore) -> AgentContext:
    return AgentContext(
        store=store,
        window=(date(2026, 3, 1), TODAY),
        gates=ColdStartReport.from_coverage(store.coverage()),
        run_id="synthesis-recall-test",
    )


def test_recall_payload_includes_ranked_connections() -> None:
    store = _store()
    store.insert_finding(_finding())
    save(
        store,
        SynthesisResult(
            connections=[
                "Post-meal spikes follow larger dinners.",
                "Overnight drift aligns with the weeknight window.",
            ]
        ),
        today=TODAY,
    )

    payload, _numbers = _recall(_ctx(store), "overnight")
    assert "connections" in payload
    # The query-relevant connection ranks first.
    assert payload["connections"][0].startswith("Overnight drift")
    assert "findings" in payload


def test_recall_omits_connections_when_none_saved() -> None:
    store = _store()
    store.insert_finding(_finding())
    payload, _numbers = _recall(_ctx(store), "overnight")
    assert "connections" not in payload


def test_recall_excludes_synthesis_findings_from_findings_list() -> None:
    store = _store()
    store.insert_finding(_finding())
    save(store, SynthesisResult(connections=["a connection"]), today=TODAY)
    payload, _numbers = _recall(_ctx(store), "drift")
    kinds = {item["headline"] for item in payload["findings"]}
    assert all("Wiki synthesis" not in headline for headline in kinds)
