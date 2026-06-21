"""Clinical Advisory (AMIE-style discussion brief) + the Reports page.

Deterministic and model-free by default; a scripted model exercises the refine
path; a stub evidence backend exercises PubMed citations (no network). The hard
invariant across every path: the brief never contains dosing advice.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from dexta_intelligence.agents.advisory import (
    SAFETY_LINE,
    ClinicalAdvisoryAgent,
    render_markdown,
)
from dexta_intelligence.agents.brief import _ADVICE_RE
from dexta_intelligence.evidence.base import EvidenceHit
from dexta_intelligence.models import CoverageStats, Finding, FindingStats, FindingStatus

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dexta_intelligence.config import Config
    from dexta_intelligence.store.port import StoragePort

_NOW = datetime(2026, 6, 16, 12, 0, tzinfo=UTC)


def _coverage() -> CoverageStats:
    return CoverageStats(
        first_ts=datetime(2026, 3, 1, tzinfo=UTC),
        last_ts=_NOW,
        span_days=90.0,
        n_glucose=25000,
        glucose_coverage_pct=95.0,
        n_insulin=1000,
        days_with_insulin_pct=90.0,
        n_meals=270,
        n_sleep=89,
        n_activity=40,
    )


def _finding(headline: str, *, kind: str = "pattern", scope: str = "overnight") -> Finding:
    return Finding(
        agent="observation",
        kind=kind,
        scope=scope,
        headline=headline,
        body_md=f"Evidence body for {headline}.",
        stats=FindingStats(n=24, effect_size=0.6),
        confidence=0.8,
        status=FindingStatus.ACTIVE,
    )


@dataclass
class _StubEvidence:
    """A literature backend that returns fixed PubMed hits (no network)."""

    source: str = "pubmed"

    def search(self, query: str, *, limit: int = 5) -> list[EvidenceHit]:
        return [
            EvidenceHit(title="A relevant trial", source="pubmed", id="25188375", snippet="…"),
        ][:limit]


# ── deterministic core ──────────────────────────────────────────────────────


def test_build_produces_grounded_discussion_items() -> None:
    brief = ClinicalAdvisoryAgent().build(
        [_finding("Overnight lows cluster after evening exercise")],
        _coverage(),
        question="prep for endo visit",
        now=_NOW,
    )
    assert brief.question == "prep for endo visit"
    assert brief.discuss_now
    item = brief.discuss_now[0]
    assert item.evidence_refs  # every item is grounded in the patient's findings
    assert "Overnight lows" in item.evidence_refs[0]
    assert brief.goals and brief.questions_for_clinician
    assert SAFETY_LINE in brief.limitations


def test_empty_findings_is_honest() -> None:
    brief = ClinicalAdvisoryAgent().build([], _coverage(), now=_NOW)
    assert brief.discuss_now == []
    assert "not enough" in brief.analysis[0].lower()
    assert SAFETY_LINE in brief.limitations


def test_dosing_phrased_finding_is_gated_out() -> None:
    # A finding whose headline reads as dosing advice must not become an item.
    brief = ClinicalAdvisoryAgent().build(
        [
            _finding("Increase basal insulin overnight", scope="nocturnal"),
            _finding("Dinner highs follow late boluses", scope="dinner"),
        ],
        _coverage(),
        now=_NOW,
    )
    texts = [it.item for it in brief.discuss_now]
    assert any("Dinner highs" in t for t in texts)
    assert not any("Increase basal" in t for t in texts)  # dropped by the gate


def test_no_brief_text_reads_as_dosing_advice() -> None:
    findings = [_finding(f"Pattern {i}", scope=f"s{i}") for i in range(4)]
    brief = ClinicalAdvisoryAgent().build(findings, _coverage(), now=_NOW)
    assert not _ADVICE_RE.search(render_markdown(brief))


def test_dosing_headline_gated_from_monitoring_and_questions_refs() -> None:
    # A dosing-phrased headline must not leak through evidence_refs into the
    # monitoring / questions sections (the gate covers every text field).
    brief = ClinicalAdvisoryAgent().build(
        [
            _finding("Increase basal insulin overnight", scope="nocturnal"),
            _finding("Dinner highs follow late boluses", scope="dinner"),
        ],
        _coverage(),
        now=_NOW,
    )
    refs = [
        ref
        for section in (brief.monitoring, brief.questions_for_clinician)
        for it in section
        for ref in it.evidence_refs
    ]
    assert not any("Increase basal" in r for r in refs)
    assert not _ADVICE_RE.search(" ".join(refs))


def test_pubmed_citations_attach_when_backend_present() -> None:
    brief = ClinicalAdvisoryAgent(evidence=_StubEvidence()).build(
        [_finding("Dawn rise before breakfast")], _coverage(), now=_NOW
    )
    assert brief.discuss_now[0].citations == ["25188375"]
    md = render_markdown(brief)
    assert "PMID 25188375" in md


def test_citation_lookup_failure_degrades(monkeypatch: pytest.MonkeyPatch) -> None:
    @dataclass
    class _Boom:
        source: str = "pubmed"

        def search(self, _q: str, *, limit: int = 5) -> list[EvidenceHit]:
            raise RuntimeError("network down")

    brief = ClinicalAdvisoryAgent(evidence=_Boom()).build(
        [_finding("Some pattern")], _coverage(), now=_NOW
    )
    assert brief.discuss_now[0].citations == []  # no crash, no citation


# ── optional model refine (scripted) ─────────────────────────────────────────


@dataclass
class _Reply:
    content: str


class _ScriptedModel:
    def __init__(self, payload: str) -> None:
        self._payload = payload

    def invoke(self, _messages: object) -> _Reply:
        return _Reply(self._payload)


def test_model_refines_analysis_and_goals() -> None:
    model = _ScriptedModel(
        '{"analysis": ["Three months of overnight lows."], '
        '"goals": ["Review the overnight pattern with the team."]}'
    )
    brief = ClinicalAdvisoryAgent(model=model).build(
        [_finding("Overnight lows")], _coverage(), now=_NOW
    )
    assert "Three months" in brief.analysis[0]
    assert "Review the overnight pattern" in brief.goals[0]
    assert brief.discuss_now  # grounded items still present


def test_model_dosing_goal_is_filtered() -> None:
    model = _ScriptedModel(
        '{"analysis": ["ok"], "goals": ["Increase basal insulin by 10%."]}'
    )
    brief = ClinicalAdvisoryAgent(model=model).build(
        [_finding("Overnight lows")], _coverage(), now=_NOW
    )
    # The dosing goal is dropped; the deterministic goals are kept instead.
    assert all(not _ADVICE_RE.search(g) for g in brief.goals)


def test_model_garbage_falls_back_to_deterministic() -> None:
    brief = ClinicalAdvisoryAgent(model=_ScriptedModel("not json")).build(
        [_finding("Overnight lows")], _coverage(), now=_NOW
    )
    assert brief.analysis  # deterministic analysis survives


# ── Reports route + export ────────────────────────────────────────────────────

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from dexta_intelligence.config import Config, EvidenceConfig  # noqa: E402
from dexta_intelligence.server import create_app  # noqa: E402
from dexta_intelligence.store import SQLiteStore  # noqa: E402


def _opener(db_path: Path) -> Callable[[Config, Path | None], StoragePort]:
    def _open(_config: Config, _db: Path | None = None) -> StoragePort:
        s = SQLiteStore(db_path)
        s.migrate()
        return s

    return _open


def _config_no_network() -> Config:
    # Disable the evidence backend so the Reports page never hits the network.
    return Config(evidence=EvidenceConfig(enabled=False))


def _seeded(db: Path) -> None:
    store = SQLiteStore(db)
    store.migrate()
    store.insert_finding(_finding("Dinner highs follow late boluses", scope="dinner"))


def test_reports_route_renders(tmp_path: Path) -> None:
    db = tmp_path / "reports.db"
    _seeded(db)
    client = TestClient(create_app(_config_no_network(), store_opener=_opener(db)))
    resp = client.get("/reports")
    assert resp.status_code == 200
    assert "Discussion brief" in resp.text
    assert "Dinner highs" in resp.text
    assert "Not a dosing recommendation" in resp.text


def test_reports_export_returns_markdown(tmp_path: Path) -> None:
    db = tmp_path / "reports.db"
    _seeded(db)
    client = TestClient(create_app(_config_no_network(), store_opener=_opener(db)))
    resp = client.get("/actions/reports/export")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/markdown")
    assert "attachment" in resp.headers.get("content-disposition", "")
    assert SAFETY_LINE in resp.text
    assert not _ADVICE_RE.search(resp.text)


def test_reports_empty_store(tmp_path: Path) -> None:
    db = tmp_path / "empty.db"
    SQLiteStore(db).migrate()
    client = TestClient(create_app(_config_no_network(), store_opener=_opener(db)))
    resp = client.get("/reports")
    assert resp.status_code == 200
    assert "not enough" in resp.text.lower()


def test_reports_get_does_no_literature_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The /reports GET is deterministic: even with evidence enabled it never
    constructs the backend, so a slow NCBI cannot stall first paint. Citations
    are deferred to the HTMX fragment."""
    import dexta_intelligence.agents.tools.toolkit as tk  # noqa: PLC0415

    db = tmp_path / "reports.db"
    _seeded(db)

    def _tracker(*, interactive: bool = False) -> object:
        raise AssertionError("evidence backend must not be built on the page GET")

    monkeypatch.setattr(tk, "evidence_backend", _tracker)
    client = TestClient(create_app(Config(), store_opener=_opener(db)))  # evidence on
    resp = client.get("/reports")
    assert resp.status_code == 200
    assert 'hx-get="/reports/citations"' in resp.text  # deferred lookup wired


def test_reports_citations_fragment_renders_pmids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dexta_intelligence.agents.tools.toolkit as tk  # noqa: PLC0415

    db = tmp_path / "reports.db"
    _seeded(db)
    hit = EvidenceHit(title="Prebolus trial", source="pubmed", id="25188375", snippet="x")

    class _Backend:
        source = "pubmed"

        def search(self, _query: str, *, limit: int = 5) -> list[EvidenceHit]:
            return [hit]

    def _factory(*, interactive: bool = False) -> _Backend:
        return _Backend()

    monkeypatch.setattr(tk, "evidence_backend", _factory)
    client = TestClient(create_app(Config(), store_opener=_opener(db)))
    resp = client.get("/reports/citations")
    assert resp.status_code == 200
    assert "PMID 25188375" in resp.text
    assert "pubmed.ncbi.nlm.nih.gov/25188375" in resp.text
    assert "<html" not in resp.text.lower()  # a fragment, not the full page
