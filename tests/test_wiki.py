"""Tests for the wiki projection of the findings store."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

from dexta_intelligence.memory.wiki import (
    STALE_THRESHOLD,
    generate_wiki,
    staleness,
    topic_slug,
)
from dexta_intelligence.models import (
    Finding,
    FindingStats,
    FindingStatus,
    Hypothesis,
    HypothesisStatus,
)
from dexta_intelligence.store import SQLiteStore

if TYPE_CHECKING:
    from pathlib import Path

TODAY = date(2026, 6, 11)


def _store() -> SQLiteStore:
    store = SQLiteStore(":memory:")
    store.migrate()
    return store


def _finding(
    *,
    kind: str = "pattern_tod_drift",
    headline: str = "Overnight drift +28 mg/dL on weeknights",
    status: FindingStatus = FindingStatus.ACTIVE,
    confidence: float = 0.8,
    window_end: datetime | None = datetime(2026, 6, 10, tzinfo=UTC),
    skeptic_notes: str | None = None,
) -> Finding:
    return Finding(
        agent="pattern",
        kind=kind,
        scope="pattern_analysis",
        headline=headline,
        evidence={"drift_mg_dl": 28.0, "nights": 46},
        stats=FindingStats(effect_size=0.71, n=46, p_perm=0.003, q_fdr=0.04, replicated=True),
        confidence=confidence,
        status=status,
        skeptic_notes=skeptic_notes,
        window_start=datetime(2026, 3, 12, tzinfo=UTC),
        window_end=window_end,
    )


def test_topic_slug() -> None:
    assert topic_slug("pattern_tod_drift") == "pattern-tod-drift"
    assert topic_slug("Post-Meal (dinner)") == "post-meal-dinner"
    assert topic_slug("!!!") == "general"


def test_staleness_decays_with_age_and_is_rescued_by_recurrence() -> None:
    old = _finding(confidence=0.5, window_end=datetime(2026, 4, 12, tzinfo=UTC))  # 60d old
    assert staleness(old, today=TODAY) > STALE_THRESHOLD
    assert staleness(old, today=TODAY, recurrence=2) <= STALE_THRESHOLD
    undated = _finding(window_end=None)
    assert staleness(undated, today=TODAY) == 0.0


def test_empty_store_renders_skeleton(tmp_path: Path) -> None:
    report = generate_wiki(_store(), root=tmp_path / "wiki", today=TODAY, git=False)
    pages = {p.relative_to(report.root).as_posix() for p in report.pages}
    assert pages == {"index.md", "hypotheses.md", "graveyard.md"}
    assert "No active findings yet" in (report.root / "index.md").read_text()
    assert "no finding has been retracted" in (report.root / "graveyard.md").read_text()
    assert not report.committed


def test_active_finding_lands_on_index_and_topic_page(tmp_path: Path) -> None:
    store = _store()
    store.insert_finding(_finding())
    store.insert_hypothesis(
        Hypothesis(statement="Site-day-3 sensitivity shift", status=HypothesisStatus.OPEN)
    )
    report = generate_wiki(store, root=tmp_path / "wiki", today=TODAY, git=False)

    index = (report.root / "index.md").read_text()
    assert "Overnight drift +28 mg/dL on weeknights" in index
    assert "[pattern-tod-drift](topics/pattern-tod-drift.md)" in index
    assert "1 open" in index

    topic = (report.root / "topics" / "pattern-tod-drift.md").read_text()
    assert "confidence 0.80" in topic
    assert "effect=0.71 · n=46 · p_perm=0.003 · q_fdr=0.04 · replicated ✓" in topic
    assert "drift_mg_dl: 28" in topic
    assert "window 2026-03-12 → 2026-06-10" in topic

    hypotheses = (report.root / "hypotheses.md").read_text()
    assert "Site-day-3 sensitivity shift" in hypotheses


def test_rejected_finding_is_buried_with_skeptic_notes(tmp_path: Path) -> None:
    store = _store()
    store.insert_finding(
        _finding(
            status=FindingStatus.REJECTED,
            skeptic_notes="confounded with weekend effect",
        )
    )
    report = generate_wiki(store, root=tmp_path / "wiki", today=TODAY, git=False)

    graveyard = (report.root / "graveyard.md").read_text()
    assert "✗ Overnight drift" in graveyard
    assert "skeptic: confounded with weekend effect" in graveyard
    index = (report.root / "index.md").read_text()
    assert "No active findings yet" in index
    assert "1 retracted or superseded" in index


def test_stale_active_finding_is_demoted_on_index(tmp_path: Path) -> None:
    store = _store()
    store.insert_finding(_finding(headline="Fresh belief"))
    store.insert_finding(
        _finding(
            kind="meal_post_dinner",
            headline="Ancient belief",
            confidence=0.5,
            window_end=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )
    report = generate_wiki(store, root=tmp_path / "wiki", today=TODAY, git=False)

    index = (report.root / "index.md").read_text()
    assert "| Fresh belief |" in index
    assert "| Ancient belief |" not in index
    assert "## Stale — awaiting fresh data" in index
    assert "- Ancient belief" in index


def test_rebuild_is_byte_identical(tmp_path: Path) -> None:
    store = _store()
    store.insert_finding(_finding())
    first = generate_wiki(store, root=tmp_path / "wiki", today=TODAY, git=False)
    snapshots = {p: p.read_bytes() for p in first.pages}
    second = generate_wiki(store, root=tmp_path / "wiki", today=TODAY, git=False)
    assert second.pages == first.pages
    assert all(p.read_bytes() == snapshots[p] for p in second.pages)


def test_run_changelog_records_survivors_and_rejections(tmp_path: Path) -> None:
    survivor = _finding(headline="Survived")
    rejected = _finding(
        headline="Killed",
        status=FindingStatus.REJECTED,
        skeptic_notes="did not replicate",
    )
    report = generate_wiki(
        _store(),
        root=tmp_path / "wiki",
        today=TODAY,
        new_findings=(survivor, rejected),
        git=False,
    )
    run_page = (report.root / "runs" / "2026-06-11.md").read_text()
    assert "1 finding(s) survived the skeptic; 1 did not." in run_page
    assert "- Survived (pattern)" in run_page
    assert "- Killed (pattern) — skeptic: did not replicate" in run_page


def test_git_history_commits_only_on_change(tmp_path: Path) -> None:
    store = _store()
    store.insert_finding(_finding())
    first = generate_wiki(store, root=tmp_path / "wiki", today=TODAY, git=True)
    assert first.committed
    assert (first.root / ".git").is_dir()
    unchanged = generate_wiki(store, root=tmp_path / "wiki", today=TODAY, git=True)
    assert not unchanged.committed
