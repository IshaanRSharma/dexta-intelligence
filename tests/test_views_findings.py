"""Tests for the Findings page view-model.

Deterministic and model-free: a real in-memory SQLiteStore is seeded, a fixed
``now`` is passed for stable relative-time strings, and the returned dict shape
is asserted directly.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from dexta_intelligence.models import (
    Finding,
    FindingStats,
    FindingStatus,
    Hypothesis,
    InvestigationRun,
    RunFinding,
)
from dexta_intelligence.server.views_findings import (
    evidence_strength,
    findings_page_view,
    lifecycle_label,
)
from dexta_intelligence.store import SQLiteStore

NOW = datetime(2025, 6, 20, 12, 0, tzinfo=UTC)


def _seed() -> SQLiteStore:
    s = SQLiteStore(":memory:")
    s.migrate()

    s.insert_finding(
        Finding(
            agent="patterns",
            kind="overnight",
            scope="nocturnal",
            headline="Nocturnal lows after late workouts",
            body_md="Glucose **drops** overnight.",
            stats=FindingStats(effect_size=0.4, n=30, p_perm=0.01, q_fdr=0.03, replicated=True),
            confidence=0.82,
            status=FindingStatus.ACTIVE,
            skeptic_notes="Holds up under scrutiny.",
            window_start=datetime(2025, 6, 1, 0, 0, tzinfo=UTC),
            window_end=datetime(2025, 6, 10, 0, 0, tzinfo=UTC),
            last_verified=datetime(2025, 6, 19, 12, 0, tzinfo=UTC),
            seen_count=3,
        )
    )
    s.insert_finding(
        Finding(
            agent="patterns",
            kind="meal",
            scope="breakfast",
            headline="Rejected breakfast claim",
            status=FindingStatus.REJECTED,
            skeptic_notes="reject: underpowered.",
        )
    )
    s.insert_finding(
        Finding(
            agent="coordinator",
            kind="investigation",
            scope="whole",
            headline="Internal investigation finding",
            status=FindingStatus.ACTIVE,
        )
    )
    s.insert_finding(
        Finding(
            agent="patterns",
            kind="trend",
            scope="weekly",
            headline="Stale trend",
            status=FindingStatus.STALE,
        )
    )

    s.insert_hypothesis(
        Hypothesis(
            statement="Exercise lowers next-morning glucose.",
            status="open",
            source_finding_id=1,
            tests=[],
        )
    )

    s.insert_investigation_run(
        InvestigationRun(
            run_id="run-1",
            kind="deep",
            status="complete",
            question="Why are mornings high?",
            window_start=date(2025, 6, 1),
            window_end=date(2025, 6, 10),
            plan=["scan glucose", "correlate meals"],
            trace=["step a", "step b"],
            findings=[
                RunFinding(headline="Morning rise", kind="trend", confidence=0.7, status="active")
            ],
            n_findings=1,
            started_at=datetime(2025, 6, 19, 11, 0, tzinfo=UTC),
            finished_at=datetime(2025, 6, 19, 11, 30, tzinfo=UTC),
        )
    )
    return s


def test_counts_and_partitioning() -> None:
    view = findings_page_view(_seed(), now=NOW)
    counts = view["counts"]
    assert counts == {"active": 1, "hypotheses": 1, "rejected": 1, "runs": 1}


def test_active_excludes_investigation_and_stale() -> None:
    view = findings_page_view(_seed(), now=NOW)
    headlines = [c["headline"] for c in view["active"]]
    assert headlines == ["Nocturnal lows after late workouts"]
    assert "Internal investigation finding" not in headlines
    assert "Stale trend" not in headlines


def test_rejected_includes_rejected() -> None:
    view = findings_page_view(_seed(), now=NOW)
    rejected = view["rejected"]
    assert len(rejected) == 1
    row = rejected[0]
    assert row["headline"] == "Rejected breakfast claim"
    assert row["status"] == FindingStatus.REJECTED.value
    assert row["agent"] == "patterns"
    assert row["skeptic_notes"] == "reject: underpowered."


def test_active_card_shape() -> None:
    view = findings_page_view(_seed(), now=NOW)
    card = view["active"][0]
    # confidence 0.82 + replicated + survived -> strong; seen x3 -> verified
    assert card["strength"] == "strong"
    assert card["lifecycle"] == "verified"
    assert card["stats_line"] == "effect 0.4 · n=30 · p=0.01 · q=0.03 · replicated"
    assert card["skeptic_survived"] is True
    assert "<strong>drops</strong>" in card["body_html"]
    assert card["seen_count"] == 3
    assert card["last_verified_rel"] == "1d ago"
    assert card["window_label"] == "2025-06-01 to 2025-06-10"
    assert card["scope"] == "nocturnal"


def _finding(**kw: object) -> Finding:
    base: dict[str, object] = {
        "agent": "patterns",
        "kind": "overnight",
        "scope": "nocturnal",
        "headline": "h",
    }
    base.update(kw)
    return Finding(**base)  # type: ignore[arg-type]


def test_evidence_strength_bands() -> None:
    # skeptic-flagged is never above weak, regardless of confidence
    flagged = _finding(
        confidence=0.99, skeptic_notes="reject: confounded", stats=FindingStats(n=50)
    )
    assert evidence_strength(flagged) == "weak"
    # strong needs high confidence AND (replication or a real sample)
    strong = _finding(confidence=0.8, stats=FindingStats(n=30, replicated=True))
    assert evidence_strength(strong) == "strong"
    strong_n = _finding(confidence=0.8, stats=FindingStats(n=25))
    assert evidence_strength(strong_n) == "strong"
    # high confidence but thin, unreplicated sample is only moderate
    moderate = _finding(confidence=0.8, stats=FindingStats(n=5))
    assert evidence_strength(moderate) == "moderate"
    weak = _finding(confidence=0.3, stats=FindingStats(n=40))
    assert evidence_strength(weak) == "weak"


def test_lifecycle_label_reflects_reconfirmation() -> None:
    assert lifecycle_label(_finding(seen_count=1)) == "supported"
    assert lifecycle_label(_finding(seen_count=4)) == "verified"


def test_runs_reflect_seeded_run() -> None:
    view = findings_page_view(_seed(), now=NOW)
    runs = view["runs"]
    assert len(runs) == 1
    run = runs[0]
    assert run["question"] == "Why are mornings high?"
    assert run["kind"] == "deep"
    assert run["status"] == "complete"
    assert run["n_findings"] == 1
    assert run["plan"] == ["scan glucose", "correlate meals"]
    assert run["when"].endswith("ago")


def test_hypotheses_reflect_seeded() -> None:
    view = findings_page_view(_seed(), now=NOW)
    hyps = view["hypotheses"]
    assert len(hyps) == 1
    h = hyps[0]
    assert h["statement"] == "Exercise lowers next-morning glucose."
    assert h["status"] == "open"
    assert h["source_finding_id"] == 1
