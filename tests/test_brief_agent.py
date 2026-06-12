"""Tests for the Clinical Brief Agent (LLM rank/explain, guard + safety, fallback)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from dexta_intelligence.agents.brief import (
    BriefSection,
    ClinicalBrief,
    build_brief,
    render_markdown,
)
from dexta_intelligence.models import CoverageStats, Finding, FindingStats, FindingStatus

TODAY = date(2026, 6, 11)


@dataclass
class _Response:
    content: str


class _FakeModel:
    """Scripted model: returns a fixed JSON payload, records the messages."""

    model = "fake-clinical-model"

    def __init__(self, payload: dict[str, Any] | str) -> None:
        self._payload = payload if isinstance(payload, str) else json.dumps(payload)
        self.calls: list[Any] = []

    def invoke(self, messages: Any) -> _Response:
        self.calls.append(messages)
        return _Response(content=self._payload)


class _BoomModel:
    """Model that raises — exercises the total-failure fallback path."""

    def invoke(self, messages: Any) -> _Response:
        msg = "model exploded"
        raise RuntimeError(msg)


def _coverage(
    *, n_glucose: int = 25000, n_insulin: int = 400, span_days: float = 90.0
) -> CoverageStats:
    return CoverageStats(
        first_ts=datetime(2026, 3, 13, tzinfo=UTC),
        last_ts=datetime(2026, 6, 11, tzinfo=UTC),
        span_days=span_days,
        n_glucose=n_glucose,
        glucose_coverage_pct=94.0,
        n_insulin=n_insulin,
        days_with_insulin_pct=98.0,
        n_meals=0,
        n_sleep=0,
        n_activity=0,
    )


def _finding(
    *,
    kind: str = "pattern_tod_drift",
    headline: str = "Overnight drift +28 mg/dL on weeknights",
    confidence: float = 0.8,
    status: FindingStatus = FindingStatus.ACTIVE,
    fid: int | None = None,
) -> Finding:
    return Finding(
        id=fid,
        agent="pattern",
        kind=kind,
        scope="pattern_analysis",
        headline=headline,
        evidence={"drift_mg_dl": 28.0, "nights": 46},
        stats=FindingStats(effect_size=0.71, n=46, p_perm=0.003, q_fdr=0.04, replicated=True),
        confidence=confidence,
        status=status,
        window_start=datetime(2026, 3, 12, tzinfo=UTC),
        window_end=datetime(2026, 6, 10, tzinfo=UTC),
    )


def test_deterministic_brief_from_seeded_findings() -> None:
    findings = [
        _finding(kind="pattern_tod_drift", confidence=0.9, fid=1),
        _finding(kind="pattern_dawn", headline="Dawn rise of 35 mg/dL", confidence=0.6, fid=2),
    ]
    brief = build_brief(findings, _coverage(), model=None, today=TODAY)

    assert brief.provenance["model"] == "deterministic"
    assert brief.provenance["findings_considered"] == 2
    assert brief.provenance["generated"] == "2026-06-11"
    assert brief.data_sources_line == "glucose + insulin, 90 days, 94% coverage"
    # One section per finding, highest confidence ranked first.
    assert len(brief.sections) == 2
    assert brief.sections[0].title == "Pattern Tod Drift"
    assert "28" in brief.sections[0].body
    # Stats line present.
    assert "n=46" in brief.sections[0].body
    # Summary is the safe counts line.
    assert "2 active finding(s)" in brief.headline_summary


def test_fake_model_ranked_brief_renders() -> None:
    findings = [_finding(fid=1)]
    model = _FakeModel(
        {
            "summary": "One overnight pattern stands out across 46 nights.",
            "sections": [
                {
                    "title": "Overnight Drift",
                    "body": "Glucose drifted up 28 mg/dL across 46 nights (effect 0.71).",
                    "finding_idx": 0,
                }
            ],
        }
    )
    brief = build_brief(findings, _coverage(), model=model, today=TODAY)

    assert brief.provenance["model"] == "fake-clinical-model"
    assert brief.headline_summary == "One overnight pattern stands out across 46 nights."
    assert len(brief.sections) == 1
    assert brief.sections[0].title == "Overnight Drift"
    assert "28 mg/dL across 46 nights" in brief.sections[0].body

    md = render_markdown(brief)
    assert "# Clinical Brief" in md
    assert "## Overnight Drift" in md
    assert "fake-clinical-model" in md
    assert "glucose + insulin, 90 days, 94% coverage" in md


def test_fabricated_number_section_falls_back() -> None:
    findings = [_finding(fid=1)]
    model = _FakeModel(
        {
            "summary": "A clean overnight summary with no numbers.",
            "sections": [
                {
                    "title": "Fabricated",
                    # 999 is nowhere in the evidence pool — guard must reject it.
                    "body": "An unexplained spike of 999 mg/dL appeared overnight.",
                    "finding_idx": 0,
                }
            ],
        }
    )
    brief = build_brief(findings, _coverage(), model=model, today=TODAY)

    assert len(brief.sections) == 1
    # Fabricated body dropped; deterministic render used instead.
    assert "999" not in brief.sections[0].body
    assert "28" in brief.sections[0].body
    assert brief.sections[0].title == "Pattern Tod Drift"


def test_dosing_advice_section_rejected() -> None:
    findings = [_finding(fid=1)]
    model = _FakeModel(
        {
            "summary": "Overnight drift observed across 46 nights.",
            "sections": [
                {
                    "title": "Advice",
                    # Treatment advice — hard regex must refuse it even though
                    # every number traces to the evidence pool.
                    "body": "Given the 28 mg/dL drift, increase overnight basal by 0.71 units.",
                    "finding_idx": 0,
                }
            ],
        }
    )
    brief = build_brief(findings, _coverage(), model=model, today=TODAY)

    assert len(brief.sections) == 1
    body = brief.sections[0].body
    assert "increase overnight basal" not in body
    # Deterministic fallback render used instead.
    assert "28" in body
    assert brief.sections[0].title == "Pattern Tod Drift"


def test_empty_findings_graceful_brief() -> None:
    brief = build_brief([], _coverage(n_glucose=0, n_insulin=0, span_days=0.0), today=TODAY)

    assert isinstance(brief, ClinicalBrief)
    assert brief.sections == []
    assert "Insufficient data" in brief.headline_summary
    assert brief.provenance["model"] == "deterministic"
    assert brief.provenance["findings_considered"] == 0

    md = render_markdown(brief)
    assert "Insufficient data" in md
    assert "No findings to report" in md


def test_inactive_findings_excluded() -> None:
    findings = [
        _finding(status=FindingStatus.REJECTED, fid=1),
        _finding(status=FindingStatus.SUPERSEDED, fid=2),
    ]
    brief = build_brief(findings, _coverage(), today=TODAY)
    assert brief.sections == []
    assert "Insufficient data" in brief.headline_summary


def test_model_total_failure_falls_fully_deterministic() -> None:
    findings = [_finding(fid=1)]
    brief = build_brief(findings, _coverage(), model=_BoomModel(), today=TODAY)
    assert brief.provenance["model"] == "deterministic"
    assert len(brief.sections) == 1
    assert "28" in brief.sections[0].body


def test_brief_section_dataclass_is_frozen() -> None:
    section = BriefSection(title="t", body="b")
    assert section.evidence == {}
    try:
        section.title = "x"  # type: ignore[misc]
    except Exception as exc:  # frozen dataclass raises on attribute set
        assert "cannot assign" in str(exc) or "frozen" in str(exc).lower()
    else:  # pragma: no cover - frozen guarantees the except branch
        raise AssertionError("BriefSection should be frozen")
