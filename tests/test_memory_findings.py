"""Tests for memory finding helpers."""

from __future__ import annotations

from dexta_intelligence.memory.findings import (
    count_recurrence,
    find_contradictions,
    find_similar,
    recurrence_headline_suffix,
)
from dexta_intelligence.models import Finding, FindingStats, FindingStatus


def _finding(
    *,
    kind: str = "pattern_tod_drift",
    effect: float | None = 5.0,
    fid: int | None = None,
) -> Finding:
    return Finding(
        id=fid,
        agent="pattern",
        kind=kind,
        scope="pattern_analysis",
        headline="h",
        body_md="",
        stats=FindingStats(effect_size=effect, n=20),
        status=FindingStatus.ACTIVE,
    )


def test_count_recurrence() -> None:
    current = _finding(fid=3)
    prior = [_finding(fid=1), _finding(fid=2), _finding(kind="other")]
    assert count_recurrence(current, prior) == 2


def test_find_contradictions() -> None:
    current = _finding(effect=10.0)
    prior = [_finding(effect=-8.0, fid=1), _finding(effect=5.0, fid=2)]
    contradictions = find_contradictions(current, prior)
    assert len(contradictions) == 1
    assert contradictions[0].id == 1


def test_recurrence_suffix() -> None:
    assert recurrence_headline_suffix(0) == ""
    assert "3 occurrence" in recurrence_headline_suffix(2)


def test_find_similar_excludes_self() -> None:
    current = _finding(fid=1)
    prior = [current, _finding(fid=2)]
    similar = find_similar(current, prior)
    assert len(similar) == 1
    assert similar[0].id == 2
