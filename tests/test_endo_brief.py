"""The endo-visit brief: top discussion items with receipts and neutral questions.

Deterministic composition from findings + episodes, ranked by severity /
recurrence / recency. Two rails are asserted adversarially: a dosing-bait
finding never surfaces (treatment gate), and a finding whose headline cites a
number its evidence lacks is dropped (faithfulness).
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from dexta_intelligence.agents.brief import _ADVICE_RE
from dexta_intelligence.agents.endo_brief import build_endo_brief, render_markdown
from dexta_intelligence.analytics.episodes import Episode
from dexta_intelligence.guard.faithfulness import audit
from dexta_intelligence.models import Finding, FindingStats, FindingStatus

_TODAY = date(2026, 6, 1)
_END = datetime(2026, 6, 1, tzinfo=UTC)
_START = datetime(2026, 5, 12, tzinfo=UTC)


def _finding(**overrides: object) -> Finding:
    base: dict[str, object] = {
        "agent": "pattern",
        "kind": "post_run_low",
        "scope": "overnight",
        "headline": "Lows follow evening runs by about 5 hours",
        "evidence": {"n_events": 7, "lag_hours": 5},
        "stats": FindingStats(effect_size=0.6, n=9),
        "confidence": 0.9,
        "seen_count": 7,
        "window_start": _START,
        "window_end": _END,
    }
    base.update(overrides)
    return Finding.model_validate(base)


def _hypo(day: int, *, severe: bool = False) -> Episode:
    start = _END - timedelta(days=day, hours=3)
    end = start + timedelta(minutes=40)
    return Episode(
        id=f"hypo:{start.isoformat()}",
        kind="hypo",
        start=start,
        end=end,
        duration_min=40.0,
        n_readings=8,
        severe=severe,
        clinically_significant=True,
        extreme_mg_dl=48.0 if severe else 62.0,
        extreme_ts=start,
    )


def test_composes_top_items_from_findings_and_episodes() -> None:
    findings = [_finding()]
    episodes = [_hypo(2, severe=True), _hypo(5), _hypo(9)]
    brief = build_endo_brief(findings, episodes, today=_TODAY)

    assert 1 <= len(brief.items) <= 3
    sources = {item.source for item in brief.items}
    assert sources == {"finding", "episodes"}
    patterns = " ".join(item.pattern for item in brief.items)
    assert "Lows follow evening runs" in patterns
    assert "3 clinically significant hypo episode(s)" in patterns
    assert "1 severe" in patterns
    assert all(item.question.endswith("?") for item in brief.items)


def test_ranking_prefers_severity_recurrence_recency() -> None:
    weak = _finding(
        headline="Weekend mornings run slightly different",
        kind="weekend_shift",
        confidence=0.3,
        seen_count=1,
        window_end=_END - timedelta(days=80),
    )
    strong = _finding()
    brief = build_endo_brief([weak, strong], [], today=_TODAY)
    assert brief.items[0].pattern.startswith("Lows follow evening runs")


def test_every_item_pattern_is_faithful_to_its_receipts() -> None:
    brief = build_endo_brief([_finding()], [_hypo(2, severe=True), _hypo(5)], today=_TODAY)
    assert brief.items
    for item in brief.items:
        assert audit(item.pattern, item.receipts).ok


def test_unfaithful_finding_is_dropped() -> None:
    liar = _finding(
        headline="TIR was 83% across 41 nights",
        kind="tir_claim",
        evidence={},
        stats=FindingStats(),
        seen_count=1,
    )
    brief = build_endo_brief([liar], [], today=_TODAY)
    assert brief.items == ()


def test_dosing_bait_finding_never_surfaces() -> None:
    bait = _finding(
        headline="Increase basal by 2 units overnight to stop these lows",
        kind="dosing_bait",
        confidence=0.99,
        seen_count=9,
    )
    brief = build_endo_brief([bait, _finding()], [_hypo(2)], today=_TODAY)

    rendered = render_markdown(brief)
    assert "Increase basal" not in rendered
    assert _ADVICE_RE.search(rendered) is None
    assert brief.items  # the gate removed the bait, not the brief


def test_inactive_findings_are_ignored() -> None:
    stale = _finding(status=FindingStatus.STALE)
    brief = build_endo_brief([stale], [], today=_TODAY)
    assert brief.items == ()


def test_render_markdown_shape() -> None:
    brief = build_endo_brief([_finding()], [_hypo(2, severe=True)], today=_TODAY)
    rendered = render_markdown(brief)

    assert rendered.startswith("# Endo Visit Brief")
    assert "Ask the care team:" in rendered
    assert "Receipts:" in rendered
    assert "Observation only" in rendered
    assert "2026-06-01" in rendered


def test_empty_inputs_render_gracefully() -> None:
    brief = build_endo_brief([], [], today=_TODAY)
    rendered = render_markdown(brief)
    assert brief.items == ()
    assert "Not enough verified patterns" in rendered
