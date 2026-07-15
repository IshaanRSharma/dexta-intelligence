"""Recurrence lines on every finding surface.

``recurrence_line`` turns a finding's own lifecycle fields (seen_count +
window bounds) into the "seen 7 times since May 12" receipt, and the surfaces
that already render findings (clinical brief sections, the recall tool) carry
it. Deterministic: no store query, no clock.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.agents.brief import build_brief
from dexta_intelligence.agents.tools.toolkit import _recall
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.memory.findings import recurrence_line
from dexta_intelligence.models import CoverageStats, Finding, FindingStats
from dexta_intelligence.store import SQLiteStore

_END = datetime(2026, 6, 1, tzinfo=UTC)
_START = datetime(2026, 5, 12, tzinfo=UTC)


def _finding(**overrides: object) -> Finding:
    base: dict[str, object] = {
        "agent": "pattern",
        "kind": "post_run_low",
        "scope": "overnight",
        "headline": "Lows follow evening runs by about 5 hours",
        "stats": FindingStats(effect_size=0.6, n=9),
        "confidence": 0.85,
        "seen_count": 7,
        "window_start": _START,
        "window_end": _END,
    }
    base.update(overrides)
    return Finding.model_validate(base)


def test_recurrence_line_from_lifecycle_fields() -> None:
    assert recurrence_line(_finding()) == "seen 7 times since May 12"


def test_recurrence_line_empty_on_first_sighting() -> None:
    assert recurrence_line(_finding(seen_count=1)) == ""


def test_recurrence_line_without_window_still_counts() -> None:
    line = recurrence_line(_finding(window_start=None, window_end=None, seen_count=3))
    assert line == "seen 3 times"


def test_brief_section_carries_recurrence() -> None:
    coverage = CoverageStats(
        first_ts=_START,
        last_ts=_END,
        span_days=20.0,
        n_glucose=5000,
        glucose_coverage_pct=95.0,
        n_insulin=0,
        days_with_insulin_pct=0.0,
        n_meals=0,
        n_sleep=0,
        n_activity=0,
    )
    brief = build_brief([_finding()], coverage, model=None, today=_END.date())
    assert brief.sections
    assert "Seen 7 times since May 12." in brief.sections[0].body


def test_recall_items_carry_recurrence() -> None:
    store = SQLiteStore(":memory:")
    store.migrate()
    store.insert_finding(_finding())
    ctx = AgentContext(
        store=store,
        window=((_END - timedelta(days=20)).date(), _END.date()),
        gates=ColdStartReport.from_coverage(store.coverage()),
        run_id="recurrence-test",
    )
    payload, _numbers = _recall(ctx, "evening runs")
    assert payload["findings"]
    assert payload["findings"][0]["recurrence"] == "seen 7 times since May 12"
