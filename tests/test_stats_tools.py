"""Tests for the inferential-statistics tools on the DiscoveryToolkit.

Covers the new ``correlate`` tool (planted correlation, sign/strength/p, and
the too-few-days guard) and the significance fields ``_two_group`` now adds to
every group comparison (p_welch, cliffs_delta).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.agents.discovery_tools import DiscoveryToolkit, tool_specs
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.models import GlucoseEvent, SleepEvent
from dexta_intelligence.store import SQLiteStore

_DAYS = 12
_END = datetime(2026, 6, 1, tzinfo=UTC)
_START = _END - timedelta(days=_DAYS)


def _ctx(store: SQLiteStore) -> AgentContext:
    coverage = store.coverage()
    return AgentContext(
        store=store,
        window=(_START.date(), _END.date()),
        gates=ColdStartReport.from_coverage(coverage),
        run_id="test-run",
    )


def _fill_day(glucose: list[GlucoseEvent], base: datetime, level: float) -> None:
    """Enough readings for the daily aggregate to be trusted, centered on ``level``."""
    for hour in range(0, 24, 2):
        for minute in (0, 30):
            ts = base.replace(hour=hour, minute=minute)
            glucose.append(GlucoseEvent(ts=ts, mg_dl=int(level)))


def test_correlate_planted_negative_relationship() -> None:
    """mean_glucose rises as sleep_score falls → strong negative correlation."""
    store = SQLiteStore(":memory:")
    store.migrate()
    glucose: list[GlucoseEvent] = []
    sleep: list[SleepEvent] = []
    for day in range(10):
        base = _START + timedelta(days=day)
        score = 90.0 - day * 5.0  # 90 → 45
        level = 110.0 + day * 8.0  # 110 → 182, perfectly anti-correlated with score
        _fill_day(glucose, base, level)
        # ts_end lands on the same local date as the glucose (toolkit buckets by ts_end)
        sleep.append(
            SleepEvent(
                ts_start=base.replace(hour=1),
                ts_end=base.replace(hour=8),
                duration_min=420,
                score=score,
            )
        )
    store.insert_glucose(glucose)
    store.insert_sleep(sleep)

    toolkit = DiscoveryToolkit(_ctx(store))
    result = toolkit.correlate("mean_glucose", "sleep_score")

    assert "error" not in result
    assert result["n"] == 10
    assert result["direction"] == "negative"
    assert result["pearson_r"] is not None
    assert result["pearson_r"] < -0.9
    assert result["p"] is not None
    assert result["p"] < 0.05


def test_correlate_too_few_days_returns_explanatory_dict() -> None:
    """With < 4 overlapping days correlate explains itself rather than crashing."""
    store = SQLiteStore(":memory:")
    store.migrate()
    glucose: list[GlucoseEvent] = []
    sleep: list[SleepEvent] = []
    for day in range(2):  # only 2 overlapping days
        base = _START + timedelta(days=day)
        _fill_day(glucose, base, 120.0)
        sleep.append(
            SleepEvent(
                ts_start=base.replace(hour=1),
                ts_end=base.replace(hour=8),
                duration_min=420,
                score=70.0,
            )
        )
    store.insert_glucose(glucose)
    store.insert_sleep(sleep)

    toolkit = DiscoveryToolkit(_ctx(store))
    result = toolkit.correlate("mean_glucose", "sleep_score")

    assert result["n"] < 4
    assert "note" in result
    assert "pearson_r" not in result  # guard path, no fabricated stat


def test_groupby_compare_includes_significance_fields() -> None:
    """A group comparison now surfaces p_welch and cliffs_delta keys."""
    store = SQLiteStore(":memory:")
    store.migrate()
    glucose: list[GlucoseEvent] = []
    sleep: list[SleepEvent] = []
    for day in range(_DAYS):
        base = _START + timedelta(days=day)
        # poorer-sleep days run higher → a real, testable group difference
        good = day % 2 == 0
        level = 110.0 if good else 175.0
        _fill_day(glucose, base, level)
        sleep.append(
            SleepEvent(
                ts_start=base.replace(hour=1),
                ts_end=base.replace(hour=8),
                duration_min=420,
                score=85.0 if good else 50.0,
            )
        )
    store.insert_glucose(glucose)
    store.insert_sleep(sleep)

    toolkit = DiscoveryToolkit(_ctx(store))
    result = toolkit.run("groupby_compare", {"group_by": "sleep_bucket", "target": "mean_glucose"})

    assert result.ok
    summary = result.summary
    assert "p_welch" in summary
    assert "cliffs_delta" in summary
    assert "mann_whitney_p" in summary
    assert "rank_biserial" in summary
    assert "welch_t" in summary
    assert "significant" in summary
    # existing keys preserved
    assert "cohen_d" in summary and "interpretation" in summary
    # these fields are auditable through the evidence pool
    evidence = result.evidence()
    assert "p_welch" in evidence and "cliffs_delta" in evidence


def test_correlate_spec_is_exposed() -> None:
    """The correlate ToolSpec is wired into the belt and runs end to end."""
    store = SQLiteStore(":memory:")
    store.migrate()
    glucose: list[GlucoseEvent] = []
    for day in range(6):
        _fill_day(glucose, _START + timedelta(days=day), 120.0 + day)
    store.insert_glucose(glucose)

    ctx = _ctx(store)
    toolkit = DiscoveryToolkit(ctx)
    specs = tool_specs(ctx, toolkit)
    correlate_spec = next((s for s in specs if s.name == "correlate"), None)
    assert correlate_spec is not None
    result, numbers = correlate_spec.fn({"x": "mean_glucose", "y": "tir"})
    assert result["x"] == "mean_glucose"
    assert "n" in numbers
