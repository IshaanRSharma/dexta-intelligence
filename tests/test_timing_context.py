"""Timing context - deterministic, observation-only time-bucket briefings."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.agents.brief import _ADVICE_RE
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.demo import build_demo_store
from dexta_intelligence.investigations.timing_context import (
    OUTPUT_KEYS,
    SAFETY_LINE,
    resolve_bucket,
    timing_report,
)
from dexta_intelligence.models import GlucoseEvent, PredictionEvent
from dexta_intelligence.store import SQLiteStore


def _demo_ctx(timezone: str = "UTC") -> AgentContext:
    store = build_demo_store()
    cov = store.coverage()
    return AgentContext(
        store=store,
        window=(cov.first_ts.date(), cov.last_ts.date()),
        gates=ColdStartReport.from_coverage(cov),
        run_id="timing-test",
        timezone=timezone,
    )


def _offgrid_forecast_ctx() -> AgentContext:
    """A store whose oref cycle is off the CGM 5-min grid (the real-data case)."""
    store = SQLiteStore(":memory:")
    store.migrate()
    base = datetime(2026, 2, 1, 17, 0, tzinfo=UTC)
    store.insert_glucose(
        [GlucoseEvent(ts=base + timedelta(minutes=5 * i), mg_dl=150) for i in range(40)]
    )
    cycle = base + timedelta(minutes=62)  # 18:02 - two minutes off the grid
    store.insert_predictions(
        [PredictionEvent(ts=cycle, source="openaps", curve_kind="cob", values_mg_dl=[160.0] * 13)]
    )
    return AgentContext(
        store=store,
        window=(date(2026, 2, 1), date(2026, 2, 1)),
        gates=ColdStartReport.from_coverage(store.coverage()),
        run_id="offgrid",
    )


def _empty_ctx() -> AgentContext:
    store = SQLiteStore(":memory:")
    store.migrate()
    return AgentContext(
        store=store,
        window=(date(2026, 1, 1), date(2026, 2, 1)),
        gates=ColdStartReport.from_coverage(store.coverage()),
        run_id="empty",
    )


def _card_ids(report: dict) -> set[str]:
    return {c["id"] for c in report["cards"]}


# ── bucket parsing ──────────────────────────────────────────────────────────────


def test_preset_resolves() -> None:
    b = resolve_bucket("dinner")
    assert b is not None
    assert (b.start_hour, b.end_hour) == (17, 22)


def test_hour_range_resolves_both_forms() -> None:
    assert resolve_bucket("17-22").start_hour == 17  # type: ignore[union-attr]
    b = resolve_bucket("17:30-22:00")  # minutes truncate to the hour in v1
    assert b is not None
    assert (b.start_hour, b.end_hour) == (17, 22)


def test_bad_buckets_return_none() -> None:
    assert resolve_bucket("nonsense") is None
    assert resolve_bucket("22-17") is None  # reversed
    assert resolve_bucket("5-30") is None  # out of range


# ── report shape + content ──────────────────────────────────────────────────────


def test_output_schema_is_frozen() -> None:
    report = timing_report(_demo_ctx(), resolve_bucket("dinner"), intent="meal")  # type: ignore[arg-type]
    assert tuple(report) == OUTPUT_KEYS


def test_dinner_meal_surfaces_profile_glucose_and_timing() -> None:
    report = timing_report(_demo_ctx(), resolve_bucket("dinner"), intent="meal")  # type: ignore[arg-type]
    ids = _card_ids(report)
    assert {"P", "G", "U"} <= ids  # profile, glucose, usual-timing always present here
    glucose = next(c for c in report["cards"] if c["id"] == "G")
    assert glucose["n"] > 0


def test_overnight_basal_includes_basal_card() -> None:
    report = timing_report(_demo_ctx(), resolve_bucket("overnight"), intent="basal")  # type: ignore[arg-type]
    assert "B" in _card_ids(report)


def test_no_card_line_reads_as_dosing_advice() -> None:
    for intent in ("general", "meal", "basal"):
        report = timing_report(_demo_ctx(), resolve_bucket("dinner"), intent=intent)  # type: ignore[arg-type]
        for card in report["cards"]:
            for line in card["lines"]:
                assert not _ADVICE_RE.search(line), f"dosing-like line: {line!r}"
        assert report["safety"] == SAFETY_LINE


def test_is_deterministic() -> None:
    ctx_a, ctx_b = _demo_ctx(), _demo_ctx()
    bucket = resolve_bucket("dinner")
    assert timing_report(ctx_a, bucket, intent="meal") == timing_report(
        ctx_b, bucket, intent="meal"
    )  # type: ignore[arg-type]


def test_meal_intent_outside_meal_hours_skips_meal_cards() -> None:
    report = timing_report(_demo_ctx(), resolve_bucket("overnight"), intent="meal")  # type: ignore[arg-type]
    assert "U" not in _card_ids(report)
    assert any("meal" in lim.lower() for lim in report["limitations"])


def test_oref_card_present_and_labeled_with_predictions() -> None:
    report = timing_report(_demo_ctx(), resolve_bucket("dinner"), intent="meal")  # type: ignore[arg-type]
    oref = [c for c in report["cards"] if c["id"] == "O"]
    assert oref, "expected an oref0 forecast card on demo data (it logs predBGs)"
    card = oref[0]
    assert card["n"] > 0
    text = " ".join(card["lines"]).lower()
    assert "forecast error" in text
    assert "never a dose" in text  # the provenance + safety stamp is always present


def test_oref_card_absent_without_predictions() -> None:
    report = timing_report(_empty_ctx(), resolve_bucket("dinner"), intent="general")  # type: ignore[arg-type]
    assert not [c for c in report["cards"] if c["id"] == "O"]
    assert any("oref0" in lim.lower() for lim in report["limitations"])


def test_non_utc_timezone_runs_and_is_deterministic() -> None:
    bucket = resolve_bucket("dinner")
    report = timing_report(_demo_ctx("America/Los_Angeles"), bucket, intent="meal")  # type: ignore[arg-type]
    assert tuple(report) == OUTPUT_KEYS  # local-tz windows resolve without crashing
    again = timing_report(_demo_ctx("America/Los_Angeles"), bucket, intent="meal")  # type: ignore[arg-type]
    assert report == again


def test_oref_aligns_forecast_off_the_cgm_grid() -> None:
    # cycle at 18:02, horizon 19:02; nearest CGM is 19:00 (2 min) -> within tolerance.
    report = timing_report(_offgrid_forecast_ctx(), resolve_bucket("dinner"), intent="general")  # type: ignore[arg-type]
    oref = [c for c in report["cards"] if c["id"] == "O"]
    assert oref, "off-grid forecast should still align to CGM within tolerance"
    assert oref[0]["n"] == 1
    # |predicted 160 - realized 150| = 10
    assert "10" in " ".join(oref[0]["lines"])


def test_empty_store_degrades_without_crashing() -> None:
    report = timing_report(_empty_ctx(), resolve_bucket("dinner"), intent="meal")  # type: ignore[arg-type]
    assert tuple(report) == OUTPUT_KEYS
    glucose = next(c for c in report["cards"] if c["id"] == "G")
    assert glucose["n"] == 0
    assert report["limitations"]  # at least the missing-data notes
