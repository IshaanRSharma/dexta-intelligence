"""Tests for the Prediction Reconciliation Agent."""

from __future__ import annotations

import time
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from dexta_intelligence.agents.base import AgentContext, AgentRegistry
from dexta_intelligence.agents.reconciliation import (
    AGENT_NAME,
    reconciliation_agent,
)
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.models import (
    CoverageStats,
    Finding,
    GlucoseEvent,
    InsulinEvent,
    InsulinKind,
    PredictionEvent,
)
from dexta_intelligence.store import SQLiteStore
from dexta_intelligence.testing.synthetic import generate_null, scenario_all

if TYPE_CHECKING:
    from collections.abc import Iterator

T0 = datetime(2025, 6, 1, 12, 0, tzinfo=UTC)
WINDOW = (date(2025, 6, 1), date(2025, 6, 3))

_CARB_ACTUAL = [120, 125, 130, 140, 155, 175, 195, 210, 215, 210, 205, 200, 195]
_CARB_IOB = [
    120.0, 122.0, 125.0, 130.0, 140.0, 155.0, 175.0, 195.0,
    210.0, 212.0, 208.0, 203.0, 198.0,
]
_CARB_COB = [
    120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0,
    128.0, 128.0, 128.0, 128.0, 128.0,
]
_CARB_UAM = [
    120.0, 122.0, 125.0, 132.0, 150.0, 172.0, 193.0, 208.0,
    214.0, 210.0, 206.0, 201.0, 196.0,
]
_SENSITIVITY_DROP = [
    150.0, 140.0, 130.0, 120.0, 110.0, 100.0, 90.0, 85.0,
    82.0, 80.0, 78.0, 76.0, 75.0,
]


@pytest.fixture
def store() -> Iterator[SQLiteStore]:
    s = SQLiteStore(":memory:")
    s.migrate()
    yield s
    s.close()


def _ctx(store: SQLiteStore, *, window: tuple[date, date] = WINDOW) -> AgentContext:
    coverage = store.coverage()
    if coverage.span_days < 1.0:
        coverage = CoverageStats(
            first_ts=coverage.first_ts or T0,
            last_ts=coverage.last_ts or T0 + timedelta(days=2),
            span_days=2.0,
            n_glucose=max(coverage.n_glucose, 100),
            glucose_coverage_pct=max(coverage.glucose_coverage_pct, 80.0),
            n_insulin=coverage.n_insulin,
            days_with_insulin_pct=coverage.days_with_insulin_pct,
            n_meals=coverage.n_meals,
            n_sleep=0,
            n_activity=0,
        )
    return AgentContext(
        store=store,
        window=window,
        gates=ColdStartReport.from_coverage(coverage),
        run_id="test-run",
    )


def _glucose_series(base: datetime, values: list[int]) -> list[GlucoseEvent]:
    return [
        GlucoseEvent(ts=base + timedelta(minutes=5 * i), mg_dl=v) for i, v in enumerate(values)
    ]


def _openaps_curves(
    ts: datetime,
    *,
    iob: list[float],
    cob: list[float],
    uam: list[float],
    zt: list[float] | None = None,
) -> list[PredictionEvent]:
    zt_vals = zt if zt is not None else iob
    return [
        PredictionEvent(ts=ts, source="openaps", curve_kind="iob", values_mg_dl=iob),
        PredictionEvent(ts=ts, source="openaps", curve_kind="cob", values_mg_dl=cob),
        PredictionEvent(ts=ts, source="openaps", curve_kind="uam", values_mg_dl=uam),
        PredictionEvent(ts=ts, source="openaps", curve_kind="zt", values_mg_dl=zt_vals),
    ]


def _plant_carb_miss_cycles(store: SQLiteStore) -> None:
    cycles: list[PredictionEvent] = []
    for offset in range(12):
        ts = T0 + timedelta(hours=offset)
        store.insert_glucose(_glucose_series(ts, _CARB_ACTUAL))
        cycles.extend(
            _openaps_curves(ts, iob=_CARB_IOB, cob=_CARB_COB, uam=_CARB_UAM)
        )
    store.insert_predictions(cycles)


class TestTierA:
    def test_carb_underestimate_when_uam_fits_and_cob_misses(self, store: SQLiteStore) -> None:
        """UAM tracks an unannounced rise; COB misses → carb_underestimate."""
        _plant_carb_miss_cycles(store)

        findings = reconciliation_agent.run(_ctx(store))
        kinds = {f.kind for f in findings}
        assert "prediction_miss_carb_underestimate" in kinds
        hit = next(f for f in findings if f.kind == "prediction_miss_carb_underestimate")
        assert hit.evidence["tier"] == "A"
        assert hit.evidence["contributor"] == "carb_underestimate"

    def test_sensitivity_shift_when_all_curves_miss_same_direction(
        self, store: SQLiteStore
    ) -> None:
        """All predBG curves overshoot a flat actual → sensitivity_shift."""
        flat = [150] * 13
        cycles: list[PredictionEvent] = []
        for offset in range(12):
            ts = T0 + timedelta(hours=offset)
            store.insert_glucose(_glucose_series(ts, flat))
            cycles.extend(
                _openaps_curves(
                    ts,
                    iob=_SENSITIVITY_DROP,
                    cob=_SENSITIVITY_DROP,
                    uam=_SENSITIVITY_DROP,
                    zt=_SENSITIVITY_DROP,
                )
            )
        store.insert_predictions(cycles)

        findings = reconciliation_agent.run(_ctx(store))
        assert any(f.kind == "prediction_miss_sensitivity_shift" for f in findings)
        hit = next(f for f in findings if f.kind == "prediction_miss_sensitivity_shift")
        assert hit.evidence["tier"] == "A"

    def test_perfect_predictions_emit_no_findings(self, store: SQLiteStore) -> None:
        actual = [120, 122, 125, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146]
        ts = T0
        store.insert_glucose(_glucose_series(ts, actual))
        floats = [float(v) for v in actual]
        store.insert_predictions(_openaps_curves(ts, iob=floats, cob=floats, uam=floats))

        findings = reconciliation_agent.run(_ctx(store))
        assert findings == []


class TestTierB:
    def test_tier_b_finding_labeled_weaker_evidence(self, store: SQLiteStore) -> None:
        """No logged predictions: expectations from oref; planted rise is caught."""
        base = T0
        values = [120, 118, 116, 115, 114, 113, 160, 185, 200, 195, 190, 185, 180]
        store.insert_glucose(_glucose_series(base, values))
        store.insert_insulin(
            [
                InsulinEvent(ts=base - timedelta(hours=2), kind=InsulinKind.BOLUS, units=2.0),
                InsulinEvent(ts=base - timedelta(hours=1), kind=InsulinKind.BOLUS, units=1.0),
            ]
        )
        for offset in range(1, 12):
            ts = base + timedelta(hours=offset)
            store.insert_glucose(_glucose_series(ts, values))
            store.insert_insulin(
                [InsulinEvent(ts=ts - timedelta(hours=1), kind=InsulinKind.BOLUS, units=1.0)]
            )

        findings = reconciliation_agent.run(_ctx(store))
        assert findings, "expected at least one Tier B reconciliation finding"
        assert all(f.evidence["tier"] == "B" for f in findings)
        assert all(f.confidence <= 0.6 for f in findings)
        assert all("weaker evidence" in f.body_md for f in findings)


class TestRecurrence:
    def test_recurrence_count_in_evidence(self, store: SQLiteStore) -> None:
        _plant_carb_miss_cycles(store)

        for _ in range(3):
            store.insert_finding(
                Finding(
                    agent=AGENT_NAME,
                    kind="prediction_miss_carb_underestimate",
                    scope="prediction_reconciliation",
                    headline="prior",
                    evidence={"contributor": "carb_underestimate"},
                )
            )

        findings = reconciliation_agent.run(_ctx(store))
        hit = next(f for f in findings if f.kind == "prediction_miss_carb_underestimate")
        assert hit.evidence["recurrence_count"] == 4
        assert "4 occurrence" in hit.body_md


class TestRigorNull:
    def test_noise_only_data_emits_nothing(self, store: SQLiteStore) -> None:
        events, _manifest = generate_null(seed=7, n_days=14, start=T0)
        store.insert_glucose(events["glucose"])
        preds: list[PredictionEvent] = []
        for g in events["glucose"][::12]:
            vals = [float(g.mg_dl)] * 13
            preds.extend(_openaps_curves(g.ts, iob=vals, cob=vals, uam=vals))
        store.insert_predictions(preds)

        null_window = (T0.date(), (T0 + timedelta(days=14)).date())
        findings = reconciliation_agent.run(_ctx(store, window=null_window))
        assert findings == []


class TestDataRequirementGating:
    def test_under_data_agent_skipped_by_registry(self, store: SQLiteStore) -> None:
        store.insert_glucose([GlucoseEvent(ts=T0, mg_dl=120)])
        coverage = store.coverage()
        gates = ColdStartReport.from_coverage(coverage)
        ctx = AgentContext(store=store, window=WINDOW, gates=gates, run_id="gated")
        skipped: list[str] = []

        def on_skip(name: str, reasons: list[str]) -> None:
            skipped.append(name)

        registry = AgentRegistry()
        registry.register(reconciliation_agent)
        findings = registry.run_all(ctx, on_skip=on_skip)
        assert findings == []
        assert skipped == [AGENT_NAME]

    def test_missing_tier_inputs_returns_empty_without_crash(self, store: SQLiteStore) -> None:
        store.insert_glucose(
            [
                GlucoseEvent(ts=T0, mg_dl=120),
                GlucoseEvent(ts=T0 + timedelta(minutes=5), mg_dl=122),
            ]
        )
        assert reconciliation_agent.run(_ctx(store)) == []


class TestAgentContract:
    def test_name_and_requires(self) -> None:
        assert reconciliation_agent.name == AGENT_NAME
        assert reconciliation_agent.requires.min_span_days >= 1.0
        assert reconciliation_agent.requires.min_glucose_coverage_pct >= 50.0


class TestPerformance:
    def test_90_day_scenario_completes_under_15s(self, store: SQLiteStore) -> None:
        """Tier B reconciliation on 90 days must not be O(n²) (regression guard).

        The old code rebuilt the deviation series per meal per cycle and shuffled
        an unbounded null/effect group, which never completed on this input. The
        bound is intentionally generous (real runtime is well under it) so it
        catches algorithmic blowups, not minor perf drift, on slow CI hardware.
        """
        events, _ = scenario_all(seed=42, n_days=90)
        store.insert_glucose(events["glucose"])
        store.insert_insulin(events["insulin"])
        store.insert_meals(events["meal"])
        coverage = store.coverage()
        first = coverage.first_ts or events["glucose"][0].ts
        last = coverage.last_ts or events["glucose"][-1].ts
        window = (first.date(), last.date())

        start = time.monotonic()
        findings = reconciliation_agent.run(_ctx(store, window=window))
        elapsed = time.monotonic() - start

        assert elapsed < 15.0, f"reconciliation on 90d took {elapsed:.1f}s (expected < 15s)"
        # Sanity: the planted effects still surface (the fix preserved behavior).
        assert findings, "expected Tier B findings on the 90-day all-effects scenario"
