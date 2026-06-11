"""Tests for the pure daily-rollup computation."""

from __future__ import annotations

import statistics
from datetime import UTC, date, datetime, timedelta

import pytest

from dexta_intelligence.analytics.rollups import (
    EXPECTED_READINGS_PER_DAY,
    coverage_fraction,
    daily_rollup,
)
from dexta_intelligence.models import (
    GlucoseEvent,
    InsulinEvent,
    InsulinKind,
    MealEvent,
    RollupPeriod,
)
from dexta_intelligence.testing.synthetic import generate_baseline

DAY = date(2025, 3, 1)
DAY_START = datetime(2025, 3, 1, tzinfo=UTC)


def _readings(values: list[int], start: datetime = DAY_START) -> list[GlucoseEvent]:
    return [
        GlucoseEvent(ts=start + timedelta(minutes=5 * i), mg_dl=v)
        for i, v in enumerate(values)
    ]


# Hand-built day with known band membership:
#   <54: 50 | 54-69: 60 | in [70, 180]: 70, 100, 120, 140, 160, 180
#   >180: 200 | >250: 260
KNOWN_VALUES = [50, 60, 100, 120, 140, 160, 180, 200, 260, 70]


class TestKnownDay:
    def test_band_percentages(self) -> None:
        rollup = daily_rollup(DAY, _readings(KNOWN_VALUES))
        assert rollup is not None
        assert rollup.n == 10
        assert rollup.tbr2 == pytest.approx(10.0)  # 50
        assert rollup.tbr == pytest.approx(20.0)  # 50, 60
        assert rollup.tir == pytest.approx(60.0)
        assert rollup.tar == pytest.approx(20.0)  # 200, 260
        assert rollup.tar2 == pytest.approx(10.0)  # 260
        assert rollup.tir + rollup.tar + rollup.tbr == pytest.approx(100.0)

    def test_mean_sd_cv_gmi(self) -> None:
        rollup = daily_rollup(DAY, _readings(KNOWN_VALUES))
        assert rollup is not None
        expected_mean = statistics.fmean(KNOWN_VALUES)
        expected_sd = statistics.stdev(KNOWN_VALUES)
        assert rollup.mean == pytest.approx(expected_mean)
        assert rollup.sd == pytest.approx(expected_sd)
        assert rollup.cv == pytest.approx(expected_sd / expected_mean * 100.0)
        assert rollup.gmi == pytest.approx(3.31 + 0.02392 * expected_mean)

    def test_period_fields(self) -> None:
        rollup = daily_rollup(DAY, _readings(KNOWN_VALUES))
        assert rollup is not None
        assert rollup.period is RollupPeriod.DAILY
        assert rollup.period_start == DAY_START
        assert rollup.period_start.tzinfo is not None

    def test_excursions(self) -> None:
        # Time order: [50, 60] low run, then in-range, then [200, 260] high
        # run, then back in range -> exactly 2 excursions.
        rollup = daily_rollup(DAY, _readings(KNOWN_VALUES))
        assert rollup is not None
        assert rollup.excursion_count == 2

    def test_insulin_and_meal_totals(self) -> None:
        insulin = [
            InsulinEvent(ts=DAY_START + timedelta(hours=8), kind=InsulinKind.BOLUS, units=3.5),
            InsulinEvent(
                ts=DAY_START + timedelta(hours=12),
                kind=InsulinKind.BOLUS,
                units=2.0,
                automatic=True,
            ),
            InsulinEvent(ts=DAY_START + timedelta(hours=22), kind=InsulinKind.BASAL, units=18.0),
            InsulinEvent(ts=DAY_START + timedelta(hours=2), kind=InsulinKind.SUSPEND),
        ]
        meals = [
            MealEvent(ts=DAY_START + timedelta(hours=8), carbs_g=45.0),
            MealEvent(ts=DAY_START + timedelta(hours=12), carbs_g=30.0),
            MealEvent(ts=DAY_START + timedelta(hours=19), note="no carbs logged"),
        ]
        rollup = daily_rollup(DAY, _readings(KNOWN_VALUES), insulin=insulin, meals=meals)
        assert rollup is not None
        assert rollup.bolus_units == pytest.approx(5.5)
        assert rollup.basal_units == pytest.approx(18.0)
        assert rollup.carbs_g == pytest.approx(75.0)

    def test_absent_insulin_and_meals_are_none_not_zero(self) -> None:
        rollup = daily_rollup(DAY, _readings(KNOWN_VALUES))
        assert rollup is not None
        assert rollup.bolus_units is None
        assert rollup.basal_units is None
        assert rollup.carbs_g is None


class TestEdgeCases:
    def test_empty_day_returns_none(self) -> None:
        assert daily_rollup(DAY, []) is None

    def test_events_outside_day_are_filtered(self) -> None:
        next_day = _readings([400], start=DAY_START + timedelta(days=1))
        assert daily_rollup(DAY, next_day) is None

        mixed = _readings(KNOWN_VALUES) + next_day
        rollup = daily_rollup(DAY, mixed)
        assert rollup is not None
        assert rollup.n == 10
        assert rollup.tar2 == pytest.approx(10.0)  # the 400 next-day reading excluded

    def test_single_reading_has_no_sd_or_cv(self) -> None:
        rollup = daily_rollup(DAY, _readings([120]))
        assert rollup is not None
        assert rollup.n == 1
        assert rollup.mean == pytest.approx(120.0)
        assert rollup.sd is None
        assert rollup.cv is None
        assert rollup.tir == pytest.approx(100.0)

    def test_custom_target_band(self) -> None:
        rollup = daily_rollup(DAY, _readings([100, 150]), target_low=70, target_high=140)
        assert rollup is not None
        assert rollup.tir == pytest.approx(50.0)
        assert rollup.tar == pytest.approx(50.0)


class TestCoverage:
    def test_full_day(self) -> None:
        assert coverage_fraction(EXPECTED_READINGS_PER_DAY) == pytest.approx(1.0)

    def test_sparse_day(self) -> None:
        assert coverage_fraction(144) == pytest.approx(0.5)
        assert coverage_fraction(72) == pytest.approx(0.25)

    def test_empty_day(self) -> None:
        assert coverage_fraction(0) == 0.0

    def test_overfull_day_clamped(self) -> None:
        assert coverage_fraction(EXPECTED_READINGS_PER_DAY + 50) == pytest.approx(1.0)


class TestDeterminism:
    def test_same_inputs_same_rollup(self) -> None:
        events = generate_baseline(seed=42, n_days=2)
        glucose = events["glucose"]
        day = glucose[0].ts.date()
        first = daily_rollup(day, glucose, insulin=events["insulin"], meals=events["meal"])
        second = daily_rollup(day, glucose, insulin=events["insulin"], meals=events["meal"])
        assert first is not None
        assert first == second

    def test_synthetic_full_day_coverage(self) -> None:
        glucose = generate_baseline(seed=7, n_days=2)["glucose"]
        day = glucose[0].ts.date()
        rollup = daily_rollup(day, glucose)
        assert rollup is not None
        assert rollup.n == EXPECTED_READINGS_PER_DAY
        assert coverage_fraction(rollup.n) == pytest.approx(1.0)
