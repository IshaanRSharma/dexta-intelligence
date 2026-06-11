"""Tests for the oref0 physiological-math port (`dexta_intelligence.analytics.oref`).

Golden values marked "verified against oref0 JS" were produced by executing the
actual ``lib/iob/calculate.js`` from https://github.com/openaps/oref0 under
node and match this port bit-for-bit.
"""

from __future__ import annotations

import itertools
import math
from datetime import UTC, datetime, timedelta

import pytest

from dexta_intelligence.analytics.oref import (
    bgi,
    bilinear_activity,
    bilinear_iob,
    carb_sensitivity_factor,
    carbs_on_board,
    deviation_series,
    eventual_bg,
    exponential_activity,
    exponential_constants,
    exponential_iob,
    insulin_totals,
    predict_glucose,
    temp_basal_to_microboluses,
)

T0 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)


def mins(m: float) -> timedelta:
    return timedelta(minutes=m)


# ── exponential curve ────────────────────────────────────────────────────────


class TestExponentialCurve:
    def test_golden_values_rapid_acting(self) -> None:
        # Verified against oref0 JS: iobCalc(1U, t=60m, 'rapid-acting', dia=5h, peak=75)
        assert exponential_activity(60, 300, 75) == pytest.approx(0.005987443000582721, rel=1e-12)
        assert exponential_iob(60, 300, 75) == pytest.approx(0.7640057035577161, rel=1e-12)

    def test_golden_values_ultra_rapid(self) -> None:
        # Verified against oref0 JS: iobCalc(1U, t=120m, 'ultra-rapid', dia=6h, peak=55)
        assert exponential_activity(120, 360, 55) == pytest.approx(0.004689418536193994, rel=1e-12)
        assert exponential_iob(120, 360, 55) == pytest.approx(0.3143821335622309, rel=1e-12)

    def test_iob_boundary_conditions(self) -> None:
        for dia, peak in [(300, 75), (300, 55), (360, 75), (480, 120)]:
            assert exponential_iob(0, dia, peak) == pytest.approx(1.0, abs=1e-12)
            # IOB(end) = 0 by construction of the curve family
            assert exponential_iob(dia - 1e-6, dia, peak) == pytest.approx(0.0, abs=1e-9)
            assert exponential_iob(dia, dia, peak) == 0.0
            assert exponential_iob(-1, dia, peak) == 0.0
            assert exponential_activity(dia, dia, peak) == 0.0
            assert exponential_activity(-1, dia, peak) == 0.0

    def test_activity_integrates_to_one(self) -> None:
        for dia, peak in [(300, 75), (300, 55), (360, 100), (600, 120)]:
            total = sum(exponential_activity(t + 0.5, dia, peak) for t in range(dia))
            assert total == pytest.approx(1.0, abs=1e-3)

    def test_peak_activity_at_configured_peak(self) -> None:
        for dia, peak in [(300, 75), (300, 55), (360, 75), (420, 100)]:
            argmax = max(range(dia), key=lambda t, d=dia, p=peak: exponential_activity(t, d, p))
            assert abs(argmax - peak) <= 1

    def test_constants_finite_for_all_legal_combos(self) -> None:
        for peak in (35, 50, 55, 75, 100, 120):
            for dia in (300, 360, 420, 480, 600):
                tau, a, s = exponential_constants(dia, peak)
                assert math.isfinite(tau) and tau > 0
                assert math.isfinite(a) and a > 0
                assert math.isfinite(s) and s > 0
                # the curve must stay valid: IOB pinned to [1, 0] at the ends
                assert exponential_iob(0, dia, peak) == pytest.approx(1.0, abs=1e-12)
                assert exponential_iob(dia - 1e-6, dia, peak) == pytest.approx(0.0, abs=1e-9)

    def test_singular_peak_rejected(self) -> None:
        with pytest.raises(ValueError, match="peak"):
            exponential_constants(300, 150)  # peak == dia/2 → tau singular
        with pytest.raises(ValueError, match="peak"):
            exponential_constants(300, 200)


# ── bilinear curve ───────────────────────────────────────────────────────────


class TestBilinearCurve:
    def test_golden_values(self) -> None:
        # Verified against oref0 JS: iobCalc(1U, t=75m, 'bilinear', dia=3h)
        assert bilinear_iob(75, 180) == pytest.approx(0.55556, rel=1e-12)
        assert bilinear_activity(75, 180) == pytest.approx(0.011111111111111112, rel=1e-12)
        # Verified against oref0 JS: iobCalc(1U, t=100m, 'bilinear', dia=4h)
        # (scaled time = 75 → same IOB, peak activity = 2 / 240)
        assert bilinear_iob(100, 240) == pytest.approx(0.55556, rel=1e-12)
        assert bilinear_activity(100, 240) == pytest.approx(0.008333333333333333, rel=1e-12)

    def test_iob_boundary_conditions(self) -> None:
        for dia in (180, 240, 300):
            assert bilinear_iob(0, dia) == pytest.approx(1.0, abs=1e-12)
            # the empirical quadratics end at ~1e-4, not exactly 0
            assert bilinear_iob(dia - 1e-3, dia) == pytest.approx(0.0, abs=5e-4)
            assert bilinear_iob(dia, dia) == 0.0
            assert bilinear_iob(-1, dia) == 0.0
            assert bilinear_activity(dia, dia) == 0.0
            assert bilinear_activity(-1, dia) == 0.0

    def test_activity_integrates_to_one(self) -> None:
        for dia in (180, 240, 360):
            total = sum(bilinear_activity(t + 0.5, dia) for t in range(dia))
            assert total == pytest.approx(1.0, abs=1e-6)

    def test_peak_activity_position_and_height(self) -> None:
        for dia in (180, 240, 360):
            expected_peak_min = dia * 75 / 180
            argmax = max(range(dia), key=lambda t, d=dia: bilinear_activity(t, d))
            assert abs(argmax - expected_peak_min) <= 1
            assert bilinear_activity(expected_peak_min, dia) == pytest.approx(2 / dia)


# ── totals, superposition, clamping ──────────────────────────────────────────


class TestInsulinTotals:
    def test_superposition_of_two_boluses(self) -> None:
        d1 = [(T0, 1.5)]
        d2 = [(T0 + mins(30), 0.75)]
        at = T0 + mins(90)
        single1 = insulin_totals(d1, at)
        single2 = insulin_totals(d2, at)
        combined = insulin_totals(d1 + d2, at)
        assert combined.iob == pytest.approx(single1.iob + single2.iob, rel=1e-12)
        assert combined.activity_per_min == pytest.approx(
            single1.activity_per_min + single2.activity_per_min, rel=1e-12
        )

    def test_future_doses_ignored(self) -> None:
        totals = insulin_totals([(T0 + mins(10), 1.0)], T0)
        assert totals.iob == 0.0
        assert totals.activity_per_min == 0.0

    def test_bolus_at_query_time_is_full_iob(self) -> None:
        assert insulin_totals([(T0, 2.0)], T0).iob == pytest.approx(2.0)

    def test_negative_deltas_give_negative_iob(self) -> None:
        # a low temp below scheduled basal is net-negative insulin
        totals = insulin_totals([(T0, -0.1)], T0 + mins(30))
        assert totals.iob < 0
        assert totals.activity_per_min < 0

    def test_peak_clamping_rapid_acting(self) -> None:
        doses = [(T0, 1.0)]
        at = T0 + mins(60)
        # rapid-acting custom peaks clamp to [50, 120] per calculate.js
        assert insulin_totals(doses, at, peak_min=30) == insulin_totals(doses, at, peak_min=50)
        assert insulin_totals(doses, at, peak_min=200) == insulin_totals(doses, at, peak_min=120)
        # ultra-rapid clamps to [35, 100]
        assert insulin_totals(doses, at, curve="ultra-rapid", peak_min=20) == insulin_totals(
            doses, at, curve="ultra-rapid", peak_min=35
        )

    def test_dia_floor_for_exponential(self) -> None:
        doses = [(T0, 1.0)]
        at = T0 + mins(60)
        # total.js forces 5h minimum DIA for exponential curves
        assert insulin_totals(doses, at, dia_min=180) == insulin_totals(doses, at, dia_min=300)

    def test_unsupported_curve_rejected(self) -> None:
        with pytest.raises(ValueError, match="unsupported curve"):
            insulin_totals([(T0, 1.0)], T0, curve="walsh")

    def test_bilinear_totals(self) -> None:
        totals = insulin_totals([(T0, 1.0)], T0 + mins(75), curve="bilinear", dia_min=180)
        assert totals.iob == pytest.approx(0.55556)
        assert totals.activity_per_min == pytest.approx(2 / 180)


class TestTempBasalToMicroboluses:
    def test_half_hour_high_temp(self) -> None:
        out = temp_basal_to_microboluses(T0, T0 + mins(30), 2.0, 1.0)
        assert len(out) == 6
        assert all(units == pytest.approx(1.0 * 5 / 60) for _, units in out)
        assert sum(units for _, units in out) == pytest.approx(0.5)
        assert out[0][0] == T0
        assert out[-1][0] == T0 + mins(25)

    def test_partial_final_interval(self) -> None:
        out = temp_basal_to_microboluses(T0, T0 + mins(17), 0.0, 1.2)
        assert len(out) == 4  # 5 + 5 + 5 + 2 minutes
        assert sum(units for _, units in out) == pytest.approx(-1.2 * 17 / 60)
        assert out[-1][1] == pytest.approx(-1.2 * 2 / 60)

    def test_zero_duration(self) -> None:
        assert temp_basal_to_microboluses(T0, T0, 2.0, 1.0) == []


# ── BGI and deviations ───────────────────────────────────────────────────────


class TestBgiAndDeviations:
    def test_bgi_sign_convention(self) -> None:
        # positive insulin activity drives BG down → negative BGI
        assert bgi(0.01, 50.0) == pytest.approx(-2.5)
        assert bgi(0.0, 50.0) == 0.0
        assert bgi(-0.01, 50.0) == pytest.approx(2.5)

    def test_round_trip_zero_deviations(self) -> None:
        """A glucose series generated FROM the model must yield ~zero deviations."""
        isf = 40.0
        doses = [(T0, 2.0), (T0 + mins(45), 1.0)]
        glucose: list[tuple[datetime, float]] = [(T0, 180.0)]
        for i in range(1, 61):
            ts = T0 + mins(5 * i)
            activity = insulin_totals(doses, ts).activity_per_min
            glucose.append((ts, glucose[-1][1] + bgi(activity, isf)))

        devs = deviation_series(glucose, doses, isf)
        assert len(devs) == 60
        for _, dev in devs:
            assert dev == pytest.approx(0.0, abs=1e-9)

    def test_positive_deviation_without_insulin(self) -> None:
        glucose = [(T0, 100.0), (T0 + mins(5), 110.0)]
        devs = deviation_series(glucose, [], 50.0)
        assert devs == [(T0 + mins(5), pytest.approx(10.0))]


# ── carb absorption ──────────────────────────────────────────────────────────


def flat_glucose(start: datetime, n_points: int, value: float = 120.0,
                 slope_per_5m: float = 0.0) -> list[tuple[datetime, float]]:
    return [(start + mins(5 * i), value + slope_per_5m * i) for i in range(n_points)]


class TestCarbsOnBoard:
    ISF = 50.0
    CR = 10.0  # CSF = 5 mg/dL per gram

    def test_csf(self) -> None:
        assert carb_sensitivity_factor(self.ISF, self.CR) == pytest.approx(5.0)

    def test_announced_carbs_fully_decay(self) -> None:
        """With flat BG (zero deviations) the 8 mg/dL/5m floor still absorbs carbs."""
        glucose = flat_glucose(T0, 49)  # 4 hours
        # floor absorbs 8 / 5 = 1.6 g per 5-min window → 50 g gone in ~157 min
        result = carbs_on_board(50.0, T0, glucose, [], self.ISF, self.CR, T0 + mins(240))
        assert result.cob_g == 0.0
        assert result.absorbed_g == pytest.approx(48 * 1.6)

    def test_min_5m_carbimpact_floor_honored(self) -> None:
        """Deviations below the floor still absorb min_5m_carbimpact worth of carbs."""
        glucose = flat_glucose(T0, 7, slope_per_5m=-1.0)  # deviation = -1 each window
        result = carbs_on_board(50.0, T0, glucose, [], self.ISF, self.CR, T0 + mins(30))
        assert result.absorbed_g == pytest.approx(6 * 8.0 / 5.0)
        assert result.cob_g == pytest.approx(50.0 - 9.6)

    def test_positive_deviations_absorb_faster_than_floor(self) -> None:
        glucose = flat_glucose(T0, 7, slope_per_5m=10.0)  # deviation = +10 each window
        result = carbs_on_board(50.0, T0, glucose, [], self.ISF, self.CR, T0 + mins(30))
        assert result.absorbed_g == pytest.approx(6 * 10.0 / 5.0)

    def test_cob_capped_at_max_cob(self) -> None:
        glucose = flat_glucose(T0, 2)
        result = carbs_on_board(200.0, T0, glucose, [], self.ISF, self.CR, T0 + mins(5))
        assert result.cob_g == 120.0  # oref0 maxCOB default

    def test_six_hour_carb_window(self) -> None:
        """Absorption stops counting 6 h after the carb entry (meal/total.js)."""
        glucose = flat_glucose(T0, 85)  # 7 hours
        result = carbs_on_board(200.0, T0, glucose, [], self.ISF, self.CR, T0 + mins(420),
                                max_cob=1000.0)
        assert result.absorbed_g == pytest.approx(72 * 1.6)  # only 6 h of windows


# ── prediction curves ────────────────────────────────────────────────────────


class TestPredictionCurves:
    ISF = 50.0
    BG = 180.0

    def test_bolus_only_iobpred_falls_to_eventual_bg(self) -> None:
        doses = [(T0, 1.0)]
        # horizon 300 = full DIA so the bolus completely decays within the curve
        curves = predict_glucose(self.BG, doses, T0, self.ISF, horizon_min=300.0)
        iob_curve = curves.iob
        assert iob_curve[0] == self.BG
        assert all(b <= a + 1e-12 for a, b in itertools.pairwise(iob_curve))
        assert iob_curve[-1] < iob_curve[0]

        iob_now = insulin_totals(doses, T0).iob
        expected = eventual_bg(self.BG, iob_now, self.ISF)
        assert expected == pytest.approx(self.BG - 50.0)
        assert iob_curve[-1] == pytest.approx(expected, abs=1.0)

    def test_zt_equals_iob_without_deviation_or_future_insulin(self) -> None:
        curves = predict_glucose(self.BG, [(T0 - mins(30), 1.0)], T0, self.ISF)
        assert curves.zt == pytest.approx(curves.iob)

    def test_zt_zero_temp_is_above_iob_curve(self) -> None:
        """Zero-temping (negative future net insulin) means BG falls less."""
        curves = predict_glucose(
            self.BG, [(T0, 1.0)], T0, self.ISF, zt_basal_u_hr=1.0
        )
        assert curves.zt[-1] > curves.iob[-1]

    def test_uam_deviation_decays_over_three_hours(self) -> None:
        """With no insulin, UAM adds uci*(1 - n/36) per step; closed-form sum = 17.5*uci."""
        uci = 6.0
        curves = predict_glucose(120.0, [], T0, self.ISF, deviation_5m=uci)
        assert curves.uam[-1] == pytest.approx(120.0 + uci * 17.5)
        # IOBpredBG's deviation term decays over 60 min instead: sum = 5.5*uci
        assert curves.iob[-1] == pytest.approx(120.0 + uci * 5.5)
        # ZT ignores deviations entirely
        assert curves.zt[-1] == pytest.approx(120.0)

    def test_cob_curve_conserves_carb_impact(self) -> None:
        """Total COB-curve rise ≈ COB * CSF (all carbs eventually hit BG)."""
        csf = carb_sensitivity_factor(self.ISF, 10.0)
        cob_g = 50.0
        curves = predict_glucose(
            120.0, [], T0, self.ISF,
            carb_ratio=10.0, cob_g=cob_g, deviation_5m=5.0,
        )
        assert curves.cob[-1] > curves.iob[-1]
        assert curves.cob[-1] == pytest.approx(120.0 + cob_g * csf, abs=10.0)

    def test_cob_requires_carb_ratio(self) -> None:
        with pytest.raises(ValueError, match="carb_ratio"):
            predict_glucose(120.0, [], T0, self.ISF, cob_g=20.0)

    def test_default_horizon_and_step_count(self) -> None:
        curves = predict_glucose(120.0, [], T0, self.ISF)
        assert len(curves.iob) == 49  # starting BG + 48 five-minute steps = 240 min
        assert len(curves.zt) == len(curves.cob) == len(curves.uam) == 49

    def test_eventual_bg_arithmetic(self) -> None:
        assert eventual_bg(120.0, 2.0, 50.0) == pytest.approx(20.0)
        assert eventual_bg(120.0, 2.0, 50.0, remaining_carb_impact=30.0) == pytest.approx(50.0)
        assert eventual_bg(120.0, -0.5, 40.0) == pytest.approx(140.0)
