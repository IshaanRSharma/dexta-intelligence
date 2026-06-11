"""Known-answer and edge-case tests for the deterministic statistics core.

Reference values come from three sources, noted per test:
- hand computation (exact closed forms, shown in comments),
- published examples (Anscombe's quartet, Wikipedia worked examples),
- exact distribution identities (Cauchy/t closed forms, normal erfc).

The edge-case sections pin the hardened contract: primitives raise on
degenerate input, analyses return None, and nothing ever emits NaN or inf.
"""

from __future__ import annotations

import math
from typing import ClassVar

import pytest

from dexta_intelligence.stats.core import (
    BootstrapCI,
    MannWhitneyResult,
    SummaryStats,
    WelchTTestResult,
    bootstrap_diff_ci,
    bootstrap_mean_ci,
    cliffs_delta,
    cohen_d,
    confidence_from_n_and_d,
    hedges_g,
    mann_whitney_u,
    mean,
    median,
    normal_two_sided_p,
    pearson_r,
    spearman_rho,
    stdev,
    strength_from_effect,
    student_t_two_sided_p,
    summarize,
    welch_t_test,
    zscores,
)

# ─────────────────────────────────────────────────────────────────────────────
# Descriptive primitives
# ─────────────────────────────────────────────────────────────────────────────


class TestMean:
    def test_known_answer(self) -> None:
        assert mean([1.0, 2.0, 3.0, 4.0]) == pytest.approx(2.5)

    def test_single_element(self) -> None:
        assert mean([7.0]) == 7.0

    def test_negative_values(self) -> None:
        assert mean([-2.0, 2.0]) == 0.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            mean([])


class TestMedian:
    def test_odd_n(self) -> None:
        assert median([3.0, 1.0, 2.0]) == 2.0

    def test_even_n_interpolates(self) -> None:
        assert median([1.0, 2.0, 3.0, 4.0]) == pytest.approx(2.5)

    def test_single_element(self) -> None:
        assert median([5.0]) == 5.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            median([])


class TestStdev:
    def test_known_answer(self) -> None:
        # [2, 4, 4, 4, 5, 5, 7, 9]: mean 5, sum of squares 32, sample var 32/7.
        xs = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        assert stdev(xs) == pytest.approx(math.sqrt(32.0 / 7.0))

    def test_constant_series_is_zero(self) -> None:
        assert stdev([3.0, 3.0, 3.0]) == 0.0

    def test_single_element_raises(self) -> None:
        with pytest.raises(ValueError):
            stdev([1.0])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            stdev([])


class TestSummarize:
    def test_full_series(self) -> None:
        s = summarize([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert isinstance(s, SummaryStats)
        assert s.n == 8
        assert s.mean == pytest.approx(5.0)
        assert s.sd == pytest.approx(math.sqrt(32.0 / 7.0))
        assert s.cv_pct == pytest.approx(math.sqrt(32.0 / 7.0) / 5.0 * 100.0)
        assert s.minimum == 2.0
        assert s.maximum == 9.0
        assert s.median == pytest.approx(4.5)

    def test_empty_is_all_none(self) -> None:
        s = summarize([])
        assert s.n == 0
        assert s.mean is None
        assert s.sd is None
        assert s.cv_pct is None
        assert s.minimum is None
        assert s.maximum is None
        assert s.median is None

    def test_single_element(self) -> None:
        s = summarize([42.0])
        assert s.n == 1
        assert s.mean == 42.0
        assert s.sd is None  # no spread from one observation
        assert s.cv_pct is None
        assert s.minimum == 42.0
        assert s.maximum == 42.0
        assert s.median == 42.0

    def test_zero_mean_has_no_cv(self) -> None:
        s = summarize([-1.0, 1.0])
        assert s.mean == 0.0
        assert s.sd is not None
        assert s.cv_pct is None  # CV undefined at mean 0


class TestZscores:
    def test_known_answer(self) -> None:
        # [1, 2, 3]: mean 2, sample sd 1 → z = [-1, 0, 1].
        assert zscores([1.0, 2.0, 3.0]) == pytest.approx([-1.0, 0.0, 1.0])

    def test_constant_series_is_none(self) -> None:
        # Donor patched sd=0 with 1.0, fabricating z-scores. We refuse.
        assert zscores([5.0, 5.0, 5.0]) is None

    def test_single_element_is_none(self) -> None:
        assert zscores([1.0]) is None

    def test_empty_is_none(self) -> None:
        assert zscores([]) is None


# ─────────────────────────────────────────────────────────────────────────────
# Correlation
# ─────────────────────────────────────────────────────────────────────────────


class TestPearson:
    def test_perfect_positive(self) -> None:
        assert pearson_r([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)

    def test_perfect_negative(self) -> None:
        assert pearson_r([1.0, 2.0, 3.0], [6.0, 4.0, 2.0]) == pytest.approx(-1.0)

    def test_hand_computed(self) -> None:
        # x=[1..5], y=[2,1,4,3,5]: cov num = 8, dx = dy = sqrt(10) → r = 0.8.
        assert pearson_r([1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 1.0, 4.0, 3.0, 5.0]) == pytest.approx(
            0.8
        )

    def test_anscombe_quartet_i(self) -> None:
        # Published reference: Anscombe (1973) dataset I, r = 0.816 (3 d.p.).
        x = [10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0]
        y = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
        assert pearson_r(x, y) == pytest.approx(0.81642, abs=1e-5)

    def test_constant_series_is_none(self) -> None:
        assert pearson_r([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
        assert pearson_r([1.0, 2.0, 3.0], [4.0, 4.0, 4.0]) is None

    def test_too_short_is_none(self) -> None:
        assert pearson_r([1.0], [2.0]) is None
        assert pearson_r([], []) is None

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            pearson_r([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_clamped_to_unit_interval(self) -> None:
        r = pearson_r([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0])
        assert r is not None
        assert -1.0 <= r <= 1.0


class TestSpearman:
    def test_monotonic_nonlinear_is_one(self) -> None:
        # Exponential growth is far from linear but perfectly monotonic.
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [math.exp(v) for v in x]
        assert spearman_rho(x, y) == pytest.approx(1.0)

    def test_wikipedia_iq_tv_example(self) -> None:
        # Published reference (Wikipedia, Spearman's rho worked example):
        # rho = -29/165 ≈ -0.1758, no ties.
        iq = [106.0, 86.0, 100.0, 101.0, 99.0, 103.0, 97.0, 113.0, 112.0, 110.0]
        tv = [7.0, 0.0, 27.0, 50.0, 28.0, 29.0, 20.0, 12.0, 6.0, 17.0]
        assert spearman_rho(iq, tv) == pytest.approx(-29.0 / 165.0)

    def test_hand_computed_with_ties(self) -> None:
        # y ranks with average-tie handling: [1, 2, 3.5, 5, 3.5];
        # Pearson on ranks = 8 / sqrt(10 * 9.5) = 0.820783 (6 d.p.).
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 6.0, 7.0, 8.0, 7.0]
        assert spearman_rho(x, y) == pytest.approx(8.0 / math.sqrt(95.0))

    def test_constant_series_is_none(self) -> None:
        assert spearman_rho([2.0, 2.0, 2.0], [1.0, 2.0, 3.0]) is None

    def test_too_short_is_none(self) -> None:
        assert spearman_rho([1.0], [2.0]) is None
        assert spearman_rho([], []) is None

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            spearman_rho([1.0], [1.0, 2.0])


# ─────────────────────────────────────────────────────────────────────────────
# Effect sizes
# ─────────────────────────────────────────────────────────────────────────────


class TestCohenD:
    def test_hand_computed(self) -> None:
        # a=[1..4], b=[3..6]: mean diff -2, pooled var 5/3 → d = -2/sqrt(5/3).
        a = [1.0, 2.0, 3.0, 4.0]
        b = [3.0, 4.0, 5.0, 6.0]
        assert cohen_d(a, b) == pytest.approx(-2.0 / math.sqrt(5.0 / 3.0))

    def test_symmetry(self) -> None:
        a = [1.0, 2.0, 3.0, 4.0]
        b = [3.0, 4.0, 5.0, 6.0]
        d_ab = cohen_d(a, b)
        d_ba = cohen_d(b, a)
        assert d_ab is not None and d_ba is not None
        assert d_ab == pytest.approx(-d_ba)

    def test_identical_groups_is_zero(self) -> None:
        xs = [1.0, 2.0, 3.0]
        assert cohen_d(xs, xs) == pytest.approx(0.0)

    def test_both_constant_is_none(self) -> None:
        # Pooled sd = 0: the standardized difference is undefined, not 0.
        assert cohen_d([5.0, 5.0], [3.0, 3.0]) is None

    def test_small_groups_are_none(self) -> None:
        assert cohen_d([1.0], [2.0, 3.0]) is None
        assert cohen_d([1.0, 2.0], [3.0]) is None
        assert cohen_d([], []) is None


class TestHedgesG:
    def test_shrinks_cohen_d(self) -> None:
        # df = 6: J = 1 - 3/23 = 20/23 exactly.
        a = [1.0, 2.0, 3.0, 4.0]
        b = [3.0, 4.0, 5.0, 6.0]
        d = cohen_d(a, b)
        g = hedges_g(a, b)
        assert d is not None and g is not None
        assert g == pytest.approx(d * 20.0 / 23.0)
        assert abs(g) < abs(d)

    def test_degenerate_is_none(self) -> None:
        assert hedges_g([1.0], [2.0, 3.0]) is None
        assert hedges_g([5.0, 5.0], [3.0, 3.0]) is None


class TestCliffsDelta:
    def test_complete_separation(self) -> None:
        assert cliffs_delta([4.0, 5.0, 6.0], [1.0, 2.0, 3.0]) == 1.0
        assert cliffs_delta([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == -1.0

    def test_hand_computed_with_ties(self) -> None:
        # a=[1,2,3] vs b=[2,3,4]: 1 win, 6 losses, 2 ties over 9 pairs.
        assert cliffs_delta([1.0, 2.0, 3.0], [2.0, 3.0, 4.0]) == pytest.approx(-5.0 / 9.0)

    def test_identical_groups_is_zero(self) -> None:
        assert cliffs_delta([1.0, 2.0], [1.0, 2.0]) == 0.0

    def test_empty_is_none(self) -> None:
        assert cliffs_delta([], [1.0]) is None
        assert cliffs_delta([1.0], []) is None

    def test_single_elements_work(self) -> None:
        # Distribution-free: defined even for n=1 per group.
        assert cliffs_delta([2.0], [1.0]) == 1.0

    def test_bounded(self) -> None:
        d = cliffs_delta([1.0, 5.0, 2.0, 8.0], [3.0, 3.0, 6.0])
        assert d is not None
        assert -1.0 <= d <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Group comparison tests
# ─────────────────────────────────────────────────────────────────────────────


class TestWelchTTest:
    def test_hand_computed_symmetric_case(self) -> None:
        # Equal n and equal variance: a=[1..5], b=[2..6].
        # var = 2.5 each, SE = sqrt(0.5+0.5) = 1 → t = -1; Welch df = 8.
        # P(|T_8| >= 1) = 0.34659 (R: 2*pt(-1, 8)).
        res = welch_t_test([1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0])
        assert isinstance(res, WelchTTestResult)
        assert res.t == pytest.approx(-1.0)
        assert res.df == pytest.approx(8.0)
        assert res.p_two_sided == pytest.approx(0.34659, abs=1e-4)
        assert res.mean_a == pytest.approx(3.0)
        assert res.mean_b == pytest.approx(4.0)
        assert (res.n_a, res.n_b) == (5, 5)

    def test_unequal_variance_unequal_n(self) -> None:
        # Reference values verified independently of core.py: t and df from a
        # direct evaluation of the Welch formulas, p by trapezoidal numerical
        # integration of the t density (2M steps): t = -2.707778,
        # df = 26.9527, p = 0.011616.
        a = [
            27.5, 21.0, 19.0, 23.6, 17.0, 17.9, 16.9, 20.1,
            21.9, 22.6, 23.1, 19.6, 19.0, 21.7, 21.4,
        ]
        b = [
            27.1, 22.0, 20.8, 23.4, 23.4, 23.5, 25.8,
            22.0, 24.8, 20.2, 21.9, 22.1, 22.9, 30.5,
        ]
        res = welch_t_test(a, b)
        assert res is not None
        assert res.t == pytest.approx(-2.707778, abs=1e-6)
        assert res.df == pytest.approx(26.9527, abs=1e-4)
        assert res.p_two_sided == pytest.approx(0.011616, abs=1e-6)

    def test_no_difference_gives_p_one(self) -> None:
        xs = [1.0, 2.0, 3.0, 4.0]
        res = welch_t_test(xs, xs)
        assert res is not None
        assert res.t == 0.0
        assert res.p_two_sided == pytest.approx(1.0)

    def test_both_constant_is_none(self) -> None:
        assert welch_t_test([2.0, 2.0, 2.0], [5.0, 5.0]) is None

    def test_one_constant_group_still_works(self) -> None:
        # Welch only needs total SE > 0; one group may be constant.
        res = welch_t_test([2.0, 2.0, 2.0], [4.0, 5.0, 6.0])
        assert res is not None
        assert math.isfinite(res.t)
        assert 0.0 <= res.p_two_sided <= 1.0

    def test_small_groups_are_none(self) -> None:
        assert welch_t_test([1.0], [2.0, 3.0]) is None
        assert welch_t_test([], [2.0, 3.0]) is None


class TestMannWhitney:
    def test_complete_separation(self) -> None:
        # a=[1,2,3] vs b=[4,5,6]: U = 0, mean U = 4.5, var = 5.25,
        # z = -(4.5-0.5)/sqrt(5.25) = -1.7457, p = 0.08086
        # (matches scipy.stats.mannwhitneyu, method="asymptotic").
        res = mann_whitney_u([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        assert isinstance(res, MannWhitneyResult)
        assert res.u == 0.0
        assert res.z == pytest.approx(-4.0 / math.sqrt(5.25))
        assert res.p_two_sided == pytest.approx(0.08086, abs=1e-4)
        assert res.rank_biserial == pytest.approx(-1.0)

    def test_rank_biserial_equals_cliffs_delta(self) -> None:
        # Algebraic identity 2U/(n_a*n_b) - 1 == Cliff's delta, ties included.
        a = [1.0, 3.0, 3.0, 7.0, 9.0]
        b = [2.0, 3.0, 5.0, 8.0]
        res = mann_whitney_u(a, b)
        delta = cliffs_delta(a, b)
        assert res is not None and delta is not None
        assert res.rank_biserial == pytest.approx(delta)

    def test_symmetric_input_is_centered(self) -> None:
        xs = [1.0, 2.0, 3.0, 4.0]
        res = mann_whitney_u(xs, xs)
        assert res is not None
        assert res.u == pytest.approx(8.0)  # mean U = n_a*n_b/2
        assert res.z == 0.0
        assert res.p_two_sided == pytest.approx(1.0)
        assert res.rank_biserial == pytest.approx(0.0)

    def test_all_values_identical_is_none(self) -> None:
        # Tie correction kills the variance: nothing to rank.
        assert mann_whitney_u([3.0, 3.0], [3.0, 3.0, 3.0]) is None

    def test_empty_is_none(self) -> None:
        assert mann_whitney_u([], [1.0]) is None
        assert mann_whitney_u([1.0], []) is None


# ─────────────────────────────────────────────────────────────────────────────
# Distribution tails
# ─────────────────────────────────────────────────────────────────────────────


class TestNormalP:
    def test_z_zero(self) -> None:
        assert normal_two_sided_p(0.0) == pytest.approx(1.0)

    def test_z_196(self) -> None:
        # The canonical 5% two-sided critical value.
        assert normal_two_sided_p(1.96) == pytest.approx(0.0499958, abs=1e-6)

    def test_sign_invariant(self) -> None:
        assert normal_two_sided_p(-2.5) == normal_two_sided_p(2.5)

    def test_nonfinite_raises(self) -> None:
        with pytest.raises(ValueError):
            normal_two_sided_p(math.nan)


class TestStudentTP:
    def test_cauchy_exact(self) -> None:
        # df=1 is Cauchy: P(|T| >= 1) = 2*(1 - 3/4) = 0.5 exactly.
        assert student_t_two_sided_p(1.0, 1.0) == pytest.approx(0.5)

    def test_df2_exact(self) -> None:
        # df=2 closed form: CDF(t) = 1/2 + t / (2*sqrt(2 + t^2)).
        # P(|T| >= 1) = 1 - 1/sqrt(3) = 0.42265.
        assert student_t_two_sided_p(1.0, 2.0) == pytest.approx(1.0 - 1.0 / math.sqrt(3.0))

    def test_t_zero_is_one(self) -> None:
        assert student_t_two_sided_p(0.0, 10.0) == 1.0

    def test_converges_to_normal(self) -> None:
        assert student_t_two_sided_p(1.96, 1e6) == pytest.approx(
            normal_two_sided_p(1.96), abs=1e-5
        )

    def test_sign_invariant(self) -> None:
        assert student_t_two_sided_p(-2.0, 7.0) == pytest.approx(student_t_two_sided_p(2.0, 7.0))

    def test_extreme_t_is_tiny_but_valid(self) -> None:
        p = student_t_two_sided_p(50.0, 10.0)
        assert 0.0 <= p < 1e-9

    def test_bad_df_raises(self) -> None:
        with pytest.raises(ValueError):
            student_t_two_sided_p(1.0, 0.0)
        with pytest.raises(ValueError):
            student_t_two_sided_p(1.0, -3.0)


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap confidence intervals
# ─────────────────────────────────────────────────────────────────────────────


class TestBootstrapMeanCI:
    def test_basic_properties(self) -> None:
        xs = [4.0, 8.0, 6.0, 5.0, 3.0, 7.0, 9.0, 5.0, 6.0, 4.0]
        ci = bootstrap_mean_ci(xs, seed=42)
        assert isinstance(ci, BootstrapCI)
        assert ci.point == pytest.approx(5.7)
        assert ci.lower <= ci.point <= ci.upper
        assert min(xs) <= ci.lower and ci.upper <= max(xs)
        assert ci.confidence == 0.95
        assert ci.n_resamples == 2000

    def test_deterministic_for_fixed_seed(self) -> None:
        xs = [1.0, 5.0, 3.0, 8.0, 2.0]
        a = bootstrap_mean_ci(xs, seed=7)
        b = bootstrap_mean_ci(xs, seed=7)
        assert a == b

    def test_different_seeds_differ(self) -> None:
        # Irregular values so resample means rarely collide; with few
        # round-numbered points the percentile bounds can coincide by chance.
        xs = [1.31, 5.77, 3.14, 8.92, 2.23, 9.41, 4.86, 6.18, 0.73, 7.39]
        a = bootstrap_mean_ci(xs, seed=1)
        b = bootstrap_mean_ci(xs, seed=2)
        assert a is not None and b is not None
        assert (a.lower, a.upper) != (b.lower, b.upper)

    def test_constant_series_collapses(self) -> None:
        ci = bootstrap_mean_ci([5.0, 5.0, 5.0], seed=0)
        assert ci is not None
        assert ci.lower == ci.point == ci.upper == 5.0

    def test_single_element(self) -> None:
        ci = bootstrap_mean_ci([3.5], seed=0)
        assert ci is not None
        assert ci.lower == ci.point == ci.upper == 3.5

    def test_empty_is_none(self) -> None:
        assert bootstrap_mean_ci([], seed=0) is None

    def test_wider_at_higher_confidence(self) -> None:
        xs = [1.0, 9.0, 4.0, 7.0, 2.0, 8.0, 3.0, 6.0]
        narrow = bootstrap_mean_ci(xs, confidence=0.80, seed=3)
        wide = bootstrap_mean_ci(xs, confidence=0.99, seed=3)
        assert narrow is not None and wide is not None
        assert (wide.upper - wide.lower) > (narrow.upper - narrow.lower)

    def test_bad_args_raise(self) -> None:
        with pytest.raises(ValueError):
            bootstrap_mean_ci([1.0, 2.0], confidence=1.0)
        with pytest.raises(ValueError):
            bootstrap_mean_ci([1.0, 2.0], confidence=0.0)
        with pytest.raises(ValueError):
            bootstrap_mean_ci([1.0, 2.0], n_resamples=0)


class TestBootstrapDiffCI:
    def test_clear_separation_excludes_zero(self) -> None:
        a = [10.0, 11.0, 12.0, 10.5, 11.5, 12.5, 10.2, 11.8]
        b = [1.0, 2.0, 1.5, 2.5, 1.2, 2.2, 1.8, 0.8]
        ci = bootstrap_diff_ci(a, b, seed=11)
        assert ci is not None
        assert ci.point == pytest.approx(mean(a) - mean(b))
        assert ci.lower > 0.0  # no overlap → CI excludes zero

    def test_identical_groups_straddle_zero(self) -> None:
        xs = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 2.5, 4.5]
        ci = bootstrap_diff_ci(xs, xs, seed=11)
        assert ci is not None
        assert ci.point == pytest.approx(0.0)
        assert ci.lower < 0.0 < ci.upper

    def test_deterministic_for_fixed_seed(self) -> None:
        a = [1.0, 2.0, 3.0]
        b = [2.0, 3.0, 4.0]
        assert bootstrap_diff_ci(a, b, seed=5) == bootstrap_diff_ci(a, b, seed=5)

    def test_empty_group_is_none(self) -> None:
        assert bootstrap_diff_ci([], [1.0], seed=0) is None
        assert bootstrap_diff_ci([1.0], [], seed=0) is None

    def test_bad_args_raise(self) -> None:
        with pytest.raises(ValueError):
            bootstrap_diff_ci([1.0], [2.0], confidence=2.0)
        with pytest.raises(ValueError):
            bootstrap_diff_ci([1.0], [2.0], n_resamples=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Donor heuristics
# ─────────────────────────────────────────────────────────────────────────────


class TestConfidenceFromNAndD:
    def test_known_answers(self) -> None:
        # sqrt((8/30) * (0.3/0.8)) = sqrt(0.1) = 0.316 etc. — hand-computed
        # (the donor docstring's own examples were wrong; these are right).
        assert confidence_from_n_and_d(8, 0.3) == pytest.approx(0.316)
        assert confidence_from_n_and_d(20, 0.5) == pytest.approx(0.645)
        assert confidence_from_n_and_d(40, 0.8) == pytest.approx(1.0)

    def test_saturates(self) -> None:
        assert confidence_from_n_and_d(1000, 10.0) == 1.0

    def test_zero_inputs(self) -> None:
        assert confidence_from_n_and_d(0, 0.5) == 0.0
        assert confidence_from_n_and_d(10, 0.0) == 0.0

    def test_negative_n_clamped(self) -> None:
        assert confidence_from_n_and_d(-5, 0.5) == 0.0

    def test_sign_of_d_irrelevant(self) -> None:
        assert confidence_from_n_and_d(15, -0.6) == confidence_from_n_and_d(15, 0.6)

    def test_bounded(self) -> None:
        for n in (0, 1, 8, 30, 100):
            for d in (-2.0, -0.5, 0.0, 0.3, 5.0):
                c = confidence_from_n_and_d(n, d)
                assert 0.0 <= c <= 1.0


class TestStrengthFromEffect:
    def test_known_answers(self) -> None:
        # 1 - e^-1 = 0.632 at value == scale; 1 - e^-2 = 0.865 at 2x scale.
        assert strength_from_effect(10.0, scale=10.0) == pytest.approx(0.632)
        assert strength_from_effect(20.0, scale=10.0) == pytest.approx(0.865)

    def test_zero_effect(self) -> None:
        assert strength_from_effect(0.0, scale=10.0) == 0.0

    def test_sign_invariant(self) -> None:
        assert strength_from_effect(-7.0, scale=10.0) == strength_from_effect(7.0, scale=10.0)

    def test_degenerate_scale(self) -> None:
        assert strength_from_effect(5.0, scale=0.0) == 0.0
        assert strength_from_effect(5.0, scale=-1.0) == 0.0

    def test_bounded(self) -> None:
        for v in (0.0, 0.1, 5.0, 100.0, -100.0):
            assert 0.0 <= strength_from_effect(v, scale=15.0) <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# NaN-free guarantee
# ─────────────────────────────────────────────────────────────────────────────


def _assert_all_finite(obj: object) -> None:
    """Walk any result object and assert every float in it is finite."""
    if obj is None:
        return
    if isinstance(obj, float):
        assert math.isfinite(obj), f"non-finite value {obj!r}"
        return
    if isinstance(obj, (list, tuple)):
        for item in obj:
            _assert_all_finite(item)
        return
    if hasattr(obj, "__dataclass_fields__"):
        for name in obj.__dataclass_fields__:
            _assert_all_finite(getattr(obj, name))


class TestNaNFreeGuarantee:
    """Adversarial inputs (constant, tiny, huge, near-collinear) never leak
    NaN or infinity — undefined results are always spelled None."""

    CASES: ClassVar[list[list[float]]] = [
        [],
        [0.0],
        [1.0],
        [5.0, 5.0],
        [5.0, 5.0, 5.0, 5.0],
        [1e-12, 2e-12, 3e-12],
        [1e75, 2e75],  # large magnitudes; intermediate squares must stay finite
        [-1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 2.0],
    ]

    def test_unary_functions(self) -> None:
        for xs in self.CASES:
            _assert_all_finite(summarize(xs))
            _assert_all_finite(zscores(xs))
            _assert_all_finite(bootstrap_mean_ci(xs, n_resamples=50, seed=0))

    def test_binary_functions(self) -> None:
        for a in self.CASES:
            for b in self.CASES:
                _assert_all_finite(cliffs_delta(a, b))
                _assert_all_finite(cohen_d(a, b))
                _assert_all_finite(hedges_g(a, b))
                _assert_all_finite(welch_t_test(a, b))
                _assert_all_finite(mann_whitney_u(a, b))
                _assert_all_finite(bootstrap_diff_ci(a, b, n_resamples=50, seed=0))
                if len(a) == len(b):
                    _assert_all_finite(pearson_r(a, b))
                    _assert_all_finite(spearman_rho(a, b))

    def test_p_values_always_in_unit_interval(self) -> None:
        for t in (-100.0, -3.2, 0.0, 0.001, 7.5, 500.0):
            for df in (0.5, 1.0, 2.0, 8.0, 24.9, 1e6):
                assert 0.0 <= student_t_two_sided_p(t, df) <= 1.0
        for z in (-50.0, -1.96, 0.0, 3.0, 50.0):
            assert 0.0 <= normal_two_sided_p(z) <= 1.0
