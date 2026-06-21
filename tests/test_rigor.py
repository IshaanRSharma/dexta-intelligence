"""Tests for the statistical rigor layer (``stats/rigor.py``).

Two tiers:

- Unit tests: exact behavior against worked examples and constructed data.
- Calibration tests (``@pytest.mark.calibration``): do the procedures hold
  their advertised error rates on synthetic data? Everything is seeded.
"""

from __future__ import annotations

import random

import pytest

from dexta_intelligence.stats.rigor import (
    assess,
    benjamini_hochberg,
    mean_difference,
    permutation_pvalue,
    power_gate,
    split_half_replication,
)

# ─────────────────────────────────────────────────────────────────────────────
# Benjamini-Hochberg
# ─────────────────────────────────────────────────────────────────────────────

#: The 15 p-values from Benjamini & Hochberg (1995), section 4 (data of
#: Neuhaus et al. 1992). At alpha = 0.05 the step-up procedure rejects
#: exactly the four smallest.
BH_1995_PVALUES = [
    0.0001, 0.0004, 0.0019, 0.0095, 0.0201,
    0.0278, 0.0298, 0.0344, 0.0459, 0.3240,
    0.4262, 0.5719, 0.6528, 0.7590, 1.0000,
]


def test_bh_published_example_rejections() -> None:
    result = benjamini_hochberg(BH_1995_PVALUES, alpha=0.05)
    assert result.reject == (True,) * 4 + (False,) * 11


def test_bh_published_example_qvalues() -> None:
    # Reference q-values computed by R's p.adjust(p, method = "BH").
    result = benjamini_hochberg(BH_1995_PVALUES, alpha=0.05)
    assert result.qvalues[0] == pytest.approx(0.0015)
    assert result.qvalues[1] == pytest.approx(0.0030)
    assert result.qvalues[2] == pytest.approx(0.0095)
    assert result.qvalues[3] == pytest.approx(0.035625)
    # Monotonization: rank 5 raw value 0.0603 survives, ranks 6 and 7 share
    # the running minimum 0.0298 * 15 / 7.
    assert result.qvalues[4] == pytest.approx(0.0603)
    assert result.qvalues[5] == pytest.approx(0.0298 * 15 / 7)
    assert result.qvalues[6] == pytest.approx(0.0298 * 15 / 7)
    assert result.qvalues[14] == pytest.approx(1.0)
    # reject flags are exactly "q <= alpha"
    assert result.reject == tuple(q <= 0.05 for q in result.qvalues)


def test_bh_preserves_input_order() -> None:
    result = benjamini_hochberg([0.9, 0.01, 0.5], alpha=0.05)
    assert result.qvalues[1] == pytest.approx(0.03)  # 0.01 * 3 / 1
    assert result.reject == (False, True, False)


def test_bh_empty_input() -> None:
    result = benjamini_hochberg([], alpha=0.05)
    assert result.qvalues == ()
    assert result.reject == ()


def test_bh_validates_inputs() -> None:
    with pytest.raises(ValueError, match="alpha"):
        benjamini_hochberg([0.5], alpha=1.5)
    with pytest.raises(ValueError, match="p-values"):
        benjamini_hochberg([0.5, 1.2], alpha=0.05)


# ─────────────────────────────────────────────────────────────────────────────
# Permutation p-value
# ─────────────────────────────────────────────────────────────────────────────


def test_permutation_obvious_effect() -> None:
    rng = random.Random(42)
    group_a = [10.0 + 0.1 * i for i in range(12)]
    group_b = [0.0 + 0.1 * i for i in range(12)]
    observed = mean_difference(group_a, group_b)
    p = permutation_pvalue(observed, mean_difference, group_a, group_b, rng=rng)
    # Fully separated groups: essentially no relabeling reaches the observed
    # statistic, so p sits at (or within a tie or two of) the 1/2001 floor.
    assert p <= 0.005
    assert p >= 1 / 2001


def test_permutation_no_effect() -> None:
    rng = random.Random(7)
    group_a = [rng.gauss(0.0, 1.0) for _ in range(15)]
    group_b = [rng.gauss(0.0, 1.0) for _ in range(15)]
    observed = mean_difference(group_a, group_b)
    p = permutation_pvalue(
        observed, mean_difference, group_a, group_b, rng=rng, n_permutations=1000
    )
    assert p > 0.05


def test_permutation_never_exactly_zero() -> None:
    rng = random.Random(1)
    group_a = [100.0, 101.0, 102.0]
    group_b = [0.0, 1.0, 2.0]
    observed = mean_difference(group_a, group_b)
    p = permutation_pvalue(
        observed, mean_difference, group_a, group_b, rng=rng, n_permutations=10
    )
    assert p >= 1 / 11
    assert p > 0.0


def test_permutation_one_sided() -> None:
    rng = random.Random(3)
    group_a = [0.0] * 10
    group_b = [10.0] * 10
    observed = mean_difference(group_a, group_b)  # strongly negative
    p = permutation_pvalue(
        observed, mean_difference, group_a, group_b, rng=rng,
        n_permutations=500, two_sided=False,
    )
    # One-sided "greater": almost every shuffled statistic exceeds -10.
    assert p > 0.95


def test_permutation_validates_inputs() -> None:
    rng = random.Random(0)
    with pytest.raises(ValueError, match="non-empty"):
        permutation_pvalue(0.0, mean_difference, [], [1.0], rng=rng)
    with pytest.raises(ValueError, match="n_permutations"):
        permutation_pvalue(0.0, mean_difference, [1.0], [1.0], rng=rng, n_permutations=0)


# ─────────────────────────────────────────────────────────────────────────────
# Split-half replication
# ─────────────────────────────────────────────────────────────────────────────


def test_split_half_stable_effect_replicates() -> None:
    group_a = [10.0] * 20  # consistently higher across the whole window
    group_b = [0.0] * 20
    result = split_half_replication(mean_difference, group_a, group_b)
    assert result.replicated is True
    assert result.effect_first == pytest.approx(10.0)
    assert result.effect_second == pytest.approx(10.0)


def test_split_half_unstable_effect_fails() -> None:
    # Effect direction flips between the temporal halves: +10 then -10.
    group_a = [10.0] * 10 + [0.0] * 10
    group_b = [0.0] * 10 + [10.0] * 10
    result = split_half_replication(mean_difference, group_a, group_b)
    assert result.replicated is False
    assert result.effect_first == pytest.approx(10.0)
    assert result.effect_second == pytest.approx(-10.0)
    assert "flips" in result.reason


def test_split_half_too_few_samples() -> None:
    result = split_half_replication(mean_difference, [1.0, 2.0], [3.0, 4.0])
    assert result.replicated is False
    assert result.effect_first is None
    assert result.effect_second is None
    assert "too few" in result.reason


def test_split_half_zero_effect_in_one_half() -> None:
    group_a = [5.0] * 6 + [0.0] * 6
    group_b = [0.0] * 12
    result = split_half_replication(mean_difference, group_a, group_b)
    assert result.replicated is False
    assert "vanishes" in result.reason


# ─────────────────────────────────────────────────────────────────────────────
# Power gate
# ─────────────────────────────────────────────────────────────────────────────


def test_power_gate_passes_at_exact_boundary() -> None:
    result = power_gate([8, 8], min_per_group=8, min_total=16)
    assert result.passed is True
    assert "powered" in result.reason


def test_power_gate_fails_one_below_per_group_floor() -> None:
    result = power_gate([7, 20], min_per_group=8, min_total=16)
    assert result.passed is False
    assert "7 of 8" in result.reason
    assert "collecting more data" in result.reason


def test_power_gate_fails_below_total_floor() -> None:
    result = power_gate([10, 10], min_per_group=8, min_total=24)
    assert result.passed is False
    assert "20 total samples of 24" in result.reason


def test_power_gate_no_groups() -> None:
    result = power_gate([])
    assert result.passed is False
    assert "collecting more data" in result.reason


# ─────────────────────────────────────────────────────────────────────────────
# assess() - full pipeline
# ─────────────────────────────────────────────────────────────────────────────


def test_assess_pass_on_strong_stable_effect() -> None:
    rng = random.Random(99)
    group_a = [10.0 + 0.5 * (i % 4) for i in range(16)]
    group_b = [0.0 + 0.5 * (i % 4) for i in range(16)]
    verdict = assess(group_a, group_b, rng=rng)
    assert verdict.verdict == "pass"
    assert verdict.powered is True
    assert verdict.replicated is True
    assert verdict.p is not None and verdict.p < 0.01
    assert verdict.q is not None and verdict.q <= 0.10


def test_assess_weak_on_significant_but_unreplicated_effect() -> None:
    # Big early effect that reverses sign late: globally significant under
    # permutation, but the direction flips between temporal halves.
    rng = random.Random(123)
    group_a = [10.0] * 8 + [-1.0] * 8
    group_b = [0.0] * 16
    verdict = assess(group_a, group_b, rng=rng)
    assert verdict.powered is True
    assert verdict.q is not None and verdict.q <= 0.10
    assert verdict.replicated is False
    assert verdict.verdict == "weak"


def test_assess_fail_when_underpowered() -> None:
    rng = random.Random(5)
    verdict = assess([10.0] * 4, [0.0] * 4, rng=rng)
    assert verdict.verdict == "fail"
    assert verdict.powered is False
    assert verdict.p is None
    assert verdict.q is None
    assert verdict.replicated is None
    assert any("collecting more data" in r for r in verdict.reasons)


def test_assess_fail_when_not_significant() -> None:
    rng = random.Random(11)
    values = [float(i % 5) for i in range(20)]
    verdict = assess(values, list(values), rng=rng)
    assert verdict.verdict == "fail"
    assert verdict.powered is True
    assert verdict.q is not None and verdict.q > 0.10


def test_assess_sibling_pvalues_tighten_q() -> None:
    # The same finding assessed inside a run of many null hypotheses must
    # carry a larger q than when assessed alone.
    group_a = [10.0 + 0.5 * (i % 4) for i in range(16)]
    group_b = [0.0 + 0.5 * (i % 4) for i in range(16)]
    alone = assess(group_a, group_b, rng=random.Random(99))
    in_run = assess(
        group_a, group_b, rng=random.Random(99),
        sibling_pvalues=[0.2, 0.4, 0.6, 0.8],
    )
    assert alone.q is not None and in_run.q is not None
    assert in_run.q >= alone.q


# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.calibration
def test_permutation_null_rejection_rate_calibrated() -> None:
    """Under a pure null, P(p <= alpha) must be ~alpha (or conservative).

    200 synthetic datasets with both groups drawn from the same N(0, 1).
    The +1 correction makes the test slightly conservative, so the observed
    rejection rate at alpha = 0.05 should land near (or just under) 5%.
    With 200 trials the binomial standard error is ~1.5%, so [0.005, 0.105]
    is a sensible tolerance band.
    """
    rng = random.Random(20260610)
    n_datasets = 200
    rejections = 0
    for _ in range(n_datasets):
        group_a = [rng.gauss(0.0, 1.0) for _ in range(15)]
        group_b = [rng.gauss(0.0, 1.0) for _ in range(15)]
        observed = mean_difference(group_a, group_b)
        p = permutation_pvalue(
            observed, mean_difference, group_a, group_b, rng=rng, n_permutations=500
        )
        if p <= 0.05:
            rejections += 1
    rate = rejections / n_datasets
    assert 0.005 <= rate <= 0.105, f"null rejection rate {rate:.3f} outside tolerance"


@pytest.mark.calibration
def test_bh_controls_fdr_on_synthetic_mixture() -> None:
    """BH must hold the FDR at or below alpha on a true/false mixture.

    300 simulated runs of m = 20 hypotheses: 5 true effects (p-values near
    zero) and 15 nulls (p ~ Uniform(0, 1), the exact null distribution of a
    valid p-value). BH's guarantee for this mixture is FDR <= alpha * m0/m
    = 0.10 * 15/20 = 0.075, so the empirical mean false-discovery proportion
    should sit near 0.075 - comfortably below 0.12 and well above zero (the
    test would be vacuous if nothing was ever falsely rejected).
    """
    rng = random.Random(8675309)
    n_runs = 300
    m, n_true = 20, 5
    alpha = 0.10
    fdp_sum = 0.0
    true_rejections = 0
    for _ in range(n_runs):
        pvalues = [rng.random() * 1e-4 for _ in range(n_true)]
        pvalues += [rng.random() for _ in range(m - n_true)]
        result = benjamini_hochberg(pvalues, alpha=alpha)
        rejected = sum(result.reject)
        false_rejected = sum(result.reject[n_true:])
        fdp_sum += false_rejected / rejected if rejected else 0.0
        true_rejections += sum(result.reject[:n_true])
    fdr = fdp_sum / n_runs
    power = true_rejections / (n_runs * n_true)
    assert 0.02 <= fdr <= 0.12, f"empirical FDR {fdr:.3f} outside tolerance"
    assert power >= 0.95, f"power {power:.3f} too low for the FDR check to be meaningful"
