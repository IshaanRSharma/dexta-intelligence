"""Deterministic statistics core — pure functions, stdlib only, never NaN.

This module is the numeric foundation every agent builds on. The LLM never
computes statistics; it picks which of these functions to call. That is the
whole point — models hallucinate means, this module does not.

Ported and hardened from the donor codebase:

- ``agents/coach/correlators/_stats.py`` → :func:`mean`, :func:`stdev`,
  :func:`cohen_d`, :func:`confidence_from_n_and_d`,
  :func:`strength_from_effect`.
- ``agents/researcher/tools.py`` → :func:`pearson_r` and the z-score
  baseline logic behind anomaly detection (:func:`zscores`), with all DB /
  Supabase / LLM-context plumbing stripped.

New for the OSS core (spec §8): :func:`spearman_rho`, :func:`welch_t_test`,
:func:`mann_whitney_u`, :func:`cliffs_delta`, :func:`hedges_g`,
:func:`bootstrap_mean_ci`, :func:`bootstrap_diff_ci`, :func:`summarize`.

Design rules
------------
1. **Stdlib only.** ``math`` + ``statistics`` + ``random``. No numpy/scipy —
   self-hosters get a zero-dependency install and auditable math.
2. **Primitives raise, analyses return ``None``.** :func:`mean`,
   :func:`median`, :func:`stdev` raise :class:`ValueError` on degenerate
   input, like ``statistics`` does. Everything higher-level (correlations,
   tests, effect sizes, bootstraps) returns ``None`` when the statistic is
   mathematically undefined. The donor returned ``0.0`` in those cases,
   which silently conflates "no effect" with "cannot compute" — that
   convention is deliberately not inherited.
3. **Never NaN, never infinity.** Every float that comes out of this module
   is finite. Undefined is spelled ``None``, not ``nan``.
4. **Determinism.** Bootstrap functions take an explicit ``seed`` and are
   reproducible by default. Same inputs, same outputs, always.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "BootstrapCI",
    "MannWhitneyResult",
    "SummaryStats",
    "WelchTTestResult",
    "bootstrap_diff_ci",
    "bootstrap_mean_ci",
    "cliffs_delta",
    "cohen_d",
    "confidence_from_n_and_d",
    "hedges_g",
    "mann_whitney_u",
    "mean",
    "median",
    "normal_two_sided_p",
    "pearson_r",
    "spearman_rho",
    "stdev",
    "strength_from_effect",
    "student_t_two_sided_p",
    "summarize",
    "welch_t_test",
    "zscores",
]


# ─────────────────────────────────────────────────────────────────────────────
# Result records
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SummaryStats:
    """Descriptive summary of one series. Fields are ``None`` when undefined.

    ``sd`` needs n ≥ 2; ``cv_pct`` (coefficient of variation, %) additionally
    needs a nonzero mean. CV is a first-class CGM metric — the consensus
    glycemic-variability target is CV ≤ 36%.
    """

    n: int
    mean: float | None
    sd: float | None
    cv_pct: float | None
    minimum: float | None
    maximum: float | None
    median: float | None


@dataclass(frozen=True, slots=True)
class WelchTTestResult:
    """Welch's unequal-variance t-test for a difference in means.

    ``t`` is signed ``mean_a - mean_b``; ``df`` is the Welch-Satterthwaite
    approximation; ``p_two_sided`` comes from the Student-t distribution.
    """

    t: float
    df: float
    p_two_sided: float
    mean_a: float
    mean_b: float
    n_a: int
    n_b: int


@dataclass(frozen=True, slots=True)
class MannWhitneyResult:
    """Mann-Whitney U test (normal approximation, tie-corrected).

    ``u`` counts pairwise wins of group *a* over group *b* (ties count ½),
    so ``rank_biserial = 2u/(n_a·n_b) - 1`` equals Cliff's delta and is the
    natural effect size to report alongside the p-value.
    """

    u: float
    z: float
    p_two_sided: float
    rank_biserial: float
    n_a: int
    n_b: int


@dataclass(frozen=True, slots=True)
class BootstrapCI:
    """Percentile-bootstrap confidence interval. ``lower ≤ upper`` always."""

    point: float
    lower: float
    upper: float
    confidence: float
    n_resamples: int


# ─────────────────────────────────────────────────────────────────────────────
# Descriptive primitives
# ─────────────────────────────────────────────────────────────────────────────


def mean(xs: Sequence[float]) -> float:
    """Arithmetic mean.

    Raises:
        ValueError: on empty input. The donor returned ``0.0`` here, which
            silently poisoned downstream deltas; that behavior is gone.
    """
    if not xs:
        raise ValueError("mean() requires at least one value")
    return statistics.fmean(xs)


def median(xs: Sequence[float]) -> float:
    """Median (average of the middle two for even n).

    Raises:
        ValueError: on empty input.
    """
    if not xs:
        raise ValueError("median() requires at least one value")
    return float(statistics.median(xs))


def stdev(xs: Sequence[float]) -> float:
    """Sample standard deviation (n - 1 denominator).

    Raises:
        ValueError: if n < 2 — one observation has no spread. (The donor
            returned ``0.0``, conflating "constant" with "unknown".)
    """
    if len(xs) < 2:
        raise ValueError("stdev() requires at least two values")
    return statistics.stdev(xs)


def summarize(xs: Sequence[float]) -> SummaryStats:
    """Descriptive summary that is total: any input, including empty, works.

    This is the safe agent-facing entry point — fields the data cannot
    support are ``None`` rather than fabricated.
    """
    n = len(xs)
    if n == 0:
        return SummaryStats(
            n=0, mean=None, sd=None, cv_pct=None, minimum=None, maximum=None, median=None
        )
    m = statistics.fmean(xs)
    sd = statistics.stdev(xs, m) if n >= 2 else None
    cv = (sd / abs(m)) * 100.0 if sd is not None and m != 0.0 else None
    return SummaryStats(
        n=n,
        mean=m,
        sd=sd,
        cv_pct=cv,
        minimum=min(xs),
        maximum=max(xs),
        median=float(statistics.median(xs)),
    )


def zscores(xs: Sequence[float]) -> list[float] | None:
    """Standard scores ``(x - mean) / sd`` for every element.

    This is the baseline-deviation primitive behind the donor's
    ``anomaly_days`` tool. Returns ``None`` when standardization is
    undefined: n < 2 or a constant series (sd = 0). The donor patched
    constant series with ``stdev or 1.0``, which manufactured fake z-scores;
    that hack is not ported.
    """
    if len(xs) < 2:
        return None
    m = statistics.fmean(xs)
    sd = statistics.stdev(xs, m)
    if sd == 0.0:
        return None
    return [(x - m) / sd for x in xs]


# ─────────────────────────────────────────────────────────────────────────────
# Correlation
# ─────────────────────────────────────────────────────────────────────────────


def pearson_r(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Pearson product-moment correlation in [-1, 1].

    Measures *linear* association; sensitive to outliers, so prefer
    :func:`spearman_rho` for skewed lifestyle signals (workout strain,
    sleep debt) or whenever only monotonicity matters.

    Returns ``None`` when undefined: n < 2 or either series is constant.
    (The donor returned ``0.0`` for those *and* for mismatched lengths;
    a length mismatch is a caller bug and now raises.)

    Raises:
        ValueError: if the series have different lengths.
    """
    if len(xs) != len(ys):
        msg = f"pearson_r() needs equal-length series, got {len(xs)} and {len(ys)}"
        raise ValueError(msg)
    if len(xs) < 2:
        return None
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    num = math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    dx = math.sqrt(math.fsum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(math.fsum((y - my) ** 2 for y in ys))
    if dx == 0.0 or dy == 0.0:
        return None
    # Clamp floating-point drift so callers can trust the documented range.
    return max(-1.0, min(1.0, num / (dx * dy)))


def spearman_rho(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Spearman rank correlation: Pearson on average ranks (ties shared).

    Captures any monotonic relationship, linear or not, and is robust to
    outliers — the right default for ordinal scores (recovery 0-100) and
    heavy-tailed daily metrics.

    Returns ``None`` when undefined (n < 2 or a constant series).

    Raises:
        ValueError: if the series have different lengths.
    """
    if len(xs) != len(ys):
        msg = f"spearman_rho() needs equal-length series, got {len(xs)} and {len(ys)}"
        raise ValueError(msg)
    if len(xs) < 2:
        return None
    return pearson_r(_average_ranks(xs), _average_ranks(ys))


def _average_ranks(values: Sequence[float]) -> list[float]:
    """1-based ranks; tied values share the average of their rank positions."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


# ─────────────────────────────────────────────────────────────────────────────
# Effect sizes
# ─────────────────────────────────────────────────────────────────────────────


def cohen_d(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Cohen's d: ``(mean_a - mean_b) / pooled_sd``, the standardized mean
    difference.

    Rule-of-thumb bands: |d| ≈ 0.2 small, 0.5 medium, 0.8 large. Assumes
    roughly similar group variances; for ordinal or heavy-tailed data prefer
    :func:`cliffs_delta`. For small samples (n ≲ 20 total) report
    :func:`hedges_g`, which corrects d's upward bias.

    Returns ``None`` when undefined: either group has n < 2, or both groups
    are constant (pooled sd = 0). The donor returned ``0.0`` and carried a
    dead ``max(1, …)`` guard on the df denominator; both are fixed.
    """
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return None
    var_a = statistics.variance(a)
    var_b = statistics.variance(b)
    pooled = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled == 0.0:
        return None
    return (statistics.fmean(a) - statistics.fmean(b)) / pooled


def hedges_g(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Hedges' g: Cohen's d with the small-sample bias correction
    ``J = 1 - 3 / (4·df - 1)`` where ``df = n_a + n_b - 2``.

    Cohen's d overestimates the population effect for small n; g shrinks it.
    With CGM day-level comparisons (often 5-15 days per group) the
    correction is material. Returns ``None`` whenever :func:`cohen_d` does.
    """
    d = cohen_d(a, b)
    if d is None:
        return None
    df = len(a) + len(b) - 2
    return d * (1.0 - 3.0 / (4.0 * df - 1.0))


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Cliff's delta: ``P(a > b) - P(a < b)`` over all cross-group pairs.

    A distribution-free ordinal effect size in [-1, 1]: +1 means every value
    in *a* exceeds every value in *b*. Bands (Romano et al.): |δ| < 0.147
    negligible, < 0.33 small, < 0.474 medium, else large. Use when normality
    is implausible or outliers would distort :func:`cohen_d`.

    O(n_a · n_b) — fine for day-level series. Returns ``None`` if either
    group is empty.
    """
    if not a or not b:
        return None
    wins = 0
    losses = 0
    for x in a:
        for y in b:
            if x > y:
                wins += 1
            elif x < y:
                losses += 1
    return (wins - losses) / (len(a) * len(b))


# ─────────────────────────────────────────────────────────────────────────────
# Group comparison tests
# ─────────────────────────────────────────────────────────────────────────────


def welch_t_test(a: Sequence[float], b: Sequence[float]) -> WelchTTestResult | None:
    """Welch's unequal-variance t-test for a difference in group means.

    Preferred over Student's pooled t because it does not assume equal
    variances — "workout days" and "rest days" rarely share a variance.
    Appropriate when group means are the quantity of interest and the data
    are roughly symmetric or n is moderate (CLT); otherwise reach for
    :func:`mann_whitney_u`.

    Returns ``None`` when undefined: either group has n < 2, or both groups
    are constant (zero standard error — no sampling noise to test against).
    """
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return None
    mean_a = statistics.fmean(a)
    mean_b = statistics.fmean(b)
    va_n = statistics.variance(a, mean_a) / n_a
    vb_n = statistics.variance(b, mean_b) / n_b
    se_sq = va_n + vb_n
    if se_sq == 0.0:
        return None
    t = (mean_a - mean_b) / math.sqrt(se_sq)
    df = se_sq**2 / (va_n**2 / (n_a - 1) + vb_n**2 / (n_b - 1))
    return WelchTTestResult(
        t=t,
        df=df,
        p_two_sided=student_t_two_sided_p(t, df),
        mean_a=mean_a,
        mean_b=mean_b,
        n_a=n_a,
        n_b=n_b,
    )


def mann_whitney_u(a: Sequence[float], b: Sequence[float]) -> MannWhitneyResult | None:
    """Mann-Whitney U test: do values from *a* tend to exceed values from *b*?

    Distribution-free alternative to :func:`welch_t_test` — compares the
    whole distributions via ranks, so outliers and skew (post-meal spikes,
    hypo tails) cannot dominate. Uses the normal approximation with tie
    correction and a 0.5 continuity correction; treat p-values as
    approximate below ~8 observations per group.

    Returns ``None`` when undefined: either group is empty, or every value
    across both groups is identical (tie-corrected variance = 0).
    """
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return None
    combined = list(a) + list(b)
    ranks = _average_ranks(combined)
    rank_sum_a = math.fsum(ranks[:n_a])
    u = rank_sum_a - n_a * (n_a + 1) / 2.0

    n = n_a + n_b
    tie_term = math.fsum(t**3 - t for t in Counter(combined).values())
    var_u = (n_a * n_b / 12.0) * ((n + 1) - tie_term / (n * (n - 1)))
    if var_u <= 0.0:
        return None

    diff = u - n_a * n_b / 2.0
    # Continuity correction: shrink |diff| by 0.5, never across zero.
    diff = math.copysign(max(0.0, abs(diff) - 0.5), diff)
    z = diff / math.sqrt(var_u)
    return MannWhitneyResult(
        u=u,
        z=z,
        p_two_sided=normal_two_sided_p(z),
        rank_biserial=2.0 * u / (n_a * n_b) - 1.0,
        n_a=n_a,
        n_b=n_b,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Distribution tails (stdlib implementations — no scipy)
# ─────────────────────────────────────────────────────────────────────────────


def normal_two_sided_p(z: float) -> float:
    """Two-sided standard-normal p-value: ``P(|Z| ≥ |z|) = erfc(|z|/√2)``.

    Exact via :func:`math.erfc` (no lookup tables, no approximation error
    beyond float precision).
    """
    if not math.isfinite(z):
        raise ValueError("normal_two_sided_p() requires a finite z")
    return math.erfc(abs(z) / math.sqrt(2.0))


def student_t_two_sided_p(t: float, df: float) -> float:
    """Two-sided Student-t p-value: ``P(|T_df| ≥ |t|)``.

    Computed as the regularized incomplete beta ``I_x(df/2, 1/2)`` with
    ``x = df/(df + t²)`` — the standard closed form — using a Lentz
    continued-fraction evaluation accurate to ~1e-12. Valid for any real
    ``df > 0`` (Welch df is fractional).

    Raises:
        ValueError: if ``df <= 0`` or ``t`` is not finite.
    """
    if df <= 0.0 or not math.isfinite(df):
        raise ValueError("student_t_two_sided_p() requires df > 0")
    if not math.isfinite(t):
        raise ValueError("student_t_two_sided_p() requires a finite t")
    if t == 0.0:
        return 1.0
    x = df / (df + t * t)
    return _regularized_incomplete_beta(df / 2.0, 0.5, x)


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """``I_x(a, b)`` via the continued fraction (Numerical Recipes §6.4)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    front = math.exp(ln_front)
    # Use the fraction on whichever side converges fast, mirror the other.
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Lentz's algorithm for the incomplete-beta continued fraction."""
    # O(sqrt(max(a, b))) iterations needed near the branch point; 2000 covers
    # Welch df up to ~10^6 (beyond that the t-dist is the normal anyway).
    max_iter = 2000
    eps = 1e-14
    tiny = 1e-300

    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        # Even step.
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        # Odd step.
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap confidence intervals
# ─────────────────────────────────────────────────────────────────────────────


def bootstrap_mean_ci(
    xs: Sequence[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> BootstrapCI | None:
    """Percentile-bootstrap CI for the mean of one series.

    Resamples with replacement ``n_resamples`` times and takes the
    ``(1 - confidence)/2`` and ``(1 + confidence)/2`` empirical quantiles.
    Distribution-free and honest about skew, but anti-conservative for very
    small n (≲ 10) — pair with a minimum-n gate upstream. Deterministic for
    a fixed ``seed``.

    Returns ``None`` on empty input.

    Raises:
        ValueError: if ``confidence`` is outside (0, 1) or ``n_resamples < 1``.
    """
    _validate_bootstrap_args(confidence, n_resamples)
    if not xs:
        return None
    rng = random.Random(seed)
    data = list(xs)
    n = len(data)
    means = sorted(statistics.fmean(rng.choices(data, k=n)) for _ in range(n_resamples))
    alpha = (1.0 - confidence) / 2.0
    return BootstrapCI(
        point=statistics.fmean(data),
        lower=_percentile(means, alpha),
        upper=_percentile(means, 1.0 - alpha),
        confidence=confidence,
        n_resamples=n_resamples,
    )


def bootstrap_diff_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> BootstrapCI | None:
    """Percentile-bootstrap CI for ``mean(a) - mean(b)`` (independent groups).

    Each group is resampled independently. A CI excluding zero is bootstrap
    evidence of a real mean difference — the natural companion to
    :func:`cohen_d` when distributional assumptions are shaky.

    Returns ``None`` if either group is empty.

    Raises:
        ValueError: if ``confidence`` is outside (0, 1) or ``n_resamples < 1``.
    """
    _validate_bootstrap_args(confidence, n_resamples)
    if not a or not b:
        return None
    rng = random.Random(seed)
    da, db = list(a), list(b)
    n_a, n_b = len(da), len(db)
    diffs = sorted(
        statistics.fmean(rng.choices(da, k=n_a)) - statistics.fmean(rng.choices(db, k=n_b))
        for _ in range(n_resamples)
    )
    alpha = (1.0 - confidence) / 2.0
    return BootstrapCI(
        point=statistics.fmean(da) - statistics.fmean(db),
        lower=_percentile(diffs, alpha),
        upper=_percentile(diffs, 1.0 - alpha),
        confidence=confidence,
        n_resamples=n_resamples,
    )


def _validate_bootstrap_args(confidence: float, n_resamples: int) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile of an already-sorted list, q in [0, 1]."""
    pos = q * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


# ─────────────────────────────────────────────────────────────────────────────
# Donor heuristics (observation language, not inference)
# ─────────────────────────────────────────────────────────────────────────────


def confidence_from_n_and_d(n: int, d: float) -> float:
    """Map ``(n, |d|)`` → a 0-1 confidence score. Heuristic, NOT a p-value.

    Geometric mean of a sample-size score (saturates at n = 30) and an
    effect-size score (saturates at |d| = 0.8), so a weak n or a weak d
    cannot be averaged away by the other. Donor port; the donor's docstring
    examples were wrong and are corrected here:

    - n=8,  |d|=0.3 → 0.316
    - n=20, |d|=0.5 → 0.645
    - n=40, |d|=0.8 → 1.0

    Rounded to 3 decimals, matching the donor's presentation contract.
    """
    n_score = min(1.0, max(0, n) / 30.0)
    d_score = min(1.0, abs(d) / 0.8)
    return round(math.sqrt(n_score * d_score), 3)


def strength_from_effect(value: float, *, scale: float) -> float:
    """Normalize an effect magnitude to [0, 1] with a saturating curve.

    ``1 - exp(-|value|/scale)``: ``scale`` is the magnitude that reads as
    "strong" (≈ 0.632). Donor convention: scale=10 for TIR percentage-point
    deltas, scale=15 for mg/dL mean deltas. Returns 0.0 for ``scale <= 0``.

    Hardened: the donor used a hand-typed ``2.718281828`` for *e*; this uses
    :func:`math.exp`. Rounded to 3 decimals, matching the donor.
    """
    if scale <= 0:
        return 0.0
    return round(1.0 - math.exp(-abs(value) / scale), 3)
