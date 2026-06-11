"""Statistical rigor layer — the gate every discovery must clear.

A discovery engine mining months of CGM data across dozens of dimensions is a
p-hacking machine unless every candidate finding is forced through the same
four checks. This module is that gate. Discovery agents may only emit findings
whose :class:`~dexta_intelligence.models.FindingStats` were filled in here.

The four checks
---------------
1. **Permutation p-values** (:func:`permutation_pvalue`). Distribution-free:
   instead of assuming normality, we ask "if group labels were meaningless,
   how often would shuffled labels produce a statistic at least this extreme?"
   The Monte Carlo estimate uses the +1 correction of Phipson & Smyth (2010),
   ``p = (b + 1) / (m + 1)``, which is a valid p-value (never exactly zero and
   never anti-conservative, no matter how few permutations were run).
2. **Benjamini-Hochberg FDR** (:func:`benjamini_hochberg`). When an analysis
   run tests many hypotheses, raw p-values lie: at alpha = 0.05, fifty null
   comparisons yield ~2.5 false discoveries. BH (1995) bounds the *expected
   fraction* of false discoveries among everything surfaced. Findings carry
   the adjusted ``q`` value, not the raw ``p``.
3. **Split-half replication** (:func:`split_half_replication`). Even a
   q-significant effect can be an artifact of one anomalous week. We split
   the window temporally, recompute the effect in each disjoint half, and
   require the *direction* to agree. Non-replicated findings are demoted to
   hypotheses rather than surfaced as discoveries.
4. **Power gate** (:func:`power_gate`). With tiny groups, both significance
   and replication are noise. Below a minimum-n policy we refuse to claim
   anything and emit a human-readable "collecting more data" reason that
   surfaces can show as cold-start progress messaging.

:func:`assess` chains all four into a single structured
:class:`RigorVerdict` for one candidate finding.

Dependencies: Python stdlib + pydantic only. This module is self-contained
by design — it must never import from sibling analytics modules.
"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    import random
    from collections.abc import Callable, Sequence

    #: A two-sample statistic, e.g. difference of group means. Both rigor and
    #: discovery code pass groups in *time order* (oldest first) so that
    #: split-half replication can cut them temporally.
    TwoSampleStatistic = Callable[[Sequence[float], Sequence[float]], float]

__all__ = [
    "BHResult",
    "PowerGateResult",
    "RigorVerdict",
    "SplitHalfResult",
    "assess",
    "benjamini_hochberg",
    "mean_difference",
    "permutation_pvalue",
    "power_gate",
    "split_half_replication",
]

#: Slack when comparing permuted statistics against the observed one, so that
#: exact ties (common with discrete CGM values) are counted as "at least as
#: extreme". Counting ties is the conservative, standard choice.
_TIE_EPS = 1e-12


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────


def mean_difference(group_a: Sequence[float], group_b: Sequence[float]) -> float:
    """Difference of group means, ``mean(group_a) - mean(group_b)``.

    The default effect statistic for the rigor pipeline: positive values mean
    group A runs higher. It is deliberately unstandardized — prose layers
    report it in the metric's native units (mg/dL, %TIR, ...).

    :raises statistics.StatisticsError: if either group is empty.
    """
    return statistics.fmean(group_a) - statistics.fmean(group_b)


def permutation_pvalue(
    observed: float,
    statistic: TwoSampleStatistic,
    group_a: Sequence[float],
    group_b: Sequence[float],
    *,
    rng: random.Random,
    n_permutations: int = 2000,
    two_sided: bool = True,
) -> float:
    """Monte Carlo permutation p-value for a two-sample statistic.

    Under the null hypothesis that group membership is exchangeable (the
    labels carry no information), every relabeling of the pooled data is
    equally likely. We therefore shuffle the pooled values ``n_permutations``
    times, recompute ``statistic`` on each relabeled split, and count how
    often the permuted statistic is at least as extreme as ``observed``.

    The returned estimate uses the +1 correction of Phipson & Smyth (2010)::

        p = (count_at_least_as_extreme + 1) / (n_permutations + 1)

    which treats the observed labeling as one more (valid) permutation. This
    guarantees ``p > 0`` — a Monte Carlo test can never honestly report
    ``p == 0`` — and keeps the test exact-or-conservative regardless of how
    few permutations were run.

    No distributional assumptions are made; this works for any statistic the
    discovery tools compute (mean differences, TIR deltas, medians, ...).

    :param observed: the statistic computed on the real labeling. Passed in
        (rather than recomputed) so callers can reuse the value they already
        have and so ``assess`` reports exactly what was tested.
    :param statistic: callable mapping ``(group_a, group_b)`` to a float.
    :param group_a: first group's values.
    :param group_b: second group's values.
    :param rng: seeded :class:`random.Random`; the caller owns reproducibility.
    :param n_permutations: number of random relabelings (default 2000, giving
        a p-value floor of 1/2001 ≈ 0.0005).
    :param two_sided: if true (default), extremity is ``abs(stat) >=
        abs(observed)``; otherwise one-sided, ``stat >= observed``.
    :returns: p-value in ``(0, 1]``.
    :raises ValueError: if either group is empty or ``n_permutations < 1``.
    """
    if not group_a or not group_b:
        raise ValueError("permutation test requires two non-empty groups")
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be >= 1, got {n_permutations}")

    pool = list(group_a) + list(group_b)
    n_a = len(group_a)
    threshold = abs(observed) if two_sided else observed

    at_least_as_extreme = 0
    for _ in range(n_permutations):
        rng.shuffle(pool)
        stat = statistic(pool[:n_a], pool[n_a:])
        value = abs(stat) if two_sided else stat
        if value >= threshold - _TIE_EPS:
            at_least_as_extreme += 1

    return (at_least_as_extreme + 1) / (n_permutations + 1)


class BHResult(_FrozenModel):
    """Benjamini-Hochberg output, index-aligned with the input p-values."""

    qvalues: tuple[float, ...]
    """Adjusted q-values: the smallest FDR level at which each test would
    be rejected. Monotone in the p-value ranking and capped at 1."""
    reject: tuple[bool, ...]
    """Whether each hypothesis is rejected at the requested ``alpha``
    (equivalent to ``q <= alpha``)."""


def benjamini_hochberg(pvalues: Sequence[float], alpha: float = 0.10) -> BHResult:
    """Benjamini-Hochberg step-up FDR correction (BH 1995).

    Applied once per Deep-Analysis run across *all* hypotheses tested in that
    run. Controls the false discovery rate — the expected fraction of false
    positives among the rejected hypotheses — at level ``alpha``, assuming
    independent (or positively dependent, PRDS) tests.

    Procedure: sort the m p-values ascending, find the largest rank ``k``
    with ``p_(k) <= k * alpha / m``, and reject hypotheses with rank <= k.
    Equivalently, the adjusted q-value is::

        q_(i) = min_{j >= i} ( p_(j) * m / j )

    and a hypothesis is rejected iff ``q <= alpha``. Output order matches
    input order, so callers can zip results back onto their findings.

    :param pvalues: raw p-values, one per hypothesis tested in the run.
    :param alpha: target FDR level (project default 0.10 — the surfacing
        threshold for discoveries).
    :returns: :class:`BHResult` with q-values and reject flags in input order.
    :raises ValueError: if alpha is outside ``(0, 1)`` or any p-value is
        outside ``[0, 1]``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    for p in pvalues:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p-values must be in [0, 1], got {p}")

    m = len(pvalues)
    if m == 0:
        return BHResult(qvalues=(), reject=())

    order = sorted(range(m), key=lambda i: pvalues[i])
    qvalues = [0.0] * m
    running_min = 1.0
    for rank in range(m, 0, -1):  # walk from the largest p-value down
        idx = order[rank - 1]
        running_min = min(running_min, pvalues[idx] * m / rank)
        qvalues[idx] = running_min

    return BHResult(
        qvalues=tuple(qvalues),
        reject=tuple(q <= alpha for q in qvalues),
    )


class SplitHalfResult(_FrozenModel):
    """Outcome of temporal split-half replication for one effect."""

    effect_first: float | None
    """Statistic recomputed on the first (earlier) half, if computable."""
    effect_second: float | None
    """Statistic recomputed on the second (later) half, if computable."""
    replicated: bool
    """True iff both halves were computable and the effect direction agrees."""
    reason: str
    """Human-readable explanation of the outcome."""


def split_half_replication(
    statistic: TwoSampleStatistic,
    group_a: Sequence[float],
    group_b: Sequence[float],
    *,
    min_per_half: int = 3,
) -> SplitHalfResult:
    """Re-test an effect on temporally disjoint halves of the data.

    A pattern mined from the whole window may be driven entirely by one
    anomalous stretch (a site change, an illness, a vacation). The cheapest
    honest defense is replication on disjoint time: split each group at its
    temporal midpoint (callers pass values oldest-first), recompute the
    statistic on the *first halves* and on the *second halves*, and require
    the sign of the effect to agree.

    This intentionally checks **direction only**, not magnitude: with half
    the data per side, magnitudes are noisy, and demanding agreement there
    would reject real effects. Magnitudes are still reported so the skeptic
    agent and prose layers can see how stable the effect is.

    :param statistic: the same two-sample statistic used for the main test.
    :param group_a: first group's values in time order (oldest first).
    :param group_b: second group's values in time order (oldest first).
    :param min_per_half: minimum samples per group per half; below this the
        split is meaningless and replication fails with a reason rather than
        pretending to confirm anything.
    :returns: :class:`SplitHalfResult` with both effect magnitudes, the
        replication flag, and a human-readable reason.
    """
    half_a = len(group_a) // 2
    half_b = len(group_b) // 2
    smallest = min(half_a, len(group_a) - half_a, half_b, len(group_b) - half_b)
    if smallest < min_per_half:
        return SplitHalfResult(
            effect_first=None,
            effect_second=None,
            replicated=False,
            reason=(
                f"too few samples to split: smallest group-half has {smallest} "
                f"of {min_per_half} required"
            ),
        )

    effect_first = statistic(group_a[:half_a], group_b[:half_b])
    effect_second = statistic(group_a[half_a:], group_b[half_b:])

    if effect_first == 0.0 or effect_second == 0.0:
        replicated = False
        reason = (
            f"effect vanishes in at least one half "
            f"({effect_first:+.4g} then {effect_second:+.4g})"
        )
    elif (effect_first > 0.0) == (effect_second > 0.0):
        replicated = True
        reason = (
            f"direction replicates across halves "
            f"({effect_first:+.4g} then {effect_second:+.4g})"
        )
    else:
        replicated = False
        reason = f"direction flips between halves ({effect_first:+.4g} then {effect_second:+.4g})"

    return SplitHalfResult(
        effect_first=effect_first,
        effect_second=effect_second,
        replicated=replicated,
        reason=reason,
    )


class PowerGateResult(_FrozenModel):
    """Pass/fail of the minimum-sample gate, with a surfaceable reason."""

    passed: bool
    reason: str
    """Human-readable; failure reasons are written as cold-start progress
    messages ("collecting more data"), not as errors."""


def power_gate(
    group_sizes: Sequence[int],
    *,
    min_per_group: int = 8,
    min_total: int = 16,
) -> PowerGateResult:
    """Minimum-sample gate: refuse to claim anything from thin data.

    A full power analysis needs an effect-size assumption; for a discovery
    harness the honest, assumption-free proxy is a minimum-n policy: below a
    floor of observations per group, *any* significance or replication result
    is noise, so we refuse to test at all and tell the user what to collect.

    The default floor of 8 per group mirrors the platform's correlation gates
    (e.g. "8 nights per sleep bucket") and keeps the permutation null
    distribution rich enough to be meaningful (C(16, 8) = 12870 relabelings).

    :param group_sizes: observation counts for each comparison group.
    :param min_per_group: floor for the smallest group.
    :param min_total: floor for the total sample across groups.
    :returns: :class:`PowerGateResult`; the failure reason is phrased for
        cold-start messaging.
    """
    if not group_sizes:
        return PowerGateResult(
            passed=False,
            reason="no comparison groups present — collecting more data",
        )
    smallest = min(group_sizes)
    total = sum(group_sizes)
    if smallest < min_per_group:
        return PowerGateResult(
            passed=False,
            reason=(
                f"underpowered: smallest group has {smallest} of {min_per_group} "
                f"required samples — collecting more data"
            ),
        )
    if total < min_total:
        return PowerGateResult(
            passed=False,
            reason=(
                f"underpowered: {total} total samples of {min_total} required "
                f"— collecting more data"
            ),
        )
    return PowerGateResult(
        passed=True,
        reason=(
            f"powered: {len(group_sizes)} groups, smallest n={smallest} "
            f"(>= {min_per_group}), total n={total} (>= {min_total})"
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────


class RigorVerdict(_FrozenModel):
    """Structured outcome of the full rigor pipeline for one candidate finding.

    Maps directly onto :class:`~dexta_intelligence.models.FindingStats`
    (``p`` → ``p_perm``, ``q`` → ``q_fdr``, ``replicated`` → ``replicated``).
    Verdict semantics:

    - ``"pass"`` — powered, q-significant, and replicated: may be surfaced
      as a discovery.
    - ``"weak"`` — powered and q-significant but **not** replicated: demote
      to a :class:`~dexta_intelligence.models.Hypothesis` (status ``open``).
    - ``"fail"`` — underpowered, or not significant after FDR correction:
      must not be surfaced as an effect claim.
    """

    p: float | None
    """Permutation p-value; ``None`` when the power gate refused to test."""
    q: float | None
    """BH-adjusted q-value across the run; ``None`` when untested."""
    replicated: bool | None
    """Split-half direction agreement; ``None`` when untested."""
    powered: bool
    """Whether the minimum-sample gate passed."""
    verdict: Literal["pass", "weak", "fail"]
    reasons: tuple[str, ...]
    """Human-readable trail of every check, in pipeline order."""


def assess(
    group_a: Sequence[float],
    group_b: Sequence[float],
    *,
    rng: random.Random,
    statistic: TwoSampleStatistic = mean_difference,
    alpha: float = 0.10,
    n_permutations: int = 2000,
    min_per_group: int = 8,
    min_total: int = 16,
    min_per_half: int = 3,
    sibling_pvalues: Sequence[float] = (),
) -> RigorVerdict:
    """Run the full rigor pipeline on one candidate two-group finding.

    Pipeline order (matching the Deep-Analysis contract):

    1. :func:`power_gate` on the group sizes. If it fails, nothing else is
       computed — testing thin data would only manufacture false confidence —
       and the verdict is ``"fail"`` with a cold-start reason.
    2. :func:`permutation_pvalue` on ``statistic(group_a, group_b)``.
    3. :func:`benjamini_hochberg` over this p-value **plus**
       ``sibling_pvalues`` — the raw p-values of every other hypothesis
       tested in the same analysis run. FDR correction is only honest at the
       run level; a finding assessed alone (no siblings) has ``q == p``.
    4. :func:`split_half_replication` on temporally ordered groups.

    Verdict: ``"pass"`` if q-significant and replicated; ``"weak"`` if
    q-significant but not replicated (demote to hypothesis); ``"fail"``
    otherwise.

    :param group_a: first group's values in time order (oldest first).
    :param group_b: second group's values in time order (oldest first).
    :param rng: seeded :class:`random.Random` for the permutation test. The
        skeptic agent re-runs with a different seed by design.
    :param statistic: two-sample effect statistic (default
        :func:`mean_difference`).
    :param alpha: FDR surfacing threshold (project default 0.10).
    :param n_permutations: permutation count for the p-value.
    :param min_per_group: power-gate floor per group.
    :param min_total: power-gate floor for total samples.
    :param min_per_half: replication floor per group per temporal half.
    :param sibling_pvalues: raw p-values of the other hypotheses tested in
        this analysis run, for run-level FDR correction.
    :returns: a frozen :class:`RigorVerdict`.
    """
    reasons: list[str] = []

    gate = power_gate(
        [len(group_a), len(group_b)],
        min_per_group=min_per_group,
        min_total=min_total,
    )
    reasons.append(gate.reason)
    if not gate.passed:
        return RigorVerdict(
            p=None,
            q=None,
            replicated=None,
            powered=False,
            verdict="fail",
            reasons=tuple(reasons),
        )

    observed = statistic(group_a, group_b)
    p = permutation_pvalue(
        observed,
        statistic,
        group_a,
        group_b,
        rng=rng,
        n_permutations=n_permutations,
    )
    q = benjamini_hochberg([p, *sibling_pvalues], alpha=alpha).qvalues[0]

    split = split_half_replication(statistic, group_a, group_b, min_per_half=min_per_half)
    reasons.append(split.reason)

    verdict: Literal["pass", "weak", "fail"]
    if q > alpha:
        verdict = "fail"
        reasons.append(
            f"not significant after FDR correction: q={q:.4g} > alpha={alpha:g} "
            f"(raw p={p:.4g}, effect={observed:+.4g})"
        )
    elif not split.replicated:
        verdict = "weak"
        reasons.append(
            f"significant (q={q:.4g}) but not replicated — demote to hypothesis"
        )
    else:
        verdict = "pass"
        reasons.append(
            f"significant after FDR (q={q:.4g}) and replicated (effect={observed:+.4g})"
        )

    return RigorVerdict(
        p=p,
        q=q,
        replicated=split.replicated,
        powered=True,
        verdict=verdict,
        reasons=tuple(reasons),
    )
