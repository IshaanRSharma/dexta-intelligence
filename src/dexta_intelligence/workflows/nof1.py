"""Single-subject (n-of-1) research mode: pre-register, test, report.

A clinician or self-experimenter pre-registers one comparison over the
subject's own data - a grouping of days or events and an outcome metric - and
this workflow runs the full statistical-rigor battery on the two resulting
conditions, then emits a reproducible verdict.

What this is, honestly
----------------------
This is an **observational, within-subject association test**, not a randomized
trial. Conditions are sorted *post hoc* from naturally-occurring data (weekends
that happened to be weekends, nights that happened to be poorly slept), never
randomly assigned, so the result surfaces a rigorously-tested *association* and
nothing more. It never claims causation and never recommends a treatment
change. Pre-registering the single comparison up front is what keeps it honest:
it removes the multiple-comparisons freedom that turns observational mining into
p-hacking.

The rigor is reused, not reinvented. Each n-of-1 test is one
:func:`dexta_intelligence.stats.rigor.assess` call - permutation p-value,
split-half replication, and the minimum-sample power gate - over the two groups
the :class:`~dexta_intelligence.agents.tools.toolkit.DiscoveryToolkit`
produces for the registered comparison. FDR correction is intentionally absent:
a single pre-registered hypothesis tests exactly one thing, so ``q == p`` and
the BH step would be a no-op (and dishonest dressing).

Deterministic: no LLM. Given the same store and seed, two runs are identical.
Never raises on thin data - a comparison the toolkit cannot form, or groups
below the power floor, comes back as an ``"underpowered"`` verdict with a
human-readable reason, never an exception.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from typing import TYPE_CHECKING, Any, Literal

from dexta_intelligence.agents.tools.toolkit import DiscoveryToolkit
from dexta_intelligence.models import Finding, FindingStats
from dexta_intelligence.stats.core import cohen_d
from dexta_intelligence.stats.rigor import assess

if TYPE_CHECKING:
    from dexta_intelligence.agents.base import AgentContext

__all__ = [
    "COMPARISONS",
    "METRICS",
    "Hypothesis",
    "Nof1Result",
    "parse_hypothesis",
    "result_to_finding",
    "run_nof1",
]

#: Default permutation seed. The caller owns reproducibility; a fixed default
#: makes the CLI deterministic out of the box.
_DEFAULT_SEED = 1_729

Verdict = Literal["supported", "not_supported", "underpowered"]

#: Pre-registerable comparisons → the DiscoveryToolkit instrument + fixed args
#: that splits the subject's data into the two conditions. Each entry reuses the
#: toolkit's existing grouping vocabulary; n-of-1 invents no new splitters.
#:
#: ``metric_aware`` comparisons take the outcome metric through to the tool
#: (only ``groupby_compare`` supports mean_glucose vs tir); the rest fix their
#: own outcome (an excursion, a per-day window mean) and ignore the metric arg.
COMPARISONS: dict[str, dict[str, Any]] = {
    "weekend": {
        "tool": "groupby_compare",
        "args": {"group_by": "weekend"},
        "metric_aware": True,
        "labels": ("weekend", "weekday"),
        "phrase": "weekends differ from weekdays",
    },
    "sleep": {
        "tool": "groupby_compare",
        "args": {"group_by": "sleep_bucket"},
        "metric_aware": True,
        "labels": ("poorer-sleep days", "better-sleep days"),
        "phrase": "poorer-sleep days differ from better-sleep days",
    },
    "workout": {
        "tool": "groupby_compare",
        "args": {"group_by": "workout_day"},
        "metric_aware": True,
        "labels": ("workout days", "rest days"),
        "phrase": "workout days differ from rest days",
    },
    "meal_carbs": {
        "tool": "meal_response",
        "args": {"window_min": 120},
        "metric_aware": False,
        "labels": ("bigger-carb meals", "smaller-carb meals"),
        "phrase": "bigger-carb meals produce a larger post-meal excursion than smaller-carb meals",
        "outcome": "post-meal glucose excursion (mg/dL)",
    },
}

#: Outcome metrics a metric-aware comparison may register, with prose labels.
METRICS: dict[str, str] = {
    "mean_glucose": "mean glucose (mg/dL)",
    "tir": "time in range (%)",
}

#: ``tir`` is the user-facing name; the toolkit's groupby target is ``tir_pct``.
_METRIC_TARGET = {"mean_glucose": "mean_glucose", "tir": "tir_pct"}


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """A pre-registered single-subject comparison.

    ``comparison`` names an entry in :data:`COMPARISONS` (which daily/event
    grouping to test); ``metric`` names the outcome in :data:`METRICS` for
    metric-aware comparisons (ignored by comparisons that fix their own outcome,
    e.g. ``meal_carbs``, whose outcome is the post-meal excursion).

    ``statement`` is the human-readable pre-registration - what the subject
    committed to testing before looking. It is recorded verbatim in the result
    so the report shows exactly what was registered.
    """

    comparison: str
    metric: str = "mean_glucose"
    statement: str = ""

    def registered_statement(self) -> str:
        """The pre-registered claim as prose (falls back to a generated one)."""
        return self.statement or self._default_statement()

    def _default_statement(self) -> str:
        spec = COMPARISONS.get(self.comparison)
        if spec is None:
            return f"unknown comparison {self.comparison!r}"
        if spec["metric_aware"]:
            metric_label = METRICS.get(self.metric, self.metric)
            return f"{spec['phrase']} in {metric_label}"
        return str(spec["phrase"])

    def outcome_label(self) -> str:
        """Prose name of the outcome being compared."""
        spec = COMPARISONS.get(self.comparison)
        if spec is None:
            return self.metric
        if spec["metric_aware"]:
            return METRICS.get(self.metric, self.metric)
        return str(spec.get("outcome", self.metric))


@dataclass(frozen=True, slots=True)
class Nof1Result:
    """Outcome of one pre-registered single-subject test.

    Always returned - a comparison that could not be formed or that fell below
    the power floor comes back with ``verdict="underpowered"`` and an empty
    ``stats``, never an exception.
    """

    hypothesis: Hypothesis
    ok: bool
    """True iff the comparison formed two usable groups *and* cleared the power
    gate so the rigor battery actually ran."""
    label_a: str
    label_b: str
    n_a: int
    n_b: int
    mean_a: float | None
    mean_b: float | None
    effect_size: float | None
    """Difference of group means in the outcome's native units (mean_a - mean_b)."""
    cohen_d: float | None
    p_perm: float | None
    replicated: bool | None
    powered: bool
    verdict: Verdict
    reason: str
    """Plain-English summary of the verdict and why."""
    seed: int
    n_permutations: int

    def disclaimer(self) -> str:
        """The standing honesty caveat for every n-of-1 result."""
        return (
            "Single-subject observational association from naturally-occurring "
            "data, not a randomized trial - describes a pattern, not a cause, and "
            "is not treatment advice."
        )


def parse_hypothesis(text: str) -> Hypothesis | None:
    """Best-effort free-text → :class:`Hypothesis` for the CLI's positional form.

    Recognizes a registered comparison keyword (weekend/sleep/workout/meal) and,
    for metric-aware comparisons, an outcome keyword (tir / time-in-range, else
    mean glucose). The original text is kept as the pre-registered statement.
    Returns ``None`` when no comparison keyword is present so the caller can ask
    for structured flags instead of guessing.
    """
    lowered = text.lower()
    comparison: str | None = None
    if "weekend" in lowered or "weekday" in lowered:
        comparison = "weekend"
    elif "sleep" in lowered:
        comparison = "sleep"
    elif "workout" in lowered or "exercise" in lowered or "activity" in lowered:
        comparison = "workout"
    elif "meal" in lowered or "carb" in lowered:
        comparison = "meal_carbs"
    if comparison is None:
        return None

    metric = "tir" if ("tir" in lowered or "time in range" in lowered) else "mean_glucose"
    return Hypothesis(comparison=comparison, metric=metric, statement=text.strip())


def run_nof1(
    ctx: AgentContext,
    hypothesis: Hypothesis,
    *,
    seed: int = _DEFAULT_SEED,
    n_permutations: int = 2000,
    min_per_group: int = 8,
    min_total: int = 16,
) -> Nof1Result:
    """Run the full rigor battery on one pre-registered single-subject comparison.

    Splits the subject's data into the two registered conditions via the
    :class:`~dexta_intelligence.agents.tools.toolkit.DiscoveryToolkit`
    instrument, then runs :func:`dexta_intelligence.stats.rigor.assess` - power
    gate, permutation p-value, split-half replication - on the two groups in
    time order. No FDR step: a single pre-registered hypothesis tests one thing,
    so ``q == p``.

    Verdict mapping:

    - ``"underpowered"`` - the comparison could not be formed, or a group fell
      below the power floor. Collect more data; no effect claim is made.
    - ``"supported"`` - powered, permutation-significant, and the direction
      replicated on a temporally disjoint split.
    - ``"not_supported"`` - powered but not significant, or significant without
      replication (an unreplicated single-subject signal is not support).

    Deterministic given ``seed``; never raises on thin data.
    """
    spec = COMPARISONS.get(hypothesis.comparison)
    if spec is None:
        return _underpowered(
            hypothesis,
            f"unknown comparison {hypothesis.comparison!r}; "
            f"choose one of {', '.join(sorted(COMPARISONS))}",
            seed,
            n_permutations,
        )
    if spec["metric_aware"] and hypothesis.metric not in METRICS:
        return _underpowered(
            hypothesis,
            f"unknown metric {hypothesis.metric!r}; choose one of {', '.join(sorted(METRICS))}",
            seed,
            n_permutations,
        )

    toolkit = DiscoveryToolkit(ctx)
    args = dict(spec["args"])
    if spec["metric_aware"]:
        args["target"] = _METRIC_TARGET[hypothesis.metric]
    tool_result = toolkit.run(str(spec["tool"]), args)

    if not tool_result.ok:
        reason = tool_result.error or "comparison could not be formed"
        return _underpowered(
            hypothesis, f"{reason} - collecting more data", seed, n_permutations
        )

    group_a = tool_result.group_a
    group_b = tool_result.group_b
    labels = spec["labels"]

    verdict_obj = assess(
        group_a,
        group_b,
        rng=random.Random(seed),
        n_permutations=n_permutations,
        min_per_group=min_per_group,
        min_total=min_total,
    )

    if not verdict_obj.powered:
        return Nof1Result(
            hypothesis=hypothesis,
            ok=False,
            label_a=labels[0],
            label_b=labels[1],
            n_a=len(group_a),
            n_b=len(group_b),
            mean_a=None,
            mean_b=None,
            effect_size=None,
            cohen_d=None,
            p_perm=None,
            replicated=None,
            powered=False,
            verdict="underpowered",
            reason=verdict_obj.reasons[0] if verdict_obj.reasons else "underpowered",
            seed=seed,
            n_permutations=n_permutations,
        )

    mean_a = sum(group_a) / len(group_a)
    mean_b = sum(group_b) / len(group_b)
    effect = mean_a - mean_b
    d = cohen_d(group_a, group_b)

    verdict: Verdict
    if verdict_obj.verdict == "pass":
        verdict = "supported"
        reason = (
            f"supported: {labels[0]} vs {labels[1]} differ by {effect:+.1f} in "
            f"{hypothesis.outcome_label()} (p_perm={verdict_obj.p:.4g}), and the "
            f"direction replicated on a temporally disjoint split"
        )
    elif verdict_obj.verdict == "weak":
        verdict = "not_supported"
        reason = (
            f"not supported: a {effect:+.1f} difference reached significance "
            f"(p_perm={verdict_obj.p:.4g}) but did not replicate on a disjoint split, "
            f"so it is not a stable single-subject association"
        )
    else:
        verdict = "not_supported"
        reason = (
            f"not supported: the {effect:+.1f} difference in {hypothesis.outcome_label()} "
            f"is not distinguishable from chance (p_perm={verdict_obj.p:.4g})"
        )

    return Nof1Result(
        hypothesis=hypothesis,
        ok=True,
        label_a=labels[0],
        label_b=labels[1],
        n_a=len(group_a),
        n_b=len(group_b),
        mean_a=mean_a,
        mean_b=mean_b,
        effect_size=effect,
        cohen_d=d,
        p_perm=verdict_obj.p,
        replicated=verdict_obj.replicated,
        powered=True,
        verdict=verdict,
        reason=reason,
        seed=seed,
        n_permutations=n_permutations,
    )


def result_to_finding(result: Nof1Result, ctx: AgentContext) -> Finding:
    """Persist an n-of-1 result as a ``kind="nof1"`` :class:`Finding`.

    The verdict reason becomes the headline; the registered statement, the
    honesty disclaimer, and every tested number go into ``body_md`` / ``evidence``
    so the wiki and physician brief can surface a reproducible record.
    """
    window_start = datetime.combine(ctx.window[0], time.min, tzinfo=UTC)
    window_end = datetime.combine(ctx.window[1], time.min, tzinfo=UTC) + timedelta(days=1)
    confidence = 0.7 if result.verdict == "supported" else 0.3

    evidence: dict[str, Any] = {
        "comparison": result.hypothesis.comparison,
        "metric": result.hypothesis.metric,
        "registered_statement": result.hypothesis.registered_statement(),
        "verdict": result.verdict,
        "label_a": result.label_a,
        "label_b": result.label_b,
        "n_a": result.n_a,
        "n_b": result.n_b,
        "seed": result.seed,
        "n_permutations": result.n_permutations,
    }
    if result.effect_size is not None:
        evidence["effect_size"] = round(result.effect_size, 3)
    if result.p_perm is not None:
        evidence["p_perm"] = result.p_perm
    if result.replicated is not None:
        evidence["replicated"] = result.replicated

    body = (
        f"Pre-registered: {result.hypothesis.registered_statement()}\n\n"
        f"{result.reason}\n\n"
        f"{result.disclaimer()}"
    )

    return Finding(
        agent="nof1",
        kind="nof1",
        scope=result.hypothesis.comparison,
        headline=result.reason,
        body_md=body,
        evidence=evidence,
        stats=FindingStats(
            effect_size=result.effect_size,
            n=result.n_a + result.n_b,
            p_perm=result.p_perm,
            replicated=result.replicated,
        ),
        confidence=confidence,
        window_start=window_start,
        window_end=window_end,
    )


def _underpowered(
    hypothesis: Hypothesis, reason: str, seed: int, n_permutations: int
) -> Nof1Result:
    """Build an ``underpowered`` result for a comparison that never ran."""
    spec = COMPARISONS.get(hypothesis.comparison)
    labels = spec["labels"] if spec else ("group A", "group B")
    return Nof1Result(
        hypothesis=hypothesis,
        ok=False,
        label_a=labels[0],
        label_b=labels[1],
        n_a=0,
        n_b=0,
        mean_a=None,
        mean_b=None,
        effect_size=None,
        cohen_d=None,
        p_perm=None,
        replicated=None,
        powered=False,
        verdict="underpowered",
        reason=reason,
        seed=seed,
        n_permutations=n_permutations,
    )
