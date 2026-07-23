"""CGM metric ontology: the small deterministic graph the faithfulness guard uses
to bind numbers to the metric they describe.

The numeric-faithfulness guard on its own is set-membership: a number passes if it
appears anywhere in the evidence pool. That misses "right number, wrong metric"
errors, the exact class the LLM-CGM benchmark (Healey & Kohane, PSB 2025) found a
code-execution agent still gets wrong (it reported the standard deviation when
asked for the coefficient of variation, scoring 0/10). Closing that hole needs to
know which metric a number *is*, and which metric a sentence is *about*.

This module is that knowledge, as an explicit graph:

- a :class:`Metric` node per CGM statistic,
- edges to the evidence-dict keys that carry its value (``key_aliases``),
- edges to the phrases a model uses to name it in prose (``prose_aliases``).

It is deliberately small, deterministic, and free of the model: an ontology of
conventions, not data. ``is_percent`` records whether a metric is a ratio, so a
value stored as a fraction (0.287) and cited as a percent (28.7%) still matches.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "METRICS",
    "Metric",
    "compute_metric",
    "derived_values",
    "inputs_of",
    "is_percent",
    "metric_for_key",
    "metrics_in_context",
]


@dataclass(frozen=True, slots=True)
class Metric:
    """One CGM statistic and the tokens that denote it in data and in prose.

    ``derives_from`` names the metrics this one is computed from and ``formula``
    computes it from their values. Together they let the guard check a claim
    against the metric's *recomputed* value, not just a stored one, and identify a
    number that is actually one of the metric's inputs (SD reported as CV).
    """

    canonical: str
    key_aliases: frozenset[str]
    prose_aliases: tuple[str, ...]
    is_percent: bool = False
    derives_from: tuple[str, ...] = ()
    formula: Callable[[dict[str, float]], float] | None = field(default=None, compare=False)


#: The ontology. Order matters only for deterministic iteration; resolution is by
#: exact key or by longest prose alias, not by position. Prose aliases are kept
#: specific to avoid cross-metric collisions (e.g. "above 250" only for tar250).
METRICS: tuple[Metric, ...] = (
    Metric("mean", frozenset({"mean", "mean_glucose", "avg", "average", "average_glucose"}),
           ("mean glucose", "average glucose", "mean", "average")),
    Metric("sd", frozenset({"sd", "std", "stdev", "standard_deviation"}),
           ("standard deviation", "std dev", "sd")),
    # CV is a coefficient: SD normalized by the mean, as a percent. The formula
    # lets the guard recompute the true CV from sd + mean, so an SD reported as a
    # CV is caught even when no CV value was computed to compare against.
    Metric("cv", frozenset({"cv", "cv_pct", "coefficient_of_variation"}),
           ("coefficient of variation", "glycemic variability", "cv"), is_percent=True,
           derives_from=("sd", "mean"),
           formula=lambda v: 100.0 * v["sd"] / v["mean"]),
    Metric("gmi", frozenset({"gmi", "gmi_pct", "a1c", "ea1c", "estimated_a1c", "eag"}),
           ("glucose management indicator", "estimated a1c", "gmi", "hba1c", "a1c"),
           is_percent=True),
    Metric("tir", frozenset({"tir", "tir_pct", "time_in_range"}),
           ("time in range", "in range", "tir"), is_percent=True),
    Metric("tar", frozenset({"tar", "tar_pct", "tar180", "time_above_range"}),
           ("time above range", "above range", "percent time in hyperglycemia"),
           is_percent=True),
    Metric("tar250", frozenset({"tar250", "tar250_pct", "severe_hyper", "pct_very_high"}),
           ("above 250", "severe hyperglycemia", "very high"), is_percent=True),
    Metric("tbr", frozenset({"tbr", "tbr_pct", "tbr70", "time_below_range"}),
           ("time below range", "below 70", "time in hypoglycemia", "percent time in hypoglycemia"),
           is_percent=True),
    Metric("tbr54", frozenset({"tbr54", "tbr54_pct", "severe_hypo", "pct_very_low"}),
           ("below 54", "severe hypoglycemia", "very low"), is_percent=True),
    Metric("max", frozenset({"max", "maximum", "max_glucose", "peak"}),
           ("maximum", "highest", "peak", "max glucose")),
    Metric("min", frozenset({"min", "minimum", "min_glucose", "nadir"}),
           ("minimum", "lowest", "nadir", "min glucose")),
    Metric("overnight_mean", frozenset({"overnight_mean", "nocturnal_mean"}),
           ("overnight mean", "overnight average", "nocturnal mean")),
    Metric("dinner_max", frozenset({"dinner_max", "dinner_maximum"}),
           ("dinner", "during dinner")),
    Metric("longest_hyper", frozenset({"longest_hyper_min", "longest_hyperglycemia"}),
           ("longest continuous", "longest time in hyperglycemia", "longest stretch")),
    Metric("num_hypo", frozenset({"n_lows", "num_hypo", "n_hypo", "hypo_episodes"}),
           ("number of hypoglycemia", "hypoglycemic episodes", "separate lows",
            "times i experienced hypoglycemia")),
    # Klonoff Glycemia Risk Index. gri and its components are 0-100 scores, not
    # ratios (never cited as fractions). Banded gri_low/gri_high are percent-of-time
    # but exclude the severe bands, so they stay distinct from cumulative tbr/tar;
    # very-low/very-high map onto tbr54/tar250 above.
    # GRI = min(100, 3.0*hypo_component + 1.6*hyper_component), the weighting in
    # toolkit.glycemia_risk_index (Klonoff). Deriving it catches a single
    # component reported as the whole index.
    Metric("gri", frozenset({"gri", "gri_score", "glycemia_risk_index", "glycemic_risk_index"}),
           ("glycemia risk index", "glycemic risk index", "gri"),
           derives_from=("gri_hypo_component", "gri_hyper_component"),
           formula=lambda v: min(
               100.0, 3.0 * v["gri_hypo_component"] + 1.6 * v["gri_hyper_component"]
           )),
    Metric("gri_hypo_component", frozenset({"gri_hypo_component", "hypo_component"}),
           ("hypoglycemia component", "hypo component")),
    Metric("gri_hyper_component", frozenset({"gri_hyper_component", "hyper_component"}),
           ("hyperglycemia component", "hyper component")),
    Metric("gri_low", frozenset({"gri_low", "pct_low"}),
           ("gri low band", "banded low"), is_percent=True),
    Metric("gri_high", frozenset({"gri_high", "pct_high"}),
           ("gri high band", "banded high"), is_percent=True),
)

_KEY_TO_METRIC: dict[str, str] = {k: m.canonical for m in METRICS for k in m.key_aliases}
_BY_CANONICAL: dict[str, Metric] = {m.canonical: m for m in METRICS}

# (alias, canonical, needs_word_boundary) sorted longest-first so a specific
# multi-word phrase wins over a short abbreviation sharing a substring.
_PROSE_INDEX: tuple[tuple[str, str, bool], ...] = tuple(sorted(
    ((alias, m.canonical, len(alias) <= 3) for m in METRICS for alias in m.prose_aliases),
    key=lambda t: -len(t[0]),
))

_IS_PERCENT: dict[str, bool] = {m.canonical: m.is_percent for m in METRICS}


def is_percent(canonical: str) -> bool:
    """Whether a canonical metric is a ratio/percentage (for fraction-vs-% matching)."""
    return _IS_PERCENT.get(canonical, False)


def metric_for_key(key: str) -> str | None:
    """Resolve an evidence-dict key to a canonical metric, or ``None``.

    Matches the whole key first, then its final ``_``-segment path component, so
    both ``cv_pct`` and a wrapped ``glucose_stats.cv_pct`` resolve to ``cv``.
    """
    norm = key.strip().lower()
    if norm in _KEY_TO_METRIC:
        return _KEY_TO_METRIC[norm]
    tail = norm.rsplit(".", 1)[-1]
    return _KEY_TO_METRIC.get(tail)


def inputs_of(canonical: str) -> tuple[str, ...]:
    """The metrics ``canonical`` is derived from, empty when it has no formula."""
    metric = _BY_CANONICAL.get(canonical)
    return metric.derives_from if metric else ()


def compute_metric(canonical: str, values_by_metric: dict[str, list[float]]) -> float | None:
    """Recompute a metric from its inputs, or ``None`` when it has no formula or an
    input is missing.

    Takes one representative value per input (the first recorded). A division by
    zero or malformed input yields ``None`` rather than raising, so the guard
    degrades to its stored-value check instead of failing.
    """
    metric = _BY_CANONICAL.get(canonical)
    if metric is None or metric.formula is None:
        return None
    inputs: dict[str, float] = {}
    for name in metric.derives_from:
        values = values_by_metric.get(name)
        if not values:
            return None
        inputs[name] = values[0]
    try:
        return float(metric.formula(inputs))
    except (ArithmeticError, KeyError, ValueError, TypeError):
        return None


_DERIVABLE: tuple[str, ...] = tuple(m.canonical for m in METRICS if m.formula is not None)


def derived_values(values_by_metric: dict[str, list[float]]) -> list[float]:
    """Every metric value derivable from the evidence (CV from sd+mean, GRI from
    its components). A faithful answer may legitimately cite these even when no
    tool emitted them, so they belong in the traceability pool."""
    out: list[float] = []
    for canonical in _DERIVABLE:
        value = compute_metric(canonical, values_by_metric)
        if value is not None:
            out.append(value)
    return out


def metrics_in_context(window: str) -> set[str]:
    """Canonical metrics named in a snippet of prose (a number's neighbourhood).

    Short aliases (<= 3 chars, e.g. "cv", "sd", "tir") require word boundaries so
    they do not fire inside unrelated words; longer phrases match as substrings.
    """
    low = window.lower()
    found: set[str] = set()
    for alias, canonical, boundary in _PROSE_INDEX:
        if boundary:
            if re.search(rf"\b{re.escape(alias)}\b", low):
                found.add(canonical)
        elif alias in low:
            found.add(canonical)
    return found
