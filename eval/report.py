"""Emit the headline eval table (spec §14)."""

from __future__ import annotations

import json

from eval.metrics.e1_faithfulness import E1FaithfulnessResult
from eval.metrics.e4_null_fdr import E4NullResult
from eval.metrics.e5_perturbation import E5PerturbationResult

__all__ = [
    "E1FaithfulnessResult",
    "E4NullResult",
    "E5PerturbationResult",
    "e1_row",
    "e4_row",
    "e5_row",
    "render_json",
    "render_markdown",
]


def render_markdown(rows: dict[str, object]) -> str:
    """Render a minimal markdown results table."""
    lines = ["| Metric | Value |", "| --- | --- |"]
    for key, value in rows.items():
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def render_json(rows: dict[str, object]) -> str:
    return json.dumps(rows, indent=2, sort_keys=True)


def e4_row(result: E4NullResult) -> dict[str, object]:
    return {
        "E4_null_datasets": result.n_datasets,
        "E4_false_discoveries": result.n_false_discoveries,
        "E4_empirical_fdr": round(result.empirical_fdr, 4),
        "E4_alpha": result.alpha,
        "E4_passed": result.passed,
    }


def e1_row(result: E1FaithfulnessResult) -> dict[str, object]:
    return {
        "E1_n_per_class": result.n_faithful,
        "E1_catch_rate_fabricated": round(result.catch_rate_fabricated, 4),
        "E1_catch_rate_miscontextualized": round(result.catch_rate_miscontextualized, 4),
        "E1_false_rejection_rate": round(result.false_rejection_rate, 4),
        "E1_catch_target": result.catch_target,
        "E1_false_reject_target": result.false_reject_target,
        "E1_passed": result.passed,
    }


def e5_row(result: E5PerturbationResult) -> dict[str, object]:
    return {
        "E5_n_days": result.n_days,
        "E5_clean_kinds": len(result.clean_kinds),
        "E5_min_jaccard": round(result.min_jaccard, 4),
        "E5_total_new_kinds": result.total_new_kinds,
        "E5_jaccard_target": result.jaccard_target,
        "E5_per_corruption_jaccard": {
            r.name: round(r.jaccard, 4) for r in result.corruptions
        },
        "E5_passed": result.passed,
    }
