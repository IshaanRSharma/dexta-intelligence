"""Emit the headline eval table."""

from __future__ import annotations

import json

from eval.metrics.e1_faithfulness import E1FaithfulnessResult
from eval.metrics.e2_power import E2PowerResult
from eval.metrics.e3_accuracy import E3AccuracyResult
from eval.metrics.e4_null_fdr import E4NullResult
from eval.metrics.e5_perturbation import E5PerturbationResult
from eval.metrics.e6_attribution import E6AttributionResult
from eval.metrics.e6_faithfulness import E6FaithfulnessResult
from eval.metrics.e6_safety import E6SafetyResult
from eval.metrics.e7_reasoning import E7ReasoningResult
from eval.metrics.e_consensus import EConsensusResult

__all__ = [
    "E1FaithfulnessResult",
    "E2PowerResult",
    "E3AccuracyResult",
    "E4NullResult",
    "E5PerturbationResult",
    "E6AttributionResult",
    "E6FaithfulnessResult",
    "E6SafetyResult",
    "E7ReasoningResult",
    "EConsensusResult",
    "e1_row",
    "e2_row",
    "e3_row",
    "e4_row",
    "e5_row",
    "e6_attribution_row",
    "e6_faithfulness_row",
    "e6_safety_row",
    "e7_reasoning_row",
    "e_consensus_row",
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


def e2_row(result: E2PowerResult) -> dict[str, object]:
    return {
        "E2_best_recall": round(result.best_recall, 4),
        "E2_recall_target": result.recall_target,
        "E2_cells": {
            f"{c.scenario}@{c.effect_size:g}mgdl/{c.n_days}d": round(c.recall, 4)
            for c in result.cells
        },
        "E2_passed": result.passed,
    }


def e3_row(result: E3AccuracyResult) -> dict[str, object]:
    return {
        "E3_n_pairs": result.n_pairs,
        "E3_horizon_min": result.horizon_min,
        "E3_mard_pct": round(result.mard_pct, 2),
        "E3_clarke_zones": result.clarke,
        "E3_clarke_ab_pct": round(result.clarke_ab_pct, 2),
        "E3_parkes_zones": result.parkes,
        "E3_parkes_ab_pct": round(result.parkes_ab_pct, 2),
    }


def e_consensus_row(result: EConsensusResult) -> dict[str, object]:
    return {
        "Econsensus_n_days": result.n_days,
        "Econsensus_n_checks": result.n_checks,
        "Econsensus_n_disagreements": result.n_disagreements,
        "Econsensus_passed": result.passed,
    }


def e5_row(result: E5PerturbationResult) -> dict[str, object]:
    return {
        "E5_n_days": result.n_days,
        "E5_clean_kinds": len(result.clean_kinds),
        "E5_min_jaccard": round(result.min_jaccard, 4),
        "E5_total_new_kinds": result.total_new_kinds,
        "E5_jaccard_target": result.jaccard_target,
        "E5_per_corruption_jaccard": {r.name: round(r.jaccard, 4) for r in result.corruptions},
        "E5_passed": result.passed,
    }


def e6_attribution_row(result: E6AttributionResult) -> dict[str, object]:
    return {
        "E6_attribution_accuracy": round(result.accuracy, 4),
        "E6_attribution_target": result.accuracy_target,
        "E6_attribution_cells": {c.scenario: c.hit for c in result.cells},
        "E6_attribution_passed": result.passed,
    }


def e6_safety_row(result: E6SafetyResult) -> dict[str, object]:
    return {
        "E6_safety_prompts": result.n_prompts,
        "E6_safety_violations": result.violations,
        "E6_safety_violation_rate": round(result.violation_rate, 4),
        "E6_safety_passed": result.passed,
    }


def e6_faithfulness_row(result: E6FaithfulnessResult) -> dict[str, object]:
    return {
        "E6_faithfulness_rate": round(result.faithful_rate, 4),
        "E6_faithfulness_target": result.faithful_target,
        "E6_faithfulness_passed": result.passed,
    }


def e7_reasoning_row(result: E7ReasoningResult) -> dict[str, object]:
    return {
        "E7_mean_coverage": round(result.mean_coverage, 4),
        "E7_coverage_target": result.coverage_target,
        "E7_mean_efficiency": round(result.mean_efficiency, 4),
        "E7_soundness_rate": round(result.soundness_rate, 4),
        "E7_soundness_target": result.soundness_target,
        "E7_gap_handling_rate": round(result.gap_handling_rate, 4),
        "E7_cells": {
            c.scenario: {
                "coverage": round(c.coverage, 2),
                "probes": c.probes,
                "sound": c.sound,
            }
            for c in result.cells
        },
        "E7_passed": result.passed,
    }
