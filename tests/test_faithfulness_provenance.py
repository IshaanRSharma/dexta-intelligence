"""Provenance layer of the faithfulness guard: right-number-wrong-metric catch.

The membership guard (tested in test_faithfulness.py) passes any number that
exists in the evidence pool. This suite covers the opt-in provenance layer, which
additionally requires a number cited next to a metric name to match *that* metric,
closing the "cited the standard deviation as the coefficient of variation" hole
the LLM-CGM benchmark found a code-execution agent still falls into.

Two invariants are load-bearing and tested explicitly:
- provenance is OFF by default, so no existing caller changes behaviour;
- it only fires when the ontology resolves both the claimed metric and a
  conflicting one, so it is precise (no false positives on ambiguous prose).
"""

from __future__ import annotations

import pytest

from dexta_intelligence.guard.faithfulness import audit, build_provenance
from dexta_intelligence.guard.metrics import metric_for_key, metrics_in_context

# A realistic tool-result evidence bundle (keys as the glucose tool emits them).
EV = {
    "glucose_stats_1": {
        "mean": 135.2, "sd": 38.8, "cv": 0.287, "cv_pct": 28.7,
        "tir_pct": 81.1, "tbr_pct": 4.3, "maximum": 224, "minimum": 43,
        "gmi_pct": 6.5, "overnight_mean": 148.6,
    },
    "find_lows_2": {"n_lows": 65},
}


# ── ontology ──────────────────────────────────────────────────────────────────


def test_metric_for_key_resolves_aliases() -> None:
    assert metric_for_key("cv_pct") == "cv"
    assert metric_for_key("mean_glucose") == "mean"
    assert metric_for_key("standard_deviation") == "sd"
    assert metric_for_key("maximum") == "max"
    assert metric_for_key("n_lows") == "num_hypo"


def test_metric_for_key_unknown_is_none() -> None:
    assert metric_for_key("coverage") is None
    assert metric_for_key("timestamp") is None


def test_metrics_in_context_matches_phrases() -> None:
    assert metrics_in_context("your coefficient of variation was high") == {"cv"}
    assert "mean" in metrics_in_context("the average glucose over the period")
    assert metrics_in_context("time in range 70-180") == {"tir"}


def test_metrics_in_context_short_alias_needs_word_boundary() -> None:
    # "sd" must not fire inside "Thursday"; "cv" must not fire inside "recovery"
    assert "sd" not in metrics_in_context("on thursday it improved")
    assert "cv" not in metrics_in_context("a full recovery followed")
    assert "cv" in metrics_in_context("the cv was 30%")


# ── build_provenance ──────────────────────────────────────────────────────────


def test_build_provenance_binds_numbers_to_metrics() -> None:
    prov = build_provenance(EV)
    assert prov["mean"] == [135.2]
    assert prov["sd"] == [38.8]
    assert 0.287 in prov["cv"] and 28.7 in prov["cv"]
    assert prov["max"] == [224.0]
    assert prov["num_hypo"] == [65.0]


def test_build_provenance_ignores_unknown_keys() -> None:
    prov = build_provenance({"coverage": 0.99, "mean": 120.0})
    assert "mean" in prov
    assert all(k != "coverage" for k in prov)


def test_build_provenance_propagates_key_through_lists() -> None:
    prov = build_provenance({"mean_glucose": [130.0, 140.0]})
    assert prov["mean"] == [130.0, 140.0]


# ── the wrong-metric catch ────────────────────────────────────────────────────


def test_sd_cited_as_cv_is_flagged() -> None:
    # the LLM-CGM Q8 failure: reporting the SD (38.8) as the CV
    report = audit("Your coefficient of variation was 38.8%.", EV, check_provenance=True)
    assert not report.ok
    assert not report.violations  # membership passes: 38.8 is in the pool
    assert len(report.provenance_violations) == 1
    pv = report.provenance_violations[0]
    assert pv.claimed_metric == "cv" and pv.matched_metric == "sd"


def test_mean_cited_as_maximum_is_flagged() -> None:
    report = audit("Your maximum glucose was 135.2 mg/dL.", EV, check_provenance=True)
    assert not report.ok
    assert report.provenance_violations[0].matched_metric == "mean"


def test_min_cited_as_mean_is_flagged() -> None:
    report = audit("Your mean glucose was 43 mg/dL.", EV, check_provenance=True)
    assert not report.ok
    assert report.provenance_violations[0].matched_metric == "min"


# ── no false positives ────────────────────────────────────────────────────────


def test_correct_percent_citation_passes() -> None:
    assert audit("Your coefficient of variation was 28.7%.", EV, check_provenance=True).ok


def test_correct_fraction_citation_passes() -> None:
    # CV stored as fraction (0.287); a fraction citation must also pass
    assert audit("Your coefficient of variation was 0.287.", EV, check_provenance=True).ok


def test_correct_maximum_passes() -> None:
    assert audit("Your maximum glucose was 224 mg/dL.", EV, check_provenance=True).ok


def test_gmi_proxy_for_a1c_passes() -> None:
    # eA1c is reported as GMI (6.5); the ontology maps both to the same metric
    assert audit("Your estimated A1C is 6.5%.", EV, check_provenance=True).ok


def test_number_without_named_metric_is_not_a_provenance_violation() -> None:
    # 38.8 appears, but no metric is named near it -> membership only, no prov flag
    report = audit("There were 38.8 units of something noteworthy.", EV, check_provenance=True)
    assert not report.provenance_violations


def test_unheld_claimed_metric_does_not_fire() -> None:
    # "median" is named but we hold no median value -> cannot judge -> no flag
    report = audit("Your median glucose was 224.", EV, check_provenance=True)
    assert not report.provenance_violations


def test_fabricated_number_is_membership_not_provenance() -> None:
    report = audit("Your mean glucose was 999 mg/dL.", EV, check_provenance=True)
    assert not report.ok
    assert len(report.violations) == 1
    assert not report.provenance_violations


# ── backward compatibility ────────────────────────────────────────────────────


def test_provenance_off_by_default() -> None:
    # same wrong-metric prose, default call: unchanged membership-only behaviour
    report = audit("Your coefficient of variation was 38.8%.", EV)
    assert report.ok
    assert report.provenance_violations == ()


def test_report_has_empty_provenance_by_default() -> None:
    report = audit("mean 135.2", EV)
    assert report.provenance_violations == ()
    assert bool(report) is report.ok


# ── Glycemia Risk Index (GRI) provenance ──────────────────────────────────────

# A GRI tool-result bundle (keys as glycemia_risk_index emits them). Distinct
# values so a component cited as the score (or vice versa) is caught.
GRI_EV = {
    "glycemia_risk_index_1": {
        "gri": 42.0, "hypo_component": 10.0, "hyper_component": 8.0,
        "pct_very_low": 3.0, "pct_low": 6.0, "pct_high": 12.0,
        "pct_very_high": 2.0, "n_readings": 288,
    }
}


def test_gri_keys_resolve_to_distinct_metrics() -> None:
    assert metric_for_key("gri") == "gri"
    assert metric_for_key("hypo_component") == "gri_hypo_component"
    assert metric_for_key("hyper_component") == "gri_hyper_component"
    # banded low/high are distinct from cumulative tbr/tar
    assert metric_for_key("pct_low") == "gri_low"
    assert metric_for_key("pct_high") == "gri_high"
    assert metric_for_key("tbr_pct") == "tbr"
    assert metric_for_key("tar_pct") == "tar"
    # severe bands map straight onto tbr54 / tar250
    assert metric_for_key("pct_very_low") == "tbr54"
    assert metric_for_key("pct_very_high") == "tar250"


def test_build_provenance_binds_gri_bundle() -> None:
    prov = build_provenance(GRI_EV)
    assert prov["gri"] == [42.0]
    assert prov["gri_hypo_component"] == [10.0]
    assert prov["gri_low"] == [6.0]
    assert prov["tbr54"] == [3.0]  # pct_very_low binds to the severe-hypo metric


def test_gri_score_cited_as_hypo_component_is_flagged() -> None:
    report = audit(
        "Your hypoglycemia component was 42.0.", GRI_EV, check_provenance=True
    )
    assert not report.ok
    assert not report.violations  # membership passes: 42.0 is in the pool
    pv = report.provenance_violations[0]
    assert pv.claimed_metric == "gri_hypo_component" and pv.matched_metric == "gri"


def test_component_cited_as_gri_is_flagged() -> None:
    report = audit(
        "Your glycemia risk index was 10.0.", GRI_EV, check_provenance=True
    )
    assert not report.ok
    pv = report.provenance_violations[0]
    assert pv.claimed_metric == "gri" and pv.matched_metric == "gri_hypo_component"


def test_correct_gri_citations_pass() -> None:
    assert audit("Your glycemia risk index was 42.", GRI_EV, check_provenance=True).ok
    assert audit(
        "Your hypoglycemia component was 10.", GRI_EV, check_provenance=True
    ).ok


# ── derivation: recompute a metric from its inputs, not just its stored value ──

def test_sd_reported_as_cv_caught_even_when_cv_was_never_computed() -> None:
    # Only sd + mean are in evidence; no CV was computed. The guard derives the
    # true CV (28.7) from them and flags the SD (38.8) reported as CV. The old
    # membership-only check could not: it had no CV value to compare against.
    ev = {"mean": 135.2, "sd": 38.8}
    report = audit("Your coefficient of variation is 38.8.", ev, check_provenance=True)
    assert not report.ok
    pv = report.provenance_violations[0]
    assert pv.claimed_metric == "cv" and pv.matched_metric == "sd"


def test_correct_cv_is_traceable_via_derivation() -> None:
    # 28.7 is not literally in the evidence, but it is derivable from sd+mean, so
    # a faithful answer that cites it must pass both checks.
    ev = {"mean": 135.2, "sd": 38.8}
    report = audit("Your coefficient of variation is 28.7.", ev, check_provenance=True)
    assert report.ok
    assert not report.violations and not report.provenance_violations


def test_component_reported_as_gri_caught_by_derivation() -> None:
    # No GRI stored; derived GRI = min(100, 3*0.2 + 1.6*0.7) = 1.72. A component
    # (0.7) reported as the whole index is flagged.
    ev = {"gri_hypo_component": 0.2, "gri_hyper_component": 0.7}
    report = audit("Your glycemia risk index is 0.7.", ev, check_provenance=True)
    assert not report.ok
    pv = report.provenance_violations[0]
    assert pv.claimed_metric == "gri" and pv.matched_metric == "gri_hyper_component"


def test_correct_derived_gri_passes() -> None:
    ev = {"gri_hypo_component": 0.2, "gri_hyper_component": 0.7}
    assert audit("Your glycemia risk index is 1.7.", ev, check_provenance=True).ok


def test_compute_metric_and_derived_values() -> None:
    from dexta_intelligence.guard.metrics import (  # noqa: PLC0415
        compute_metric,
        derived_values,
        inputs_of,
    )

    assert inputs_of("cv") == ("sd", "mean")
    assert inputs_of("mean") == ()  # no formula
    assert compute_metric("cv", {"sd": [38.8], "mean": [135.2]}) == pytest.approx(28.7, abs=0.05)
    assert compute_metric("cv", {"sd": [38.8]}) is None  # missing input
    assert compute_metric("cv", {"sd": [1.0], "mean": [0.0]}) is None  # div by zero -> None
    assert compute_metric("mean", {"mean": [135.2]}) is None  # no formula
    assert pytest.approx(28.7, abs=0.05) in derived_values({"sd": [38.8], "mean": [135.2]})
