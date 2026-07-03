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
