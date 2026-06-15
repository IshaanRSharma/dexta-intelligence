"""Direct unit tests for the faithfulness guard — the central safety module.

`guard.faithfulness.audit` is the honesty mechanism the whole thesis rests on;
it deserves a fast, deterministic suite over its edge cases (number extraction,
tolerance, allowed constants, sign/rounding, citations) independent of the E1
eval that exercises it end to end.
"""

from __future__ import annotations

from dexta_intelligence.guard.faithfulness import (
    DEFAULT_ALLOWED_CONSTANTS,
    audit,
    extract_numbers,
)

# ── extract_numbers (the evidence pool builder) ───────────────────────────────


def test_extract_walks_nested_structures() -> None:
    pool = extract_numbers({"a": [1, 2, {"b": 3.5}], "c": "peak 246 mg/dL"})
    assert set(pool) == {1.0, 2.0, 3.5, 246.0}


def test_extract_excludes_bools() -> None:
    # bools are not citable figures even though they're int subclasses
    assert extract_numbers({"flag": True, "n": 5}) == [5.0]


def test_extract_ignores_non_finite() -> None:
    assert extract_numbers(float("inf")) == []


# ── audit: the gate ───────────────────────────────────────────────────────────


def test_traceable_number_passes() -> None:
    report = audit("Your peak was 246 mg/dL.", {"peak": 246})
    assert report.ok and not report.violations


def test_fabricated_number_is_flagged() -> None:
    report = audit("Your peak was 999 mg/dL.", {"peak": 246})
    assert not report.ok
    assert report.violations[0].number == 999.0
    assert report.violations[0].nearest_pool_value == 246.0


def test_rounding_within_tolerance_passes() -> None:
    # 246.3 vs pool 246 → |0.3| <= max(0.05*246, 1)
    assert audit("peak 246.3", {"peak": 246}).ok


def test_sign_formatting_does_not_false_reject() -> None:
    # prose "-0.4" vs pool 0.4 — magnitudes compared, so the minus is fine
    assert audit("the delta was -0.4", {"delta": 0.4}).ok


def test_abs_floor_allows_tiny_pool_values() -> None:
    # cited 1.4 vs pool 0.5: within the abs_floor (1.0), not the rel tolerance
    assert audit("about 1.4 units", {"x": 0.5}).ok


def test_allowed_constants_pass_without_pool() -> None:
    # clinical/clock constants need no evidence
    report = audit("In range 70-180 over 24 hours.", {})
    assert report.ok
    assert {70, 180, 24} <= DEFAULT_ALLOWED_CONSTANTS


def test_prose_without_numbers_passes_trivially() -> None:
    report = audit("The pattern looks like late meal insulin.", {})
    assert report.ok and report.n_numbers_checked == 0


def test_multiple_violations_all_reported() -> None:
    report = audit("Saw 999 then 888.", {"peak": 246})
    assert not report.ok
    assert {v.number for v in report.violations} == {999.0, 888.0}
    assert report.n_numbers_checked == 2


def test_documented_limit_miscontextualized_number_still_passes() -> None:
    # set-membership, not semantic: a real pool number cited in the wrong place
    # passes the guard (E2 consistency invariants exist for that class).
    assert audit("carbs were 246 g", {"glucose_peak": 246}).ok


def test_list_of_texts_is_joined_and_audited() -> None:
    report = audit(["peak 246", "and a stray 777"], {"peak": 246})
    assert not report.ok
    assert report.violations[0].number == 777.0


def test_report_is_falsy_when_unfaithful() -> None:
    assert not audit("999", {"peak": 246})
    assert audit("246", {"peak": 246})
