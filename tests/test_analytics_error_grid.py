"""Known-point zone and MARD checks for the clinical-accuracy error grids."""

from __future__ import annotations

import pytest

from dexta_intelligence.analytics.error_grid import (
    clarke_zone,
    mard,
    parkes_zone,
    zone_distribution,
)


def test_perfect_prediction_is_zone_a_both_grids() -> None:
    for bg in (45.0, 70.0, 100.0, 150.0, 250.0, 380.0):
        assert clarke_zone(bg, bg) == "A"
        assert parkes_zone(bg, bg) == "A"


def test_clarke_within_20_percent_is_zone_a() -> None:
    # 100 -> 115 is +15%, inside the 20% band.
    assert clarke_zone(100.0, 115.0) == "A"
    assert clarke_zone(200.0, 170.0) == "A"


def test_clarke_both_hypo_is_zone_a() -> None:
    # Both <= 70: agreement in the hypo band even if relative error is large.
    assert clarke_zone(65.0, 50.0) == "A"


def test_clarke_zone_e_opposite_treatment() -> None:
    # True hyper predicted as hypo, and the reverse.
    assert clarke_zone(260.0, 50.0) == "E"
    assert clarke_zone(50.0, 260.0) == "E"


def test_clarke_zone_d_failure_to_detect() -> None:
    # Reference very high but predicted in-range -> no treatment when needed.
    assert clarke_zone(300.0, 150.0) == "D"
    # Reference hypo but predicted in-range.
    assert clarke_zone(50.0, 120.0) == "D"


def test_clarke_zone_c_overcorrection() -> None:
    # Reference in range, prediction wildly high -> would overtreat.
    assert clarke_zone(120.0, 240.0) == "C"


def test_clarke_zone_b_benign() -> None:
    # 25% off but not into C/D/E territory.
    assert clarke_zone(150.0, 200.0) == "B"


def test_parkes_zone_e_extreme_opposite() -> None:
    # True hypo (~35) predicted as hyper (~200): above the DE boundary -> E.
    assert parkes_zone(35.0, 200.0) == "E"


def test_parkes_more_lenient_than_clarke_near_target() -> None:
    # A modest over-read near the target stays in A/B for Parkes.
    assert parkes_zone(100.0, 120.0) in {"A", "B"}


def test_parkes_severe_under_read() -> None:
    # True very high, predicted low -> dangerous (D or worse).
    assert parkes_zone(350.0, 70.0) in {"D", "E"}


def test_mard_zero_for_identical() -> None:
    ref = [80.0, 120.0, 200.0]
    assert mard(ref, list(ref)) == 0.0


def test_mard_known_value() -> None:
    # Each prediction 10% high -> MARD = 10%.
    ref = [100.0, 200.0, 50.0]
    pred = [110.0, 220.0, 55.0]
    assert mard(ref, pred) == pytest.approx(10.0)


def test_mard_mixed() -> None:
    ref = [100.0, 100.0]
    pred = [110.0, 80.0]  # +10%, -20%
    assert mard(ref, pred) == pytest.approx(15.0)


def test_mard_rejects_mismatched_length() -> None:
    with pytest.raises(ValueError, match="equal length"):
        mard([100.0], [100.0, 110.0])


def test_mard_rejects_nonpositive_reference() -> None:
    with pytest.raises(ValueError, match="positive"):
        mard([0.0], [100.0])


def test_zone_distribution_sums_to_one() -> None:
    ref = [100.0, 100.0, 300.0, 50.0]
    pred = [100.0, 130.0, 150.0, 260.0]
    dist = zone_distribution(ref, pred, grid="clarke")
    assert set(dist) == {"A", "B", "C", "D", "E"}
    assert sum(dist.values()) == pytest.approx(1.0)


def test_zone_distribution_perfect_all_a() -> None:
    ref = [80.0, 150.0, 250.0]
    dist = zone_distribution(ref, list(ref), grid="parkes")
    assert dist["A"] == pytest.approx(1.0)


def test_zone_arguments_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        clarke_zone(0.0, 100.0)
    with pytest.raises(ValueError, match="positive"):
        parkes_zone(100.0, -1.0)
