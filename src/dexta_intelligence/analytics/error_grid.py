"""Clinical-accuracy error grids and MARD - pure, deterministic, stdlib-only.

Three standard tools for scoring a predicted/measured glucose against a
reference (mg/dL):

- :func:`clarke_zone` - Clarke Error Grid Analysis (Clarke et al., *Diabetes
  Care* 1987;10(5):622-628). Classifies a (reference, predicted) pair into one
  of five zones A-E by clinical consequence.
- :func:`parkes_zone` - Parkes (Consensus) Error Grid, type 1 / insulin-using
  diabetes (Parkes et al., *Diabetes Care* 2000;23(8):1143-1148; boundary
  coordinates per Pfutzner et al., *J Diabetes Sci Technol* 2013;7(5):1275-1281).
  Same A-E zone scheme, smoother clinically-derived boundaries.
- :func:`mard` - Mean Absolute Relative Difference (%), the headline CGM
  accuracy summary: mean over pairs of ``|predicted - reference| / reference``.

Zone meaning (both grids): A - clinically accurate; B - benign error (no or
benign treatment change); C - overcorrection; D - dangerous failure to detect;
E - erroneous treatment (opposite of correct). A+B is the standard "clinically
acceptable" fraction.

No I/O, no randomness, no clamping of inputs beyond what the definitions
require.
"""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "Zone",
    "clarke_zone",
    "mard",
    "parkes_zone",
    "zone_distribution",
]

Zone = Literal["A", "B", "C", "D", "E"]


def clarke_zone(reference: float, predicted: float) -> Zone:
    """Clarke Error Grid zone for one (reference, predicted) pair in mg/dL.

    Implements the original Clarke et al. (1987) decision rules. ``reference``
    is the true/measured BG; ``predicted`` is the estimate being scored. Both
    must be positive.

    Zone A is a perfect prediction's home: ``predicted == reference`` always
    returns ``"A"``.
    """
    if reference <= 0.0 or predicted <= 0.0:
        msg = f"reference and predicted must be positive, got {reference}, {predicted}"
        raise ValueError(msg)

    # Canonical Clarke decision tree (Clarke et al. 1987). Order matters: the
    # accurate zone A is tested first, then the two erroneous-treatment wedges
    # (E), the overcorrection wedges (C), the failure-to-detect wedges (D), and
    # everything else is the benign zone B.

    # Zone A: within 20% of reference, or both readings <= 70 (hypo agreement).
    if (predicted <= 70.0 and reference <= 70.0) or abs(predicted - reference) <= 0.2 * reference:
        return "A"

    # Zone E: opposite treatment - true hyper read as hypo, or true hypo read
    # as hyper.
    if (reference >= 180.0 and predicted <= 70.0) or (reference <= 70.0 and predicted >= 180.0):
        return "E"

    # Zone C: overcorrection.
    if (70.0 <= reference <= 290.0 and predicted >= reference + 110.0) or (
        130.0 <= reference <= 180.0 and predicted <= (7.0 / 5.0) * reference - 182.0
    ):
        return "C"

    # Zone D: failure to detect a value that needs treatment - reference out of
    # range, prediction inside the target range.
    if (reference > 240.0 and 70.0 <= predicted <= 180.0) or (
        reference < 70.0 and 70.0 <= predicted <= 180.0
    ):
        return "D"

    # Zone B: benign error (everything remaining).
    return "B"


# Parkes type-1 zone boundaries are defined as polylines in the (reference,
# predicted) plane. A point's zone is the most severe region it falls into.
# Boundary coordinates from Pfutzner et al. (2013), Table 1 (type 1 diabetes).

# Upper boundaries (predicted above reference): crossing a line moves the point
# into the next-worse zone above.
_PARKES_T1_UP_AB: tuple[tuple[float, float], ...] = (
    (0.0, 50.0),
    (30.0, 50.0),
    (140.0, 170.0),
    (280.0, 380.0),
    (430.0, 550.0),
)
_PARKES_T1_UP_BC: tuple[tuple[float, float], ...] = (
    (0.0, 60.0),
    (30.0, 60.0),
    (50.0, 80.0),
    (70.0, 110.0),
    (260.0, 550.0),
)
_PARKES_T1_UP_CD: tuple[tuple[float, float], ...] = (
    (0.0, 100.0),
    (25.0, 100.0),
    (50.0, 125.0),
    (80.0, 215.0),
    (125.0, 550.0),
)
_PARKES_T1_UP_DE: tuple[tuple[float, float], ...] = (
    (0.0, 150.0),
    (35.0, 155.0),
    (50.0, 550.0),
)

# Lower boundaries (predicted below reference): crossing moves to next-worse
# zone below.
_PARKES_T1_DOWN_AB: tuple[tuple[float, float], ...] = (
    (50.0, 0.0),
    (50.0, 30.0),
    (170.0, 145.0),
    (385.0, 300.0),
    (550.0, 450.0),
)
_PARKES_T1_DOWN_BC: tuple[tuple[float, float], ...] = (
    (120.0, 0.0),
    (120.0, 30.0),
    (260.0, 130.0),
    (550.0, 250.0),
)
_PARKES_T1_DOWN_CD: tuple[tuple[float, float], ...] = (
    (250.0, 0.0),
    (250.0, 40.0),
    (550.0, 150.0),
)


def _interp_y(polyline: Sequence[tuple[float, float]], x: float) -> float:
    """Predicted-axis value of ``polyline`` at reference ``x``.

    Clamps to the polyline's endpoints outside its x-range, matching how the
    Parkes boundaries extend flat past their defined extent.
    """
    if x <= polyline[0][0]:
        return polyline[0][1]
    if x >= polyline[-1][0]:
        return polyline[-1][1]
    for (x0, y0), (x1, y1) in pairwise(polyline):
        if x0 <= x <= x1:
            if x1 == x0:
                return y0
            frac = (x - x0) / (x1 - x0)
            return y0 + frac * (y1 - y0)
    return polyline[-1][1]


# Each entry pairs a boundary polyline with the zone above it; the first whose
# boundary the prediction exceeds wins (most severe first).
_PARKES_T1_UPPER: tuple[tuple[tuple[tuple[float, float], ...], Zone], ...] = (
    (_PARKES_T1_UP_DE, "E"),
    (_PARKES_T1_UP_CD, "D"),
    (_PARKES_T1_UP_BC, "C"),
    (_PARKES_T1_UP_AB, "B"),
)
_PARKES_T1_LOWER: tuple[tuple[tuple[tuple[float, float], ...], Zone], ...] = (
    (_PARKES_T1_DOWN_CD, "D"),
    (_PARKES_T1_DOWN_BC, "C"),
    (_PARKES_T1_DOWN_AB, "B"),
)


def parkes_zone(reference: float, predicted: float) -> Zone:
    """Parkes (Consensus) Error Grid zone, type 1, for one pair in mg/dL.

    ``reference`` is the true BG, ``predicted`` the estimate. The plane is
    partitioned by clinically-derived polylines; a point's zone is the most
    severe region it lands in. A perfect prediction (on the identity line) is
    always zone A. Both inputs must be positive.
    """
    if reference <= 0.0 or predicted <= 0.0:
        msg = f"reference and predicted must be positive, got {reference}, {predicted}"
        raise ValueError(msg)

    if predicted > reference:
        for polyline, zone in _PARKES_T1_UPPER:
            if predicted > _interp_y(polyline, reference):
                return zone
        return "A"

    for polyline, zone in _PARKES_T1_LOWER:
        if predicted < _interp_y(polyline, reference):
            return zone
    return "A"


def mard(reference: Sequence[float], predicted: Sequence[float]) -> float:
    """Mean Absolute Relative Difference (%) over paired readings.

    ``MARD = mean( |predicted_i - reference_i| / reference_i ) * 100``. Equal
    sequences → 0.0. Both sequences must be the same non-zero length and every
    reference must be positive (relative difference is undefined at zero).
    """
    if len(reference) != len(predicted):
        msg = (
            f"reference and predicted must be equal length, "
            f"got {len(reference)} and {len(predicted)}"
        )
        raise ValueError(msg)
    if not reference:
        msg = "MARD requires at least one paired reading"
        raise ValueError(msg)

    total = 0.0
    for ref, pred in zip(reference, predicted, strict=True):
        if ref <= 0.0:
            msg = f"reference values must be positive for MARD, got {ref}"
            raise ValueError(msg)
        total += abs(pred - ref) / ref
    return 100.0 * total / len(reference)


def zone_distribution(
    reference: Sequence[float],
    predicted: Sequence[float],
    *,
    grid: Literal["clarke", "parkes"] = "clarke",
) -> dict[Zone, float]:
    """Fraction of paired readings in each zone A-E for the chosen grid.

    Returns a dict with every zone key present (zeros included), summing to 1.0
    for a non-empty input.
    """
    if len(reference) != len(predicted):
        msg = "reference and predicted must be equal length"
        raise ValueError(msg)
    if not reference:
        msg = "zone_distribution requires at least one paired reading"
        raise ValueError(msg)

    classify = clarke_zone if grid == "clarke" else parkes_zone
    counts: dict[Zone, int] = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
    for ref, pred in zip(reference, predicted, strict=True):
        counts[classify(ref, pred)] += 1
    n = len(reference)
    return {zone: count / n for zone, count in counts.items()}
