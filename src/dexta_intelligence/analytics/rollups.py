"""Daily rollup computation - pure, deterministic, stdlib-only.

Turns one UTC day of timeline events into a
:class:`~dexta_intelligence.models.Rollup` row. The sync workflow
(``workflows.sync``) calls this after every ingest; nothing here touches
storage, clocks, or randomness, so identical events always produce an
identical rollup.

Glycemic bands (mg/dL, international consensus; Battelino et al. 2019)
----------------------------------------------------------------------
- ``tir``  - % of readings in the target range [70, 180], bounds inclusive.
- ``tbr``  - % below 70 (includes very-low readings).
- ``tbr2`` - % below 54 (very low).
- ``tar``  - % above 180 (includes very-high readings).
- ``tar2`` - % above 250 (very high).

So ``tir + tar + tbr == 100`` for any non-empty day. The target band
defaults match :class:`~dexta_intelligence.config.AnalysisConfig`
(``target_low=70`` / ``target_high=180``); callers holding a config can pass
overrides.

House philosophy: a metric the data cannot support is ``None``, never a
fabricated ``0.0``. An empty day yields no rollup at all; a single-reading
day has ``sd is None`` and ``cv is None``.
"""

from __future__ import annotations

import statistics
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from dexta_intelligence.models import InsulinKind, Rollup, RollupPeriod

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date

    from dexta_intelligence.models import GlucoseEvent, InsulinEvent, MealEvent

__all__ = [
    "EXPECTED_READINGS_PER_DAY",
    "TARGET_HIGH_MG_DL",
    "TARGET_LOW_MG_DL",
    "TIGHT_HIGH_MG_DL",
    "VERY_HIGH_MG_DL",
    "VERY_LOW_MG_DL",
    "coverage_fraction",
    "daily_rollup",
]

#: Standard T1D target range (mg/dL); mirrors ``AnalysisConfig`` defaults.
TARGET_LOW_MG_DL = 70
TARGET_HIGH_MG_DL = 180
#: Upper bound of the "tight" range 70-140 (time-in-tight-range reporting).
TIGHT_HIGH_MG_DL = 140
#: Clinically very low (level-2 hypoglycemia) threshold.
VERY_LOW_MG_DL = 54
#: Clinically very high (level-2 hyperglycemia) threshold.
VERY_HIGH_MG_DL = 250

#: Expected readings per UTC day at the native 5-minute CGM cadence.
EXPECTED_READINGS_PER_DAY = 288

#: GMI (Bergenstal et al. 2018): GMI(%) = 3.31 + 0.02392 x mean mg/dL.
_GMI_INTERCEPT = 3.31
_GMI_SLOPE = 0.02392


def coverage_fraction(n_readings: int, *, expected: int = EXPECTED_READINGS_PER_DAY) -> float:
    """Fraction of expected CGM slots present, clamped to [0.0, 1.0].

    Duplicate or overlapping-sensor readings can push the raw ratio above
    1.0; coverage is capped because "more than complete" is meaningless.
    """
    if n_readings <= 0:
        return 0.0
    return min(1.0, n_readings / expected)


def daily_rollup(
    day: date,
    glucose: Sequence[GlucoseEvent],
    *,
    insulin: Sequence[InsulinEvent] = (),
    meals: Sequence[MealEvent] = (),
    target_low: int = TARGET_LOW_MG_DL,
    target_high: int = TARGET_HIGH_MG_DL,
) -> Rollup | None:
    """Compute the daily :class:`Rollup` for one UTC calendar day.

    Events are filtered to ``day`` defensively (timestamps are UTC by model
    validation), so callers may pass wider windows without skewing the
    result. Returns ``None`` when the day has no glucose readings - an
    absent day is not a day of zeros.

    Insulin totals: ``bolus_units`` sums every bolus (manual and automatic
    SMB alike); ``basal_units`` sums basal and temp-basal deliveries;
    suspends deliver nothing and are ignored. Totals are ``None`` when the
    day has no events of that kind with a known dose.

    ``gmi`` is the standard affine map of the day's mean; note the GMI
    formula was derived on >=14-day windows, so single-day values are
    indicative only. ``excursion_count`` is the number of contiguous
    out-of-range runs (a direct swing from above-range to below-range
    counts as two excursions).
    """
    readings = sorted((g for g in glucose if g.ts.date() == day), key=lambda g: g.ts)
    if not readings:
        return None

    values = [g.mg_dl for g in readings]
    n = len(values)
    mean = statistics.fmean(values)
    sd = statistics.stdev(values, mean) if n >= 2 else None
    cv = (sd / mean) * 100.0 if sd is not None else None

    def pct(count: int) -> float:
        return 100.0 * count / n

    tbr2_count = sum(1 for v in values if v < VERY_LOW_MG_DL)
    tbr_count = sum(1 for v in values if v < target_low)
    tar2_count = sum(1 for v in values if v > VERY_HIGH_MG_DL)
    tar_count = sum(1 for v in values if v > target_high)

    return Rollup(
        period=RollupPeriod.DAILY,
        period_start=datetime(day.year, day.month, day.day, tzinfo=UTC),
        n=n,
        mean=mean,
        sd=sd,
        cv=cv,
        tir=pct(n - tbr_count - tar_count),
        tar=pct(tar_count),
        tar2=pct(tar2_count),
        tbr=pct(tbr_count),
        tbr2=pct(tbr2_count),
        gmi=_GMI_INTERCEPT + _GMI_SLOPE * mean,
        excursion_count=_excursion_count(values, target_low, target_high),
        bolus_units=_insulin_total(insulin, day, (InsulinKind.BOLUS,)),
        basal_units=_insulin_total(insulin, day, (InsulinKind.BASAL, InsulinKind.TEMP_BASAL)),
        carbs_g=_carb_total(meals, day),
    )


def _excursion_count(values: Sequence[int], low: int, high: int) -> int:
    """Count contiguous out-of-range runs in a time-ordered series.

    Each maximal run of consecutive readings on the same side of the target
    range (all below ``low``, or all above ``high``) is one excursion.
    """
    count = 0
    prev = 0
    for v in values:
        state = -1 if v < low else (1 if v > high else 0)
        if state not in (0, prev):
            count += 1
        prev = state
    return count


def _insulin_total(
    insulin: Sequence[InsulinEvent], day: date, kinds: tuple[InsulinKind, ...]
) -> float | None:
    """Sum dosed units for the given kinds on ``day``; ``None`` if no doses."""
    doses = [
        e.units
        for e in insulin
        if e.ts.date() == day and e.kind in kinds and e.units is not None
    ]
    if not doses:
        return None
    return sum(doses)


def _carb_total(meals: Sequence[MealEvent], day: date) -> float | None:
    """Sum logged carbs on ``day``; ``None`` if no meal carries a carb count."""
    carbs = [m.carbs_g for m in meals if m.ts.date() == day and m.carbs_g is not None]
    if not carbs:
        return None
    return sum(carbs)
