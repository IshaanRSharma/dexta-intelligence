"""Pure-Python port of the core oref0 physiological math (insulin, carbs, predictions).

Provenance
----------
This module is an analytical port of the documented, MIT-licensed math from the
OpenAPS **oref0** reference implementation (https://github.com/openaps/oref0):

- ``lib/iob/calculate.js`` - exponential and bilinear insulin activity/IOB curves.
- ``lib/iob/total.js`` - DIA/peak clamping rules and IOB/activity summation.
- ``lib/determine-basal/cob.js`` - deviation calculation and carb-absorption detection.
- ``lib/meal/total.js`` - COB accounting (clamping at 0, ``maxCOB`` cap).
- ``lib/determine-basal/determine-basal.js`` - BGI, eventual BG, and the
  IOB/ZT/COB/UAM prediction curves.
- Docs: https://openaps.readthedocs.io/en/master/docs/While%20You%20Wait%20For%20Gear/Understand-determine-basal.html

oref0 is Copyright (c) 2015-2017 OpenAPS contributors, released under the MIT
license. This port is used by the Prediction Reconciliation Agent to compute an
*expected glucose trajectory* for retrospective analysis and reconciliation.
**It is NOT a dosing algorithm and must never be used to recommend or deliver
insulin.**

Faithfulness notes (where this port deviates from the JS source)
----------------------------------------------------------------
1. oref0 rounds many intermediate values for display (BGI to 2 decimals,
   deviations to 2-3 decimals, predBGs to integers clamped to [39, 401]).
   This port keeps full float precision throughout and does not clamp
   predicted BGs, since the goal is analysis rather than pump display.
2. ``cob.js`` buckets/interpolates raw CGM data into 5-minute slots and also
   considers ``currentDeviation / 2`` when computing carbs absorbed
   (``ci = max(deviation, currentDeviation/2, min_5m_carbimpact)``). This port
   expects an already-regular 5-minute glucose series and omits the
   ``currentDeviation/2`` term, keeping ``ci = max(deviation, min_5m_carbimpact)``.
3. The ``min_5m_carbimpact`` default here is 8 mg/dL/5m (the modern oref0
   preferences default). A stale comment in ``cob.js`` still says 3 mg/dL/5m,
   which was the pre-0.6.0 default.
4. Custom exponential peak times are clamped per-curve exactly as in
   ``calculate.js``: rapid-acting to [50, 120] minutes, ultra-rapid to
   [35, 100] minutes (not a single global 35-120 range).
5. Autosens (``sensitivityRatio``) adjustments to ISF / ``remainingCATime`` and
   the SMB-specific bookkeeping in determine-basal are out of scope and omitted.
"""

from __future__ import annotations

import math
from datetime import timedelta
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from datetime import datetime

__all__ = [
    "CobResult",
    "InsulinTotals",
    "PredictionCurves",
    "bgi",
    "bilinear_activity",
    "bilinear_iob",
    "carb_sensitivity_factor",
    "carbs_on_board",
    "deviation_series",
    "eventual_bg",
    "exponential_activity",
    "exponential_constants",
    "exponential_iob",
    "insulin_totals",
    "predict_glucose",
    "temp_basal_to_microboluses",
]

# ── oref0 constants ──────────────────────────────────────────────────────────

MIN_EXPONENTIAL_DIA_MIN = 300.0
"""lib/iob/total.js forces DIA >= 5 h (300 min) for the exponential curves."""

MIN_BILINEAR_DIA_MIN = 180.0
"""lib/iob/total.js forces a global minimum DIA of 3 h (180 min)."""

DEFAULT_PEAK_MIN = {"rapid-acting": 75.0, "ultra-rapid": 55.0}
"""Default activity-peak times (minutes) per lib/iob/calculate.js."""

PEAK_CLAMP_MIN = {"rapid-acting": (50.0, 120.0), "ultra-rapid": (35.0, 100.0)}
"""Custom peak-time clamps (minutes) per curve, from lib/iob/calculate.js."""

MIN_5M_CARBIMPACT_DEFAULT = 8.0
"""Default carb-impact floor in mg/dL per 5 min (oref0 preferences default)."""

MAX_COB_DEFAULT = 120.0
"""Default hard cap on carbs on board, grams (oref0 profile ``maxCOB``)."""

MAX_CARB_ABSORPTION_RATE_G_HR = 30.0
"""g/h cap on observed carb impact (determine-basal.js ``maxCarbAbsorptionRate``)."""

ASSUMED_CARB_ABSORPTION_RATE_G_HR = 20.0
"""g/h used to size remaining-carb absorption time (determine-basal.js)."""

REMAINING_CARBS_CAP_G = 90.0
"""Cap on not-yet-observed remaining carbs (determine-basal.js ``remainingCarbsCap``)."""

CARB_WINDOW_MIN = 6.0 * 60.0
"""Carbs are only considered for 6 h after entry (lib/meal/total.js, cob.js)."""


# ── result types ─────────────────────────────────────────────────────────────


class InsulinTotals(NamedTuple):
    """Total insulin on board and activity at a point in time.

    ``iob`` is in units; ``activity_per_min`` is in units of insulin used per
    minute (oref0's ``activity``).
    """

    iob: float
    activity_per_min: float


class CobResult(NamedTuple):
    """Carbs on board (grams) and total carbs absorbed (grams) at a point in time."""

    cob_g: float
    absorbed_g: float


class PredictionCurves(NamedTuple):
    """Predicted BG trajectories (mg/dL) at 5-minute steps, index 0 = starting BG.

    Mirrors oref0's ``predBGs`` output: ``iob`` (insulin only, plus decaying
    deviation), ``zt`` (zero-temp worst case, no deviations), ``cob``
    (insulin + predicted carb impact), ``uam`` (insulin + unannounced-meal
    deviation decay).
    """

    iob: list[float]
    zt: list[float]
    cob: list[float]
    uam: list[float]


# ── insulin activity curves (lib/iob/calculate.js) ───────────────────────────


def exponential_constants(dia_min: float, peak_min: float) -> tuple[float, float, float]:
    """Return ``(tau, a, S)`` for the oref0 exponential insulin curve.

    From ``iobCalcExponential`` in lib/iob/calculate.js (formula source:
    https://github.com/LoopKit/Loop/issues/388#issuecomment-317938473), with
    ``end = dia_min`` (DIA in minutes) and ``peak = peak_min``::

        tau = peak * (1 - peak/end) / (1 - 2*peak/end)   # decay time constant
        a   = 2 * tau / end                              # rise time factor
        S   = 1 / (1 - a + (1 + a) * exp(-end/tau))      # auxiliary scale factor

    ``tau`` is singular at ``peak = end/2``, so ``peak < end/2`` is required;
    oref0 guarantees this by forcing DIA >= 300 min while peaks max out at
    120 min.
    """
    if dia_min <= 0 or not 0 < peak_min < dia_min / 2:
        msg = f"need 0 < peak ({peak_min}) < dia/2 ({dia_min / 2}) for exponential curve"
        raise ValueError(msg)
    tau = peak_min * (1 - peak_min / dia_min) / (1 - 2 * peak_min / dia_min)
    a = 2 * tau / dia_min
    s = 1 / (1 - a + (1 + a) * math.exp(-dia_min / tau))
    return tau, a, s


def exponential_activity(minutes_since: float, dia_min: float, peak_min: float) -> float:
    """Fraction of a dose used per minute, ``minutes_since`` minutes after delivery.

    From lib/iob/calculate.js::

        activityContrib(t) = (S / tau^2) * t * (1 - t/end) * exp(-t/tau)

    Zero outside ``0 <= t < end``. The integral over [0, end] is 1 by
    construction (the curve is normalized so a dose is fully used over DIA).
    """
    if minutes_since < 0 or minutes_since >= dia_min:
        return 0.0
    tau, _a, s = exponential_constants(dia_min, peak_min)
    t = minutes_since
    return (s / tau**2) * t * (1 - t / dia_min) * math.exp(-t / tau)


def exponential_iob(minutes_since: float, dia_min: float, peak_min: float) -> float:
    """Fraction of a dose still on board ``minutes_since`` minutes after delivery.

    From lib/iob/calculate.js::

        iobContrib(t) = 1 - S * (1 - a) *
            ((t^2 / (tau * end * (1 - a)) - t/tau - 1) * exp(-t/tau) + 1)

    ``iob(0) = 1`` and ``iob(end) = 0`` exactly; zero for ``t >= end`` and
    ``t < 0`` (future doses contribute nothing).
    """
    if minutes_since < 0 or minutes_since >= dia_min:
        return 0.0
    tau, a, s = exponential_constants(dia_min, peak_min)
    t = minutes_since
    inner = (t**2 / (tau * dia_min * (1 - a)) - t / tau - 1) * math.exp(-t / tau) + 1
    return 1 - s * (1 - a) * inner


def bilinear_activity(minutes_since: float, dia_min: float) -> float:
    """Per-minute activity fraction for the legacy bilinear curve.

    From ``iobCalcBilinear`` in lib/iob/calculate.js: time is scaled by
    ``3 h / dia`` so fixed constants (peak 75 min, end 180 min on the scaled
    axis) apply to any DIA. On the *unscaled* axis activity rises linearly to
    its peak at ``dia_min * 75/180`` minutes and falls linearly to zero at
    ``dia_min``. Peak height is ``2 / dia_min`` per minute (triangle area = 1).
    """
    if dia_min <= 0:
        msg = f"dia_min must be positive, got {dia_min}"
        raise ValueError(msg)
    scaled = (180.0 / dia_min) * minutes_since
    if minutes_since < 0 or scaled >= 180.0:
        return 0.0
    activity_peak = 2.0 / dia_min
    if scaled < 75.0:
        return activity_peak / 75.0 * scaled
    return activity_peak + (-activity_peak / (180.0 - 75.0)) * (scaled - 75.0)


def bilinear_iob(minutes_since: float, dia_min: float) -> float:
    """Fraction of a dose on board under the legacy bilinear curve.

    From ``iobCalcBilinear``: two empirical quadratics in scaled time
    (coefficients estimated on 5-minute increments)::

        pre-peak  (x1 = scaled/5 + 1):       -0.001852*x1^2 + 0.001852*x1 + 1.0
        post-peak (x2 = (scaled - 75) / 5):   0.001323*x2^2 - 0.054233*x2 + 0.555560

    ``iob(0) = 1`` exactly; ``iob(end)`` is ~1e-4 (the quadratics are an
    approximation and do not hit zero exactly, matching the JS). There is a
    tiny discontinuity (~4e-5) at the peak where the two quadratics meet.
    """
    if dia_min <= 0:
        msg = f"dia_min must be positive, got {dia_min}"
        raise ValueError(msg)
    scaled = (180.0 / dia_min) * minutes_since
    if minutes_since < 0 or scaled >= 180.0:
        return 0.0
    if scaled < 75.0:
        x1 = scaled / 5.0 + 1.0
        return -0.001852 * x1 * x1 + 0.001852 * x1 + 1.0
    x2 = (scaled - 75.0) / 5.0
    return 0.001323 * x2 * x2 - 0.054233 * x2 + 0.555560


# ── totals over a dose schedule (lib/iob/total.js) ───────────────────────────


def _resolve_curve(curve: str, dia_min: float, peak_min: float | None) -> tuple[str, float, float]:
    """Apply oref0's DIA/peak clamping rules; return ``(curve, dia_min, peak_min)``.

    Mirrors lib/iob/total.js (global 3 h DIA floor; 5 h floor for exponential
    curves) and lib/iob/calculate.js (per-curve custom peak clamps).
    """
    curve = curve.lower()
    if curve == "bilinear":
        return curve, max(dia_min, MIN_BILINEAR_DIA_MIN), 75.0
    if curve not in DEFAULT_PEAK_MIN:
        msg = f"unsupported curve {curve!r}; expected bilinear, rapid-acting, or ultra-rapid"
        raise ValueError(msg)
    dia = max(dia_min, MIN_EXPONENTIAL_DIA_MIN)
    if peak_min is None:
        peak = DEFAULT_PEAK_MIN[curve]
    else:
        lo, hi = PEAK_CLAMP_MIN[curve]
        peak = min(hi, max(lo, peak_min))
    return curve, dia, peak


def insulin_totals(
    doses: Iterable[tuple[datetime, float]],
    at: datetime,
    *,
    curve: str = "rapid-acting",
    dia_min: float = 300.0,
    peak_min: float | None = None,
) -> InsulinTotals:
    """Total IOB (units) and activity (units/min) at ``at`` from insulin deltas.

    ``doses`` is a list of ``(timestamp, units)`` insulin deltas: boluses, plus
    net basal micro-boluses relative to the scheduled basal (see
    :func:`temp_basal_to_microboluses`). Negative units (e.g. a low temp below
    scheduled basal) are allowed and contribute negative IOB, as in oref0.

    Mirrors lib/iob/total.js: future doses are ignored, minutes-since-dose is
    rounded to the nearest whole minute (as the JS does), and each dose's
    contribution comes from the configured activity curve. By superposition,
    totals are simple sums over doses.
    """
    curve, dia, peak = _resolve_curve(curve, dia_min, peak_min)
    iob = 0.0
    activity = 0.0
    for ts, units in doses:
        if units == 0.0 or ts > at:
            continue
        mins_ago = round((at - ts).total_seconds() / 60.0)
        if curve == "bilinear":
            iob += units * bilinear_iob(mins_ago, dia)
            activity += units * bilinear_activity(mins_ago, dia)
        else:
            iob += units * exponential_iob(mins_ago, dia, peak)
            activity += units * exponential_activity(mins_ago, dia, peak)
    return InsulinTotals(iob=iob, activity_per_min=activity)


def temp_basal_to_microboluses(
    start: datetime,
    end: datetime,
    temp_rate_u_hr: float,
    scheduled_rate_u_hr: float,
    interval_min: float = 5.0,
) -> list[tuple[datetime, float]]:
    """Convert a temp basal into net insulin deltas relative to scheduled basal.

    oref0 (lib/iob/history.js) models temp basals by splitting the *difference*
    from the scheduled basal rate into small "micro-boluses" at regular
    intervals; a high temp yields positive deltas, a low/zero temp yields
    negative deltas. Each slice of ``interval_min`` minutes contributes::

        units = (temp_rate - scheduled_rate) * slice_minutes / 60

    stamped at the slice start. The final slice is shortened if the duration is
    not a multiple of ``interval_min``.
    """
    if end <= start:
        return []
    if interval_min <= 0:
        msg = f"interval_min must be positive, got {interval_min}"
        raise ValueError(msg)
    rate_delta = temp_rate_u_hr - scheduled_rate_u_hr
    out: list[tuple[datetime, float]] = []
    t = start
    while t < end:
        slice_min = min(interval_min, (end - t).total_seconds() / 60.0)
        out.append((t, rate_delta * slice_min / 60.0))
        t += timedelta(minutes=interval_min)
    return out


# ── glucose impact and deviations (cob.js / determine-basal.js) ──────────────


def bgi(activity_per_min: float, isf: float) -> float:
    """Blood-glucose impact of insulin, mg/dL per 5 minutes.

    From determine-basal.js / cob.js: ``BGI = -activity * sens * 5``. Insulin
    activity drives BG *down*, hence the negative sign: positive activity gives
    a negative BGI.
    """
    return -activity_per_min * isf * 5.0


def deviation_series(
    glucose: Sequence[tuple[datetime, float]],
    doses: Sequence[tuple[datetime, float]],
    isf: float,
    *,
    curve: str = "rapid-acting",
    dia_min: float = 300.0,
    peak_min: float | None = None,
) -> list[tuple[datetime, float]]:
    """Per-interval deviations: observed BG delta minus expected insulin impact.

    From cob.js: for each glucose point, ``deviation = delta - BGI`` where
    ``delta`` is the observed change since the previous point and BGI is
    computed from total insulin activity *at the newer point*. A positive
    deviation means BG rose more (or fell less) than insulin alone explains -
    the signal used to detect carb absorption.

    ``glucose`` must be sorted oldest-to-newest at (roughly) 5-minute spacing;
    cob.js's raw-CGM bucketing/interpolation is not reproduced here. (cob.js
    also computes a 15-minute average-delta deviation, ``avgDelta = (bg -
    bucketed[i+3])/3 - bgi``, for its ``currentDeviation``; this port uses the
    plain 5-minute delta.)

    Returns ``[(timestamp, deviation_mg_dl_per_5m), ...]`` with one entry per
    consecutive pair, stamped at the newer point.
    """
    out: list[tuple[datetime, float]] = []
    for i in range(1, len(glucose)):
        ts, bg = glucose[i]
        _, prev_bg = glucose[i - 1]
        activity = insulin_totals(
            doses, ts, curve=curve, dia_min=dia_min, peak_min=peak_min
        ).activity_per_min
        out.append((ts, (bg - prev_bg) - bgi(activity, isf)))
    return out


# ── carb absorption (cob.js / meal/total.js) ─────────────────────────────────


def carb_sensitivity_factor(isf: float, carb_ratio: float) -> float:
    """CSF in mg/dL per gram: ``ISF (mg/dL/U) / CR (g/U)``, per determine-basal.js."""
    return isf / carb_ratio


def carbs_on_board(
    carbs_g: float,
    carb_time: datetime,
    glucose: Sequence[tuple[datetime, float]],
    doses: Sequence[tuple[datetime, float]],
    isf: float,
    carb_ratio: float,
    at: datetime,
    *,
    min_5m_carbimpact: float = MIN_5M_CARBIMPACT_DEFAULT,
    max_cob: float = MAX_COB_DEFAULT,
    curve: str = "rapid-acting",
    dia_min: float = 300.0,
    peak_min: float | None = None,
) -> CobResult:
    """COB for a single announced carb entry, deviation-based as in oref0.

    Port of the core of cob.js ``detectCarbAbsorption`` + meal/total.js COB
    accounting. For every 5-minute interval after ``carb_time`` (up to ``at``,
    capped at 6 h after the meal), carbs absorbed are::

        ci       = max(deviation, min_5m_carbimpact)   # mg/dL per 5 min
        absorbed = ci / CSF = ci * carb_ratio / isf     # grams

    The ``min_5m_carbimpact`` floor (default 8 mg/dL/5m) guarantees announced
    carbs always decay, even when deviations are flat or negative. Then::

        COB = min(max_cob, max(0, carbs_g - total_absorbed))

    Simplifications vs. the JS (documented in the module docstring): no raw-CGM
    bucketing/interpolation, and the ``currentDeviation/2`` term in cob.js's
    ``ci`` is omitted. The floor applies to *announced* (entered) carbs only -
    callers tracking unannounced meals should rely on the UAM prediction curve
    instead.
    """
    deadline = min(at, carb_time + timedelta(minutes=CARB_WINDOW_MIN))
    devs = deviation_series(glucose, doses, isf, curve=curve, dia_min=dia_min, peak_min=peak_min)
    absorbed = 0.0
    for ts, dev in devs:
        if carb_time < ts <= deadline:
            ci = max(dev, min_5m_carbimpact)
            absorbed += ci * carb_ratio / isf
    cob = min(max_cob, max(0.0, carbs_g - absorbed))
    return CobResult(cob_g=cob, absorbed_g=absorbed)


# ── prediction curves (determine-basal.js) ───────────────────────────────────


def eventual_bg(bg: float, iob: float, isf: float, remaining_carb_impact: float = 0.0) -> float:
    """Naive eventual BG: ``bg - iob * isf + remaining_carb_impact`` (mg/dL).

    determine-basal.js: ``naive_eventualBG = bg - iob * sens`` (bolus-calculator
    math: all IOB eventually acts at full ISF), then adjusted upward by
    expected remaining carb impact / projected deviations.
    """
    return bg - iob * isf + remaining_carb_impact


def _remaining_carb_params(
    cob_g: float,
    ci: float,
    csf: float,
    last_carb_age_min: float,
) -> tuple[float, float, float]:
    """Compute ``(remaining_ca_time_hr, remaining_ci_peak, cid)`` per determine-basal.js.

    ``remainingCATime`` is the assumed window (hours) over which not-yet-observed
    carb absorption completes: at least 3 h, raised to ``COB / 20 g/h`` for large
    meals, plus ``1.5 * last_carb_age / 60`` (absorption windows stretch as the
    meal ages). ``remainingCIpeak`` is the apex of the /\\-shaped (triangular)
    remaining-carb impact curve::

        totalCI         = max(0, ci/5 * 60 * remainingCATime / 2)     # mg/dL
        totalCA         = totalCI / CSF                               # g
        remainingCarbs  = min(90, max(0, COB - totalCA))
        remainingCIpeak = remainingCarbs * CSF * 5/60 / (remainingCATime/2)

    ``cid`` is the carb-impact duration in 5-minute half-intervals: observed CI
    is assumed to decay linearly to zero over ``cid * 2`` intervals, sized so
    the area under the decay covers the full COB, limited so the rest is
    handled by the remaining-carb triangle::

        cid = min(remainingCATime * 60/5 / 2, max(0, COB * CSF / ci))
    """
    remaining_ca_time = 3.0
    if cob_g > 0:
        remaining_ca_time = max(remaining_ca_time, cob_g / ASSUMED_CARB_ABSORPTION_RATE_G_HR)
        remaining_ca_time += 1.5 * last_carb_age_min / 60.0
    total_ci = max(0.0, ci / 5.0 * 60.0 * remaining_ca_time / 2.0)
    total_ca = total_ci / csf if csf > 0 else 0.0
    remaining_carbs = min(REMAINING_CARBS_CAP_G, max(0.0, cob_g - total_ca))
    remaining_ci_peak = remaining_carbs * csf * 5.0 / 60.0 / (remaining_ca_time / 2.0)
    cid = 0.0 if ci <= 0 else min(remaining_ca_time * 60.0 / 5.0 / 2.0, cob_g * csf / ci)
    return remaining_ca_time, remaining_ci_peak, cid


def predict_glucose(
    bg: float,
    doses: Sequence[tuple[datetime, float]],
    at: datetime,
    isf: float,
    *,
    horizon_min: float = 240.0,
    curve: str = "rapid-acting",
    dia_min: float = 300.0,
    peak_min: float | None = None,
    carb_ratio: float | None = None,
    cob_g: float = 0.0,
    deviation_5m: float = 0.0,
    slope_from_deviations: float = 0.0,
    last_carb_age_min: float = 0.0,
    zt_basal_u_hr: float = 0.0,
) -> PredictionCurves:
    """Generate oref0's IOB / ZT / COB / UAM predicted BG curves.

    Port of the prediction loop in determine-basal.js. All four curves start at
    ``bg`` and advance in 5-minute steps for ``horizon_min`` minutes; at each
    step ``k`` (array length ``n = k + 1`` before appending, matching the JS)::

        predBGI   = -activity(t_k) * isf * 5          # insulin impact
        predDev   = ci * (1 - min(1, n / 12))         # deviation decays over 60 min
        IOBpredBG = prev + predBGI + predDev
        ZTpredBG  = prev + predZTBGI                  # zero-temp IOB, no deviations
        predCI    = max(0, max(0, ci) * (1 - n / max(cid*2, 1)))
        remainCI  = max(0, min(n, T*12 - n) / (T/2*12) * remainingCIpeak)
        COBpredBG = prev + predBGI + min(0, predDev) + predCI + remainCI
        predUCI   = min(max(0, uci + n*slopeFromDeviations),
                        max(0, uci * (1 - n / 36)))   # UAM decay, capped at 3 h
        UAMpredBG = prev + predBGI + min(0, predDev) + predUCI

    where ``ci = uci = deviation_5m`` (oref0's ``minDelta - bgi``, mg/dL per
    5 min), ``ci`` additionally capped at ``30 g/h * CSF * 5/60`` when a carb
    ratio is given, and ``T = remainingCATime`` (see
    :func:`_remaining_carb_params`).

    Inputs:

    - ``doses``: insulin deltas as in :func:`insulin_totals`. Future-dated
      deltas (e.g. an in-flight temp basal converted via
      :func:`temp_basal_to_microboluses`) are honored by the IOB/COB/UAM
      curves as they come due.
    - ``zt_basal_u_hr``: scheduled basal rate for the zero-temp curve. The ZT
      curve uses only doses at/before ``at`` plus, if this is > 0, negative
      deltas modeling a zero temp (no basal) over the whole horizon - oref0's
      ``iobWithZeroTemp`` worst case. With the default 0, ZT is simply
      "existing IOB only, no future insulin".
    - ``deviation_5m``: current deviation in mg/dL per 5 min. Default 0 gives a
      pure insulin-only IOB curve.
    - ``slope_from_deviations``: oref0's ``min(slopeFromMaxDeviation,
      -slopeFromMinDeviation/3)`` (mg/dL/5m per 5m interval, <= 0). Default 0
      lets the UAM decay fall back to the linear 3-hour ramp.

    Unlike oref0 this port does not round or clamp predicted values to
    [39, 401], and does not truncate flat curve tails.
    """
    if cob_g > 0 and carb_ratio is None:
        msg = "carb_ratio is required when cob_g > 0"
        raise ValueError(msg)
    ci = deviation_5m
    uci = deviation_5m
    csf = carb_sensitivity_factor(isf, carb_ratio) if carb_ratio else 0.0
    if csf > 0:
        ci = min(ci, MAX_CARB_ABSORPTION_RATE_G_HR * csf * 5.0 / 60.0)
    remaining_ca_time, remaining_ci_peak, cid = _remaining_carb_params(
        cob_g, ci, csf, last_carb_age_min
    )

    zt_doses = [d for d in doses if d[0] <= at]
    if zt_basal_u_hr > 0:
        zt_end = at + timedelta(minutes=horizon_min)
        zt_doses += temp_basal_to_microboluses(at, zt_end, 0.0, zt_basal_u_hr)

    iob_curve, zt_curve, cob_curve, uam_curve = [bg], [bg], [bg], [bg]
    n_steps = round(horizon_min / 5.0)
    for k in range(n_steps):
        t = at + timedelta(minutes=5.0 * k)
        act = insulin_totals(doses, t, curve=curve, dia_min=dia_min, peak_min=peak_min)
        act_zt = insulin_totals(zt_doses, t, curve=curve, dia_min=dia_min, peak_min=peak_min)
        pred_bgi = bgi(act.activity_per_min, isf)
        pred_zt_bgi = bgi(act_zt.activity_per_min, isf)
        n = k + 1
        pred_dev = ci * (1 - min(1.0, n / (60.0 / 5.0)))
        iob_curve.append(iob_curve[-1] + pred_bgi + pred_dev)
        zt_curve.append(zt_curve[-1] + pred_zt_bgi)
        pred_ci = max(0.0, max(0.0, ci) * (1 - n / max(cid * 2.0, 1.0)))
        intervals = min(float(n), remaining_ca_time * 12.0 - n)
        remaining_ci = max(0.0, intervals / (remaining_ca_time / 2.0 * 12.0) * remaining_ci_peak)
        cob_curve.append(cob_curve[-1] + pred_bgi + min(0.0, pred_dev) + pred_ci + remaining_ci)
        pred_uci_slope = max(0.0, uci + n * slope_from_deviations)
        pred_uci_max = max(0.0, uci * (1 - n / (3.0 * 60.0 / 5.0)))
        pred_uci = min(pred_uci_slope, pred_uci_max)
        uam_curve.append(uam_curve[-1] + pred_bgi + min(0.0, pred_dev) + pred_uci)
    return PredictionCurves(iob=iob_curve, zt=zt_curve, cob=cob_curve, uam=uam_curve)
