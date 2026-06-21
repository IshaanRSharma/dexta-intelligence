"""E3 clinical accuracy - oref0 forecast vs realized glucose on synthetic data.

Report-style: there is no hard pass/fail bar, since absolute forecast accuracy
depends on assumed physiology (ISF, DIA) that synthetic data only approximates.
The job is to *emit the numbers* a clinician would judge a CGM/predictor by:

- Clarke and Parkes error-grid zone distributions (and the headline A+B %).
- MARD (mean absolute relative difference, %).

Ground truth is non-LLM: the synthetic generator produces a deterministic
glucose series with known meals/insulin. At a grid of anchor points we run
:func:`~dexta_intelligence.analytics.oref.predict_glucose` over a fixed horizon
and compare its forecast at the horizon to the *realized* synthetic value at the
same future timestamp. The realized value is the reference; the forecast is the
predicted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from dexta_intelligence.analytics.error_grid import mard, zone_distribution
from dexta_intelligence.analytics.oref import predict_glucose
from dexta_intelligence.models import InsulinKind
from dexta_intelligence.testing.synthetic import generate_baseline

__all__ = ["E3AccuracyResult", "run_e3_accuracy"]

#: Assumed insulin sensitivity factor (mg/dL per unit) for the oref0 forecast.
#: A representative T1D value; the synthetic kernel uses 12 mg/dL/U per bolus,
#: so this is deliberately a held-out, imperfect assumption.
_ISF_MG_DL_PER_U = 50.0

#: Forecast horizon (minutes) scored against the realized value.
_HORIZON_MIN = 60.0

#: Lookback window (minutes) of insulin doses fed to the forecast.
_DOSE_LOOKBACK_MIN = 360.0


@dataclass(frozen=True, slots=True)
class E3AccuracyResult:
    """Outcome of one E3 accuracy sweep."""

    n_pairs: int
    horizon_min: float
    mard_pct: float
    clarke: dict[str, float]
    parkes: dict[str, float]
    clarke_ab_pct: float
    parkes_ab_pct: float
    reference: tuple[float, ...] = field(repr=False, default=())
    predicted: tuple[float, ...] = field(repr=False, default=())


def _doses_before(
    insulin: list[tuple[datetime, float]], at: datetime
) -> list[tuple[datetime, float]]:
    lo = at - timedelta(minutes=_DOSE_LOOKBACK_MIN)
    return [(ts, units) for ts, units in insulin if lo <= ts <= at]


def run_e3_accuracy(
    *,
    seed: int = 8800,
    n_days: int = 30,
    sample_every_min: float = 60.0,
) -> E3AccuracyResult:
    """Score oref0 forecasts against realized synthetic glucose.

    Anchors are taken every ``sample_every_min`` minutes; at each, the forecast
    horizon value is compared to the realized series value at that future time.
    """
    events = generate_baseline(seed=seed, n_days=n_days)
    glucose = sorted(events["glucose"], key=lambda g: g.ts)
    by_ts = {g.ts: float(g.mg_dl) for g in glucose}
    insulin_deltas = [
        (e.ts, e.units)
        for e in events["insulin"]
        if e.kind is InsulinKind.BOLUS and e.units is not None
    ]

    step = round(sample_every_min / 5.0)
    horizon = timedelta(minutes=_HORIZON_MIN)

    reference: list[float] = []
    predicted: list[float] = []
    for i in range(0, len(glucose) - 1, max(step, 1)):
        anchor = glucose[i]
        realized_ts = anchor.ts + horizon
        realized = by_ts.get(realized_ts)
        if realized is None:
            continue
        curves = predict_glucose(
            float(anchor.mg_dl),
            _doses_before(insulin_deltas, anchor.ts),
            anchor.ts,
            _ISF_MG_DL_PER_U,
            horizon_min=_HORIZON_MIN,
        )
        forecast = curves.iob[-1]
        if forecast <= 0.0:
            continue
        reference.append(realized)
        predicted.append(forecast)

    if not reference:
        msg = "E3 produced no scorable forecast/realized pairs"
        raise RuntimeError(msg)

    clarke = zone_distribution(reference, predicted, grid="clarke")
    parkes = zone_distribution(reference, predicted, grid="parkes")
    return E3AccuracyResult(
        n_pairs=len(reference),
        horizon_min=_HORIZON_MIN,
        mard_pct=mard(reference, predicted),
        clarke={k: round(v, 4) for k, v in clarke.items()},
        parkes={k: round(v, 4) for k, v in parkes.items()},
        clarke_ab_pct=100.0 * (clarke["A"] + clarke["B"]),
        parkes_ab_pct=100.0 * (parkes["A"] + parkes["B"]),
        reference=tuple(reference),
        predicted=tuple(predicted),
    )
