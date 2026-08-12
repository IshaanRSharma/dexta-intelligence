"""Synthetic patient for `dexta demo` - the zero-config try-it on-ramp.

Builds an in-memory :class:`SQLiteStore` loaded with ~6 months of 5-minute CGM
plus a planted recurring late-bolus dinner spike, enough that
:func:`~dexta_intelligence.investigations.spike.explain_spike` reaches the
"late/insufficient meal insulin context" attribution.

It lands at moderate confidence, not high, and that is the correct reading rather
than a shortfall to tune away. Confidence here is computed from recurrence and
from how far the bolus delay separates spiking events from quiet ones. In a record
where ordinary meals also run high, that separation is genuinely narrower than in
one where the planted dinners were the only thing that ever moved. Manufacturing
the wider separation would mean shaping the data to the number.

Around that hero spike the store is populated so every surface has something to
show: sleep and activity context, logged forecast curves (so prediction
reconciliation has real material), two therapy-profile versions (so versioned
profiles matter), and a few user-reported manual notes aligned to the spike.

The trace is generated *from* the treatment record, not alongside it. Every carb
entry produces a postprandial response damped by how promptly its bolus landed,
and the baseline between meals is a correlated wander rather than independent
per-reading jitter. That matters beyond looking right: a record whose carbs move
nothing is one where every meal-versus-glucose correlation the discovery agents
test is null by construction, and where "why did I go high?" has no answer to
find. Variability lands where a real well-controlled T1D record sits, roughly 75%
time in range at a CV near 32, instead of the near-flat 99% an independently
generated baseline produces. Those bounds are asserted in ``tests/test_demo.py``
so the record cannot quietly drift back to implausible.

Fully deterministic (seeded RNG, fixed dates - no ``random.random`` / ``now``).
This mirrors the ``late_bolus`` golden dataset; tests/ cannot be imported by
shipped code, so the planting logic lives here independently.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from typing import TYPE_CHECKING

from dexta_intelligence.analytics.rollups import daily_rollup
from dexta_intelligence.models import (
    ActivityEvent,
    GlucoseEvent,
    InsulinEvent,
    InsulinKind,
    ManualEvent,
    MealEvent,
    PredictionEvent,
    RollupPeriod,
    SleepEvent,
    TherapyProfile,
)
from dexta_intelligence.store import SQLiteStore

if TYPE_CHECKING:
    from dexta_intelligence.store.port import StoragePort

__all__ = ["DEMO_SPIKE_DATE", "DEMO_SPIKE_TS", "build_demo_store"]

_GRID_MIN = 5
_SEED = 514
_BASAL_UNITS = 24.0

#: The canonical day whose dinner spike `dexta demo` explains.
DEMO_SPIKE_TS = datetime(2026, 3, 14, 20, 42, tzinfo=UTC)
DEMO_SPIKE_DATE = DEMO_SPIKE_TS.date()
_SPIKE_PEAK = 246
_BOLUS_DELAY_MIN = 22

_START = datetime(2025, 12, 15, 0, 2, tzinfo=UTC)
#: Record length in days. The hero story runs in the first ~90 days (to the
#: DEMO_SPIKE_DATE dinner spike); the remainder (to ~mid-June 2026) carries the
#: extended excursion spread (:data:`_EXT_EXCURSIONS`).
_DAYS = 185
_N_DINNERS = 18
_ONTIME_IDX = frozenset({3, 7, 11, 15})

#: Stationary spread (mg/dL) and autocorrelation time (minutes) of the baseline
#: wander. An Ornstein-Uhlenbeck walk rather than per-reading jitter: real CGM
#: drifts over hours, and independent noise flattens to nothing over any window
#: wide enough to matter, which is what held the old trace at 99% time in range.
_WANDER_SD = 30.0
_WANDER_TAU_MIN = 110.0

#: Postprandial response per gram of carbohydrate (mg/dL), before the bolus
#: credit below. Spread deterministically per meal so days differ; the range
#: brackets the usual adult carb factor.
_CARB_RISE_MIN = 1.35
_CARB_RISE_MAX = 5.0
#: A bolus this many minutes after the carb entry (or earlier) blunts the
#: response the most; the credit decays to nothing as the bolus slips later,
#: which is the same mechanism the hero dinner spike is planted to demonstrate.
_BOLUS_ON_TIME_MIN = 10.0
_BOLUS_LATE_MIN = 60.0
#: Share of everyday meals whose bolus lands badly late.
_LATE_BOLUS_SHARE = 0.24
#: How much of the carb rise a well-timed bolus removes. Enough that prompt
#: insulin visibly works and late insulin visibly does not, since that separation
#: is the signal the whole demo is built to surface. Not so much that a bolused
#: meal reads flat: prompt insulin bounds a postprandial excursion, it does not
#: abolish one.
_BOLUS_MAX_CREDIT = 0.62
#: Minutes from carb entry to the peak of its response, and the widths of the
#: rise and the (slower) fall. A postprandial excursion takes hours to clear.
_RESPONSE_PEAK_MIN = 62.0
_RESPONSE_RISE_MIN = 40.0
_RESPONSE_FALL_MIN = 95.0

#: A bolus with no carb entry within this many minutes is a correction, and pushes
#: glucose down instead of blunting a rise. Wide enough to cover a badly late meal
#: bolus: counting one as a correction would have it both fail to blunt its meal
#: and drive a fall of its own, which is how a demo patient ends up spending 8% of
#: the record below range.
_CORRECTION_ISOLATION_MIN = 75.0
#: mg/dL removed per unit of correction insulin. A fraction of the profile's
#: ~45 mg/dL/U sensitivity, since basal and counter-regulation take the rest.
_CORRECTION_FALL_PER_U = 38.0
_CORRECTION_PEAK_MIN = 85.0
_CORRECTION_RISE_MIN = 55.0
_CORRECTION_FALL_MIN = 105.0

#: mg/dL removed by an hour of all-out activity, scaled by intensity and length.
#: Wide and slow on the way out: post-exercise insulin sensitization is the reason
#: the demo's planted post-workout lows arrive an hour and a half after the run.
_ACTIVITY_FALL_PER_HOUR = 80.0
_ACTIVITY_PEAK_MIN = 80.0
_ACTIVITY_RISE_MIN = 50.0
_ACTIVITY_FALL_MIN = 150.0


@dataclass(frozen=True, slots=True)
class _Response:
    """One event's signed contribution to the trace, as a peak and two widths.

    Meals push up, corrections and activity pull down. Modelling them the same way
    is the point: the record's events are what move the curve, so every excursion
    in the demo has something logged that accounts for it.
    """

    peak_ts: datetime
    amount: float
    rise_min: float
    fall_min: float

    def at(self, ts: datetime) -> float:
        offset_min = (ts - self.peak_ts).total_seconds() / 60.0
        width = self.rise_min if offset_min < 0 else self.fall_min
        return self.amount * math.exp(-((offset_min / width) ** 2))


#: Overnight/fasting centre of the trace (mg/dL) and the diurnal swing around it.
_BASELINE_MG_DL = 133.0
_BASELINE_SWING = 14.0


def _baseline(ts: datetime) -> float:
    hour = ts.hour + ts.minute / 60
    return _BASELINE_MG_DL + _BASELINE_SWING * math.sin(2 * math.pi * (hour - 9.0) / 24)


def _wander(n: int, rng: random.Random) -> list[float]:
    """``n`` steps of a zero-mean Ornstein-Uhlenbeck walk on the reading grid."""
    phi = math.exp(-_GRID_MIN / _WANDER_TAU_MIN)
    step_sd = _WANDER_SD * math.sqrt(1.0 - phi * phi)
    out: list[float] = []
    value = 0.0
    for _ in range(n):
        value = phi * value + rng.gauss(0.0, step_sd)
        out.append(value)
    return out


def _grid(start: datetime, days: int) -> list[datetime]:
    return [start + timedelta(minutes=_GRID_MIN * i) for i in range(days * 24 * 60 // _GRID_MIN)]


def _snap(ts: datetime, start: datetime) -> datetime:
    steps = round((ts - start).total_seconds() / 60 / _GRID_MIN)
    return start + timedelta(minutes=_GRID_MIN * steps)


def _bump(ts: datetime, peak_ts: datetime, amplitude: float, sigma_min: float = 25.0) -> float:
    offset_min = (ts - peak_ts).total_seconds() / 60
    return amplitude * math.exp(-((offset_min / sigma_min) ** 2))


def _dinner_ts(day: date, idx: int) -> datetime:
    base = datetime(day.year, day.month, day.day, 19, 45, tzinfo=UTC)
    return base + timedelta(minutes=(idx * 11) % 46)


def _nearest_min(ts: datetime, others: list[datetime]) -> float | None:
    """Minutes from ``ts`` to the closest of ``others``, or ``None`` if empty."""
    return min((abs((o - ts).total_seconds() / 60.0) for o in others), default=None)


def _event_responses(
    meals: list[MealEvent],
    boluses: list[InsulinEvent],
    activity: list[ActivityEvent],
) -> list[_Response]:
    """Every logged event's effect on the curve.

    Carbs push glucose up, damped by how promptly a manual bolus followed: the
    credit fades to nothing as the bolus slips past an hour, which is the same
    mechanism the hero dinner spike is planted to demonstrate. A bolus with no carb
    entry near it is a correction and pulls down. So does activity, slowly and for
    hours afterwards.
    """
    manual = sorted(b.ts for b in boluses if not b.automatic)
    meal_times = sorted(m.ts for m in meals if m.carbs_g)
    out: list[_Response] = []

    for i, meal in enumerate(sorted(meals, key=lambda m: m.ts)):
        carbs = meal.carbs_g
        if not carbs:
            continue
        span = _CARB_RISE_MAX - _CARB_RISE_MIN
        per_gram = _CARB_RISE_MIN + span * (((i * 37) % 100) / 99.0)
        delay = _nearest_min(meal.ts, manual)
        credit = 0.0
        if delay is not None:
            slip = (delay - _BOLUS_ON_TIME_MIN) / (_BOLUS_LATE_MIN - _BOLUS_ON_TIME_MIN)
            credit = _BOLUS_MAX_CREDIT * (1.0 - min(1.0, max(0.0, slip)))
        out.append(_Response(
            peak_ts=meal.ts + timedelta(minutes=_RESPONSE_PEAK_MIN),
            amount=carbs * per_gram * (1.0 - credit),
            rise_min=_RESPONSE_RISE_MIN, fall_min=_RESPONSE_FALL_MIN,
        ))

    for bolus in boluses:
        near_meal = _nearest_min(bolus.ts, meal_times)
        if near_meal is not None and near_meal <= _CORRECTION_ISOLATION_MIN:
            continue  # covered by the meal's credit above, not a correction
        out.append(_Response(
            peak_ts=bolus.ts + timedelta(minutes=_CORRECTION_PEAK_MIN),
            amount=-(bolus.units or 0.0) * _CORRECTION_FALL_PER_U,
            rise_min=_CORRECTION_RISE_MIN, fall_min=_CORRECTION_FALL_MIN,
        ))

    for session in activity:
        hours = (session.duration_min or 0.0) / 60.0
        out.append(_Response(
            peak_ts=session.ts + timedelta(minutes=_ACTIVITY_PEAK_MIN),
            amount=-_ACTIVITY_FALL_PER_HOUR * hours * (session.intensity or 0.5),
            rise_min=_ACTIVITY_RISE_MIN, fall_min=_ACTIVITY_FALL_MIN,
        ))
    return out


def _trace(
    grid: list[datetime],
    bumps: list[tuple[datetime, datetime, float]],
    responses: list[_Response],
    rng: random.Random,
) -> list[GlucoseEvent]:
    """CGM trace: diurnal baseline, a correlated wander, the record's own events,
    and the planted excursions exact at their peaks.

    The wander and the event responses are faded out in proportion to how strongly
    a planted bump is in effect, so a planted peak lands on its stated value to the
    mg/dL (the hero spike must read 246) while the curve stays continuous either
    side of it. Bucketing by day keeps this linear in readings rather than
    readings x events.
    """
    bumps_by_day: dict[date, list[tuple[datetime, float]]] = {}
    for anchor_ts, peak_ts, peak in bumps:
        bumps_by_day.setdefault(anchor_ts.date(), []).append(
            (peak_ts, peak - _baseline(peak_ts))
        )
    resp_by_day: dict[date, list[_Response]] = {}
    for response in responses:
        resp_by_day.setdefault(response.peak_ts.date(), []).append(response)

    wander = _wander(len(grid), rng)
    events: list[GlucoseEvent] = []
    for i, ts in enumerate(grid):
        today, yesterday = ts.date(), (ts - timedelta(days=1)).date()
        planted = 0.0
        strength = 0.0
        for peak_ts, amplitude in bumps_by_day.get(today, ()):
            share = _bump(ts, peak_ts, 1.0)
            planted += amplitude * share
            strength = max(strength, share)
        moved = 0.0
        for day in (today, yesterday):  # a late event's tail runs past midnight
            for response in resp_by_day.get(day, ()):
                moved += response.at(ts)
        damp = 1.0 - strength
        value = _baseline(ts) + planted + damp * (moved + wander[i])
        events.append(GlucoseEvent(ts=ts, mg_dl=round(min(400.0, max(40.0, value)))))
    return events


#: Ceiling (mg/dL) for DEMO_SPIKE_DATE outside its planted dinner, and the hour
#: after which the planted spike owns the day.
_HERO_DAY_CEILING = 192.0
_HERO_SPIKE_HOUR = 18


def _protect_hero_day(glucose: list[GlucoseEvent]) -> list[GlucoseEvent]:
    """Keep the planted dinner spike the largest thing on DEMO_SPIKE_DATE.

    ``explain_spike`` locates the day's spike as its highest reading above the
    threshold, so a lunch that happened to run higher would silently retarget the
    demo's headline investigation at the wrong event. Daytime deviations above the
    baseline are compressed under a ceiling rather than erased, so the day still
    carries the postprandial shape every other day has. One day of the record is
    pinned; the other 184 are whatever the treatment record produces.
    """
    def daytime(g: GlucoseEvent) -> bool:
        return g.ts.date() == DEMO_SPIKE_DATE and g.ts.hour < _HERO_SPIKE_HOUR

    # Headroom is measured against the baseline at each reading, not the daily
    # centre: the diurnal term alone moves the ceiling by a dozen mg/dL.
    worst = max(
        (
            (g.mg_dl - _baseline(g.ts)) / max(1.0, _HERO_DAY_CEILING - _baseline(g.ts))
            for g in glucose
            if daytime(g)
        ),
        default=0.0,
    )
    if worst <= 1.0:
        return glucose
    scale = 1.0 / worst
    out: list[GlucoseEvent] = []
    for g in glucose:
        deviation = g.mg_dl - _baseline(g.ts)
        if not daytime(g) or deviation <= 0.0:
            out.append(g)
            continue
        out.append(g.model_copy(
            update={"mg_dl": round(_baseline(g.ts) + deviation * scale)}
        ))
    return out


def _daily_basal(start: datetime, days: int) -> list[InsulinEvent]:
    midnight = start.replace(hour=0, minute=0)
    return [
        InsulinEvent(
            ts=midnight + timedelta(days=i),
            kind=InsulinKind.BASAL,
            units=_BASAL_UNITS,
            duration_min=1440.0,
        )
        for i in range(days)
    ]


def _planted_dinner_days() -> list[date]:
    """The every-fifth-day dinners the hero late-bolus story is planted on."""
    return [
        DEMO_SPIKE_DATE - timedelta(days=5 * (_N_DINNERS - 1 - i)) for i in range(_N_DINNERS)
    ]


def _patient(
    other_meals: list[MealEvent],
    other_boluses: list[InsulinEvent],
    other_activity: list[ActivityEvent],
) -> tuple[list[GlucoseEvent], list[InsulinEvent], list[MealEvent]]:
    """The hero timeline (planted dinners) plus a trace that answers to the rest of
    the treatment record: ``other_meals``, ``other_boluses``, ``other_activity``."""
    rng = random.Random(_SEED)
    dinner_days = _planted_dinner_days()

    meals: list[MealEvent] = []
    boluses: list[InsulinEvent] = []
    bumps: list[tuple[datetime, datetime, float]] = []
    for idx, day in enumerate(dinner_days):
        if day == DEMO_SPIKE_DATE:
            meal_ts = datetime(2026, 3, 14, 20, 0, tzinfo=UTC)
            delay_min, peak = _BOLUS_DELAY_MIN, float(_SPIKE_PEAK)
        elif idx not in _ONTIME_IDX:
            meal_ts = _dinner_ts(day, idx)
            delay_min, peak = 20 + idx % 6, 212.0 + (idx * 5) % 29
        else:
            meal_ts = _dinner_ts(day, idx)
            delay_min, peak = 1, 168.0 + (idx * 3) % 12
        meals.append(MealEvent(ts=meal_ts, carbs_g=55.0 + (idx * 7) % 20, note="dinner"))
        boluses.append(
            InsulinEvent(
                ts=meal_ts + timedelta(minutes=delay_min), kind=InsulinKind.BOLUS, units=6.0
            )
        )
        bumps.append((meal_ts, _snap(meal_ts + timedelta(minutes=42), _START), peak))

    # The planted dinners already carry their own shape, so only the rest of the
    # record drives a modelled response; otherwise a hero peak would be counted
    # twice and land off its stated value.
    responses = _event_responses(other_meals, other_boluses, other_activity)
    glucose = _trace(_grid(_START, _DAYS), bumps, responses, rng)
    insulin = _daily_basal(_START, _DAYS) + boluses
    return glucose, insulin, meals


def _demo_sleep(rng: random.Random) -> list[SleepEvent]:
    """One scored sleep event per night across the window."""
    out: list[SleepEvent] = []
    midnight = _START.replace(hour=0, minute=0)
    for i in range(_DAYS - 1):
        day = midnight + timedelta(days=i)
        ts_start = day + timedelta(hours=22, minutes=30 + rng.uniform(-30.0, 30.0))
        ts_end = day + timedelta(days=1, hours=6, minutes=30 + rng.uniform(-30.0, 30.0))
        out.append(
            SleepEvent(
                ts_start=ts_start,
                ts_end=ts_end,
                duration_min=round((ts_end - ts_start).total_seconds() / 60.0, 1),
                score=round(rng.uniform(45.0, 95.0), 1),
            )
        )
    return out


def _demo_activity(rng: random.Random) -> list[ActivityEvent]:
    """Afternoon workouts on roughly half the days."""
    out: list[ActivityEvent] = []
    midnight = _START.replace(hour=0, minute=0)
    for i in range(_DAYS):
        if rng.random() < 0.55:
            continue
        ts = midnight + timedelta(days=i, hours=14, minutes=rng.uniform(-45.0, 45.0))
        out.append(
            ActivityEvent(
                ts=ts,
                kind=rng.choice(["run", "ride", "strength"]),
                duration_min=round(rng.uniform(30.0, 75.0), 1),
                intensity=round(rng.uniform(0.4, 0.9), 2),
            )
        )
    return out


#: Evenings (days before the hero spike) given a prolonged high - a recurring
#: forecast miss for prediction reconciliation. Kept below the 246 hero peak and
#: off DEMO_SPIKE_DATE so the canonical explain_spike contract is unaffected.
_MISS_DAY_OFFSETS = (3, 9, 16, 23, 37, 51)
_MISS_ELEVATION = 78


def _miss_days() -> set[date]:
    return {DEMO_SPIKE_DATE - timedelta(days=d) for d in _MISS_DAY_OFFSETS}


#: Minutes the miss-day elevation takes to ramp in and out. A step change would
#: be a 78 mg/dL jump between two readings five minutes apart, a rate no glucose
#: physiology produces and one the error grids would rightly flag as nonsense.
_MISS_RAMP_MIN = 35.0


def _with_prolonged_highs(glucose: list[GlucoseEvent]) -> list[GlucoseEvent]:
    """Elevate 21:00-24:00 on the miss days to a prolonged high the forecast
    fails to anticipate (the reconciliation ground truth)."""
    days = _miss_days()
    out: list[GlucoseEvent] = []
    for g in glucose:
        if g.ts.date() not in days or not 21 <= g.ts.hour < 24:
            out.append(g)
            continue
        into = (g.ts.hour - 21) * 60.0 + g.ts.minute
        share = min(1.0, into / _MISS_RAMP_MIN, (180.0 - into) / _MISS_RAMP_MIN)
        elevated = g.mg_dl + _MISS_ELEVATION * max(0.0, share)
        out.append(g.model_copy(update={"mg_dl": min(350, round(elevated))}))
    return out


#: Planted "story" days for the episode graph (offsets in days before the hero
#: spike). Chosen off the dinner days (multiples of 5) and the forecast-miss
#: days so the explain_spike and reconciliation contracts are untouched, and
#: every planted peak stays below the 246 hero peak.
_CHAIN_DAY_OFFSETS = (6, 13, 27, 34)  # low -> rescue carbs -> rebound high
_SEVERE_CHAIN_OFFSET = 13  # this rebound day dips below 54 (one severe low)
_CHAIN_CORRECTION_OFFSETS = (6, 34)  # these rebounds get a manual correction
_STACK_DAY_OFFSET = 11  # evening high, stacked corrections, night low
_WORKOUT_LOW_OFFSETS = (8, 29, 43)  # afternoon run, low ~90 min later
_GAP_DAY_OFFSET = 26  # 02:00-03:30 sensor gap


def _day(offset: int) -> date:
    return DEMO_SPIKE_DATE - timedelta(days=offset)


def _at(day: date, hh: int, mm: int, plus_days: int = 0) -> datetime:
    return datetime(day.year, day.month, day.day, hh, mm, tzinfo=UTC) + timedelta(
        days=plus_days
    )


def _story_segments() -> list[list[tuple[datetime, float]]]:
    """Hand-shaped glucose checkpoints for each story day, linearly interpolated."""
    segments: list[list[tuple[datetime, float]]] = []
    for off in _CHAIN_DAY_OFFSETS:
        day = _day(off)
        nadir = 48.0 if off == _SEVERE_CHAIN_OFFSET else 55.0 + off % 3
        peak = 204.0 + (off * 3) % 11
        segments.append([
            (_at(day, 15, 30), 118.0), (_at(day, 15, 50), nadir),
            (_at(day, 16, 5), 66.0), (_at(day, 16, 10), 74.0),
            (_at(day, 16, 40), 110.0), (_at(day, 17, 10), peak),
            (_at(day, 17, 50), 150.0), (_at(day, 18, 20), 122.0),
        ])
    day = _day(_STACK_DAY_OFFSET)
    segments.append([
        (_at(day, 20, 20), 130.0), (_at(day, 21, 0), 232.0),
        (_at(day, 22, 20), 225.0), (_at(day, 23, 0), 140.0),
        (_at(day, 23, 40), 62.0), (_at(day, 0, 10, 1), 68.0),
        (_at(day, 0, 40, 1), 96.0), (_at(day, 1, 10, 1), 112.0),
    ])
    for off in _WORKOUT_LOW_OFFSETS:
        day = _day(off)
        segments.append([
            (_at(day, 16, 20), 112.0), (_at(day, 16, 50), 63.0),
            (_at(day, 17, 20), 58.0), (_at(day, 17, 50), 75.0),
            (_at(day, 18, 20), 105.0),
        ])
    return segments


def _interp(ts: datetime, points: list[tuple[datetime, float]]) -> float:
    for (t0, v0), (t1, v1) in pairwise(points):
        if t0 <= ts <= t1:
            frac = (ts - t0).total_seconds() / max(1.0, (t1 - t0).total_seconds())
            return v0 + (v1 - v0) * frac
    return points[-1][1]


def _apply_segments(
    glucose: list[GlucoseEvent], segments: list[list[tuple[datetime, float]]]
) -> list[GlucoseEvent]:
    """Overwrite each segment's span with its hand-shaped curve, jitter-free so
    episode boundaries are stable.

    Each segment's first and last checkpoint is re-anchored to the trace value
    already at that timestamp, so a planted excursion joins the surrounding curve
    instead of stepping onto it. Hard-coded endpoints were harmless against a flat
    baseline and are not against a wandering one: the seam would be a jump of tens
    of mg/dL between two readings, visible on the Timeline and wrong.
    """
    by_ts = {g.ts: g.mg_dl for g in glucose}
    anchored: list[list[tuple[datetime, float]]] = []
    for points in segments:
        head = (points[0][0], float(by_ts.get(points[0][0], points[0][1])))
        tail = (points[-1][0], float(by_ts.get(points[-1][0], points[-1][1])))
        anchored.append([head, *points[1:-1], tail])

    out: list[GlucoseEvent] = []
    for g in glucose:
        value: float | None = None
        for points in anchored:
            if points[0][0] <= g.ts <= points[-1][0]:
                value = _interp(g.ts, points)
                break
        out.append(g if value is None else g.model_copy(update={"mg_dl": round(value)}))
    return out


#: The extended record (Mar 15 - ~Jun 16 2026): a spread of excursions across every
#: severity band so the demo shows regular and severe highs and lows beyond the
#: story window. (day offset from _START, hour, minute, extreme mg/dL, cause).
#: Highs >180, very-high >250, lows <70, very-low <54. Each is held ~30 min so it
#: registers as a clinically significant (or severe) episode. Spaced so none
#: overlap. Every one carries a cause, which :func:`_ext_events` turns into the
#: context events that explain it: an excursion with nothing logged around it is
#: an episode the graph can show and no agent can account for, and a whole quarter
#: of unexplainable highs is a worse demo than none.
_EXT_EXCURSIONS: tuple[tuple[int, int, int, int, str], ...] = (
    (93, 20, 0, 216, "late_bolus"),
    (97, 8, 30, 62, "over_correction"),
    (101, 21, 30, 271, "missed_bolus"),
    (106, 3, 0, 48, "over_correction"),
    (111, 13, 0, 228, "late_bolus"),
    (116, 16, 30, 57, "post_exercise"),
    (122, 20, 30, 241, "late_bolus"),
    (127, 2, 30, 44, "over_correction"),
    (133, 19, 0, 293, "missed_bolus"),
    (139, 7, 0, 66, "over_correction"),
    (145, 21, 0, 207, "late_bolus"),
    (151, 14, 30, 51, "post_exercise"),
    (157, 20, 0, 233, "late_bolus"),
    (163, 4, 0, 60, "over_correction"),
    (169, 19, 30, 264, "missed_bolus"),
    (175, 15, 30, 55, "post_exercise"),
    (181, 20, 0, 221, "late_bolus"),
)

#: Carbs (g) behind an extended high, by cause.
_EXT_MEAL_CARBS = {"late_bolus": 78.0, "missed_bolus": 92.0}


def _ext_day(offset: int) -> date:
    return (_START + timedelta(days=offset)).date()


def _ext_events() -> tuple[list[MealEvent], list[InsulinEvent], list[ActivityEvent]]:
    """The context that accounts for each extended excursion.

    Highs get a large meal, either bolused far too late or not at all; lows get
    either a correction stacked onto an already-falling glucose about two hours
    earlier, or a hard workout the afternoon before. These are the same shapes the
    story days plant, so the same traversals that explain a January episode explain
    a June one.
    """
    meals: list[MealEvent] = []
    insulin: list[InsulinEvent] = []
    activity: list[ActivityEvent] = []
    for off, hh, mm, extreme, cause in _EXT_EXCURSIONS:
        day, onset = _ext_day(off), _at(_ext_day(off), hh, mm) - timedelta(minutes=15)
        if cause in _EXT_MEAL_CARBS:
            carbs = _EXT_MEAL_CARBS[cause]
            meal_ts = onset - timedelta(minutes=75)
            meals.append(MealEvent(ts=meal_ts, carbs_g=carbs, note="dinner out"))
            if cause == "late_bolus":
                insulin.append(InsulinEvent(
                    ts=meal_ts + timedelta(minutes=55), kind=InsulinKind.BOLUS,
                    units=round(carbs / _carb_ratio_at(meal_ts), 2), automatic=False,
                ))
        elif cause == "over_correction":
            insulin.append(InsulinEvent(
                ts=onset - timedelta(minutes=125), kind=InsulinKind.BOLUS,
                units=round(2.0 + (extreme % 7) * 0.2, 2), automatic=False,
            ))
        else:  # post_exercise
            activity.append(ActivityEvent(
                ts=_at(day, max(0, hh - 2), mm), kind="run",
                duration_min=70.0, intensity=0.85,
            ))
    return meals, insulin, activity


def _ext_segments() -> list[list[tuple[datetime, float]]]:
    """Each extended excursion as a curve: ramp to the extreme, hold it ~30 min so
    the episode clears the 15-minute clinical bar and its severity band, recover."""
    segments: list[list[tuple[datetime, float]]] = []
    for off, hh, mm, extreme, _cause in _EXT_EXCURSIONS:
        center = _at(_ext_day(off), hh, mm)
        segments.append([
            (center - timedelta(minutes=40), 118.0),
            (center - timedelta(minutes=15), float(extreme)),
            (center + timedelta(minutes=15), float(extreme)),
            (center + timedelta(minutes=40), 120.0),
        ])
    return segments


def _drop_sensor_gap(glucose: list[GlucoseEvent]) -> list[GlucoseEvent]:
    """A 90-minute pre-dawn hole so the graph has a real sensor-gap node."""
    day = _day(_GAP_DAY_OFFSET)
    lo, hi = _at(day, 2, 0), _at(day, 3, 30)
    return [g for g in glucose if not (lo <= g.ts <= hi)]


def _story_events() -> tuple[list[MealEvent], list[InsulinEvent], list[ActivityEvent]]:
    """The context events that make the story days legible in the graph: rescue
    carbs bridging each rebound (unbolused, so they stay a bare meal edge), the
    manual correction stack, the in-gap correction that bridges the night low,
    and the runs that precede the post-workout lows."""
    meals: list[MealEvent] = []
    insulin: list[InsulinEvent] = []
    activity: list[ActivityEvent] = []
    for off in _CHAIN_DAY_OFFSETS:
        day = _day(off)
        meals.append(MealEvent(ts=_at(day, 16, 15), carbs_g=16.0, note="rescue carbs"))
        if off in _CHAIN_CORRECTION_OFFSETS:
            insulin.append(InsulinEvent(
                ts=_at(day, 17, 30), kind=InsulinKind.BOLUS, units=1.5, automatic=False,
            ))
    day = _day(_STACK_DAY_OFFSET)
    # A cleanly paired dinner treatment before the high, so the graph shows a
    # "treatment" edge: 52 g and its bolus three minutes later merge into one node.
    meals.append(MealEvent(ts=_at(day, 19, 30), carbs_g=52.0, note="dinner"))
    insulin.append(InsulinEvent(
        ts=_at(day, 19, 33), kind=InsulinKind.BOLUS, units=5.2, automatic=False,
    ))
    for hh, mm, units in (
        (21, 10, 1.6), (21, 35, 1.2), (21, 55, 1.0),
        (22, 15, 0.8), (22, 35, 0.9), (22, 55, 1.4),
    ):
        insulin.append(InsulinEvent(
            ts=_at(day, hh, mm), kind=InsulinKind.BOLUS, units=units, automatic=False,
        ))
    for off in _WORKOUT_LOW_OFFSETS:
        day = _day(off)
        activity.append(ActivityEvent(
            ts=_at(day, 15, 0), kind="run", duration_min=60.0, intensity=0.8,
        ))
    return meals, insulin, activity


def _demo_predictions(glucose: list[GlucoseEvent]) -> list[PredictionEvent]:
    """Logged forecast curves anchored at 21:00 on the miss days.

    Two oref curves per cycle: COB (carbs-as-announced) predicts a return toward
    range - a big miss, because the actual CGM stays high (see
    :func:`_with_prolonged_highs`); UAM (unannounced meal) tracks the high - a
    small miss. UAM fitting far better than COB is the signature reconciliation
    attributes to carb underestimation (the planted ground truth)."""
    by_ts = {g.ts: g.mg_dl for g in glucose}
    days = _miss_days()
    by_day: dict[date, list[datetime]] = {}
    for g in glucose:
        if g.ts.date() in days and g.ts.hour >= 21:
            by_day.setdefault(g.ts.date(), []).append(g.ts)
    out: list[PredictionEvent] = []
    horizon = 36  # 3h at 5-minute spacing
    for slots in by_day.values():
        cycle = min(slots)
        start_bg = by_ts[cycle]
        cob = [
            round(110.0 + (start_bg - 110.0) * math.exp(-3.0 * step / horizon), 1)
            for step in range(horizon)
        ]
        uam = [float(start_bg)] * horizon  # tracks the sustained high (small miss)
        out.append(
            PredictionEvent(ts=cycle, source="openaps", curve_kind="cob", values_mg_dl=cob)
        )
        out.append(
            PredictionEvent(ts=cycle, source="openaps", curve_kind="uam", values_mg_dl=uam)
        )
    return out


#: Realistic t:slim X2 time-of-day schedules: (start, basal U/hr, ISF mg/dL/U,
#: carb ratio g/U, target). Spring is less insulin-sensitive (lower ISF, tighter
#: carb ratio, higher basal) - the planted sensitivity shift mid-window.
_WINTER_SCHEDULE: tuple[tuple[str, float, int, int, int], ...] = (
    ("00:00", 0.70, 50, 12, 110),
    ("06:00", 0.95, 45, 10, 110),
    ("11:00", 0.85, 48, 11, 110),
    ("17:00", 0.80, 45, 11, 110),
)
_SPRING_SCHEDULE: tuple[tuple[str, float, int, int, int], ...] = (
    ("00:00", 0.80, 45, 11, 110),
    ("06:00", 1.05, 40, 9, 110),
    ("11:00", 0.95, 43, 10, 110),
    ("17:00", 0.90, 40, 10, 110),
)


def _segments(schedule: tuple[tuple[str, float, int, int, int], ...]) -> list[dict[str, object]]:
    return [
        {
            "start": start,
            "basal_u_hr": basal,
            "isf_mg_dl_u": isf,
            "carb_ratio_g_u": cr,
            "target_mg_dl": target,
        }
        for start, basal, isf, cr, target in schedule
    ]


def _profile_payload(
    name: str, schedule: tuple[tuple[str, float, int, int, int], ...]
) -> dict[str, object]:
    return {
        "active_profile": name,
        "pump_serial": "DEMO-CIQ-0001",
        "pump_model": "Tandem t:slim X2",
        "control_iq": True,
        "profiles": [
            {
                "name": name,
                "active": True,
                "dia_hr": 5.0,
                "max_bolus_u": 10.0,
                "segments": _segments(schedule),
            }
        ],
    }


def _carb_ratio_at(ts: datetime) -> float:
    """Carb ratio (g/U) in effect at ``ts`` from the dominant (Spring) schedule."""
    hour = ts.hour + ts.minute / 60.0
    cr = _SPRING_SCHEDULE[0][3]
    for start, _basal, _isf, ratio, _target in _SPRING_SCHEDULE:
        seg_hour = int(start[:2])
        if hour >= seg_hour:
            cr = ratio
    return float(cr)


def _basal_rate_at(ts: datetime) -> float:
    hour = ts.hour + ts.minute / 60.0
    rate = _SPRING_SCHEDULE[0][1]
    for start, basal, _isf, _ratio, _target in _SPRING_SCHEDULE:
        if hour >= int(start[:2]):
            rate = basal
    return float(rate)


def _demo_profiles() -> list[TherapyProfile]:
    """Two profile versions: a spring sensitivity change splits the window."""
    v1 = _profile_payload("Winter", _WINTER_SCHEDULE)
    v2 = _profile_payload("Spring", _SPRING_SCHEDULE)
    # Fixed switch date (not _DAYS//2) so the DEMO_SPIKE_DATE spike stays under the
    # Spring profile regardless of how long the record runs.
    mid = _START + timedelta(days=45)
    return [
        TherapyProfile(
            source="tandem",
            name="Winter",
            content=v1,
            content_hash=_hash(v1),
            active_from=_START,
            created_at=_START,
        ),
        TherapyProfile(
            source="tandem",
            name="Spring",
            content=v2,
            content_hash=_hash(v2),
            active_from=mid,
            created_at=mid,
        ),
    ]


def _hash(payload: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _demo_manual() -> list[ManualEvent]:
    """User-reported context aligned to the spike (the manual-context story)."""
    return [
        ManualEvent(
            event_type="meal",
            event_ts=DEMO_SPIKE_TS - timedelta(minutes=42),
            title="High-fat dinner",
            description="Pizza night, ate out",
            tags=["fat", "dinner"],
            created_at=DEMO_SPIKE_TS,
        ),
        ManualEvent(
            event_type="stress",
            event_ts=DEMO_SPIKE_TS - timedelta(hours=6),
            description="Stressful workday",
            tags=["stress"],
            created_at=DEMO_SPIKE_TS,
        ),
        ManualEvent(
            event_type="site_change",
            event_ts=datetime.combine(
                DEMO_SPIKE_DATE - timedelta(days=27), datetime.min.time(), tzinfo=UTC
            )
            + timedelta(hours=8),
            title="Infusion site change",
            tags=["site"],
            created_at=DEMO_SPIKE_TS,
        ),
    ]


def _tandem_treatment(rng: random.Random) -> tuple[list[MealEvent], list[InsulinEvent]]:
    """Fill out the Tandem t:slim X2 / Control-IQ treatment timeline around the
    hero dinners: breakfast, lunch, and dinner carb entries with carb-ratio-matched
    boluses, Control-IQ temp-basal adjustments through the day, occasional
    automatic corrections, and the rare low-glucose suspend.

    Dinner is skipped on the days the hero story plants one of its own, so those
    keep exactly the late-bolus shape :func:`_patient` gives them. Every other day
    gets one: a record where the patient eats breakfast and lunch and then nothing,
    five days in six, is not a record anyone should be shown as realistic.
    """
    meals: list[MealEvent] = []
    insulin: list[InsulinEvent] = []
    midnight = _START.replace(hour=0, minute=0)
    planted_dinners = set(_planted_dinner_days())
    for i in range(_DAYS):
        day = midnight + timedelta(days=i)
        slots = [(7.5, 45.0, "breakfast"), (12.5, 60.0, "lunch")]
        if day.date() not in planted_dinners:
            slots.append((19.5, 65.0, "dinner"))
        for hour, base_carbs, note in slots:
            meal_ts = day + timedelta(hours=hour, minutes=rng.uniform(-20.0, 20.0))
            carbs = round(max(10.0, base_carbs + rng.uniform(-12.0, 12.0)), 1)
            meals.append(MealEvent(ts=meal_ts, carbs_g=carbs, note=note))
            # Usually prompt, sometimes badly late. A record where every bolus
            # lands within eight minutes has no bolus-timing signal to discover:
            # the variation is what makes "late insulin spikes you" a finding
            # rather than an assertion.
            late = rng.random() < _LATE_BOLUS_SHARE
            delay = rng.uniform(20.0, 48.0) if late else rng.uniform(0.0, 9.0)
            bolus_ts = meal_ts + timedelta(minutes=delay)
            insulin.append(
                InsulinEvent(
                    ts=bolus_ts,
                    kind=InsulinKind.BOLUS,
                    units=round(carbs / _carb_ratio_at(meal_ts), 2),
                    automatic=False,
                )
            )
        # Control-IQ temp basals through the morning. Kept to the 03:00-13:00 band
        # so they never fall inside the +/-6h window explain_spike inspects around
        # the ~20:00 dinner spike (which must read "basal stable").
        for slot in (3, 6, 9, 12):
            ts = day + timedelta(hours=slot, minutes=rng.uniform(0.0, 55.0))
            rate = round(_basal_rate_at(ts) * rng.uniform(0.0, 1.8), 2)
            insulin.append(
                InsulinEvent(
                    ts=ts,
                    kind=InsulinKind.TEMP_BASAL,
                    units=rate,
                    duration_min=float(rng.choice([5, 10, 15, 30])),
                    automatic=True,
                )
            )
        if rng.random() < 0.33:  # automatic correction bolus, mid-morning
            ts = day + timedelta(hours=10, minutes=rng.uniform(0.0, 90.0))
            insulin.append(
                InsulinEvent(
                    ts=ts, kind=InsulinKind.BOLUS, units=round(rng.uniform(0.3, 1.2), 2),
                    automatic=True,
                )
            )
        if rng.random() < 0.11:  # low-glucose suspend, pre-dawn
            ts = day + timedelta(hours=4, minutes=rng.uniform(0.0, 60.0))
            insulin.append(
                InsulinEvent(
                    ts=ts, kind=InsulinKind.SUSPEND, units=0.0,
                    duration_min=float(rng.choice([15, 30, 45])),
                )
            )
    return meals, insulin


def seed_demo(store: StoragePort) -> None:
    """Load the synthetic patient into ``store`` (assumed already migrated).

    Backend-agnostic: works on the in-memory showcase store, a SQLite file, or
    Postgres. Beyond the hero CGM/insulin/meal timeline it adds a full Tandem
    t:slim X2 / Control-IQ treatment record (multi-segment profile, temp basals,
    corrections, suspends, three meals a day), sleep, activity, logged forecast
    curves, two therapy-profile versions, manual notes, the episode-graph story
    days (rebound chains bridged by rescue carbs, a stacked-correction evening
    ending in a night low, post-workout lows, one sensor gap), and an extended
    Mar-Jun 2026 spread of highs, very-highs, lows, and very-lows, each with the
    context that explains it - so every surface and every severity band has data."""
    # The treatment record is built first because the trace is generated from it:
    # every carb entry outside the planted dinners drives a postprandial response.
    rng = random.Random(_SEED + 1)  # separate stream so the hero timeline is unchanged
    extra_meals, extra_insulin = _tandem_treatment(rng)
    story_meals, story_insulin, story_activity = _story_events()
    ext_meals, ext_insulin, ext_activity = _ext_events()
    base_activity = _demo_activity(rng)
    other_meals = extra_meals + story_meals + ext_meals
    other_activity = base_activity + story_activity + ext_activity
    other_boluses = [
        i for i in extra_insulin + story_insulin + ext_insulin if i.kind == InsulinKind.BOLUS
    ]

    glucose, insulin, meals = _patient(other_meals, other_boluses, other_activity)
    glucose = _protect_hero_day(glucose)
    glucose = _with_prolonged_highs(glucose)
    # Story days and the extended spread overwrite the generated curve, so they
    # land last and stay jitter-free: their episode boundaries are contracts.
    glucose = _apply_segments(glucose, _story_segments())
    glucose = _apply_segments(glucose, _ext_segments())
    glucose = _drop_sensor_gap(glucose)
    store.insert_glucose(glucose)
    store.insert_insulin(insulin + extra_insulin + story_insulin + ext_insulin)
    store.insert_meals(meals + other_meals)
    store.insert_sleep(_demo_sleep(rng))
    store.insert_activity(other_activity)
    store.insert_predictions(_demo_predictions(glucose))
    for profile in _demo_profiles():
        store.add_profile_version(profile)
    for event in _demo_manual():
        store.add_manual_event(event)
    _seed_rollups(store, glucose)


def _seed_rollups(store: StoragePort, glucose: list[GlucoseEvent]) -> None:
    """Compute the daily rollups the demo would otherwise never get.

    Rollups are normally a side effect of connector sync, which demo mode
    disables. Without them every rollup-backed surface (dashboard time in
    range, goals, trends) reads empty on a fully seeded database.
    """
    days = sorted({g.ts.date() for g in glucose})
    if not days:
        return
    start = datetime.combine(days[0], datetime.min.time(), tzinfo=UTC)
    end = datetime.combine(days[-1], datetime.max.time(), tzinfo=UTC)

    glucose_by_day: dict[date, list[GlucoseEvent]] = {}
    for reading in store.get_glucose(start, end):
        glucose_by_day.setdefault(reading.ts.date(), []).append(reading)
    insulin_by_day: dict[date, list[InsulinEvent]] = {}
    for dose in store.get_insulin(start, end):
        insulin_by_day.setdefault(dose.ts.date(), []).append(dose)
    meals_by_day: dict[date, list[MealEvent]] = {}
    for meal in store.get_meals(start, end):
        meals_by_day.setdefault(meal.ts.date(), []).append(meal)

    rollups = [
        rollup
        for day in days
        if (
            rollup := daily_rollup(
                day,
                glucose_by_day.get(day, []),
                insulin=insulin_by_day.get(day, []),
                meals=meals_by_day.get(day, []),
            )
        )
        is not None
    ]
    if rollups:
        store.upsert_rollups(rollups)


def seed_demo_if_empty(store: StoragePort) -> bool:
    """Seed the synthetic patient only when ``store`` has no glucose yet.

    Returns whether it seeded, so a one-command demo is idempotent: the first
    `serve --demo` populates the database, restarts reuse it untouched. A
    database seeded before rollups were part of the seed is repaired in place
    rather than left with every rollup-backed surface reading empty."""
    coverage = store.coverage()
    if coverage.first_ts is not None:
        if coverage.last_ts is not None and not store.get_rollups(
            RollupPeriod.DAILY, coverage.first_ts, coverage.last_ts
        ):
            _seed_rollups(store, store.get_glucose(coverage.first_ts, coverage.last_ts))
        return False
    seed_demo(store)
    return True


def build_demo_store() -> SQLiteStore:
    """An in-memory, migrated store loaded with the synthetic patient.

    Deterministic: repeated calls produce byte-identical timelines. Fast (<2s)."""
    store = SQLiteStore(":memory:")
    store.migrate()
    seed_demo(store)
    return store
