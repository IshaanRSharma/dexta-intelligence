"""Synthetic patient for `dexta demo` - the zero-config try-it on-ramp.

Builds an in-memory :class:`SQLiteStore` loaded with ~90 days of 5-minute CGM
plus a planted recurring late-bolus dinner spike, enough that
:func:`~dexta_intelligence.investigations.spike.explain_spike` produces the
"late/insufficient meal insulin context" finding with high confidence.

Around that hero spike the store is populated so every surface has something to
show: sleep and activity context, logged forecast curves (so prediction
reconciliation has real material), two therapy-profile versions (so versioned
profiles matter), and a few user-reported manual notes aligned to the spike.

Fully deterministic (seeded RNG, fixed dates - no ``random.random`` / ``now``).
This mirrors the ``late_bolus`` golden dataset; tests/ cannot be imported by
shipped code, so the planting logic lives here independently.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
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
_BUMP_PRE = timedelta(minutes=20)
_BUMP_POST = timedelta(minutes=150)


def _baseline(ts: datetime) -> float:
    hour = ts.hour + ts.minute / 60
    return 124.0 + 14.0 * math.sin(2 * math.pi * (hour - 9.0) / 24)


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


def _trace(
    grid: list[datetime],
    bumps: list[tuple[datetime, datetime, float]],
    rng: random.Random,
) -> list[GlucoseEvent]:
    """CGM trace: diurnal baseline + jitter, with jitter-free planted excursions."""
    by_day: dict[date, list[tuple[datetime, datetime, float]]] = {}
    for anchor_ts, peak_ts, peak in bumps:
        by_day.setdefault(anchor_ts.date(), []).append((anchor_ts, peak_ts, peak))

    events: list[GlucoseEvent] = []
    for ts in grid:
        base = _baseline(ts)
        value = base
        planted = False
        for anchor_ts, peak_ts, peak in by_day.get(ts.date(), ()):
            if anchor_ts - _BUMP_PRE <= ts <= anchor_ts + _BUMP_POST:
                value = base + _bump(ts, peak_ts, peak - _baseline(peak_ts))
                planted = True
                break
        if not planted:
            value += rng.randint(-6, 6)
        events.append(GlucoseEvent(ts=ts, mg_dl=round(value)))
    return events


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


def _patient() -> tuple[list[GlucoseEvent], list[InsulinEvent], list[MealEvent]]:
    rng = random.Random(_SEED)
    dinner_days = [
        DEMO_SPIKE_DATE - timedelta(days=5 * (_N_DINNERS - 1 - i)) for i in range(_N_DINNERS)
    ]

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

    glucose = _trace(_grid(_START, _DAYS), bumps, rng)
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


def _with_prolonged_highs(glucose: list[GlucoseEvent]) -> list[GlucoseEvent]:
    """Elevate 21:00-24:00 on the miss days to a prolonged high the forecast
    fails to anticipate (the reconciliation ground truth)."""
    days = _miss_days()
    out: list[GlucoseEvent] = []
    for g in glucose:
        if g.ts.date() in days and 21 <= g.ts.hour < 24:
            out.append(g.model_copy(update={"mg_dl": min(350, g.mg_dl + _MISS_ELEVATION)}))
        else:
            out.append(g)
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


#: The extended record (Mar 15 - ~Jun 16 2026): a spread of excursions across every
#: severity band so the demo shows regular and severe highs and lows beyond the
#: story window. (day offset from _START, hour, minute, extreme mg/dL). Highs >180,
#: very-high >250, lows <70, very-low <54. Each is held ~30 min so it registers as a
#: clinically significant (or severe) episode. Spaced so none overlap.
_EXT_EXCURSIONS: tuple[tuple[int, int, int, int], ...] = (
    (93, 20, 0, 216),    # high
    (97, 8, 30, 62),     # low
    (101, 21, 30, 271),  # very high
    (106, 3, 0, 48),     # very low (overnight)
    (111, 13, 0, 228),   # high
    (116, 16, 30, 57),   # low
    (122, 20, 30, 241),  # high
    (127, 2, 30, 44),    # very low (overnight)
    (133, 19, 0, 293),   # very high
    (139, 7, 0, 66),     # low
    (145, 21, 0, 207),   # high
    (151, 14, 30, 51),   # very low
    (157, 20, 0, 233),   # high
    (163, 4, 0, 60),     # low
    (169, 19, 30, 264),  # very high
    (175, 15, 30, 55),   # low
    (181, 20, 0, 221),   # high
)


def _ext_day(offset: int) -> date:
    return (_START + timedelta(days=offset)).date()


def _with_extended_excursions(glucose: list[GlucoseEvent]) -> list[GlucoseEvent]:
    """Overwrite the extended (Mar-Jun) window with the excursion spread. Each curve
    ramps to its extreme, holds it ~30 min (so the episode clears the 15-min clinical
    bar and any severe band), then recovers. Jitter-free so boundaries are stable."""
    segments: list[list[tuple[datetime, float]]] = []
    for off, hh, mm, extreme in _EXT_EXCURSIONS:
        center = _at(_ext_day(off), hh, mm)
        segments.append([
            (center - timedelta(minutes=40), 118.0),
            (center - timedelta(minutes=15), float(extreme)),
            (center + timedelta(minutes=15), float(extreme)),
            (center + timedelta(minutes=40), 120.0),
        ])
    out: list[GlucoseEvent] = []
    for g in glucose:
        value: float | None = None
        for points in segments:
            if points[0][0] <= g.ts <= points[-1][0]:
                value = _interp(g.ts, points)
                break
        out.append(g if value is None else g.model_copy(update={"mg_dl": round(value)}))
    return out


def _with_story_days(glucose: list[GlucoseEvent]) -> list[GlucoseEvent]:
    """Overwrite the story windows with their hand-shaped curves: rebound chains
    (low, rescue carbs, high), one stacked-correction evening ending in a night
    low, and post-workout lows. Jitter-free so episode boundaries are stable."""
    segments = _story_segments()
    out: list[GlucoseEvent] = []
    for g in glucose:
        value: float | None = None
        for points in segments:
            if points[0][0] <= g.ts <= points[-1][0]:
                value = _interp(g.ts, points)
                break
        out.append(g if value is None else g.model_copy(update={"mg_dl": round(value)}))
    return out


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
    hero dinners: breakfast and lunch carb entries with carb-ratio-matched
    boluses, Control-IQ temp-basal adjustments through the day, occasional
    automatic corrections, and the rare low-glucose suspend."""
    meals: list[MealEvent] = []
    insulin: list[InsulinEvent] = []
    midnight = _START.replace(hour=0, minute=0)
    for i in range(_DAYS):
        day = midnight + timedelta(days=i)
        for hour, base_carbs, note in ((7.5, 45.0, "breakfast"), (12.5, 60.0, "lunch")):
            meal_ts = day + timedelta(hours=hour, minutes=rng.uniform(-20.0, 20.0))
            carbs = round(max(10.0, base_carbs + rng.uniform(-12.0, 12.0)), 1)
            meals.append(MealEvent(ts=meal_ts, carbs_g=carbs, note=note))
            bolus_ts = meal_ts + timedelta(minutes=rng.uniform(0.0, 8.0))
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
    Mar-Jun 2026 spread of highs, very-highs, lows, and very-lows - so every
    surface and every severity band has data."""
    glucose, insulin, meals = _patient()
    glucose = _with_prolonged_highs(glucose)
    glucose = _with_story_days(glucose)
    glucose = _with_extended_excursions(glucose)
    glucose = _drop_sensor_gap(glucose)
    store.insert_glucose(glucose)
    rng = random.Random(_SEED + 1)  # separate stream so the hero timeline is unchanged
    extra_meals, extra_insulin = _tandem_treatment(rng)
    story_meals, story_insulin, story_activity = _story_events()
    store.insert_insulin(insulin + extra_insulin + story_insulin)
    store.insert_meals(meals + extra_meals + story_meals)
    store.insert_sleep(_demo_sleep(rng))
    store.insert_activity(_demo_activity(rng) + story_activity)
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
