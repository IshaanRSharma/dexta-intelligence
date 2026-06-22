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
from typing import TYPE_CHECKING

from dexta_intelligence.models import (
    ActivityEvent,
    GlucoseEvent,
    InsulinEvent,
    InsulinKind,
    ManualEvent,
    MealEvent,
    PredictionEvent,
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
_DAYS = 90
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
    mid = _START + timedelta(days=_DAYS // 2)
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
    curves, two therapy-profile versions, and manual notes - so every surface has
    data."""
    glucose, insulin, meals = _patient()
    glucose = _with_prolonged_highs(glucose)
    store.insert_glucose(glucose)
    rng = random.Random(_SEED + 1)  # separate stream so the hero timeline is unchanged
    extra_meals, extra_insulin = _tandem_treatment(rng)
    store.insert_insulin(insulin + extra_insulin)
    store.insert_meals(meals + extra_meals)
    store.insert_sleep(_demo_sleep(rng))
    store.insert_activity(_demo_activity(rng))
    store.insert_predictions(_demo_predictions(glucose))
    for profile in _demo_profiles():
        store.add_profile_version(profile)
    for event in _demo_manual():
        store.add_manual_event(event)


def seed_demo_if_empty(store: StoragePort) -> bool:
    """Seed the synthetic patient only when ``store`` has no glucose yet.

    Returns whether it seeded, so a one-command demo is idempotent: the first
    `serve --demo` populates the database, restarts reuse it untouched."""
    if store.coverage().first_ts is not None:
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
