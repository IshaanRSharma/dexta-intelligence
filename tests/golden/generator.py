"""Golden datasets — planted ground truth for Wave 5 spike explanation (WAVE5 §6).

Five deterministic builders, each planting a known contributor (or none) into a
synthetic timeline. The manifest records the planted numbers so tests and evals
assert against ground truth, never against model output. The canonical plant:
dinner on 2026-03-14 at 20:00, bolus 22 min late, spike peaking at exactly
246 mg/dL at 20:42 — recurring across 14 of 18 dinners.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

from dexta_intelligence.models import GlucoseEvent, InsulinEvent, InsulinKind, MealEvent
from dexta_intelligence.store import SQLiteStore

if TYPE_CHECKING:
    from collections.abc import Callable

    from dexta_intelligence.store.port import StoragePort

__all__ = [
    "BUILDERS",
    "GoldenDataset",
    "basal_drift",
    "late_bolus",
    "load_golden",
    "make_store",
    "missing_carb",
    "no_insulin",
    "null",
]

GRID_MIN = 5
SPIKE_TS = datetime(2026, 3, 14, 20, 42, tzinfo=UTC)
SPIKE_PEAK = 246
BOLUS_DELAY_MIN = 22
N_DINNERS = 18
N_SPIKING = 14
SPIKE_THRESHOLD = 200
BASAL_UNITS = 24.0

_LB_START = datetime(2025, 12, 15, 0, 2, tzinfo=UTC)
_LB_DAYS = 90
_30D_START = datetime(2026, 2, 13, 0, 2, tzinfo=UTC)
_30D_DAYS = 30
_ONTIME_IDX = frozenset({3, 7, 11, 15})
_BUMP_PRE = timedelta(minutes=20)
_BUMP_POST = timedelta(minutes=150)


@dataclass(frozen=True)
class GoldenDataset:
    name: str
    glucose: list[GlucoseEvent]
    insulin: list[InsulinEvent]
    meals: list[MealEvent]
    manifest: dict[str, Any]


def _baseline(ts: datetime) -> float:
    """Plausible diurnal baseline, 110-138 mg/dL."""
    hour = ts.hour + ts.minute / 60
    return 124.0 + 14.0 * math.sin(2 * math.pi * (hour - 9.0) / 24)


def _grid(start: datetime, days: int) -> list[datetime]:
    return [start + timedelta(minutes=GRID_MIN * i) for i in range(days * 24 * 60 // GRID_MIN)]


def _snap(ts: datetime, start: datetime) -> datetime:
    steps = round((ts - start).total_seconds() / 60 / GRID_MIN)
    return start + timedelta(minutes=GRID_MIN * steps)


def _bump(ts: datetime, peak_ts: datetime, amplitude: float, sigma_min: float = 25.0) -> float:
    offset_min = (ts - peak_ts).total_seconds() / 60
    return amplitude * math.exp(-((offset_min / sigma_min) ** 2))


def _trace(
    grid: list[datetime],
    bumps: list[tuple[datetime, datetime, float]],
    rng: random.Random,
    extra: Callable[[datetime], float] | None = None,
) -> list[GlucoseEvent]:
    """CGM trace: diurnal baseline + jitter, with jitter-free planted excursions.

    ``bumps`` entries are ``(anchor_ts, peak_ts, peak_target)``; inside the
    anchor's window the value is exact so planted peaks pin to the target.
    """
    by_day: dict[date, list[tuple[datetime, datetime, float]]] = {}
    for anchor_ts, peak_ts, peak in bumps:
        by_day.setdefault(anchor_ts.date(), []).append((anchor_ts, peak_ts, peak))

    events: list[GlucoseEvent] = []
    for ts in grid:
        base = _baseline(ts)
        if extra is not None:
            base += extra(ts)
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
            units=BASAL_UNITS,
            duration_min=1440.0,
        )
        for i in range(days)
    ]


def _dinner_ts(day: date, idx: int) -> datetime:
    """Dinner entry between 19:45 and 20:30."""
    base = datetime(day.year, day.month, day.day, 19, 45, tzinfo=UTC)
    return base + timedelta(minutes=(idx * 11) % 46)


def late_bolus() -> GoldenDataset:
    rng = random.Random(514)
    dinner_days = [
        SPIKE_TS.date() - timedelta(days=5 * (N_DINNERS - 1 - i)) for i in range(N_DINNERS)
    ]

    meals: list[MealEvent] = []
    boluses: list[InsulinEvent] = []
    bumps: list[tuple[datetime, datetime, float]] = []
    for idx, day in enumerate(dinner_days):
        if day == SPIKE_TS.date():
            meal_ts = datetime(2026, 3, 14, 20, 0, tzinfo=UTC)
            delay_min, peak = BOLUS_DELAY_MIN, float(SPIKE_PEAK)
        elif idx not in _ONTIME_IDX:
            meal_ts = _dinner_ts(day, idx)
            delay_min, peak = 20 + idx % 6, 212.0 + (idx * 5) % 29
        else:
            meal_ts = _dinner_ts(day, idx)
            delay_min, peak = 1, 168.0 + (idx * 3) % 12
        meals.append(MealEvent(ts=meal_ts, carbs_g=55.0 + (idx * 7) % 20, note="dinner"))
        bolus_ts = meal_ts + timedelta(minutes=delay_min)
        boluses.append(InsulinEvent(ts=bolus_ts, kind=InsulinKind.BOLUS, units=6.0))
        bumps.append((meal_ts, _snap(meal_ts + timedelta(minutes=42), _LB_START), peak))

    manifest = {
        "spike_ts": SPIKE_TS.isoformat(),
        "peak": SPIKE_PEAK,
        "bolus_delay_min": BOLUS_DELAY_MIN,
        "n_dinner_events": N_DINNERS,
        "n_spiking": N_SPIKING,
        "spike_threshold": SPIKE_THRESHOLD,
        "basal_units_per_day": BASAL_UNITS,
        "dinner_dates": [d.isoformat() for d in dinner_days],
        "ontime_dates": [dinner_days[i].isoformat() for i in sorted(_ONTIME_IDX)],
    }
    return GoldenDataset(
        name="late_bolus",
        glucose=_trace(_grid(_LB_START, _LB_DAYS), bumps, rng),
        insulin=_daily_basal(_LB_START, _LB_DAYS) + boluses,
        meals=meals,
        manifest=manifest,
    )


def _overnight_drift(ts: datetime) -> float:
    hour = ts.hour + ts.minute / 60
    if 2.0 <= hour < 6.0:
        return 55.0 * (hour - 2.0) / 4.0
    if 6.0 <= hour < 8.0:
        return 55.0 * (8.0 - hour) / 2.0
    return 0.0


def basal_drift() -> GoldenDataset:
    rng = random.Random(515)
    grid = _grid(_30D_START, _30D_DAYS)

    meals: list[MealEvent] = []
    boluses: list[InsulinEvent] = []
    bumps: list[tuple[datetime, datetime, float]] = []
    for i in range(_30D_DAYS):
        day = (_30D_START + timedelta(days=i)).date()
        for hour, minute, carbs in ((12, 30, 45.0), (19, 0, 60.0)):
            meal_ts = datetime(day.year, day.month, day.day, hour, minute, tzinfo=UTC)
            meals.append(MealEvent(ts=meal_ts, carbs_g=carbs))
            boluses.append(
                InsulinEvent(ts=meal_ts + timedelta(minutes=1), kind=InsulinKind.BOLUS, units=5.0)
            )
            peak_ts = _snap(meal_ts + timedelta(minutes=45), _30D_START)
            bumps.append((meal_ts, peak_ts, _baseline(peak_ts) + 45.0))

    manifest = {
        "drift_window": ["02:00", "06:00"],
        "drift_rise_mg_dl": 55,
        "n_nights": _30D_DAYS,
        "meal_times": ["12:30", "19:00"],
        "spike_threshold": SPIKE_THRESHOLD,
    }
    return GoldenDataset(
        name="basal_drift",
        glucose=_trace(grid, bumps, rng, extra=_overnight_drift),
        insulin=_daily_basal(_30D_START, _30D_DAYS) + boluses,
        meals=meals,
        manifest=manifest,
    )


def missing_carb() -> GoldenDataset:
    rng = random.Random(516)
    grid = _grid(_30D_START, _30D_DAYS)

    boluses: list[InsulinEvent] = []
    bumps: list[tuple[datetime, datetime, float]] = []
    for i in range(_30D_DAYS):
        day = (_30D_START + timedelta(days=i)).date()
        bolus_ts = datetime(day.year, day.month, day.day, 19, 2, tzinfo=UTC)
        boluses.append(InsulinEvent(ts=bolus_ts, kind=InsulinKind.BOLUS, units=6.0))
        bumps.append((bolus_ts, _snap(bolus_ts + timedelta(minutes=45), _30D_START), 232.0))

    manifest = {
        "n_boluses": _30D_DAYS,
        "n_meals": 0,
        "bolus_time": "19:02",
        "spike_peak": 232,
        "spike_threshold": SPIKE_THRESHOLD,
    }
    return GoldenDataset(
        name="missing_carb",
        glucose=_trace(grid, bumps, rng),
        insulin=_daily_basal(_30D_START, _30D_DAYS) + boluses,
        meals=[],
        manifest=manifest,
    )


def null() -> GoldenDataset:
    rng = random.Random(517)
    grid = _grid(_30D_START, _30D_DAYS)

    meals: list[MealEvent] = []
    boluses: list[InsulinEvent] = []
    bumps: list[tuple[datetime, datetime, float]] = []
    for i in range(_30D_DAYS):
        day = (_30D_START + timedelta(days=i)).date()
        for hour, minute, carbs in ((8, 0, 40.0), (12, 30, 35.0), (19, 0, 55.0)):
            meal_ts = datetime(day.year, day.month, day.day, hour, minute, tzinfo=UTC)
            meals.append(MealEvent(ts=meal_ts, carbs_g=carbs))
            boluses.append(
                InsulinEvent(ts=meal_ts + timedelta(minutes=1), kind=InsulinKind.BOLUS, units=4.0)
            )
            bumps.append((meal_ts, _snap(meal_ts + timedelta(minutes=45), _30D_START), 170.0))

    manifest = {
        "n_days": _30D_DAYS,
        "n_meals": _30D_DAYS * 3,
        "bolus_delay_min": 1,
        "max_peak": 170,
        "spike_threshold": SPIKE_THRESHOLD,
    }
    return GoldenDataset(
        name="null",
        glucose=_trace(grid, bumps, rng),
        insulin=_daily_basal(_30D_START, _30D_DAYS) + boluses,
        meals=meals,
        manifest=manifest,
    )


def no_insulin() -> GoldenDataset:
    base = late_bolus()
    manifest = {
        "spike_ts": SPIKE_TS.isoformat(),
        "peak": SPIKE_PEAK,
        "n_insulin": 0,
        "n_meals": 0,
        "spike_threshold": SPIKE_THRESHOLD,
    }
    return GoldenDataset(
        name="no_insulin", glucose=base.glucose, insulin=[], meals=[], manifest=manifest
    )


BUILDERS = {
    "late_bolus": late_bolus,
    "basal_drift": basal_drift,
    "missing_carb": missing_carb,
    "null": null,
    "no_insulin": no_insulin,
}


def load_golden(store: StoragePort, dataset: GoldenDataset) -> None:
    store.insert_glucose(dataset.glucose)
    if dataset.insulin:
        store.insert_insulin(dataset.insulin)
    if dataset.meals:
        store.insert_meals(dataset.meals)


def make_store(name: str) -> SQLiteStore:
    store = SQLiteStore(":memory:")
    store.migrate()
    load_golden(store, BUILDERS[name]())
    return store
