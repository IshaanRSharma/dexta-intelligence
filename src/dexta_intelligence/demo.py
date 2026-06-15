"""Synthetic patient for `dexta demo` — the zero-config try-it on-ramp.

Builds an in-memory :class:`SQLiteStore` loaded with ~90 days of 5-minute CGM
plus a planted recurring late-bolus dinner spike, enough that
:func:`~dexta_intelligence.investigations.spike.explain_spike` produces the
"late/insufficient meal insulin context" finding with high confidence.

Fully deterministic (seeded RNG, fixed dates — no ``random.random`` / ``now``).
This mirrors the ``late_bolus`` golden dataset; tests/ cannot be imported by
shipped code, so the planting logic lives here independently.
"""

from __future__ import annotations

import math
import random
from datetime import UTC, date, datetime, timedelta

from dexta_intelligence.models import GlucoseEvent, InsulinEvent, InsulinKind, MealEvent
from dexta_intelligence.store import SQLiteStore

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


def build_demo_store() -> SQLiteStore:
    """An in-memory, migrated store loaded with the synthetic patient.

    Deterministic: repeated calls produce byte-identical timelines. Fast (<2s)."""
    store = SQLiteStore(":memory:")
    store.migrate()
    glucose, insulin, meals = _patient()
    store.insert_glucose(glucose)
    store.insert_insulin(insulin)
    store.insert_meals(meals)
    return store
