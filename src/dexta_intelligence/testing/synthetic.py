"""Deterministic synthetic CGM generator with planted, ground-truth effects.

The evaluation framework (spec §14, E4/E5) scores discovery agents on whether
they recover *planted* patterns and, crucially, find *nothing* in null data.
That only works if the ground truth is non-LLM and exactly known. This module
is that ground truth.

Design
------
1. **Baseline first.** :func:`generate_baseline` synthesizes a realistic
   5-minute CGM series with diurnal structure (dawn phenomenon), meal
   excursions tied to emitted :class:`~dexta_intelligence.models.MealEvent`\\ s,
   insulin-driven descents tied to bolus
   :class:`~dexta_intelligence.models.InsulinEvent`\\ s, and AR(1) autocorrelated
   noise. Values are clamped to a physiological 40-400 mg/dL.
2. **Effects compose on top.** Each planted-effect injector is a small frozen
   dataclass parameterized by an effect size. They mutate the working glucose
   series additively, so any subset composes cleanly.
3. **Everything is recorded.** A scenario returns the events *and* a frozen
   :class:`ScenarioManifest` describing exactly what was planted (effect name,
   parameters, effect size, affected windows) so an eval harness can score
   recovery without re-deriving the truth.

Determinism: every draw comes from a single :class:`random.Random` seeded by
the caller. Same seed and parameters → byte-identical events.

Dependencies: Python standard library (``math``, ``random``, ``datetime``) plus
``pydantic``. No numpy.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, TypedDict

from pydantic import BaseModel, ConfigDict

from dexta_intelligence.models import (
    ActivityEvent,
    GlucoseEvent,
    InsulinEvent,
    InsulinKind,
    MealEvent,
    SleepEvent,
)

__all__ = [
    "DEFAULT_START",
    "BaselineConfig",
    "EventsByType",
    "PlantedEffect",
    "PostWorkoutHypo",
    "ScenarioManifest",
    "SensitivityRegimeShift",
    "SleepQualityAssociation",
    "WeekdayBreakfastSpike",
    "generate_baseline",
    "generate_dataset",
    "generate_null",
    "scenario_all",
    "scenario_post_workout_hypo",
    "scenario_sensitivity_shift",
    "scenario_sleep_quality",
    "scenario_weekday_breakfast",
]

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

#: Seconds per CGM slot — Dexcom/Libre native cadence is 5 minutes.
SLOT_SECONDS = 300
#: Slots per 24h day.
SLOTS_PER_DAY = 288

#: A Monday at midnight UTC, so weekday-keyed effects align predictably.
DEFAULT_START = datetime(2025, 1, 6, 0, 0, tzinfo=UTC)

#: Physiological clamp applied to every emitted reading (mg/dL).
GLUCOSE_MIN = 40
GLUCOSE_MAX = 400

#: Glucose response amplitudes used by the baseline kernels.
MG_DL_PER_CARB = 1.3
MG_DL_PER_UNIT = 12.0
MEAL_PEAK_MIN = 50.0
MEAL_HORIZON_MIN = 240.0
INSULIN_PEAK_MIN = 90.0
INSULIN_HORIZON_MIN = 300.0

#: Poor-sleep threshold used by the sleep-quality scenario and recoverable by
#: an eval harness via the manifest params.
DEFAULT_SLEEP_SCORE_THRESHOLD = 55.0

Window = tuple[datetime, datetime]


class EventsByType(TypedDict):
    """The five timeline event streams a scenario produces, keyed by kind."""

    glucose: list[GlucoseEvent]
    insulin: list[InsulinEvent]
    meal: list[MealEvent]
    activity: list[ActivityEvent]
    sleep: list[SleepEvent]


# ─────────────────────────────────────────────────────────────────────────────
# Manifest (frozen ground truth)
# ─────────────────────────────────────────────────────────────────────────────


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PlantedEffect(_FrozenModel):
    """One planted effect, recorded exactly as it was injected.

    ``windows`` enumerates every contiguous time span the effect touched, so a
    scorer can localize recovery without re-deriving the planting logic.
    """

    name: str
    effect_size: float
    params: dict[str, Any]
    windows: list[Window]
    description: str = ""


class ScenarioManifest(_FrozenModel):
    """Ground-truth description of a generated scenario.

    A null scenario carries an empty ``effects`` list — the assertion an eval
    harness scores against is "find nothing here".
    """

    name: str
    seed: int
    n_days: int
    start: datetime
    effects: list[PlantedEffect]

    def effect(self, name: str) -> PlantedEffect | None:
        """Return the first planted effect with ``name``, or ``None``."""
        return next((e for e in self.effects if e.name == name), None)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    """Tunable knobs for the baseline CGM series (no planted effects)."""

    fasting_mean: float = 110.0
    dawn_amplitude: float = 22.0
    dawn_peak_hour: float = 6.0
    dawn_width_hour: float = 1.5
    diurnal_amplitude: float = 6.0
    ar1_phi: float = 0.85
    ar1_sigma: float = 5.0
    carb_ratio_g_per_unit: float = 10.0
    basal_units_per_day: float = 18.0
    activity_prob_per_day: float = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Generation context
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class _GenContext:
    """Mutable working state shared by the baseline build and the injectors."""

    start: datetime
    n_days: int
    slots: list[datetime]
    glucose: list[float]
    meals: list[MealEvent]
    insulin: list[InsulinEvent]
    activities: list[ActivityEvent]
    sleep: list[SleepEvent]

    def _index_for(self, ts: datetime) -> int:
        offset = (ts - self.start).total_seconds()
        return round(offset / SLOT_SECONDS)

    def add_window(self, w_start: datetime, w_end: datetime, delta: float) -> None:
        """Add ``delta`` mg/dL to every slot whose timestamp lies in the window."""
        lo = max(0, self._index_for(w_start))
        hi = min(len(self.slots) - 1, self._index_for(w_end))
        for idx in range(lo, hi + 1):
            if w_start <= self.slots[idx] <= w_end:
                self.glucose[idx] += delta

    def add_kernel(self, anchor: datetime, amplitude: float, peak_min: float,
                   horizon_min: float) -> None:
        """Add an alpha-function impulse anchored at ``anchor`` (meals/insulin)."""
        lo = max(0, self._index_for(anchor))
        hi = min(len(self.slots) - 1, self._index_for(anchor + timedelta(minutes=horizon_min)))
        for idx in range(lo, hi + 1):
            dt_min = (self.slots[idx] - anchor).total_seconds() / 60.0
            self.glucose[idx] += amplitude * _alpha(dt_min, peak_min)


def _alpha(dt_min: float, peak_min: float) -> float:
    """Alpha-function impulse response: 0 at onset, peaks at ``peak_min`` (=1.0)."""
    if dt_min <= 0.0:
        return 0.0
    x = dt_min / peak_min
    return x * math.exp(1.0 - x)


# ─────────────────────────────────────────────────────────────────────────────
# Effect injectors (composable, parameterized by effect size)
# ─────────────────────────────────────────────────────────────────────────────


class Effect(Protocol):
    """A planted-effect injector: mutate the series, report what it planted."""

    def apply(self, ctx: _GenContext) -> PlantedEffect:
        """Mutate ``ctx.glucose`` in place and return the planted-effect record."""
        ...


@dataclass(frozen=True, slots=True)
class WeekdayBreakfastSpike:
    """Higher post-breakfast glucose on one weekday (e.g. Monday).

    ``effect_size`` is the mg/dL added flat across the post-breakfast window on
    the target weekday, so the day's breakfast-window mean exceeds other days'
    by roughly that amount.
    """

    weekday: int
    effect_size: float
    window_min: float = 120.0

    def apply(self, ctx: _GenContext) -> PlantedEffect:
        """Inflate the post-breakfast window on the configured weekday."""
        windows: list[Window] = []
        for meal in ctx.meals:
            if meal.note == "breakfast" and meal.ts.weekday() == self.weekday:
                w_start = meal.ts
                w_end = meal.ts + timedelta(minutes=self.window_min)
                ctx.add_window(w_start, w_end, self.effect_size)
                windows.append((w_start, w_end))
        return PlantedEffect(
            name="weekday_breakfast_spike",
            effect_size=self.effect_size,
            params={"weekday": self.weekday, "window_min": self.window_min},
            windows=windows,
            description=(
                f"+{self.effect_size:.0f} mg/dL across the {self.window_min:.0f}-min "
                f"post-breakfast window on weekday {self.weekday}"
            ),
        )


@dataclass(frozen=True, slots=True)
class PostWorkoutHypo:
    """Delayed lows a fixed number of hours after each activity.

    Models post-exercise insulin sensitization: ``effect_size`` mg/dL is
    subtracted across a window centered ``delay_hours`` after each activity.
    """

    effect_size: float
    delay_hours: float = 5.0
    half_width_min: float = 30.0

    def apply(self, ctx: _GenContext) -> PlantedEffect:
        """Carve a delayed hypoglycemic dip after every activity event."""
        windows: list[Window] = []
        for act in ctx.activities:
            center = act.ts + timedelta(hours=self.delay_hours)
            w_start = center - timedelta(minutes=self.half_width_min)
            w_end = center + timedelta(minutes=self.half_width_min)
            ctx.add_window(w_start, w_end, -self.effect_size)
            windows.append((w_start, w_end))
        return PlantedEffect(
            name="post_workout_hypo",
            effect_size=self.effect_size,
            params={"delay_hours": self.delay_hours, "half_width_min": self.half_width_min},
            windows=windows,
            description=(
                f"-{self.effect_size:.0f} mg/dL dip ~{self.delay_hours:.0f}h after each activity"
            ),
        )


@dataclass(frozen=True, slots=True)
class SensitivityRegimeShift:
    """A step increase in glucose after a given day (reduced insulin sensitivity).

    Same inputs, higher glucose: ``effect_size`` mg/dL is added to every slot at
    or after ``after_day`` days from the scenario start.
    """

    effect_size: float
    after_day: int

    def apply(self, ctx: _GenContext) -> PlantedEffect:
        """Add a constant offset to every slot in the post-shift regime."""
        shift_start = ctx.start + timedelta(days=self.after_day)
        w_end = ctx.slots[-1]
        ctx.add_window(shift_start, w_end, self.effect_size)
        return PlantedEffect(
            name="sensitivity_regime_shift",
            effect_size=self.effect_size,
            params={"after_day": self.after_day, "shift_start": shift_start.isoformat()},
            windows=[(shift_start, w_end)],
            description=(
                f"+{self.effect_size:.0f} mg/dL from day {self.after_day} onward "
                "(insulin-sensitivity regime shift)"
            ),
        )


@dataclass(frozen=True, slots=True)
class SleepQualityAssociation:
    """Higher next-day glucose following poor-sleep nights.

    For each sleep event scoring below ``score_threshold``, ``effect_size``
    mg/dL is added flat across the following waking day.
    """

    effect_size: float
    score_threshold: float = DEFAULT_SLEEP_SCORE_THRESHOLD
    waking_hours: float = 14.0

    def apply(self, ctx: _GenContext) -> PlantedEffect:
        """Inflate the day after each poor-sleep night."""
        windows: list[Window] = []
        for sleep in ctx.sleep:
            if sleep.score is not None and sleep.score < self.score_threshold:
                w_start = sleep.ts_end
                w_end = sleep.ts_end + timedelta(hours=self.waking_hours)
                ctx.add_window(w_start, w_end, self.effect_size)
                windows.append((w_start, w_end))
        return PlantedEffect(
            name="sleep_quality_association",
            effect_size=self.effect_size,
            params={
                "score_threshold": self.score_threshold,
                "waking_hours": self.waking_hours,
            },
            windows=windows,
            description=(
                f"+{self.effect_size:.0f} mg/dL across the waking day after nights "
                f"scoring < {self.score_threshold:.0f}"
            ),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Event synthesis (deterministic given the rng)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _MealPlan:
    """A meal's canonical time-of-day and carb load before per-day jitter."""

    note: str
    hour: float
    carbs_g: float


_MEAL_PLANS: tuple[_MealPlan, ...] = (
    _MealPlan("breakfast", 7.0, 45.0),
    _MealPlan("lunch", 12.5, 60.0),
    _MealPlan("dinner", 19.0, 70.0),
)


def _diurnal(hour: float, cfg: BaselineConfig) -> float:
    """Baseline fasting glucose at a given hour, with the dawn phenomenon."""
    dawn = cfg.dawn_amplitude * math.exp(
        -((hour - cfg.dawn_peak_hour) ** 2) / (2.0 * cfg.dawn_width_hour**2)
    )
    daily = cfg.diurnal_amplitude * math.sin(2.0 * math.pi * (hour - 3.0) / 24.0)
    return cfg.fasting_mean + dawn + daily


def _build_meals_and_insulin(
    rng: random.Random, start: datetime, n_days: int, cfg: BaselineConfig
) -> tuple[list[MealEvent], list[InsulinEvent]]:
    """Generate jittered meals plus matched boluses and a daily basal dose."""
    meals: list[MealEvent] = []
    insulin: list[InsulinEvent] = []
    for day in range(n_days):
        day_start = start + timedelta(days=day)
        for plan in _MEAL_PLANS:
            jitter_min = rng.uniform(-20.0, 20.0)
            carbs = max(5.0, plan.carbs_g + rng.uniform(-10.0, 10.0))
            ts = day_start + timedelta(hours=plan.hour, minutes=jitter_min)
            meals.append(MealEvent(ts=ts, carbs_g=round(carbs, 1), note=plan.note))
            units = round(carbs / cfg.carb_ratio_g_per_unit, 2)
            insulin.append(
                InsulinEvent(ts=ts, kind=InsulinKind.BOLUS, units=units, automatic=False)
            )
        insulin.append(
            InsulinEvent(
                ts=day_start + timedelta(hours=22),
                kind=InsulinKind.BASAL,
                units=round(cfg.basal_units_per_day, 2),
                duration_min=1440.0,
            )
        )
    return meals, insulin


def _build_activities(
    rng: random.Random, start: datetime, n_days: int, cfg: BaselineConfig
) -> list[ActivityEvent]:
    """Generate afternoon workouts on a deterministic subset of days."""
    activities: list[ActivityEvent] = []
    for day in range(n_days):
        if rng.random() >= cfg.activity_prob_per_day:
            continue
        day_start = start + timedelta(days=day)
        ts = day_start + timedelta(hours=14, minutes=rng.uniform(-45.0, 45.0))
        activities.append(
            ActivityEvent(
                ts=ts,
                kind=rng.choice(["run", "ride", "strength"]),
                duration_min=round(rng.uniform(30.0, 75.0), 1),
                intensity=round(rng.uniform(0.4, 0.9), 2),
            )
        )
    return activities


def _build_sleep(rng: random.Random, start: datetime, n_days: int) -> list[SleepEvent]:
    """Generate one scored sleep event per night, ending the following morning."""
    sleep: list[SleepEvent] = []
    for day in range(n_days - 1):
        day_start = start + timedelta(days=day)
        ts_start = day_start + timedelta(hours=22, minutes=30 + rng.uniform(-30.0, 30.0))
        ts_end = day_start + timedelta(days=1, hours=6, minutes=30 + rng.uniform(-30.0, 30.0))
        duration = (ts_end - ts_start).total_seconds() / 60.0
        sleep.append(
            SleepEvent(
                ts_start=ts_start,
                ts_end=ts_end,
                duration_min=round(duration, 1),
                score=round(rng.uniform(30.0, 95.0), 1),
            )
        )
    return sleep


def _build_baseline_series(
    ctx: _GenContext, rng: random.Random, cfg: BaselineConfig
) -> None:
    """Fill ``ctx.glucose`` with diurnal baseline + meal/insulin kernels + AR(1) noise."""
    for idx, ts in enumerate(ctx.slots):
        hour = ts.hour + ts.minute / 60.0
        ctx.glucose[idx] = _diurnal(hour, cfg)

    prev = 0.0
    for idx in range(len(ctx.slots)):
        eps = rng.gauss(0.0, cfg.ar1_sigma)
        prev = cfg.ar1_phi * prev + eps
        ctx.glucose[idx] += prev

    for meal in ctx.meals:
        if meal.carbs_g is not None:
            ctx.add_kernel(
                meal.ts, meal.carbs_g * MG_DL_PER_CARB, MEAL_PEAK_MIN, MEAL_HORIZON_MIN
            )
    for ins in ctx.insulin:
        if ins.kind is InsulinKind.BOLUS and ins.units is not None:
            ctx.add_kernel(
                ins.ts, -ins.units * MG_DL_PER_UNIT, INSULIN_PEAK_MIN, INSULIN_HORIZON_MIN
            )


def _finalize_glucose(ctx: _GenContext) -> list[GlucoseEvent]:
    """Clamp the working series to the physiological range and emit integer events."""
    events: list[GlucoseEvent] = []
    for ts, value in zip(ctx.slots, ctx.glucose, strict=True):
        clamped = min(float(GLUCOSE_MAX), max(float(GLUCOSE_MIN), value))
        events.append(GlucoseEvent(ts=ts, mg_dl=round(clamped)))
    return events


# ─────────────────────────────────────────────────────────────────────────────
# Public generators
# ─────────────────────────────────────────────────────────────────────────────


def _build_context(
    seed: int, n_days: int, start: datetime, cfg: BaselineConfig
) -> _GenContext:
    """Construct events and the baseline glucose series for ``n_days`` of data."""
    if n_days < 2:
        msg = "n_days must be at least 2 (sleep-to-next-day effects need a following day)"
        raise ValueError(msg)
    rng = random.Random(seed)
    n_slots = n_days * SLOTS_PER_DAY
    slots = [start + timedelta(seconds=SLOT_SECONDS * i) for i in range(n_slots)]
    meals, insulin = _build_meals_and_insulin(rng, start, n_days, cfg)
    activities = _build_activities(rng, start, n_days, cfg)
    sleep = _build_sleep(rng, start, n_days)
    ctx = _GenContext(
        start=start,
        n_days=n_days,
        slots=slots,
        glucose=[0.0] * n_slots,
        meals=meals,
        insulin=insulin,
        activities=activities,
        sleep=sleep,
    )
    _build_baseline_series(ctx, rng, cfg)
    return ctx


def _events_from_context(ctx: _GenContext) -> EventsByType:
    """Materialize the final, clamped event streams from a populated context."""
    return EventsByType(
        glucose=_finalize_glucose(ctx),
        insulin=ctx.insulin,
        meal=ctx.meals,
        activity=ctx.activities,
        sleep=ctx.sleep,
    )


def generate_dataset(
    *,
    seed: int,
    n_days: int = 90,
    effects: tuple[Effect, ...] = (),
    name: str = "scenario",
    start: datetime | None = None,
    config: BaselineConfig | None = None,
) -> tuple[EventsByType, ScenarioManifest]:
    """Generate a scenario: baseline data with ``effects`` planted on top.

    Returns the five event streams keyed by type and a frozen
    :class:`ScenarioManifest` recording exactly what was planted. Deterministic
    given ``seed`` and parameters.
    """
    cfg = config or BaselineConfig()
    start_ts = start or DEFAULT_START
    ctx = _build_context(seed, n_days, start_ts, cfg)
    planted = [effect.apply(ctx) for effect in effects]
    events = _events_from_context(ctx)
    manifest = ScenarioManifest(
        name=name, seed=seed, n_days=n_days, start=start_ts, effects=planted
    )
    return events, manifest


def generate_baseline(
    *,
    seed: int,
    n_days: int = 90,
    start: datetime | None = None,
    config: BaselineConfig | None = None,
) -> EventsByType:
    """Generate realistic baseline CGM + events with no planted effects."""
    events, _ = generate_dataset(
        seed=seed, n_days=n_days, name="baseline", start=start, config=config
    )
    return events


def generate_null(
    *,
    seed: int,
    n_days: int = 90,
    start: datetime | None = None,
    config: BaselineConfig | None = None,
) -> tuple[EventsByType, ScenarioManifest]:
    """Generate a null scenario: same machinery, an empty planted-effects manifest."""
    return generate_dataset(
        seed=seed, n_days=n_days, effects=(), name="null", start=start, config=config
    )


def scenario_weekday_breakfast(
    *,
    seed: int,
    n_days: int = 90,
    weekday: int = 0,
    effect_size: float = 40.0,
    start: datetime | None = None,
) -> tuple[EventsByType, ScenarioManifest]:
    """Scenario with a planted weekday-specific breakfast spike."""
    return generate_dataset(
        seed=seed,
        n_days=n_days,
        effects=(WeekdayBreakfastSpike(weekday=weekday, effect_size=effect_size),),
        name="weekday_breakfast",
        start=start,
    )


def scenario_post_workout_hypo(
    *,
    seed: int,
    n_days: int = 90,
    effect_size: float = 45.0,
    delay_hours: float = 5.0,
    start: datetime | None = None,
) -> tuple[EventsByType, ScenarioManifest]:
    """Scenario with planted post-workout delayed hypoglycemia."""
    return generate_dataset(
        seed=seed,
        n_days=n_days,
        effects=(PostWorkoutHypo(effect_size=effect_size, delay_hours=delay_hours),),
        name="post_workout_hypo",
        start=start,
    )


def scenario_sensitivity_shift(
    *,
    seed: int,
    n_days: int = 90,
    effect_size: float = 30.0,
    after_day: int | None = None,
    start: datetime | None = None,
) -> tuple[EventsByType, ScenarioManifest]:
    """Scenario with a planted insulin-sensitivity regime shift mid-window."""
    shift_day = after_day if after_day is not None else n_days // 2
    return generate_dataset(
        seed=seed,
        n_days=n_days,
        effects=(SensitivityRegimeShift(effect_size=effect_size, after_day=shift_day),),
        name="sensitivity_shift",
        start=start,
    )


def scenario_sleep_quality(
    *,
    seed: int,
    n_days: int = 90,
    effect_size: float = 35.0,
    score_threshold: float = DEFAULT_SLEEP_SCORE_THRESHOLD,
    start: datetime | None = None,
) -> tuple[EventsByType, ScenarioManifest]:
    """Scenario with a planted poor-sleep → higher-next-day association."""
    return generate_dataset(
        seed=seed,
        n_days=n_days,
        effects=(
            SleepQualityAssociation(effect_size=effect_size, score_threshold=score_threshold),
        ),
        name="sleep_quality",
        start=start,
    )


def scenario_all(
    *,
    seed: int,
    n_days: int = 120,
    start: datetime | None = None,
) -> tuple[EventsByType, ScenarioManifest]:
    """Scenario composing all four planted effects on one baseline."""
    effects: tuple[Effect, ...] = (
        WeekdayBreakfastSpike(weekday=0, effect_size=40.0),
        PostWorkoutHypo(effect_size=45.0, delay_hours=5.0),
        SensitivityRegimeShift(effect_size=30.0, after_day=n_days // 2),
        SleepQualityAssociation(effect_size=35.0),
    )
    return generate_dataset(
        seed=seed, n_days=n_days, effects=effects, name="all_effects", start=start
    )
