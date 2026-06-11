"""Tests for the synthetic CGM generator and its planted-effect ground truth."""

from __future__ import annotations

from datetime import datetime, timedelta
from statistics import mean

import pytest
from pydantic import ValidationError

from dexta_intelligence.models import (
    ActivityEvent,
    GlucoseEvent,
    InsulinEvent,
    MealEvent,
    SleepEvent,
)
from dexta_intelligence.testing import (
    EventsByType,
    ScenarioManifest,
    generate_baseline,
    generate_null,
    scenario_post_workout_hypo,
    scenario_sensitivity_shift,
    scenario_sleep_quality,
    scenario_weekday_breakfast,
)
from dexta_intelligence.testing.synthetic import (
    DEFAULT_SLEEP_SCORE_THRESHOLD,
    GLUCOSE_MAX,
    GLUCOSE_MIN,
    SLOTS_PER_DAY,
)

SEED = 1234


def _glucose_by_ts(events: list[GlucoseEvent]) -> dict[datetime, int]:
    return {g.ts: g.mg_dl for g in events}


def _window_mean(by_ts: dict[datetime, int], start: datetime, end: datetime) -> float:
    vals = [v for ts, v in by_ts.items() if start <= ts <= end]
    return mean(vals)


# ─────────────────────────────────────────────────────────────────────────────
# Structure, clamps, UTC, model validation
# ─────────────────────────────────────────────────────────────────────────────


def test_baseline_shape_and_clamps() -> None:
    events = generate_baseline(seed=SEED, n_days=10)
    assert len(events["glucose"]) == 10 * SLOTS_PER_DAY
    assert all(GLUCOSE_MIN <= g.mg_dl <= GLUCOSE_MAX for g in events["glucose"])
    assert events["meal"] and events["insulin"] and events["sleep"]


def test_all_timestamps_are_utc() -> None:
    events = generate_baseline(seed=SEED, n_days=8)
    zero = timedelta(0)
    for g in events["glucose"]:
        assert g.ts.utcoffset() == zero
    for i in events["insulin"]:
        assert i.ts.utcoffset() == zero
    for m in events["meal"]:
        assert m.ts.utcoffset() == zero
    for a in events["activity"]:
        assert a.ts.utcoffset() == zero
    for s in events["sleep"]:
        assert s.ts_start.utcoffset() == zero
        assert s.ts_end.utcoffset() == zero


def test_models_validate() -> None:
    events = generate_baseline(seed=SEED, n_days=5)
    assert all(isinstance(g, GlucoseEvent) for g in events["glucose"])
    assert all(isinstance(i, InsulinEvent) for i in events["insulin"])
    assert all(isinstance(m, MealEvent) for m in events["meal"])
    assert all(isinstance(a, ActivityEvent) for a in events["activity"])
    assert all(isinstance(s, SleepEvent) for s in events["sleep"])
    # Round-tripping through the frozen models re-runs validation.
    for g in events["glucose"][:50]:
        assert GlucoseEvent(**g.model_dump()).mg_dl == g.mg_dl


# ─────────────────────────────────────────────────────────────────────────────
# Determinism
# ─────────────────────────────────────────────────────────────────────────────


def test_same_seed_identical_output() -> None:
    a = generate_baseline(seed=SEED, n_days=12)
    b = generate_baseline(seed=SEED, n_days=12)
    assert a["glucose"] == b["glucose"]
    assert a["insulin"] == b["insulin"]
    assert a["meal"] == b["meal"]
    assert a["activity"] == b["activity"]
    assert a["sleep"] == b["sleep"]


def test_different_seed_differs() -> None:
    a = generate_baseline(seed=SEED, n_days=12)
    b = generate_baseline(seed=SEED + 1, n_days=12)
    assert a["glucose"] != b["glucose"]


def test_manifest_is_frozen() -> None:
    _events, manifest = generate_null(seed=SEED, n_days=6)
    assert isinstance(manifest, ScenarioManifest)
    assert manifest.effects == []
    with pytest.raises(ValidationError):
        manifest.name = "mutated"  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# Planted effect: weekday breakfast spike
# ─────────────────────────────────────────────────────────────────────────────


def _weekday_breakfast_gap(events: EventsByType, weekday: int, window_min: float) -> float:
    by_ts = _glucose_by_ts(events["glucose"])
    target: list[float] = []
    other: list[float] = []
    for meal in events["meal"]:
        if meal.note != "breakfast":
            continue
        m = _window_mean(by_ts, meal.ts, meal.ts + timedelta(minutes=window_min))
        (target if meal.ts.weekday() == weekday else other).append(m)
    return mean(target) - mean(other)


def test_weekday_breakfast_planted_and_null() -> None:
    size = 40.0
    events, manifest = scenario_weekday_breakfast(
        seed=SEED, n_days=90, weekday=0, effect_size=size
    )
    planted = manifest.effect("weekday_breakfast_spike")
    assert planted is not None
    assert planted.effect_size == size
    assert planted.windows

    gap = _weekday_breakfast_gap(events, weekday=0, window_min=120.0)
    assert abs(gap - size) < 8.0

    null_events, null_manifest = generate_null(seed=SEED, n_days=90)
    assert null_manifest.effects == []
    null_gap = _weekday_breakfast_gap(null_events, weekday=0, window_min=120.0)
    assert abs(null_gap) < 8.0


# ─────────────────────────────────────────────────────────────────────────────
# Planted effect: post-workout delayed hypoglycemia
# ─────────────────────────────────────────────────────────────────────────────


def test_post_workout_hypo_planted_and_null() -> None:
    size = 45.0
    events, manifest = scenario_post_workout_hypo(
        seed=SEED, n_days=90, effect_size=size, delay_hours=5.0
    )
    planted = manifest.effect("post_workout_hypo")
    assert planted is not None and planted.windows

    # Same seed → identical activities/windows in the null set, so the windows
    # isolate the planted dip.
    null_events, _ = generate_null(seed=SEED, n_days=90)
    effect_by_ts = _glucose_by_ts(events["glucose"])
    null_by_ts = _glucose_by_ts(null_events["glucose"])

    effect_vals: list[float] = []
    null_vals: list[float] = []
    for w_start, w_end in planted.windows:
        effect_vals.append(_window_mean(effect_by_ts, w_start, w_end))
        null_vals.append(_window_mean(null_by_ts, w_start, w_end))
    drop = mean(null_vals) - mean(effect_vals)
    assert abs(drop - size) < 10.0


# ─────────────────────────────────────────────────────────────────────────────
# Planted effect: insulin-sensitivity regime shift
# ─────────────────────────────────────────────────────────────────────────────


def _pre_post_shift_gap(events: EventsByType, shift_start: datetime) -> float:
    pre = [g.mg_dl for g in events["glucose"] if g.ts < shift_start]
    post = [g.mg_dl for g in events["glucose"] if g.ts >= shift_start]
    return mean(post) - mean(pre)


def test_sensitivity_shift_planted_and_null() -> None:
    size = 30.0
    after_day = 45
    events, manifest = scenario_sensitivity_shift(
        seed=SEED, n_days=90, effect_size=size, after_day=after_day
    )
    planted = manifest.effect("sensitivity_regime_shift")
    assert planted is not None
    shift_start = planted.windows[0][0]
    assert shift_start == manifest.start + timedelta(days=after_day)

    gap = _pre_post_shift_gap(events, shift_start)
    assert abs(gap - size) < 8.0

    null_events, _ = generate_null(seed=SEED, n_days=90)
    null_gap = _pre_post_shift_gap(null_events, shift_start)
    assert abs(null_gap) < 8.0


# ─────────────────────────────────────────────────────────────────────────────
# Planted effect: sleep-quality association
# ─────────────────────────────────────────────────────────────────────────────


def _poor_minus_good_next_day(events: EventsByType, threshold: float, waking_h: float) -> float:
    by_ts = _glucose_by_ts(events["glucose"])
    poor: list[float] = []
    good: list[float] = []
    for sleep in events["sleep"]:
        assert sleep.score is not None
        m = _window_mean(by_ts, sleep.ts_end, sleep.ts_end + timedelta(hours=waking_h))
        (poor if sleep.score < threshold else good).append(m)
    return mean(poor) - mean(good)


def test_sleep_quality_planted_and_null() -> None:
    size = 35.0
    events, manifest = scenario_sleep_quality(seed=SEED, n_days=90, effect_size=size)
    planted = manifest.effect("sleep_quality_association")
    assert planted is not None
    threshold = float(planted.params["score_threshold"])

    gap = _poor_minus_good_next_day(events, threshold, waking_h=14.0)
    assert abs(gap - size) < 10.0

    null_events, _ = generate_null(seed=SEED, n_days=90)
    null_gap = _poor_minus_good_next_day(
        null_events, DEFAULT_SLEEP_SCORE_THRESHOLD, waking_h=14.0
    )
    assert abs(null_gap) < 10.0
