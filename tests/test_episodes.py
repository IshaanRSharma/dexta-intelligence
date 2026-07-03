"""Temporal episode graph: excursion/gap segmentation and context edges.

These are the temporal-segmentation tasks the LLM-CGM benchmark found a code
agent gets wrong; here they are deterministic and unit-tested. Fixtures are
hand-built so episode boundaries, durations, severity, clinical significance,
sensor gaps, and the typed context edges are asserted exactly.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from dexta_intelligence.analytics.episodes import Episode, detect_episodes, summarize
from dexta_intelligence.models import (
    ActivityEvent,
    GlucoseEvent,
    InsulinEvent,
    InsulinKind,
    MealEvent,
    SleepEvent,
)
from dexta_intelligence.store.sqlite import SQLiteStore

START = datetime(2025, 1, 6, 0, 0, tzinfo=UTC)


def _ts(minute: int) -> datetime:
    return START + timedelta(minutes=minute)


def _store(
    glucose: list[tuple[int, int]], *,
    meals: list[MealEvent] | None = None,
    insulin: list[InsulinEvent] | None = None,
    activity: list[ActivityEvent] | None = None,
    sleep: list[SleepEvent] | None = None,
) -> SQLiteStore:
    """Build an in-memory store from (minute, mg_dl) readings + optional context."""
    s = SQLiteStore(":memory:")
    s.migrate()
    s.insert_glucose([GlucoseEvent(ts=_ts(m), mg_dl=v) for m, v in glucose])
    if meals:
        s.insert_meals(meals)
    if insulin:
        s.insert_insulin(insulin)
    if activity:
        s.insert_activity(activity)
    if sleep:
        s.insert_sleep(sleep)
    return s


def _detect(store: SQLiteStore) -> list[Episode]:
    return detect_episodes(store, _ts(-10), _ts(100000))


# ── excursion detection ───────────────────────────────────────────────────────


def test_hyper_run_is_one_episode() -> None:
    # 5 readings at 5-min cadence, all > 180: one hyper episode, 20-min span
    store = _store([(0, 150), (5, 200), (10, 220), (15, 210), (20, 205), (25, 150)])
    eps = [e for e in _detect(store) if e.kind == "hyper"]
    assert len(eps) == 1
    ep = eps[0]
    assert ep.start == _ts(5) and ep.end == _ts(20)
    assert ep.duration_min == 15.0
    assert ep.n_readings == 4
    assert ep.extreme_mg_dl == 220.0 and ep.extreme_ts == _ts(10)


def test_hypo_severity_and_threshold() -> None:
    # dips below 54 -> severe; strict < 70 boundary (70 is not hypo)
    store = _store([(0, 100), (5, 60), (10, 50), (15, 65), (20, 70), (25, 120)])
    eps = [e for e in _detect(store) if e.kind == "hypo"]
    assert len(eps) == 1
    assert eps[0].severe is True
    assert eps[0].extreme_mg_dl == 50.0

    mild = _store([(0, 100), (5, 65), (10, 66), (15, 68), (20, 120)])
    ep = next(e for e in _detect(mild) if e.kind == "hypo")
    assert ep.severe is False


def test_clinically_significant_threshold() -> None:
    long_run = _store([(0, 120)] + [(5 * i, 200) for i in range(1, 5)] + [(30, 120)])
    ep = next(e for e in _detect(long_run) if e.kind == "hyper")
    assert ep.duration_min >= 15.0 and ep.clinically_significant is True

    short_run = _store([(0, 120), (5, 200), (10, 205), (15, 120)])
    ep2 = next(e for e in _detect(short_run) if e.kind == "hyper")
    assert ep2.duration_min < 15.0 and ep2.clinically_significant is False


def test_separate_runs_are_separate_episodes() -> None:
    store = _store([(0, 200), (5, 205), (10, 120), (15, 210), (20, 220), (25, 120)])
    hyper = [e for e in _detect(store) if e.kind == "hyper"]
    assert len(hyper) == 2


def test_in_range_only_has_no_excursions() -> None:
    store = _store([(i * 5, 120) for i in range(6)])
    assert all(e.kind == "sensor_gap" for e in _detect(store)) or _detect(store) == []


# ── sensor gaps ───────────────────────────────────────────────────────────────


def test_sensor_gap_detected_above_threshold() -> None:
    store = _store([(0, 120), (5, 120), (60, 120), (65, 120)])  # 55-min gap
    gaps = [e for e in _detect(store) if e.kind == "sensor_gap"]
    assert len(gaps) == 1
    assert gaps[0].start == _ts(5) and gaps[0].end == _ts(60)
    assert gaps[0].duration_min == 55.0
    assert gaps[0].extreme_mg_dl is None


def test_small_gap_is_not_an_episode() -> None:
    store = _store([(0, 120), (5, 120), (25, 120), (30, 120)])  # 20-min gap < 30
    assert not any(e.kind == "sensor_gap" for e in _detect(store))


# ── context edges ─────────────────────────────────────────────────────────────


def test_meal_before_hyper_is_linked() -> None:
    meals = [MealEvent(ts=_ts(-20), carbs_g=45.0, note="breakfast")]
    store = _store([(0, 200), (5, 220), (10, 120)], meals=meals)
    ep = next(e for e in _detect(store) if e.kind == "hyper")
    meal_links = [link for link in ep.links if link.kind == "meal"]
    assert len(meal_links) == 1
    assert meal_links[0].offset_min == -20.0
    assert meal_links[0].detail["carbs_g"] == 45.0


def test_far_meal_is_not_linked() -> None:
    meals = [MealEvent(ts=_ts(-600), carbs_g=45.0, note="breakfast")]  # 10h before
    store = _store([(0, 200), (5, 220), (10, 120)], meals=meals)
    ep = next(e for e in _detect(store) if e.kind == "hyper")
    assert not [link for link in ep.links if link.kind == "meal"]


def test_activity_links_further_back_than_meals() -> None:
    # exercise 5h before a low is linked (activity pre-window is 6h)
    activity = [ActivityEvent(ts=_ts(-300), kind="run", intensity=0.6)]
    store = _store([(0, 100), (5, 60), (10, 62), (15, 120)], activity=activity)
    ep = next(e for e in _detect(store) if e.kind == "hypo")
    assert [link for link in ep.links if link.kind == "activity"]


def test_bolus_linked_and_typed() -> None:
    insulin = [InsulinEvent(ts=_ts(-30), kind=InsulinKind.BOLUS, units=6.0),
               InsulinEvent(ts=_ts(-30), kind=InsulinKind.BASAL, units=18.0)]
    store = _store([(0, 200), (5, 220), (10, 120)], insulin=insulin)
    ep = next(e for e in _detect(store) if e.kind == "hyper")
    bolus_links = [link for link in ep.links if link.kind == "bolus"]
    assert len(bolus_links) == 1  # basal is not a bolus edge
    assert bolus_links[0].detail["units"] == 6.0


def test_sleep_linked_on_overlap() -> None:
    sleep = [SleepEvent(ts_start=_ts(-30), ts_end=_ts(30), duration_min=60.0, score=40.0)]
    store = _store([(0, 60), (5, 55), (10, 120)], sleep=sleep)
    ep = next(e for e in _detect(store) if e.kind == "hypo")
    assert [link for link in ep.links if link.kind == "sleep"]


def test_sensor_gap_has_no_context_links() -> None:
    meals = [MealEvent(ts=_ts(30), carbs_g=45.0, note="lunch")]
    store = _store([(0, 120), (5, 120), (60, 120)], meals=meals)
    gap = next(e for e in _detect(store) if e.kind == "sensor_gap")
    assert gap.links == ()


# ── rollup, determinism, serialization ────────────────────────────────────────


def test_summarize_counts() -> None:
    store = _store([(0, 200), (5, 220), (10, 120), (15, 60), (20, 50), (25, 120)])
    summary = summarize(_detect(store))
    assert summary["num_hyper"] == 1
    assert summary["num_hypo"] == 1
    assert summary["n_severe_hypo"] == 1
    assert summary["longest_hyper_min"] == 5.0


def test_empty_store_returns_empty() -> None:
    store = SQLiteStore(":memory:")
    store.migrate()
    assert detect_episodes(store, _ts(0), _ts(100)) == []


def test_deterministic() -> None:
    readings = [(0, 200), (5, 220), (10, 120), (15, 60), (20, 50), (25, 120)]
    a = [e.to_dict() for e in _detect(_store(readings))]
    b = [e.to_dict() for e in _detect(_store(readings))]
    assert a == b


def test_to_dict_is_json_serializable() -> None:
    store = _store(
        [(0, 200), (5, 220), (10, 120)],
        meals=[MealEvent(ts=_ts(-15), carbs_g=45.0, note="breakfast")],
    )
    payload = [e.to_dict() for e in _detect(store)]
    assert json.loads(json.dumps(payload))
