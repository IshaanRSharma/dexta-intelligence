"""Sanity pins for the golden datasets (WAVE5 §6).

Each test asserts the *planted* ground truth — the canonical numbers the
spike-explanation workflow must recover (peak 246, bolus 22 min late,
14 of 18 dinner spikes) — so eval failures upstream can never be blamed
on the substrate.
"""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from statistics import mean

import pytest
from tests.golden import generator as gen

from dexta_intelligence.models import InsulinKind

_MEAL_TS = datetime(2026, 3, 14, 20, 0, tzinfo=UTC)
_SPIKE_TS = datetime(2026, 3, 14, 20, 42, tzinfo=UTC)


@pytest.fixture(scope="module")
def lb() -> gen.GoldenDataset:
    return gen.late_bolus()


@pytest.fixture(scope="module")
def bd() -> gen.GoldenDataset:
    return gen.basal_drift()


@pytest.fixture(scope="module")
def mc() -> gen.GoldenDataset:
    return gen.missing_carb()


@pytest.fixture(scope="module")
def nl() -> gen.GoldenDataset:
    return gen.null()


@pytest.fixture(scope="module")
def ni() -> gen.GoldenDataset:
    return gen.no_insulin()


# ── late_bolus ───────────────────────────────────────────────────────────────


def test_late_bolus_has_18_dinner_meals(lb: gen.GoldenDataset) -> None:
    assert len(lb.meals) == 18
    for meal in lb.meals:
        assert time(19, 45) <= meal.ts.time() <= time(20, 30)
    assert lb.manifest["n_dinner_events"] == 18


def test_late_bolus_canonical_peak_is_246(lb: gen.GoldenDataset) -> None:
    by_ts = {g.ts: g.mg_dl for g in lb.glucose}
    assert by_ts[_SPIKE_TS] == 246
    window = [g.mg_dl for g in lb.glucose if _MEAL_TS <= g.ts <= _MEAL_TS + timedelta(hours=2)]
    assert max(window) == 246
    assert lb.manifest["peak"] == 246


def test_late_bolus_canonical_bolus_22_min_after_meal(lb: gen.GoldenDataset) -> None:
    assert any(m.ts == _MEAL_TS for m in lb.meals)
    boluses = [
        i for i in lb.insulin if i.kind == InsulinKind.BOLUS and i.ts.date() == _MEAL_TS.date()
    ]
    assert len(boluses) == 1
    assert boluses[0].ts - _MEAL_TS == timedelta(minutes=22)
    assert lb.manifest["bolus_delay_min"] == 22


def test_late_bolus_14_of_18_dinners_spike(lb: gen.GoldenDataset) -> None:
    spiking = 0
    for meal in lb.meals:
        window = [g.mg_dl for g in lb.glucose if meal.ts <= g.ts <= meal.ts + timedelta(hours=2)]
        if max(window) > 200:
            spiking += 1
    assert spiking == 14
    assert len(lb.meals) - spiking == 4


def test_late_bolus_delay_partition_matches_spike_partition(lb: gen.GoldenDataset) -> None:
    bolus_by_date = {i.ts.date(): i for i in lb.insulin if i.kind == InsulinKind.BOLUS}
    late = ontime = 0
    for meal in lb.meals:
        delay_min = (bolus_by_date[meal.ts.date()].ts - meal.ts).total_seconds() / 60
        if 20 <= delay_min <= 25:
            late += 1
        elif delay_min <= 5:
            ontime += 1
    assert (late, ontime) == (14, 4)


def test_late_bolus_basal_present_and_uniform(lb: gen.GoldenDataset) -> None:
    basals = [i for i in lb.insulin if i.kind == InsulinKind.BASAL]
    assert len(basals) == 90
    assert {b.units for b in basals} == {24.0}
    assert {b.duration_min for b in basals} == {1440.0}


# ── basal_drift ──────────────────────────────────────────────────────────────


def test_basal_drift_overnight_rise(bd: gen.GoldenDataset) -> None:
    early = [g.mg_dl for g in bd.glucose if time(1, 0) <= g.ts.time() < time(2, 0)]
    late = [g.mg_dl for g in bd.glucose if time(5, 0) <= g.ts.time() < time(6, 0)]
    assert early and late
    assert mean(late) - mean(early) > 30


def test_basal_drift_no_treatment_near_drift_window(bd: gen.GoldenDataset) -> None:
    # drift window 02:00-06:00 plus a ±2h guard band
    for meal in bd.meals:
        assert not time(0, 0) <= meal.ts.time() < time(8, 0)
    for ev in bd.insulin:
        if ev.kind == InsulinKind.BOLUS:
            assert not time(0, 0) <= ev.ts.time() < time(8, 0)


# ── missing_carb ─────────────────────────────────────────────────────────────


def test_missing_carb_boluses_without_carb_entries(mc: gen.GoldenDataset) -> None:
    assert mc.meals == []
    boluses = [i for i in mc.insulin if i.kind == InsulinKind.BOLUS]
    assert len(boluses) == 30
    for bolus in boluses:
        window = [g.mg_dl for g in mc.glucose if bolus.ts <= g.ts <= bolus.ts + timedelta(hours=2)]
        assert max(window) > 200


# ── null ─────────────────────────────────────────────────────────────────────


def test_null_has_no_spikes_and_ontime_boluses(nl: gen.GoldenDataset) -> None:
    assert max(g.mg_dl for g in nl.glucose) <= 200
    bolus_ts = {i.ts for i in nl.insulin if i.kind == InsulinKind.BOLUS}
    for meal in nl.meals:
        assert meal.ts + timedelta(minutes=1) in bolus_ts


# ── no_insulin ───────────────────────────────────────────────────────────────


def test_no_insulin_same_glucose_empty_treatment(
    ni: gen.GoldenDataset, lb: gen.GoldenDataset
) -> None:
    assert ni.glucose == lb.glucose
    assert ni.insulin == []
    assert ni.meals == []


# ── determinism + store loading ──────────────────────────────────────────────


@pytest.mark.parametrize("name", sorted(gen.BUILDERS))
def test_builders_are_deterministic(name: str) -> None:
    first, second = gen.BUILDERS[name](), gen.BUILDERS[name]()
    assert first.glucose == second.glucose
    assert first.insulin == second.insulin
    assert first.meals == second.meals
    assert first.manifest == second.manifest


def test_make_store_loads_named_dataset() -> None:
    store = gen.make_store("late_bolus")
    glucose = store.get_glucose(_MEAL_TS, _MEAL_TS + timedelta(hours=2))
    assert max(g.mg_dl for g in glucose) == 246
    meals = store.get_meals(datetime(2025, 12, 1, tzinfo=UTC), datetime(2026, 4, 1, tzinfo=UTC))
    assert len(meals) == 18
    boluses = [
        i
        for i in store.get_insulin(_MEAL_TS, _MEAL_TS + timedelta(hours=1))
        if i.kind == InsulinKind.BOLUS
    ]
    assert len(boluses) == 1
