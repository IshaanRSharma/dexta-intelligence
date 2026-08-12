"""Temporal episode graph: excursion/gap segmentation and context edges.

These are the temporal-segmentation tasks the LLM-CGM benchmark found a code
agent gets wrong; here they are deterministic and unit-tested. Fixtures are
hand-built so episode boundaries, durations, severity, clinical significance,
sensor gaps, and the typed context edges are asserted exactly.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.agents.tools import build_belt
from dexta_intelligence.agents.tools.episodes import episode_specs
from dexta_intelligence.agents.tools.toolkit import DiscoveryToolkit
from dexta_intelligence.analytics.episodes import (
    Episode,
    EpisodeGraph,
    build_graph,
    detect_episodes,
    pair_treatments,
    summarize,
)
from dexta_intelligence.coldstart import ColdStartReport
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


def test_excursion_splits_on_intra_run_sensor_gap() -> None:
    # A low, sensor dark 50 min, low again: two observed hypo episodes, not one
    # whose duration silently spans the hole (which would falsely read as long).
    store = _store([(0, 60), (5, 55), (10, 65), (60, 60), (65, 55), (70, 68)])
    hypo = [e for e in _detect(store) if e.kind == "hypo"]
    assert len(hypo) == 2
    assert hypo[0].end == _ts(10) and hypo[1].start == _ts(60)
    assert all(e.duration_min <= 15.0 for e in hypo)  # neither spans the 60-min hole
    assert any(e.kind == "sensor_gap" for e in _detect(store))


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


# ── treatment pairing ─────────────────────────────────────────────────────────


def test_meal_and_close_bolus_pair_into_one_treatment_edge() -> None:
    meals = [MealEvent(ts=_ts(-20), carbs_g=58.0, note="dinner")]
    insulin = [InsulinEvent(ts=_ts(-17), kind=InsulinKind.BOLUS, units=6.0)]
    store = _store([(0, 200), (5, 220), (10, 120)], meals=meals, insulin=insulin)
    ep = next(e for e in _detect(store) if e.kind == "hyper")
    kinds = [link.kind for link in ep.links]
    assert kinds == ["treatment"]
    t = ep.links[0]
    assert t.offset_min == -20.0  # anchored to the meal
    assert t.detail["carbs_g"] == 58.0
    assert t.detail["units"] == 6.0
    assert t.detail["bolus_dt_min"] == 3.0


def test_shared_raw_event_id_pairs_beyond_the_time_window() -> None:
    meals = [MealEvent(ts=_ts(-40), carbs_g=30.0, raw_event_id=7)]
    insulin = [InsulinEvent(ts=_ts(-10), kind=InsulinKind.BOLUS, units=3.0, raw_event_id=7)]
    store = _store([(0, 200), (5, 220), (10, 120)], meals=meals, insulin=insulin)
    ep = next(e for e in _detect(store) if e.kind == "hyper")
    assert [link.kind for link in ep.links] == ["treatment"]
    assert ep.links[0].detail["bolus_dt_min"] == 30.0


def test_ambiguous_boluses_pair_nothing() -> None:
    # Two manual boluses inside the window of one meal: conservative, keep all
    # three as separate edges rather than guessing which one was the meal bolus.
    meals = [MealEvent(ts=_ts(-20), carbs_g=58.0)]
    insulin = [
        InsulinEvent(ts=_ts(-18), kind=InsulinKind.BOLUS, units=4.0),
        InsulinEvent(ts=_ts(-12), kind=InsulinKind.BOLUS, units=2.0),
    ]
    store = _store([(0, 200), (5, 220), (10, 120)], meals=meals, insulin=insulin)
    ep = next(e for e in _detect(store) if e.kind == "hyper")
    assert sorted(link.kind for link in ep.links) == ["bolus", "bolus", "meal"]


def test_automatic_bolus_never_time_pairs() -> None:
    meals = [MealEvent(ts=_ts(-20), carbs_g=58.0)]
    insulin = [InsulinEvent(ts=_ts(-18), kind=InsulinKind.BOLUS, units=1.5, automatic=True)]
    store = _store([(0, 200), (5, 220), (10, 120)], meals=meals, insulin=insulin)
    ep = next(e for e in _detect(store) if e.kind == "hyper")
    assert sorted(link.kind for link in ep.links) == ["bolus", "meal"]


def test_distant_meal_and_bolus_stay_separate() -> None:
    # 20 minutes apart without a shared raw id: outside the pairing window, so
    # the missed-bolus signal (a late correction is not a meal bolus) survives.
    meals = [MealEvent(ts=_ts(-40), carbs_g=58.0)]
    insulin = [InsulinEvent(ts=_ts(-20), kind=InsulinKind.BOLUS, units=6.0)]
    store = _store([(0, 200), (5, 220), (10, 120)], meals=meals, insulin=insulin)
    ep = next(e for e in _detect(store) if e.kind == "hyper")
    assert sorted(link.kind for link in ep.links) == ["bolus", "meal"]


def test_two_meals_near_one_bolus_pair_nothing() -> None:
    meals = [
        MealEvent(ts=_ts(-22), carbs_g=40.0),
        MealEvent(ts=_ts(-14), carbs_g=18.0),
    ]
    insulin = [InsulinEvent(ts=_ts(-18), kind=InsulinKind.BOLUS, units=5.0)]
    pairs, rest_meals, rest_boluses = pair_treatments(meals, insulin)
    assert pairs == []
    assert len(rest_meals) == 2 and len(rest_boluses) == 1


def _graph(store: SQLiteStore) -> EpisodeGraph:
    return build_graph(store, _ts(-10), _ts(100000))


# ── episode-to-episode chains ─────────────────────────────────────────────────


def test_low_then_carbs_then_high_is_a_rebound_chain() -> None:
    # hypo ends at min 10; 15 g rescue carbs at 20 (in the gap); hyper starts at 40
    meals = [MealEvent(ts=_ts(20), carbs_g=15.0, note="rescue")]
    store = _store(
        [(0, 60), (5, 55), (10, 65), (40, 200), (45, 220), (50, 120)], meals=meals
    )
    graph = _graph(store)
    edges = [e for e in graph.edges]
    assert len(edges) == 1
    edge = edges[0]
    assert edge.relation == "rebound_after_low"
    assert edge.src_id.startswith("hypo:") and edge.dst_id.startswith("hyper:")
    assert edge.gap_min == 30.0
    assert edge.bridge is not None
    assert edge.bridge.kind == "meal" and edge.bridge.detail["carbs_g"] == 15.0
    assert edge.bridge.offset_min == 10.0  # from the low's END


def test_high_then_insulin_then_low_is_low_after_high() -> None:
    insulin = [InsulinEvent(ts=_ts(25), kind=InsulinKind.BOLUS, units=2.0)]
    store = _store(
        [(0, 200), (5, 220), (10, 210), (30, 120), (60, 60), (65, 55), (70, 120)],
        insulin=insulin,
    )
    graph = _graph(store)
    assert len(graph.edges) == 1
    edge = graph.edges[0]
    assert edge.relation == "low_after_high"
    assert edge.bridge is not None and edge.bridge.kind == "bolus"


def test_no_bridge_is_a_weak_follows() -> None:
    store = _store([(0, 60), (5, 55), (15, 120), (40, 200), (45, 220), (50, 120)])
    graph = _graph(store)
    assert len(graph.edges) == 1
    assert graph.edges[0].relation == "follows"
    assert graph.edges[0].bridge is None


def test_chain_across_a_sensor_gap_is_only_a_weak_follows() -> None:
    # low ends at 10, sensor dark until 60, a rescue carb logged at 30 in the
    # unseen gap, high starts at 60. The trajectory through the hole is
    # unobserved, so the edge stays "follows" and never rebound_after_low.
    meals = [MealEvent(ts=_ts(30), carbs_g=15.0, note="rescue")]
    store = _store(
        [(0, 60), (5, 55), (10, 65), (60, 200), (65, 220), (70, 120)], meals=meals
    )
    graph = _graph(store)
    assert len(graph.edges) == 1
    assert graph.edges[0].relation == "follows"
    assert graph.edges[0].bridge is None


def test_far_apart_episodes_do_not_chain() -> None:
    store = _store(
        [(0, 200), (5, 210), (10, 120)]
        + [(10 + 30 * i, 120) for i in range(1, 8)]
        + [(250, 60), (255, 55), (260, 120)]
    )
    graph = _graph(store)
    assert graph.edges == ()


def test_largest_carb_event_in_gap_is_the_bridge() -> None:
    meals = [
        MealEvent(ts=_ts(15), carbs_g=8.0),
        MealEvent(ts=_ts(22), carbs_g=20.0),
    ]
    store = _store(
        [(0, 60), (5, 55), (10, 65), (40, 200), (45, 220), (50, 120)], meals=meals
    )
    edge = _graph(store).edges[0]
    assert edge.relation == "rebound_after_low"
    assert edge.bridge is not None and edge.bridge.detail["carbs_g"] == 20.0


def test_edges_for_walks_both_directions() -> None:
    meals = [MealEvent(ts=_ts(20), carbs_g=15.0)]
    store = _store(
        [(0, 60), (5, 55), (10, 65), (40, 200), (45, 220), (50, 120)], meals=meals
    )
    graph = _graph(store)
    low = next(e for e in graph.episodes if e.kind == "hypo")
    high = next(e for e in graph.episodes if e.kind == "hyper")
    assert [e.dst_id for e in graph.edges_for(low.id)["out"]] == [high.id]
    assert [e.src_id for e in graph.edges_for(high.id)["in"]] == [low.id]
    assert graph.to_dict()["edges"][0]["relation"] == "rebound_after_low"


def test_two_clean_pairs_both_pair() -> None:
    meals = [
        MealEvent(ts=_ts(0), carbs_g=40.0),
        MealEvent(ts=_ts(240), carbs_g=60.0),
    ]
    insulin = [
        InsulinEvent(ts=_ts(2), kind=InsulinKind.BOLUS, units=4.0),
        InsulinEvent(ts=_ts(243), kind=InsulinKind.BOLUS, units=6.5),
    ]
    pairs, rest_meals, rest_boluses = pair_treatments(meals, insulin)
    assert len(pairs) == 2
    assert rest_meals == [] and rest_boluses == []
    assert pairs[0][0].carbs_g == 40.0 and pairs[0][1].units == 4.0
    assert pairs[1][0].carbs_g == 60.0 and pairs[1][1].units == 6.5


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


# ── addressable graph ─────────────────────────────────────────────────────────


def _graph(store: SQLiteStore) -> EpisodeGraph:
    return build_graph(store, _ts(-500), _ts(100000))


def test_episode_ids_are_stable_and_unique() -> None:
    store = _store([(0, 200), (5, 220), (10, 120), (15, 60), (20, 50), (25, 120)])
    a = [e.id for e in _graph(store).episodes]
    b = [e.id for e in _graph(store).episodes]
    assert a == b  # stable
    assert len(a) == len(set(a))  # unique
    assert all(":" in eid for eid in a)


def test_graph_node_lookup() -> None:
    graph = _graph(_store([(0, 200), (5, 220), (10, 120)]))
    target = graph.episodes[0]
    assert graph.node(target.id) is target
    assert graph.node("missing:2025-01-06T00:00:00+00:00") is None


def test_graph_at_covering_and_nearest() -> None:
    store = _store([(0, 200), (5, 220), (10, 120), (60, 60), (65, 55), (70, 120)])
    graph = _graph(store)
    covering = graph.at(_ts(5))
    assert covering is not None and covering.kind == "hyper"
    # a moment between episodes resolves to the nearest excursion, never a gap
    near = graph.at(_ts(40))
    assert near is not None and near.kind != "sensor_gap"


def test_graph_at_returns_nothing_past_the_tolerance() -> None:
    # An unbounded nearest match turns "what happened at 09:00?" into an
    # explanation built from an excursion hours away. Past the tolerance the
    # honest answer is that nothing was happening.
    store = _store([(0, 200), (5, 220), (10, 120)] + [(m, 120) for m in range(15, 900, 5)])
    graph = _graph(store)
    assert graph.at(_ts(600)) is None
    found = graph.nearest(_ts(600))
    assert found is not None
    episode, distance_min = found
    assert episode.kind == "hyper"
    assert distance_min == 595.0  # measured to the span's end (00:05), not its start


def test_graph_nearest_measures_distance_to_the_span() -> None:
    store = _store([(m, 220) for m in range(0, 65, 5)] + [(m, 120) for m in range(65, 200, 5)])
    graph = _graph(store)
    # 01:15 sits 15 min past a run that ended at 01:00, not 75 min from its start.
    found = graph.nearest(_ts(75))
    assert found is not None and found[1] == 15.0
    assert graph.nearest(_ts(30)) == (graph.episodes[0], 0.0)  # inside the span


def test_episode_running_past_the_window_edge_is_flagged_truncated() -> None:
    # The same physiological run gets a different id and a shorter duration under
    # a differently aligned window, so the duration is only a lower bound.
    store = _store([(-5, 120)] + [(m, 220) for m in range(0, 65, 5)] + [(65, 120)])
    whole = build_graph(store, _ts(-100), _ts(200)).episodes[0]
    assert whole.complete and whole.duration_min == 60.0

    clipped = build_graph(store, _ts(30), _ts(200)).episodes[0]
    assert clipped.duration_min == 30.0  # half the run, silently, without the flag
    assert clipped.truncated_start and not clipped.truncated_end
    assert not clipped.complete
    assert clipped.to_dict()["complete"] is False


def test_episode_starting_after_a_sensor_gap_is_truncated() -> None:
    # The far side of a dark sensor is the same fact as a window edge: nobody was
    # watching when the run began, so its span is a floor, not the event.
    store = _store([(0, 120), (300, 220), (305, 220), (310, 120)])
    hypers = [e for e in _graph(store).episodes if e.kind == "hyper"]
    assert len(hypers) == 1
    assert hypers[0].truncated_start and not hypers[0].truncated_end


def test_summary_marks_a_truncated_longest_run() -> None:
    readings = [(-5, 120)] + [(m, 220) for m in range(0, 65, 5)] + [(65, 120)]
    store = _store(readings)
    assert build_graph(store, _ts(-100), _ts(200)).summary()["longest_hyper_truncated"] is False
    clipped = build_graph(store, _ts(30), _ts(200)).summary()
    assert clipped["longest_hyper_min"] == 30.0
    assert clipped["longest_hyper_truncated"] is True
    assert clipped["n_truncated"] == 1


def test_graph_coverage_is_the_denominator_under_every_count() -> None:
    # Two lows over a day the sensor watched for an hour is not the same claim as
    # two lows over a day it watched in full.
    store = _store([(0, 60), (5, 55), (10, 120), (30, 60), (35, 55), (40, 120)])
    coverage = build_graph(store, _ts(0), _ts(1440)).coverage()
    assert coverage["n_readings"] == 6
    assert coverage["window_min"] == 1440.0
    assert coverage["observed_pct"] == 2.1  # 6 readings x 5 min over 24 h
    full = build_graph(store, _ts(0), _ts(45)).coverage()
    assert full["observed_pct"] == 66.7


# ── belt tools: query and traverse ────────────────────────────────────────────


def _ctx_toolkit(store: SQLiteStore) -> tuple[AgentContext, DiscoveryToolkit]:
    cov = store.coverage()
    assert cov.first_ts is not None and cov.last_ts is not None
    ctx = AgentContext(
        store=store, window=(cov.first_ts.date(), cov.last_ts.date()),
        gates=ColdStartReport.from_coverage(cov), run_id="test", timezone="UTC",
    )
    return ctx, DiscoveryToolkit(ctx, target_low=70, target_high=180)


def test_episodes_tool_lists_nodes_and_summary() -> None:
    store = _store([(0, 200), (5, 220), (10, 120), (15, 60), (20, 50), (25, 120)])
    ctx, tk = _ctx_toolkit(store)
    specs = {s.name: s for s in episode_specs(ctx, tk)}
    result, numbers = specs["episodes"].fn({"kind": "hyper"})
    assert result["summary"]["num_hyper"] == 1
    assert len(result["episodes"]) == 1
    assert result["episodes"][0]["kind"] == "hyper"
    assert numbers["num_hyper"] == 1  # numbers surface to the faithfulness guard


def test_explain_episode_traverses_to_context() -> None:
    meals = [MealEvent(ts=_ts(-20), carbs_g=45.0, note="breakfast")]
    store = _store([(0, 200), (5, 220), (10, 120)], meals=meals)
    ctx, tk = _ctx_toolkit(store)
    specs = {s.name: s for s in episode_specs(ctx, tk)}
    node_id = specs["episodes"].fn({})[0]["episodes"][0]["id"]
    result, numbers = specs["explain_episode"].fn({"episode_id": node_id})
    meal_edges = [link for link in result["links"] if link["kind"] == "meal"]
    assert meal_edges and meal_edges[0]["detail"]["carbs_g"] == 45.0
    assert numbers["meal_carbs_g"] == 45.0


def test_explain_episode_summary_is_first_and_model_facing() -> None:
    meals = [MealEvent(ts=_ts(-20), carbs_g=45.0, note="breakfast")]
    store = _store([(0, 200), (5, 220), (10, 120)], meals=meals)
    ctx, tk = _ctx_toolkit(store)
    specs = {s.name: s for s in episode_specs(ctx, tk)}
    node_id = specs["episodes"].fn({})[0]["episodes"][0]["id"]
    result, _ = specs["explain_episode"].fn({"episode_id": node_id})
    # summary is the first key, so it survives context-budget reduction.
    assert next(iter(result)) == "summary"
    summary = result["summary"]
    assert "high episode" in summary
    assert "220 mg/dL peak" in summary  # the same number the structured field carries
    assert "1 event(s)" in summary and "meal" in summary


def test_explain_episode_summary_narrates_the_chain() -> None:
    # low -> rescue carbs -> rebound high: the summary must name the relation and
    # the bridge event, so the model reasons over the causal chain in prose.
    meals = [MealEvent(ts=_ts(20), carbs_g=15.0, note="rescue")]
    store = _store(
        [(0, 60), (5, 55), (10, 65), (40, 200), (45, 220), (50, 120)], meals=meals
    )
    ctx, tk = _ctx_toolkit(store)
    specs = {s.name: s for s in episode_specs(ctx, tk)}
    high_id = next(
        n["id"] for n in specs["episodes"].fn({})[0]["episodes"] if n["kind"] == "hyper"
    )
    result, _ = specs["explain_episode"].fn({"episode_id": high_id})
    summary = result["summary"]
    assert "followed a low episode" in summary
    assert "rebound after low" in summary
    assert "bridged by 15 g carbs (rescue)" in summary
    assert "30 min gap" in summary  # low ends at 00:10, high starts at 00:40
    # the chain is still present as structured data alongside the prose.
    assert result["chain"]["in"][0]["relation"] == "rebound_after_low"


def test_explain_episode_summary_without_chain() -> None:
    store = _store([(0, 200), (5, 220), (10, 120)])
    ctx, tk = _ctx_toolkit(store)
    specs = {s.name: s for s in episode_specs(ctx, tk)}
    node_id = specs["episodes"].fn({})[0]["episodes"][0]["id"]
    result, _ = specs["explain_episode"].fn({"episode_id": node_id})
    summary = result["summary"]
    assert summary.startswith("A high episode")
    assert "followed" not in summary  # no chain, no chain clause


def test_explain_episode_by_timestamp() -> None:
    store = _store([(0, 200), (5, 220), (10, 120)])
    ctx, tk = _ctx_toolkit(store)
    specs = {s.name: s for s in episode_specs(ctx, tk)}
    result, _ = specs["explain_episode"].fn({"timestamp": _ts(5).isoformat()})
    assert result["kind"] == "hyper"
    assert result["match"] == {"mode": "covering", "distance_min": 0.0}


def test_explain_episode_by_timestamp_reports_a_quiet_moment() -> None:
    # The model asked what happened at a time nothing happened. Handing back a
    # 10-hour-away excursion invites it to explain the wrong event.
    store = _store([(0, 200), (5, 220), (10, 120)] + [(m, 120) for m in range(15, 900, 5)])
    ctx, tk = _ctx_toolkit(store)
    specs = {s.name: s for s in episode_specs(ctx, tk)}
    result, numbers = specs["explain_episode"].fn({"timestamp": _ts(600).isoformat()})
    assert result["matched"] is False
    assert "glucose was in range" in result["summary"]
    assert result["nearest"]["distance_min"] == 595.0
    assert numbers == {}  # nothing to quote, so nothing reaches the guard


def test_explain_episode_summary_says_when_no_context_was_logged() -> None:
    # An empty links list is easy to read past; an absent context is a finding,
    # and the alternative is a model inventing a meal nobody recorded.
    store = _store([(0, 200), (5, 220), (10, 120)])
    ctx, tk = _ctx_toolkit(store)
    specs = {s.name: s for s in episode_specs(ctx, tk)}
    node_id = specs["episodes"].fn({})[0]["episodes"][0]["id"]
    result, _ = specs["explain_episode"].fn({"episode_id": node_id})
    assert result["links"] == []
    assert "nothing in the record explains it" in result["summary"]


def test_explain_episode_summary_calls_a_truncated_duration_a_lower_bound() -> None:
    store = _store([(m, 220) for m in range(0, 65, 5)] + [(65, 120)])
    cov = store.coverage()
    assert cov.last_ts is not None
    ctx = AgentContext(
        store=store, window=(_ts(30).date(), cov.last_ts.date()),
        gates=ColdStartReport.from_coverage(cov), run_id="test", timezone="UTC",
    )
    tk = DiscoveryToolkit(ctx, target_low=70, target_high=180)
    specs = {s.name: s for s in episode_specs(ctx, tk)}
    result, _ = specs["explain_episode"].fn({"timestamp": _ts(0).isoformat()})
    assert "at least 60 min" in result["summary"]
    assert "lower bounds" in result["summary"]


def test_episodes_tool_says_when_the_row_list_is_capped() -> None:
    # Rows are capped; counts are not. Without the note a capped list reads as
    # the complete set and "you had 2 highs" is one confident sentence away.
    readings = []
    for i in range(6):
        readings += [(i * 60, 200), (i * 60 + 5, 220), (i * 60 + 10, 120)]
    ctx, tk = _ctx_toolkit(_store(readings))
    specs = {s.name: s for s in episode_specs(ctx, tk)}
    result, _ = specs["episodes"].fn({"kind": "hyper", "top_n": 2})
    assert result["summary"]["num_hyper"] == 6
    assert result["n_matching"] == 6
    assert result["n_shown"] == 2
    assert len(result["episodes"]) == 2
    assert "2 longest of 6" in result["note"]

    uncapped, _ = specs["episodes"].fn({"kind": "hyper"})
    assert "note" not in uncapped  # nothing was hidden, so nothing to say


def test_episodes_tool_surfaces_observation_coverage_to_the_guard() -> None:
    store = _store([(0, 60), (5, 55), (10, 120)])
    ctx, tk = _ctx_toolkit(store)
    specs = {s.name: s for s in episode_specs(ctx, tk)}
    result, numbers = specs["episodes"].fn({})
    assert numbers["observed_pct"] == result["summary"]["observed_pct"]
    assert numbers["n_readings"] == 3


def test_explain_episode_unknown_id_errors() -> None:
    store = _store([(0, 200), (5, 220), (10, 120)])
    ctx, tk = _ctx_toolkit(store)
    specs = {s.name: s for s in episode_specs(ctx, tk)}
    result, _ = specs["explain_episode"].fn({"episode_id": "nope"})
    assert "error" in result


def test_episode_tools_registered_on_belt() -> None:
    store = _store([(0, 200), (5, 220), (10, 120)])
    ctx, tk = _ctx_toolkit(store)
    names = {s.name for s in build_belt(ctx, tk)}
    assert {"episodes", "explain_episode"} <= names


def test_find_lows_count_agrees_with_episodes_tool_across_a_gap() -> None:
    # A low, sensor dark 50 min, low again. find_lows and the episodes tool must
    # report the same count, because both segment via the shared engine and
    # neither lets a run silently span the gap.
    store = _store([(0, 60), (5, 55), (10, 65), (60, 60), (65, 55), (70, 68)])
    ctx, tk = _ctx_toolkit(store)
    lows = tk.find_lows()
    specs = {s.name: s for s in episode_specs(ctx, tk)}
    result, _ = specs["episodes"].fn({"kind": "hypo"})
    assert lows["n_lows"] == 2
    assert result["summary"]["num_hypo"] == lows["n_lows"]
