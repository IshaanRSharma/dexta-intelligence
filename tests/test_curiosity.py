"""Autonomous curiosity over the episode graph: deterministic wonders + banking.

The scanner reads the deterministic episode graph and produces observational
"wonders" about recurring structure; banking dedupes them by a stable kind so a
standing pattern is banked once, not every cycle.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from dexta_intelligence.analytics.episodes import build_graph
from dexta_intelligence.models import GlucoseEvent, HypothesisStatus, MealEvent
from dexta_intelligence.store.sqlite import SQLiteStore
from dexta_intelligence.workflows.curiosity import bank_curiosities, scan_curiosities

START = datetime(2026, 1, 6, tzinfo=UTC)


def _store() -> SQLiteStore:
    s = SQLiteStore(":memory:")
    s.migrate()
    return s


def _rebound_day(store: SQLiteStore, day_offset: int) -> None:
    """A low -> rescue carbs -> rebound high on the given day (a rebound_after_low).

    Readings are at realistic 5-min CGM cadence so the whole trajectory is
    observed: no sensor gap between the low and the high, so the rebound relation
    is confident rather than a weak gap-crossing follows.
    """
    base = START + timedelta(days=day_offset)

    def at(h: int, m: int) -> datetime:
        return base.replace(hour=h, minute=m)

    rows = [
        (at(15, 30), 110), (at(15, 35), 85), (at(15, 40), 66), (at(15, 45), 58),
        (at(15, 50), 55), (at(15, 55), 62), (at(16, 0), 68), (at(16, 5), 90),
        (at(16, 10), 140), (at(16, 15), 185), (at(16, 20), 205), (at(16, 25), 210),
        (at(16, 30), 200), (at(16, 35), 150), (at(16, 40), 120),
    ]
    store.insert_glucose([GlucoseEvent(ts=t, mg_dl=v) for t, v in rows])
    store.insert_meals([MealEvent(ts=at(16, 2), carbs_g=15.0, note="rescue")])


def _graph(store: SQLiteStore) -> object:
    return build_graph(store, START - timedelta(days=1), START + timedelta(days=40))


def test_recurring_chain_becomes_a_wonder() -> None:
    store = _store()
    for d in (0, 5, 10):  # three rebound days
        _rebound_day(store, d)
    wonders = dict(scan_curiosities(_graph(store)))
    assert "recurring_chain:rebound_after_low" in wonders
    assert "3 times" in wonders["recurring_chain:rebound_after_low"]
    store.close()


def test_below_threshold_is_not_a_wonder() -> None:
    store = _store()
    for d in (0, 5):  # only two -> below _MIN_RECURRENCE
        _rebound_day(store, d)
    wonders = dict(scan_curiosities(_graph(store)))
    assert "recurring_chain:rebound_after_low" not in wonders
    store.close()


def test_severe_low_cluster_wonder() -> None:
    store = _store()
    for d in range(3):  # three severe lows (<54)
        t = START + timedelta(days=d)
        store.insert_glucose([
            GlucoseEvent(ts=t.replace(hour=3, minute=m), mg_dl=v)
            for m, v in [(0, 100), (5, 48), (10, 46), (15, 49), (20, 100)]
        ])
    wonders = dict(scan_curiosities(_graph(store)))
    assert "severe_hypo_cluster" in wonders
    store.close()


def test_bank_dedupes_by_kind_across_runs() -> None:
    store = _store()
    for d in (0, 5, 10):
        _rebound_day(store, d)
    graph = _graph(store)
    first = bank_curiosities(store, graph)
    assert any(t.get("curiosity_kind") == "recurring_chain:rebound_after_low"
               for h in first for t in h.tests)
    open_after_first = store.get_hypotheses(status=HypothesisStatus.OPEN.value)
    # a second scan of the same graph banks nothing new (deduped by kind)
    second = bank_curiosities(store, graph)
    assert second == []
    assert len(store.get_hypotheses(status=HypothesisStatus.OPEN.value)) == len(open_after_first)
    store.close()


def test_wonders_are_observational_not_dosing() -> None:
    store = _store()
    for d in (0, 5, 10):
        _rebound_day(store, d)
    for _kind, statement in scan_curiosities(_graph(store)):
        low = statement.lower()
        for banned in ("take ", "units", "increase your", "decrease your", "you should"):
            assert banned not in low, statement
    store.close()
