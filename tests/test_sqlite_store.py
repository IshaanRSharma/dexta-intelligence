"""SQLiteStore tests — the StoragePort contract exercised against ``:memory:``.

Window semantics under test are half-open (``start <= ts < end``); dedupe keys
per table are documented in ``store/sqlite.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import TYPE_CHECKING

import pytest

from dexta_intelligence.models import (
    ActivityEvent,
    DeviceEvent,
    Finding,
    FindingStats,
    FindingStatus,
    GlucoseEvent,
    Goal,
    GoalCheckpoint,
    GoalMetric,
    GoalStatus,
    Hypothesis,
    HypothesisStatus,
    InsulinEvent,
    InsulinKind,
    MealEvent,
    PredictionEvent,
    RawEvent,
    RecoveryEvent,
    Rollup,
    RollupPeriod,
    SleepEvent,
)
from dexta_intelligence.store import SQLiteStore, StoragePort

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

T0 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)


@pytest.fixture
def store() -> Iterator[SQLiteStore]:
    s = SQLiteStore(":memory:")
    s.migrate()
    yield s
    s.close()


def _raw(source: str, source_id: str, ts: datetime) -> RawEvent:
    return RawEvent(source=source, source_id=source_id, source_ts=ts, payload={"id": source_id})


def _finding(**overrides: object) -> Finding:
    base: dict[str, object] = {
        "agent": "pattern_miner",
        "kind": "overnight_low",
        "scope": "overnight",
        "headline": "Lows cluster after late boluses",
        "body_md": "Body **markdown**.",
        "evidence": {"tbr": 6.2, "n_nights": 14},
        "stats": FindingStats(effect_size=0.4, n=14, p_perm=0.01, q_fdr=0.04, replicated=True),
        "confidence": 0.7,
        "window_start": T0 - timedelta(days=14),
        "window_end": T0,
    }
    base.update(overrides)
    return Finding.model_validate(base)


# ─────────────────────────────────────────────────────────────────────────────
# Lifecycle / protocol
# ─────────────────────────────────────────────────────────────────────────────


class TestLifecycle:
    def test_satisfies_storage_port(self, store: SQLiteStore) -> None:
        assert isinstance(store, StoragePort)

    def test_migrate_is_idempotent(self, store: SQLiteStore) -> None:
        store.migrate()  # second call: no error
        assert list(store.upsert_raw_events([_raw("nightscout", "a", T0)])) == ["a"]
        store.migrate()  # third call: existing data untouched
        assert store.get_watermark("nightscout") == T0

    def test_file_database_persists(self, tmp_path: Path) -> None:
        db = tmp_path / "dexta.db"
        first = SQLiteStore(db)
        first.migrate()
        assert first.insert_glucose([GlucoseEvent(ts=T0, mg_dl=120)]) == 1
        first.close()

        second = SQLiteStore(db)
        second.migrate()
        events = second.get_glucose(T0 - timedelta(hours=1), T0 + timedelta(hours=1))
        assert events == [GlucoseEvent(ts=T0, mg_dl=120)]
        second.close()


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — raw events + watermark
# ─────────────────────────────────────────────────────────────────────────────


class TestRawEvents:
    def test_upsert_returns_ids_for_new_rows(self, store: SQLiteStore) -> None:
        batch = [_raw("nightscout", str(i), T0 + timedelta(minutes=5 * i)) for i in range(3)]
        id_map = store.upsert_raw_events(batch)
        assert set(id_map) == {"0", "1", "2"}
        assert all(isinstance(v, int) for v in id_map.values())
        assert len(set(id_map.values())) == 3  # distinct ids

    def test_upsert_returns_stable_ids_for_existing_rows(self, store: SQLiteStore) -> None:
        batch = [_raw("nightscout", str(i), T0 + timedelta(minutes=5 * i)) for i in range(3)]
        first = store.upsert_raw_events(batch)
        # re-upsert is a no-op but still returns the same assigned ids
        second = store.upsert_raw_events(batch)
        assert second == first

    def test_upsert_maps_new_and_preexisting_together(self, store: SQLiteStore) -> None:
        batch = [_raw("nightscout", str(i), T0 + timedelta(minutes=5 * i)) for i in range(3)]
        first = store.upsert_raw_events(batch)
        mixed = [*batch, _raw("nightscout", "3", T0)]
        id_map = store.upsert_raw_events(mixed)
        # pre-existing ids unchanged, the new key gets a fresh distinct id
        assert {k: id_map[k] for k in first} == first
        assert id_map["3"] not in first.values()

    def test_existing_raw_ids_reports_only_stored_subset(self, store: SQLiteStore) -> None:
        stored = [_raw("nightscout", str(i), T0 + timedelta(minutes=5 * i)) for i in range(2)]
        store.upsert_raw_events(stored)
        probe = [*stored, _raw("nightscout", "9", T0)]
        existing = store.existing_raw_ids(probe)
        assert set(existing) == {"0", "1"}

    def test_upsert_empty_batch(self, store: SQLiteStore) -> None:
        assert store.upsert_raw_events([]) == {}
        assert store.existing_raw_ids([]) == {}

    def test_watermark_none_when_empty(self, store: SQLiteStore) -> None:
        assert store.get_watermark("nightscout") is None

    def test_watermark_is_latest_source_ts(self, store: SQLiteStore) -> None:
        store.upsert_raw_events(
            [
                _raw("nightscout", "a", T0),
                _raw("nightscout", "b", T0 + timedelta(minutes=10)),
                _raw("nightscout", "c", T0 + timedelta(minutes=5)),
            ]
        )
        watermark = store.get_watermark("nightscout")
        assert watermark == T0 + timedelta(minutes=10)
        assert watermark is not None
        assert watermark.tzinfo == UTC

    def test_watermarks_independent_per_source(self, store: SQLiteStore) -> None:
        store.upsert_raw_events(
            [
                _raw("nightscout", "a", T0 + timedelta(hours=2)),
                _raw("whoop", "a", T0),
            ]
        )
        assert store.get_watermark("nightscout") == T0 + timedelta(hours=2)
        assert store.get_watermark("whoop") == T0
        assert store.get_watermark("dexcom") is None


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — typed event round trips
# ─────────────────────────────────────────────────────────────────────────────

WIDE_START = T0 - timedelta(days=2)
WIDE_END = T0 + timedelta(days=2)


class TestEventRoundTrips:
    def test_glucose(self, store: SQLiteStore) -> None:
        events = [
            GlucoseEvent(ts=T0, mg_dl=131, trend="Flat", raw_event_id=7),
            GlucoseEvent(ts=T0 + timedelta(minutes=5), mg_dl=128),
        ]
        assert store.insert_glucose(events) == 2
        assert store.get_glucose(WIDE_START, WIDE_END) == events

    def test_insulin(self, store: SQLiteStore) -> None:
        events = [
            InsulinEvent(ts=T0, kind=InsulinKind.BOLUS, units=4.5, automatic=False),
            InsulinEvent(
                ts=T0 + timedelta(minutes=5),
                kind=InsulinKind.TEMP_BASAL,
                units=0.925,
                duration_min=30,
                automatic=True,
                raw_event_id=3,
            ),
            InsulinEvent(ts=T0 + timedelta(minutes=10), kind=InsulinKind.SUSPEND, duration_min=45),
        ]
        assert store.insert_insulin(events) == 3
        assert store.get_insulin(WIDE_START, WIDE_END) == events

    def test_meals(self, store: SQLiteStore) -> None:
        events = [
            MealEvent(ts=T0, carbs_g=45, protein_g=22, fat_g=14, note="lunch", raw_event_id=1),
            MealEvent(ts=T0 + timedelta(hours=4)),
        ]
        assert store.insert_meals(events) == 2
        assert store.get_meals(WIDE_START, WIDE_END) == events

    def test_activity(self, store: SQLiteStore) -> None:
        events = [
            ActivityEvent(ts=T0, kind="run", duration_min=32.5, intensity=0.8, strain=12.1),
            ActivityEvent(ts=T0 + timedelta(hours=6), kind="walk"),
        ]
        assert store.insert_activity(events) == 2
        assert store.get_activity(WIDE_START, WIDE_END) == events

    def test_sleep(self, store: SQLiteStore) -> None:
        events = [
            SleepEvent(
                ts_start=T0,
                ts_end=T0 + timedelta(hours=7),
                duration_min=420,
                score=82.0,
                stages={"deep": 95.0, "rem": 110.5},
                raw_event_id=9,
            ),
        ]
        assert store.insert_sleep(events) == 1
        assert store.get_sleep(WIDE_START, WIDE_END) == events

    def test_recovery(self, store: SQLiteStore) -> None:
        events = [RecoveryEvent(ts=T0, score=61.0, hrv_ms=48.2, rhr_bpm=52.0)]
        assert store.insert_recovery(events) == 1
        assert store.get_recovery(WIDE_START, WIDE_END) == events

    def test_device_insert_and_dedupe(self, store: SQLiteStore) -> None:
        events = [DeviceEvent(ts=T0, kind="site_change", note="left arm")]
        assert store.insert_device(events) == 1
        assert store.insert_device(events) == 0

    def test_non_utc_input_normalized_to_utc(self, store: SQLiteStore) -> None:
        eastern = timezone(timedelta(hours=-4))
        event = GlucoseEvent(ts=datetime(2026, 6, 1, 6, 0, tzinfo=eastern), mg_dl=110)
        store.insert_glucose([event])
        (got,) = store.get_glucose(WIDE_START, WIDE_END)
        assert got.ts == T0
        assert got.ts.tzinfo == UTC

    def test_microsecond_fidelity(self, store: SQLiteStore) -> None:
        ts = T0.replace(microsecond=123456)
        store.insert_glucose([GlucoseEvent(ts=ts, mg_dl=99)])
        (got,) = store.get_glucose(WIDE_START, WIDE_END)
        assert got.ts == ts


class TestPredictionEvents:
    def test_round_trip(self, store: SQLiteStore) -> None:
        events = [
            PredictionEvent(
                ts=T0,
                source="openaps",
                curve_kind="iob",
                values_mg_dl=[120.0, 118.0, 115.0],
                raw_event_id=5,
            ),
            PredictionEvent(
                ts=T0,
                source="openaps",
                curve_kind="cob",
                values_mg_dl=[120.0, 122.0, 125.0],
            ),
        ]
        assert store.insert_predictions(events) == 2
        assert store.get_predictions(WIDE_START, WIDE_END) == events

    def test_dedupe_on_ts_and_curve_kind(self, store: SQLiteStore) -> None:
        event = PredictionEvent(
            ts=T0, source="openaps", curve_kind="uam", values_mg_dl=[100.0, 105.0]
        )
        assert store.insert_predictions([event]) == 1
        assert store.insert_predictions([event]) == 0

    def test_window_half_open(self, store: SQLiteStore) -> None:
        inside = PredictionEvent(ts=T0, source="loop", curve_kind="loop", values_mg_dl=[140.0])
        boundary = PredictionEvent(
            ts=T0 + timedelta(days=2), source="loop", curve_kind="loop", values_mg_dl=[140.0]
        )
        store.insert_predictions([inside, boundary])
        got = store.get_predictions(WIDE_START, T0 + timedelta(days=2))
        assert got == [inside]


class TestEventDedupe:
    def test_glucose_reinsert_is_noop(self, store: SQLiteStore) -> None:
        events = [GlucoseEvent(ts=T0, mg_dl=131, trend="Flat")]
        assert store.insert_glucose(events) == 1
        assert store.insert_glucose(events) == 0
        assert len(store.get_glucose(WIDE_START, WIDE_END)) == 1

    def test_insulin_distinct_kinds_at_same_ts_both_kept(self, store: SQLiteStore) -> None:
        events = [
            InsulinEvent(ts=T0, kind=InsulinKind.BOLUS, units=2.0),
            InsulinEvent(ts=T0, kind=InsulinKind.TEMP_BASAL, units=0.5, duration_min=30),
        ]
        assert store.insert_insulin(events) == 2
        assert store.insert_insulin(events) == 0

    def test_mixed_batch_counts_only_new(self, store: SQLiteStore) -> None:
        first = [GlucoseEvent(ts=T0, mg_dl=131)]
        store.insert_glucose(first)
        batch = [*first, GlucoseEvent(ts=T0 + timedelta(minutes=5), mg_dl=128)]
        assert store.insert_glucose(batch) == 1


class TestWindowSemantics:
    """Windows are half-open: ``start <= ts < end``, ascending order."""

    def test_start_inclusive_end_exclusive(self, store: SQLiteStore) -> None:
        times = [T0, T0 + timedelta(minutes=5), T0 + timedelta(minutes=10)]
        store.insert_glucose([GlucoseEvent(ts=t, mg_dl=100 + i) for i, t in enumerate(times)])
        got = store.get_glucose(T0, T0 + timedelta(minutes=10))
        assert [e.ts for e in got] == times[:2]

    def test_empty_window(self, store: SQLiteStore) -> None:
        store.insert_glucose([GlucoseEvent(ts=T0, mg_dl=100)])
        assert store.get_glucose(T0 + timedelta(minutes=1), T0 + timedelta(minutes=2)) == []

    def test_ascending_order(self, store: SQLiteStore) -> None:
        times = [T0 + timedelta(minutes=10), T0, T0 + timedelta(minutes=5)]
        store.insert_glucose([GlucoseEvent(ts=t, mg_dl=100) for t in times])
        got = store.get_glucose(WIDE_START, WIDE_END)
        assert [e.ts for e in got] == sorted(times)

    def test_sleep_windowed_on_ts_start(self, store: SQLiteStore) -> None:
        event = SleepEvent(ts_start=T0, ts_end=T0 + timedelta(hours=8), duration_min=480)
        store.insert_sleep([event])
        # ts_start in window even though ts_end is outside
        assert store.get_sleep(T0, T0 + timedelta(hours=1)) == [event]
        # ts_start before window start: excluded
        assert store.get_sleep(T0 + timedelta(hours=1), WIDE_END) == []


# ─────────────────────────────────────────────────────────────────────────────
# Coverage
# ─────────────────────────────────────────────────────────────────────────────


class TestCoverage:
    def test_empty_store(self, store: SQLiteStore) -> None:
        stats = store.coverage()
        assert stats.first_ts is None
        assert stats.last_ts is None
        assert stats.span_days == 0.0
        assert stats.n_glucose == 0
        assert stats.glucose_coverage_pct == 0.0
        assert stats.n_insulin == 0
        assert stats.days_with_insulin_pct == 0.0
        assert stats.n_meals == 0
        assert stats.n_sleep == 0
        assert stats.n_activity == 0

    def test_populated_store(self, store: SQLiteStore) -> None:
        day1 = datetime(2026, 6, 1, tzinfo=UTC)
        store.insert_glucose(
            [GlucoseEvent(ts=day1 + timedelta(minutes=5 * i), mg_dl=110) for i in range(13)]
        )
        store.insert_insulin(
            [
                InsulinEvent(ts=day1 + timedelta(minutes=30), kind=InsulinKind.BOLUS, units=3.0),
                InsulinEvent(ts=day1 + timedelta(hours=36), kind=InsulinKind.BOLUS, units=2.0),
            ]
        )
        store.insert_meals([MealEvent(ts=day1 + timedelta(minutes=15), carbs_g=30)])
        store.insert_sleep(
            [
                SleepEvent(
                    ts_start=day1 + timedelta(hours=2),
                    ts_end=day1 + timedelta(hours=8),
                    duration_min=360,
                )
            ]
        )
        store.insert_activity([ActivityEvent(ts=day1 + timedelta(hours=9), kind="run")])

        stats = store.coverage()
        assert stats.first_ts == day1
        assert stats.last_ts == day1 + timedelta(hours=36)
        assert stats.span_days == pytest.approx(1.5)
        assert stats.n_glucose == 13
        # 1.5 days of 5-min slots = 433 expected; 13 present
        assert stats.glucose_coverage_pct == pytest.approx(100.0 * 13 / 433)
        assert stats.n_insulin == 2
        # insulin on both UTC dates spanned → 100%
        assert stats.days_with_insulin_pct == pytest.approx(100.0)
        assert stats.n_meals == 1
        assert stats.n_sleep == 1
        assert stats.n_activity == 1

    def test_single_reading_full_coverage(self, store: SQLiteStore) -> None:
        store.insert_glucose([GlucoseEvent(ts=T0, mg_dl=100)])
        stats = store.coverage()
        assert stats.span_days == 0.0
        assert stats.glucose_coverage_pct == pytest.approx(100.0)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 — rollups
# ─────────────────────────────────────────────────────────────────────────────


def _rollup(period_start: datetime, **overrides: object) -> Rollup:
    base: dict[str, object] = {
        "period": RollupPeriod.DAILY,
        "period_start": period_start,
        "n": 288,
        "mean": 142.0,
        "sd": 38.0,
        "cv": 26.8,
        "tir": 71.5,
        "tar": 22.0,
        "tar2": 4.0,
        "tbr": 6.5,
        "tbr2": 1.0,
        "gmi": 6.7,
        "excursion_count": 3,
        "bolus_units": 21.5,
        "basal_units": 18.0,
        "carbs_g": 145.0,
    }
    base.update(overrides)
    return Rollup.model_validate(base)


class TestRollups:
    def test_upsert_and_get(self, store: SQLiteStore) -> None:
        day1 = datetime(2026, 6, 1, tzinfo=UTC)
        rollups = [_rollup(day1), _rollup(day1 + timedelta(days=1), mean=120.0)]
        assert store.upsert_rollups(rollups) == 2
        got = store.get_rollups(RollupPeriod.DAILY, day1, day1 + timedelta(days=7))
        assert got == rollups

    def test_reupsert_updates_in_place(self, store: SQLiteStore) -> None:
        day1 = datetime(2026, 6, 1, tzinfo=UTC)
        assert store.upsert_rollups([_rollup(day1)]) == 1
        # same (period, period_start): updated, not duplicated, counts 0 new rows
        assert store.upsert_rollups([_rollup(day1, mean=99.0, n=200)]) == 0
        got = store.get_rollups(RollupPeriod.DAILY, day1, day1 + timedelta(days=1))
        assert len(got) == 1
        assert got[0].mean == 99.0
        assert got[0].n == 200

    def test_get_filters_by_period_and_range(self, store: SQLiteStore) -> None:
        day1 = datetime(2026, 6, 1, tzinfo=UTC)
        store.upsert_rollups(
            [
                _rollup(day1),
                _rollup(day1, period=RollupPeriod.HOURLY),
                _rollup(day1 + timedelta(days=10)),
            ]
        )
        got = store.get_rollups(RollupPeriod.DAILY, day1, day1 + timedelta(days=7))
        assert len(got) == 1
        assert got[0].period is RollupPeriod.DAILY
        assert got[0].period_start == day1
        # period_start range is half-open too
        assert store.get_rollups(RollupPeriod.DAILY, day1, day1) == []


# ─────────────────────────────────────────────────────────────────────────────
# Layer 4 — findings & hypotheses
# ─────────────────────────────────────────────────────────────────────────────


class TestFindings:
    def test_insert_returns_id_and_round_trips(self, store: SQLiteStore) -> None:
        finding = _finding()
        fid = store.insert_finding(finding)
        assert fid == 1
        (got,) = store.get_findings()
        assert got == finding.model_copy(update={"id": fid})
        assert got.window_start is not None
        assert got.window_start.tzinfo == UTC

    def test_ids_are_sequential(self, store: SQLiteStore) -> None:
        assert store.insert_finding(_finding()) == 1
        assert store.insert_finding(_finding(headline="Another")) == 2

    def test_get_filters(self, store: SQLiteStore) -> None:
        store.insert_finding(_finding(agent="pattern_miner", kind="overnight_low"))
        store.insert_finding(_finding(agent="pattern_miner", kind="post_meal_spike"))
        store.insert_finding(_finding(agent="skeptic", kind="overnight_low"))

        assert len(store.get_findings(agent="pattern_miner")) == 2
        assert len(store.get_findings(kind="overnight_low")) == 2
        assert len(store.get_findings(agent="skeptic", kind="overnight_low")) == 1
        assert store.get_findings(agent="nobody") == []
        assert len(store.get_findings(status=FindingStatus.ACTIVE)) == 3
        assert store.get_findings(status=FindingStatus.REJECTED) == []

    def test_get_limit_newest_first(self, store: SQLiteStore) -> None:
        for i in range(5):
            store.insert_finding(_finding(headline=f"finding {i}"))
        got = store.get_findings(limit=2)
        assert [f.headline for f in got] == ["finding 4", "finding 3"]

    def test_status_transition(self, store: SQLiteStore) -> None:
        fid = store.insert_finding(_finding())
        store.set_finding_status(fid, FindingStatus.REJECTED)
        (got,) = store.get_findings(status=FindingStatus.REJECTED)
        assert got.id == fid
        assert got.status is FindingStatus.REJECTED
        assert store.get_findings(status=FindingStatus.ACTIVE) == []

    def test_supersede(self, store: SQLiteStore) -> None:
        old_id = store.insert_finding(_finding(headline="v1"))
        new_id = store.insert_finding(_finding(headline="v2"))
        store.supersede_finding(old_id, new_id)

        (old,) = store.get_findings(status=FindingStatus.SUPERSEDED)
        assert old.id == old_id
        assert old.superseded_by == new_id
        (active,) = store.get_findings(status=FindingStatus.ACTIVE)
        assert active.id == new_id
        assert active.superseded_by is None


class TestHypotheses:
    def test_insert_returns_id_and_round_trips(self, store: SQLiteStore) -> None:
        hypothesis = Hypothesis(
            statement="Late-night protein delays overnight rise",
            source_finding_id=4,
            tests=[{"name": "split_replication", "passed": True, "p": 0.03}],
        )
        hid = store.insert_hypothesis(hypothesis)
        assert hid == 1
        (got,) = store.get_hypotheses()
        assert got == hypothesis.model_copy(update={"id": hid})

    def test_get_filters_by_status(self, store: SQLiteStore) -> None:
        store.insert_hypothesis(Hypothesis(statement="open one"))
        store.insert_hypothesis(
            Hypothesis(statement="refuted one", status=HypothesisStatus.REFUTED)
        )
        assert len(store.get_hypotheses()) == 2
        got = store.get_hypotheses(status="refuted")
        assert [h.statement for h in got] == ["refuted one"]
        assert store.get_hypotheses(status="stale") == []


class TestGoals:
    def _goal(self, statement: str = "reduce my overnight lows") -> Goal:
        return Goal(
            statement=statement,
            metric=GoalMetric.NOCTURNAL_TBR,
            direction="decrease",
        )

    def test_insert_and_get_round_trips(self, store: SQLiteStore) -> None:
        gid = store.insert_goal(self._goal())
        assert gid == 1
        (got,) = store.get_goals()
        assert got.id == gid
        assert got.statement == "reduce my overnight lows"
        assert got.status is GoalStatus.ACTIVE

    def test_get_goals_orders_by_id_ascending(self, store: SQLiteStore) -> None:
        ids = [store.insert_goal(self._goal(f"goal {i}")) for i in range(5)]
        got = store.get_goals()
        assert [g.id for g in got] == sorted(ids)

    def test_get_goals_filters_by_status(self, store: SQLiteStore) -> None:
        active = store.insert_goal(self._goal("active one"))
        paused = store.insert_goal(self._goal("paused one"))
        store.set_goal_status(paused, GoalStatus.PAUSED)
        got = store.get_goals(status=GoalStatus.ACTIVE)
        assert [g.id for g in got] == [active]

    def test_checkpoint_ordering_under_same_timestamp(self, store: SQLiteStore) -> None:
        gid = store.insert_goal(self._goal())
        ts = T0
        notes = ["first", "second", "third", "fourth"]
        for note in notes:
            store.insert_goal_checkpoint(
                GoalCheckpoint(goal_id=gid, ts=ts, metric_value=1.0, note=note)
            )
        got = store.get_goal_checkpoints(gid)
        # Same ts on every row: insertion order (rowid) must break the tie so the
        # arc note ("prior -> current") compares against the true previous tick.
        assert [c.note for c in got] == notes
        assert [c.id for c in got] == sorted(c.id for c in got)

    def test_checkpoints_scoped_to_their_goal(self, store: SQLiteStore) -> None:
        g1 = store.insert_goal(self._goal("g1"))
        g2 = store.insert_goal(self._goal("g2"))
        store.insert_goal_checkpoint(
            GoalCheckpoint(goal_id=g1, ts=T0, metric_value=1.0, note="for g1")
        )
        store.insert_goal_checkpoint(
            GoalCheckpoint(goal_id=g2, ts=T0, metric_value=2.0, note="for g2")
        )
        assert [c.note for c in store.get_goal_checkpoints(g1)] == ["for g1"]
        assert [c.note for c in store.get_goal_checkpoints(g2)] == ["for g2"]
