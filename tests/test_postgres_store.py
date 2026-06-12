"""PostgresStore tests — the StoragePort contract against a real Postgres.

The parity suite is gated on ``TEST_DATABASE_URL``: without a server, CI skips
the whole class cleanly. It mirrors the key cases from ``test_sqlite_store.py``
(idempotent re-ingest, half-open windows, findings CRUD + status, hypotheses,
goals + checkpoints, rollup upsert, coverage, watermark). Tables are TRUNCATEd
between tests so each case starts empty while reusing one connection/schema.

The driver-absence test is unguarded: it asserts the helpful RuntimeError when
psycopg is missing, and skips when psycopg *is* installed (it relies on the
import failing for real).
"""

from __future__ import annotations

import importlib
import os
import sys
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
from dexta_intelligence.store import PostgresStore, StoragePort

if TYPE_CHECKING:
    from collections.abc import Iterator

T0 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
WIDE_START = T0 - timedelta(days=2)
WIDE_END = T0 + timedelta(days=2)

_DSN = os.environ.get("TEST_DATABASE_URL")
_HAS_PSYCOPG = importlib.util.find_spec("psycopg") is not None

requires_db = pytest.mark.skipif(
    _DSN is None, reason="TEST_DATABASE_URL not set; Postgres parity suite skipped"
)

_TABLES = [
    "goal_checkpoints",
    "goals",
    "hypotheses",
    "findings",
    "rollups",
    "prediction_events",
    "device_events",
    "recovery_events",
    "sleep_events",
    "activity_events",
    "meal_events",
    "insulin_events",
    "glucose_events",
    "raw_events",
]


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


# ─────────────────────────────────────────────────────────────────────────────
# Driver-absence contract (unguarded)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    _HAS_PSYCOPG, reason="psycopg installed; cannot exercise the missing-driver path"
)
def test_missing_driver_raises_helpful_error() -> None:
    with pytest.raises(RuntimeError, match=r"dexta-intelligence\[postgres\]"):
        PostgresStore("postgresql://unused")


def test_missing_driver_raises_with_monkeypatched_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Force the psycopg import to fail even if it is installed, proving the guard."""
    monkeypatch.setitem(sys.modules, "psycopg", None)
    with pytest.raises(RuntimeError, match=r"dexta-intelligence\[postgres\]"):
        PostgresStore("postgresql://unused")


# ─────────────────────────────────────────────────────────────────────────────
# Parity suite (gated on a live Postgres)
# ─────────────────────────────────────────────────────────────────────────────


@requires_db
class TestPostgresContract:
    @pytest.fixture
    def store(self) -> Iterator[PostgresStore]:
        assert _DSN is not None
        s = PostgresStore(_DSN)
        s.migrate()
        with s._conn, s._conn.cursor() as cur:
            cur.execute(f"TRUNCATE {', '.join(_TABLES)} RESTART IDENTITY CASCADE")
        yield s
        s.close()

    # ── lifecycle ────────────────────────────────────────────────────────────

    def test_satisfies_storage_port(self, store: PostgresStore) -> None:
        assert isinstance(store, StoragePort)

    def test_migrate_is_idempotent(self, store: PostgresStore) -> None:
        store.migrate()
        assert store.upsert_raw_events([_raw("nightscout", "a", T0)]) == 1
        store.migrate()
        assert store.get_watermark("nightscout") == T0

    # ── raw events + watermark ────────────────────────────────────────────────

    def test_upsert_dedupes_on_source_and_source_id(self, store: PostgresStore) -> None:
        batch = [_raw("nightscout", str(i), T0 + timedelta(minutes=5 * i)) for i in range(3)]
        assert store.upsert_raw_events(batch) == 3
        assert store.upsert_raw_events(batch) == 0
        mixed = [*batch, _raw("nightscout", "3", T0), _raw("whoop", "0", T0)]
        assert store.upsert_raw_events(mixed) == 2

    def test_upsert_empty_batch(self, store: PostgresStore) -> None:
        assert store.upsert_raw_events([]) == 0

    def test_watermark_none_when_empty(self, store: PostgresStore) -> None:
        assert store.get_watermark("nightscout") is None

    def test_watermark_is_latest_and_utc(self, store: PostgresStore) -> None:
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
        assert watermark.utcoffset() == timedelta(0)

    def test_watermarks_independent_per_source(self, store: PostgresStore) -> None:
        store.upsert_raw_events(
            [_raw("nightscout", "a", T0 + timedelta(hours=2)), _raw("whoop", "a", T0)]
        )
        assert store.get_watermark("nightscout") == T0 + timedelta(hours=2)
        assert store.get_watermark("whoop") == T0
        assert store.get_watermark("dexcom") is None

    # ── typed event round trips ────────────────────────────────────────────────

    def test_glucose_round_trip(self, store: PostgresStore) -> None:
        events = [
            GlucoseEvent(ts=T0, mg_dl=131, trend="Flat", raw_event_id=7),
            GlucoseEvent(ts=T0 + timedelta(minutes=5), mg_dl=128),
        ]
        assert store.insert_glucose(events) == 2
        assert store.get_glucose(WIDE_START, WIDE_END) == events

    def test_insulin_round_trip(self, store: PostgresStore) -> None:
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

    def test_meals_round_trip(self, store: PostgresStore) -> None:
        events = [
            MealEvent(ts=T0, carbs_g=45, protein_g=22, fat_g=14, note="lunch", raw_event_id=1),
            MealEvent(ts=T0 + timedelta(hours=4)),
        ]
        assert store.insert_meals(events) == 2
        assert store.get_meals(WIDE_START, WIDE_END) == events

    def test_activity_round_trip(self, store: PostgresStore) -> None:
        events = [
            ActivityEvent(ts=T0, kind="run", duration_min=32.5, intensity=0.8, strain=12.1),
            ActivityEvent(ts=T0 + timedelta(hours=6), kind="walk"),
        ]
        assert store.insert_activity(events) == 2
        assert store.get_activity(WIDE_START, WIDE_END) == events

    def test_sleep_round_trip(self, store: PostgresStore) -> None:
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

    def test_recovery_round_trip(self, store: PostgresStore) -> None:
        events = [RecoveryEvent(ts=T0, score=61.0, hrv_ms=48.2, rhr_bpm=52.0)]
        assert store.insert_recovery(events) == 1
        assert store.get_recovery(WIDE_START, WIDE_END) == events

    def test_device_insert_and_dedupe(self, store: PostgresStore) -> None:
        events = [DeviceEvent(ts=T0, kind="site_change", note="left arm")]
        assert store.insert_device(events) == 1
        assert store.insert_device(events) == 0

    def test_predictions_round_trip_and_dedupe(self, store: PostgresStore) -> None:
        events = [
            PredictionEvent(
                ts=T0, source="openaps", curve_kind="iob",
                values_mg_dl=[120.0, 118.0, 115.0], raw_event_id=5,
            ),
            PredictionEvent(
                ts=T0, source="openaps", curve_kind="cob", values_mg_dl=[120.0, 122.0, 125.0],
            ),
        ]
        assert store.insert_predictions(events) == 2
        assert store.get_predictions(WIDE_START, WIDE_END) == events
        assert store.insert_predictions(events) == 0

    def test_non_utc_input_normalized(self, store: PostgresStore) -> None:
        eastern = timezone(timedelta(hours=-4))
        event = GlucoseEvent(ts=datetime(2026, 6, 1, 6, 0, tzinfo=eastern), mg_dl=110)
        store.insert_glucose([event])
        (got,) = store.get_glucose(WIDE_START, WIDE_END)
        assert got.ts == T0
        assert got.ts.utcoffset() == timedelta(0)

    def test_microsecond_fidelity(self, store: PostgresStore) -> None:
        ts = T0.replace(microsecond=123456)
        store.insert_glucose([GlucoseEvent(ts=ts, mg_dl=99)])
        (got,) = store.get_glucose(WIDE_START, WIDE_END)
        assert got.ts == ts

    def test_glucose_reinsert_is_noop(self, store: PostgresStore) -> None:
        events = [GlucoseEvent(ts=T0, mg_dl=131, trend="Flat")]
        assert store.insert_glucose(events) == 1
        assert store.insert_glucose(events) == 0
        assert len(store.get_glucose(WIDE_START, WIDE_END)) == 1

    def test_insulin_distinct_kinds_same_ts_both_kept(self, store: PostgresStore) -> None:
        events = [
            InsulinEvent(ts=T0, kind=InsulinKind.BOLUS, units=2.0),
            InsulinEvent(ts=T0, kind=InsulinKind.TEMP_BASAL, units=0.5, duration_min=30),
        ]
        assert store.insert_insulin(events) == 2
        assert store.insert_insulin(events) == 0

    def test_mixed_batch_counts_only_new(self, store: PostgresStore) -> None:
        first = [GlucoseEvent(ts=T0, mg_dl=131)]
        store.insert_glucose(first)
        batch = [*first, GlucoseEvent(ts=T0 + timedelta(minutes=5), mg_dl=128)]
        assert store.insert_glucose(batch) == 1

    # ── window semantics ───────────────────────────────────────────────────────

    def test_window_start_inclusive_end_exclusive(self, store: PostgresStore) -> None:
        times = [T0, T0 + timedelta(minutes=5), T0 + timedelta(minutes=10)]
        store.insert_glucose([GlucoseEvent(ts=t, mg_dl=100 + i) for i, t in enumerate(times)])
        got = store.get_glucose(T0, T0 + timedelta(minutes=10))
        assert [e.ts for e in got] == times[:2]

    def test_window_empty(self, store: PostgresStore) -> None:
        store.insert_glucose([GlucoseEvent(ts=T0, mg_dl=100)])
        assert store.get_glucose(T0 + timedelta(minutes=1), T0 + timedelta(minutes=2)) == []

    def test_window_ascending_order(self, store: PostgresStore) -> None:
        times = [T0 + timedelta(minutes=10), T0, T0 + timedelta(minutes=5)]
        store.insert_glucose([GlucoseEvent(ts=t, mg_dl=100) for t in times])
        got = store.get_glucose(WIDE_START, WIDE_END)
        assert [e.ts for e in got] == sorted(times)

    def test_sleep_windowed_on_ts_start(self, store: PostgresStore) -> None:
        event = SleepEvent(ts_start=T0, ts_end=T0 + timedelta(hours=8), duration_min=480)
        store.insert_sleep([event])
        assert store.get_sleep(T0, T0 + timedelta(hours=1)) == [event]
        assert store.get_sleep(T0 + timedelta(hours=1), WIDE_END) == []

    def test_predictions_window_half_open(self, store: PostgresStore) -> None:
        inside = PredictionEvent(ts=T0, source="loop", curve_kind="loop", values_mg_dl=[140.0])
        boundary = PredictionEvent(
            ts=T0 + timedelta(days=2), source="loop", curve_kind="loop", values_mg_dl=[140.0]
        )
        store.insert_predictions([inside, boundary])
        got = store.get_predictions(WIDE_START, T0 + timedelta(days=2))
        assert got == [inside]

    # ── coverage ───────────────────────────────────────────────────────────────

    def test_coverage_empty(self, store: PostgresStore) -> None:
        stats = store.coverage()
        assert stats.first_ts is None
        assert stats.last_ts is None
        assert stats.span_days == 0.0
        assert stats.n_glucose == 0
        assert stats.glucose_coverage_pct == 0.0
        assert stats.days_with_insulin_pct == 0.0

    def test_coverage_populated(self, store: PostgresStore) -> None:
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
        assert stats.glucose_coverage_pct == pytest.approx(100.0 * 13 / 433)
        assert stats.n_insulin == 2
        assert stats.days_with_insulin_pct == pytest.approx(100.0)
        assert stats.n_meals == 1
        assert stats.n_sleep == 1
        assert stats.n_activity == 1

    def test_coverage_single_reading_full(self, store: PostgresStore) -> None:
        store.insert_glucose([GlucoseEvent(ts=T0, mg_dl=100)])
        stats = store.coverage()
        assert stats.span_days == 0.0
        assert stats.glucose_coverage_pct == pytest.approx(100.0)

    # ── rollups ────────────────────────────────────────────────────────────────

    def test_rollups_upsert_and_get(self, store: PostgresStore) -> None:
        day1 = datetime(2026, 6, 1, tzinfo=UTC)
        rollups = [_rollup(day1), _rollup(day1 + timedelta(days=1), mean=120.0)]
        assert store.upsert_rollups(rollups) == 2
        got = store.get_rollups(RollupPeriod.DAILY, day1, day1 + timedelta(days=7))
        assert got == rollups

    def test_rollups_reupsert_updates_in_place(self, store: PostgresStore) -> None:
        day1 = datetime(2026, 6, 1, tzinfo=UTC)
        assert store.upsert_rollups([_rollup(day1)]) == 1
        assert store.upsert_rollups([_rollup(day1, mean=99.0, n=200)]) == 0
        got = store.get_rollups(RollupPeriod.DAILY, day1, day1 + timedelta(days=1))
        assert len(got) == 1
        assert got[0].mean == 99.0
        assert got[0].n == 200

    def test_rollups_filter_by_period_and_range(self, store: PostgresStore) -> None:
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
        assert store.get_rollups(RollupPeriod.DAILY, day1, day1) == []

    # ── findings ───────────────────────────────────────────────────────────────

    def test_finding_insert_round_trip(self, store: PostgresStore) -> None:
        finding = _finding()
        fid = store.insert_finding(finding)
        (got,) = store.get_findings()
        assert got == finding.model_copy(update={"id": fid})
        assert got.window_start is not None
        assert got.window_start.utcoffset() == timedelta(0)

    def test_finding_ids_sequential(self, store: PostgresStore) -> None:
        assert store.insert_finding(_finding()) == 1
        assert store.insert_finding(_finding(headline="Another")) == 2

    def test_finding_get_filters(self, store: PostgresStore) -> None:
        store.insert_finding(_finding(agent="pattern_miner", kind="overnight_low"))
        store.insert_finding(_finding(agent="pattern_miner", kind="post_meal_spike"))
        store.insert_finding(_finding(agent="skeptic", kind="overnight_low"))
        assert len(store.get_findings(agent="pattern_miner")) == 2
        assert len(store.get_findings(kind="overnight_low")) == 2
        assert len(store.get_findings(agent="skeptic", kind="overnight_low")) == 1
        assert store.get_findings(agent="nobody") == []
        assert len(store.get_findings(status=FindingStatus.ACTIVE)) == 3
        assert store.get_findings(status=FindingStatus.REJECTED) == []

    def test_finding_limit_newest_first(self, store: PostgresStore) -> None:
        for i in range(5):
            store.insert_finding(_finding(headline=f"finding {i}"))
        got = store.get_findings(limit=2)
        assert [f.headline for f in got] == ["finding 4", "finding 3"]

    def test_finding_status_transition(self, store: PostgresStore) -> None:
        fid = store.insert_finding(_finding())
        store.set_finding_status(fid, FindingStatus.REJECTED)
        (got,) = store.get_findings(status=FindingStatus.REJECTED)
        assert got.id == fid
        assert got.status is FindingStatus.REJECTED
        assert store.get_findings(status=FindingStatus.ACTIVE) == []

    def test_finding_supersede(self, store: PostgresStore) -> None:
        old_id = store.insert_finding(_finding(headline="v1"))
        new_id = store.insert_finding(_finding(headline="v2"))
        store.supersede_finding(old_id, new_id)
        (old,) = store.get_findings(status=FindingStatus.SUPERSEDED)
        assert old.id == old_id
        assert old.superseded_by == new_id
        (active,) = store.get_findings(status=FindingStatus.ACTIVE)
        assert active.id == new_id
        assert active.superseded_by is None

    # ── hypotheses ─────────────────────────────────────────────────────────────

    def test_hypothesis_round_trip(self, store: PostgresStore) -> None:
        hypothesis = Hypothesis(
            statement="Late-night protein delays overnight rise",
            source_finding_id=4,
            tests=[{"name": "split_replication", "passed": True, "p": 0.03}],
        )
        hid = store.insert_hypothesis(hypothesis)
        (got,) = store.get_hypotheses()
        assert got == hypothesis.model_copy(update={"id": hid})

    def test_hypothesis_filter_by_status(self, store: PostgresStore) -> None:
        store.insert_hypothesis(Hypothesis(statement="open one"))
        store.insert_hypothesis(
            Hypothesis(statement="refuted one", status=HypothesisStatus.REFUTED)
        )
        assert len(store.get_hypotheses()) == 2
        got = store.get_hypotheses(status="refuted")
        assert [h.statement for h in got] == ["refuted one"]
        assert store.get_hypotheses(status="stale") == []

    # ── goals + checkpoints ────────────────────────────────────────────────────

    def test_goal_round_trip_and_status(self, store: PostgresStore) -> None:
        goal = Goal(
            statement="Cut overnight lows",
            metric=GoalMetric.NOCTURNAL_TBR,
            direction="decrease",
            target=2.0,
            tools=[{"tool": "scan_overnight", "args": {"days": 14}}],
            cadence_days=7,
            created_at=T0,
        )
        gid = store.insert_goal(goal)
        (got,) = store.get_goals()
        assert got == goal.model_copy(update={"id": gid})
        assert got.created_at is not None
        assert got.created_at.utcoffset() == timedelta(0)

        store.set_goal_status(gid, GoalStatus.ACHIEVED)
        assert store.get_goals(status=GoalStatus.ACTIVE) == []
        (achieved,) = store.get_goals(status=GoalStatus.ACHIEVED)
        assert achieved.status is GoalStatus.ACHIEVED

    def test_goal_checkpoints_ordered(self, store: PostgresStore) -> None:
        gid = store.insert_goal(
            Goal(statement="g", metric=GoalMetric.TIR, direction="increase")
        )
        store.insert_goal_checkpoint(
            GoalCheckpoint(goal_id=gid, ts=T0 + timedelta(days=1), metric_value=68.0, note="b")
        )
        store.insert_goal_checkpoint(
            GoalCheckpoint(goal_id=gid, ts=T0, metric_value=65.0, note="a")
        )
        got = store.get_goal_checkpoints(gid)
        assert [c.note for c in got] == ["a", "b"]
        assert got[0].ts == T0
        assert got[0].ts.utcoffset() == timedelta(0)
