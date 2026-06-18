"""Tests for the sync workflow against in-memory protocol fakes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone

import pytest

from dexta_intelligence.connectors.base import Connector, HealthReport, NormalizedBatch
from dexta_intelligence.models import (
    ActivityEvent,
    ChatSession,
    ChatTurn,
    CoverageStats,
    DeviceEvent,
    Finding,
    FindingStatus,
    GlucoseEvent,
    Goal,
    GoalCheckpoint,
    GoalStatus,
    Hypothesis,
    InsulinEvent,
    InsulinKind,
    InvestigationRun,
    ManualEvent,
    MealEvent,
    OpenInvestigation,
    PredictionEvent,
    RawEvent,
    RecoveryEvent,
    Rollup,
    RollupPeriod,
    SleepEvent,
    TherapyProfile,
)
from dexta_intelligence.store.port import StoragePort
from dexta_intelligence.workflows.sync import (
    DEFAULT_LOOKBACK,
    OVERLAP_MARGIN,
    SyncReport,
    sync,
    sync_all,
)

FIXED_NOW = datetime(2025, 3, 2, 1, 0, tzinfo=UTC)
DAY1 = datetime(2025, 3, 1, tzinfo=UTC)
DAY2 = datetime(2025, 3, 2, tzinfo=UTC)


# ─────────────────────────────────────────────────────────────────────────────
# Protocol-complete in-memory fakes
# ─────────────────────────────────────────────────────────────────────────────


class FakeStore:
    """Minimal but protocol-complete in-memory StoragePort.

    Idempotency mirrors the real contract: raws dedupe on
    ``(source, source_id)``; timeline inserts dedupe on natural keys and
    return the count of genuinely new rows.
    """

    def __init__(self) -> None:
        self.raw: dict[tuple[str, str], RawEvent] = {}
        self.raw_ids: dict[tuple[str, str], int] = {}
        self._next_raw_id = 0
        self.glucose: dict[datetime, GlucoseEvent] = {}
        self.insulin: dict[tuple[datetime, InsulinKind], InsulinEvent] = {}
        self.meals: dict[datetime, MealEvent] = {}
        self.activity: list[ActivityEvent] = []
        self.sleep: list[SleepEvent] = []
        self.recovery: list[RecoveryEvent] = []
        self.device: list[DeviceEvent] = []
        self.predictions: dict[tuple[datetime, str], PredictionEvent] = {}
        self.rollups: dict[tuple[RollupPeriod, datetime], Rollup] = {}
        self.rollup_upsert_calls: list[list[Rollup]] = []
        self.findings: list[Finding] = []
        self.hypotheses: list[Hypothesis] = []
        self.goals: list[Goal] = []
        self.goal_checkpoints: list[GoalCheckpoint] = []
        self.chat_turns: list[ChatTurn] = []
        self.investigation_runs: list[InvestigationRun] = []
        self.open_investigations: list[OpenInvestigation] = []
        self.manual_events: list[ManualEvent] = []
        self.profile_versions: list[TherapyProfile] = []

    def migrate(self) -> None:
        return None

    def existing_raw_ids(self, events: list[RawEvent]) -> dict[str, int]:
        return {
            e.source_id: self.raw_ids[(e.source, e.source_id)]
            for e in events
            if (e.source, e.source_id) in self.raw
        }

    def upsert_raw_events(self, events: list[RawEvent]) -> dict[str, int]:
        for event in events:
            key = (event.source, event.source_id)
            if key not in self.raw:
                self.raw[key] = event
                self._next_raw_id += 1
                self.raw_ids[key] = self._next_raw_id
        return {e.source_id: self.raw_ids[(e.source, e.source_id)] for e in events}

    def replace_raw_events(self, events: list[RawEvent]) -> dict[str, int]:
        for event in events:
            key = (event.source, event.source_id)
            if key not in self.raw_ids:
                self._next_raw_id += 1
                self.raw_ids[key] = self._next_raw_id
            self.raw[key] = event
        return {e.source_id: self.raw_ids[(e.source, e.source_id)] for e in events}

    def get_raw_event(self, source: str, source_id: str) -> RawEvent | None:
        return self.raw.get((source, source_id))

    def get_watermark(self, source: str) -> datetime | None:
        stamps = [e.source_ts for e in self.raw.values() if e.source == source]
        return max(stamps) if stamps else None

    def source_event_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for event in self.raw.values():
            counts[event.source] = counts.get(event.source, 0) + 1
        return counts

    def insert_glucose(self, events: list[GlucoseEvent]) -> int:
        new = 0
        for event in events:
            if event.ts not in self.glucose:
                self.glucose[event.ts] = event
                new += 1
        return new

    def insert_insulin(self, events: list[InsulinEvent]) -> int:
        new = 0
        for event in events:
            key = (event.ts, event.kind)
            if key not in self.insulin:
                self.insulin[key] = event
                new += 1
        return new

    def insert_meals(self, events: list[MealEvent]) -> int:
        new = 0
        for event in events:
            if event.ts not in self.meals:
                self.meals[event.ts] = event
                new += 1
        return new

    def insert_activity(self, events: list[ActivityEvent]) -> int:
        self.activity.extend(events)
        return len(events)

    def insert_sleep(self, events: list[SleepEvent]) -> int:
        self.sleep.extend(events)
        return len(events)

    def insert_recovery(self, events: list[RecoveryEvent]) -> int:
        self.recovery.extend(events)
        return len(events)

    def insert_device(self, events: list[DeviceEvent]) -> int:
        self.device.extend(events)
        return len(events)

    def insert_predictions(self, events: list[PredictionEvent]) -> int:
        new = 0
        for event in events:
            key = (event.ts, event.curve_kind)
            if key not in self.predictions:
                self.predictions[key] = event
                new += 1
        return new

    def get_glucose(self, start: datetime, end: datetime) -> list[GlucoseEvent]:
        return sorted(
            (e for e in self.glucose.values() if start <= e.ts < end), key=lambda e: e.ts
        )

    def get_insulin(self, start: datetime, end: datetime) -> list[InsulinEvent]:
        return sorted(
            (e for e in self.insulin.values() if start <= e.ts < end), key=lambda e: e.ts
        )

    def get_meals(self, start: datetime, end: datetime) -> list[MealEvent]:
        return sorted(
            (e for e in self.meals.values() if start <= e.ts < end), key=lambda e: e.ts
        )

    def get_activity(self, start: datetime, end: datetime) -> list[ActivityEvent]:
        return [e for e in self.activity if start <= e.ts < end]

    def get_sleep(self, start: datetime, end: datetime) -> list[SleepEvent]:
        return [e for e in self.sleep if start <= e.ts_start < end]

    def get_recovery(self, start: datetime, end: datetime) -> list[RecoveryEvent]:
        return [e for e in self.recovery if start <= e.ts < end]

    def get_predictions(self, start: datetime, end: datetime) -> list[PredictionEvent]:
        return sorted(
            (e for e in self.predictions.values() if start <= e.ts < end), key=lambda e: e.ts
        )

    def coverage(self) -> CoverageStats:
        stamps = sorted(self.glucose)
        first = stamps[0] if stamps else None
        last = stamps[-1] if stamps else None
        span = (last - first).total_seconds() / 86400.0 if first and last else 0.0
        return CoverageStats(
            first_ts=first,
            last_ts=last,
            span_days=span,
            n_glucose=len(self.glucose),
            glucose_coverage_pct=100.0 if self.glucose else 0.0,
            n_insulin=len(self.insulin),
            days_with_insulin_pct=100.0 if self.insulin else 0.0,
            n_meals=len(self.meals),
            n_sleep=len(self.sleep),
            n_activity=len(self.activity),
        )

    def upsert_rollups(self, rollups: list[Rollup]) -> int:
        self.rollup_upsert_calls.append(list(rollups))
        for rollup in rollups:
            self.rollups[(rollup.period, rollup.period_start)] = rollup
        return len(rollups)

    def get_rollups(self, period: RollupPeriod, start: datetime, end: datetime) -> list[Rollup]:
        return sorted(
            (
                r
                for (p, ts), r in self.rollups.items()
                if p is period and start <= ts < end
            ),
            key=lambda r: r.period_start,
        )

    def insert_finding(self, finding: Finding) -> int:
        self.findings.append(finding)
        return len(self.findings)

    def supersede_finding(self, old_id: int, new_id: int) -> None:
        return None

    def set_finding_status(self, finding_id: int, status: FindingStatus) -> None:
        return None

    def get_findings(
        self,
        *,
        agent: str | None = None,
        kind: str | None = None,
        status: FindingStatus | None = None,
        limit: int = 50,
    ) -> list[Finding]:
        return self.findings[:limit]

    def insert_hypothesis(self, hypothesis: Hypothesis) -> int:
        self.hypotheses.append(hypothesis)
        return len(self.hypotheses)

    def get_hypotheses(self, *, status: str | None = None) -> list[Hypothesis]:
        return list(self.hypotheses)

    def insert_goal(self, goal: Goal) -> int:
        self.goals.append(goal)
        return len(self.goals)

    def get_goals(self, *, status: GoalStatus | None = None) -> list[Goal]:
        if status is None:
            return list(self.goals)
        return [g for g in self.goals if g.status == status]

    def set_goal_status(self, goal_id: int, status: GoalStatus) -> None:
        return None

    def insert_goal_checkpoint(self, checkpoint: GoalCheckpoint) -> int:
        self.goal_checkpoints.append(checkpoint)
        return len(self.goal_checkpoints)

    def get_goal_checkpoints(self, goal_id: int) -> list[GoalCheckpoint]:
        return [c for c in self.goal_checkpoints if c.goal_id == goal_id]

    def append_chat_turn(self, turn: ChatTurn) -> int:
        self.chat_turns.append(turn)
        return len(self.chat_turns)

    def get_chat_turns(self, session_id: str, *, limit: int = 50) -> list[ChatTurn]:
        turns = [t for t in self.chat_turns if t.session_id == session_id]
        return turns[-limit:]

    def get_chat_sessions(self, *, limit: int = 50) -> list[ChatSession]:
        by_session: dict[str, list[ChatTurn]] = {}
        for turn in self.chat_turns:
            by_session.setdefault(turn.session_id, []).append(turn)
        sessions = [
            ChatSession(
                session_id=sid,
                last_ts=turns[-1].ts,
                turn_count=len(turns),
                preview=next((t.content for t in turns if t.role == "user"), ""),
            )
            for sid, turns in by_session.items()
        ]
        sessions.sort(key=lambda s: s.last_ts, reverse=True)
        return sessions[:limit]

    def delete_chat_session(self, session_id: str) -> int:
        before = len(self.chat_turns)
        self.chat_turns = [t for t in self.chat_turns if t.session_id != session_id]
        return before - len(self.chat_turns)

    def insert_investigation_run(self, run: InvestigationRun) -> int:
        self.investigation_runs.append(run)
        return len(self.investigation_runs)

    def get_investigation_runs(self, *, limit: int = 50) -> list[InvestigationRun]:
        return list(reversed(self.investigation_runs))[:limit]

    def get_investigation_run(self, run_db_id: int) -> InvestigationRun | None:
        idx = run_db_id - 1
        if 0 <= idx < len(self.investigation_runs):
            return self.investigation_runs[idx]
        return None

    def insert_open_investigation(self, inv: OpenInvestigation) -> int:
        new_id = len(self.open_investigations) + 1
        self.open_investigations.append(inv.model_copy(update={"id": new_id}))
        return new_id

    def get_open_investigations(
        self, *, status: str | None = None
    ) -> list[OpenInvestigation]:
        rows = list(reversed(self.open_investigations))
        if status is not None:
            rows = [r for r in rows if r.status == status]
        return rows

    def update_open_investigation(
        self,
        inv_id: int,
        *,
        current: float,
        status: str,
        promoted_run_id: str | None = None,
    ) -> None:
        for i, inv in enumerate(self.open_investigations):
            if inv.id == inv_id:
                self.open_investigations[i] = inv.model_copy(
                    update={
                        "current": current,
                        "status": status,
                        "promoted_run_id": promoted_run_id,
                    }
                )
                return

    def add_manual_event(self, event: ManualEvent) -> int:
        new_id = len(self.manual_events) + 1
        self.manual_events.append(event.model_copy(update={"id": new_id}))
        return new_id

    def get_manual_events(self, start: datetime, end: datetime) -> list[ManualEvent]:
        return sorted(
            (e for e in self.manual_events if start <= e.event_ts < end),
            key=lambda e: e.event_ts,
        )

    def add_profile_version(self, profile: TherapyProfile) -> int:
        latest = self.profile_versions[-1] if self.profile_versions else None
        if latest is not None and latest.content_hash == profile.content_hash:
            return latest.id or 0
        if latest is not None and latest.active_to is None:
            self.profile_versions[-1] = latest.model_copy(
                update={"active_to": profile.active_from}
            )
        new_id = len(self.profile_versions) + 1
        self.profile_versions.append(profile.model_copy(update={"id": new_id}))
        return new_id

    def get_profile_versions(self) -> list[TherapyProfile]:
        return sorted(self.profile_versions, key=lambda p: p.active_from)

    def get_active_profile(self, at: datetime) -> TherapyProfile | None:
        candidates = [p for p in self.profile_versions if p.active_from <= at]
        return max(candidates, key=lambda p: p.active_from) if candidates else None


@dataclass
class FakeConnector:
    """Connector fake: holds a fixed dataset, serves the slice newer than ``since``."""

    source: str
    batch: NormalizedBatch
    pull_since: list[datetime] = field(default_factory=list)

    def check(self) -> HealthReport:
        return HealthReport(ok=True, source=self.source)

    def pull(self, since: datetime) -> NormalizedBatch:
        self.pull_since.append(since)
        return NormalizedBatch(
            raw=[r for r in self.batch.raw if r.source_ts > since],
            glucose=[g for g in self.batch.glucose if g.ts > since],
            insulin=[i for i in self.batch.insulin if i.ts > since],
            meals=[m for m in self.batch.meals if m.ts > since],
            predictions=[p for p in self.batch.predictions if p.ts > since],
        )


@dataclass
class FailingConnector:
    source: str = "broken"

    def check(self) -> HealthReport:
        return HealthReport(ok=False, source=self.source)

    def pull(self, since: datetime) -> NormalizedBatch:
        raise RuntimeError("boom")


def make_batch() -> NormalizedBatch:
    """Four glucose readings straddling the UTC midnight between two days."""
    stamps = [
        DAY1 + timedelta(hours=23, minutes=50),
        DAY1 + timedelta(hours=23, minutes=55),
        DAY2,
        DAY2 + timedelta(minutes=5),
    ]
    values = [120, 130, 140, 150]
    raw = [
        RawEvent(source="fake", source_id=f"g{i}", source_ts=ts, payload={"sgv": v})
        for i, (ts, v) in enumerate(zip(stamps, values, strict=True))
    ]
    glucose = [GlucoseEvent(ts=ts, mg_dl=v) for ts, v in zip(stamps, values, strict=True)]
    meal_ts = DAY1 + timedelta(hours=23, minutes=52)
    return NormalizedBatch(
        raw=raw,
        glucose=glucose,
        insulin=[InsulinEvent(ts=meal_ts, kind=InsulinKind.BOLUS, units=3.0)],
        meals=[MealEvent(ts=meal_ts, carbs_g=30.0)],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestProtocolConformance:
    def test_fakes_satisfy_protocols(self) -> None:
        assert isinstance(FakeStore(), StoragePort)
        assert isinstance(FakeConnector(source="fake", batch=NormalizedBatch()), Connector)
        assert isinstance(FailingConnector(), Connector)


class TestFirstSync:
    def test_uses_default_lookback_when_no_watermark(self) -> None:
        connector = FakeConnector(source="fake", batch=make_batch())
        report = sync(connector, FakeStore(), now=FIXED_NOW)
        assert connector.pull_since == [FIXED_NOW - DEFAULT_LOOKBACK]
        assert report.since == FIXED_NOW - DEFAULT_LOOKBACK
        assert report.until == FIXED_NOW

    def test_report_counts(self) -> None:
        store = FakeStore()
        report = sync(FakeConnector(source="fake", batch=make_batch()), store, now=FIXED_NOW)
        assert report.source == "fake"
        assert report.raw_new == 4
        assert report.inserted["glucose"] == 4
        assert report.inserted["insulin"] == 1
        assert report.inserted["meals"] == 1
        assert report.inserted["activity"] == 0
        assert report.errors == ()
        assert report.ok
        assert report.duration_s >= 0.0

    def test_rollups_recomputed_only_for_touched_days(self) -> None:
        store = FakeStore()
        report = sync(FakeConnector(source="fake", batch=make_batch()), store, now=FIXED_NOW)
        assert report.rollup_days == 2
        assert set(store.rollups) == {
            (RollupPeriod.DAILY, DAY1),
            (RollupPeriod.DAILY, DAY2),
        }
        day1 = store.rollups[(RollupPeriod.DAILY, DAY1)]
        assert day1.n == 2
        assert day1.bolus_units == pytest.approx(3.0)
        assert day1.carbs_g == pytest.approx(30.0)
        day2 = store.rollups[(RollupPeriod.DAILY, DAY2)]
        assert day2.n == 2
        assert day2.carbs_g is None


class TestSecondSync:
    def test_uses_watermark_minus_overlap(self) -> None:
        store = FakeStore()
        connector = FakeConnector(source="fake", batch=make_batch())
        sync(connector, store, now=FIXED_NOW)
        sync(connector, store, now=FIXED_NOW + timedelta(hours=1))
        watermark = DAY2 + timedelta(minutes=5)  # max raw source_ts
        assert connector.pull_since[1] == watermark - OVERLAP_MARGIN

    def test_idempotent_rerun_inserts_nothing_new(self) -> None:
        store = FakeStore()
        connector = FakeConnector(source="fake", batch=make_batch())
        first = sync(connector, store, now=FIXED_NOW)
        second = sync(connector, store, now=FIXED_NOW + timedelta(hours=1))
        assert first.raw_new == 4
        assert second.raw_new == 0
        assert second.inserted["glucose"] == 0
        assert len(store.glucose) == 4

    def test_rollup_recomputed_from_full_stored_day(self) -> None:
        # The overlap re-pull only returns a tail slice of day 1, but the
        # rollup must be rebuilt from the store's complete day.
        store = FakeStore()
        connector = FakeConnector(source="fake", batch=make_batch())
        sync(connector, store, now=FIXED_NOW)
        second = sync(connector, store, now=FIXED_NOW + timedelta(hours=1))
        assert second.rollup_days == 2
        assert store.rollups[(RollupPeriod.DAILY, DAY1)].n == 2


class TestSyncAll:
    def test_failure_isolation(self) -> None:
        store = FakeStore()
        bad = FailingConnector()
        good = FakeConnector(source="fake", batch=make_batch())
        reports = sync_all([bad, good], store, now=FIXED_NOW)

        assert [r.source for r in reports] == ["broken", "fake"]
        assert not reports[0].ok
        assert reports[0].errors == ("RuntimeError: boom",)
        assert reports[0].raw_new == 0
        assert reports[1].ok
        assert reports[1].raw_new == 4
        assert len(store.glucose) == 4  # the failing source did not stop the good one

    def test_all_success(self) -> None:
        reports = sync_all(
            [FakeConnector(source="fake", batch=make_batch())], FakeStore(), now=FIXED_NOW
        )
        assert len(reports) == 1
        assert all(isinstance(r, SyncReport) and r.ok for r in reports)


class TestUtcEnforcement:
    def test_naive_now_rejected(self) -> None:
        connector = FakeConnector(source="fake", batch=make_batch())
        with pytest.raises(ValueError, match="timezone-aware"):
            sync(connector, FakeStore(), now=datetime(2025, 3, 2, 1, 0))

    def test_aware_non_utc_now_normalized_to_utc(self) -> None:
        est_now = FIXED_NOW.astimezone(timezone(timedelta(hours=-5)))
        report = sync(
            FakeConnector(source="fake", batch=make_batch()), FakeStore(), now=est_now
        )
        assert report.until == FIXED_NOW
        assert report.until.tzinfo == UTC


class TestProvenance:
    def test_glucose_linked_to_originating_raw(self) -> None:
        store = FakeStore()
        sync(FakeConnector(source="fake", batch=make_batch()), store, now=FIXED_NOW)
        stored = sorted(store.glucose.values(), key=lambda e: e.ts)
        assert all(g.raw_event_id is not None for g in stored)
        # each glucose points at the raw row sharing its instant
        for g in stored:
            raw_key = next(
                k for k, r in store.raw.items() if r.source_ts == g.ts
            )
            assert g.raw_event_id == store.raw_ids[raw_key]

    def test_typed_event_without_matching_raw_stays_unlinked(self) -> None:
        # make_batch's insulin/meal sit at 23:52, an instant with no raw row.
        store = FakeStore()
        sync(FakeConnector(source="fake", batch=make_batch()), store, now=FIXED_NOW)
        assert all(i.raw_event_id is None for i in store.insulin.values())
        assert all(m.raw_event_id is None for m in store.meals.values())

    def test_ambiguous_timestamp_left_unlinked(self) -> None:
        ts = DAY2 + timedelta(hours=1)
        raw = [
            RawEvent(source="fake", source_id="a", source_ts=ts, payload={}),
            RawEvent(source="fake", source_id="b", source_ts=ts, payload={}),
        ]
        batch = NormalizedBatch(raw=raw, glucose=[GlucoseEvent(ts=ts, mg_dl=110)])
        store = FakeStore()
        sync(FakeConnector(source="fake", batch=batch), store, now=FIXED_NOW)
        assert store.glucose[ts].raw_event_id is None

    def test_relink_is_idempotent_across_resync(self) -> None:
        store = FakeStore()
        connector = FakeConnector(source="fake", batch=make_batch())
        sync(connector, store, now=FIXED_NOW)
        first = {ts: g.raw_event_id for ts, g in store.glucose.items()}
        sync(connector, store, now=FIXED_NOW + timedelta(hours=1))
        second = {ts: g.raw_event_id for ts, g in store.glucose.items()}
        assert first == second
        assert all(v is not None for v in second.values())


class TestPredictions:
    def test_predictions_persisted(self) -> None:
        batch = make_batch()
        pred = PredictionEvent(
            ts=DAY2,
            source="openaps",
            curve_kind="iob",
            values_mg_dl=[140.0, 142.0, 145.0],
        )
        batch_with_preds = NormalizedBatch(
            raw=batch.raw,
            glucose=batch.glucose,
            insulin=batch.insulin,
            meals=batch.meals,
            predictions=[pred],
        )
        store = FakeStore()
        report = sync(
            FakeConnector(source="fake", batch=batch_with_preds), store, now=FIXED_NOW
        )
        assert report.ok
        assert report.notes == ()
        assert report.inserted["predictions"] == 1
        stored = list(store.predictions.values())
        assert len(stored) == 1
        assert stored[0].model_copy(update={"raw_event_id": None}) == pred
