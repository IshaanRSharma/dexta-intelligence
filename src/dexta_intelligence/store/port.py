"""StoragePort — the single seam between the system and persistence.

Everything above this module is backend-agnostic. Postgres is the reference
backend (TIMESTAMPTZ / JSONB / pgvector); SQLite is the zero-setup on-ramp.
Nothing outside ``dexta_intelligence.store`` may import a database driver —
CI enforces this.

The port is deliberately narrow: connectors write, analytics read windows,
agents read/write findings. If a feature needs a new query, it gets a new
*named* method here — agents never build SQL.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from datetime import datetime

    from dexta_intelligence.models import (
        ActivityEvent,
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
        MealEvent,
        PredictionEvent,
        RawEvent,
        RecoveryEvent,
        Rollup,
        RollupPeriod,
        SleepEvent,
    )

__all__ = ["StoragePort"]


@runtime_checkable
class StoragePort(Protocol):
    """Persistence contract. Implementations: ``PostgresStore``, ``SQLiteStore``."""

    # ── lifecycle ────────────────────────────────────────────────────────────

    def migrate(self) -> None:
        """Create or upgrade the schema. Idempotent."""
        ...

    # ── layer 1: raw events ──────────────────────────────────────────────────

    def upsert_raw_events(self, events: list[RawEvent]) -> int:
        """Insert raw events, skipping ``(source, source_id)`` duplicates.

        Returns the number of *new* rows — the idempotency contract that makes
        every connector re-run safe.
        """
        ...

    def get_watermark(self, source: str) -> datetime | None:
        """Latest ``source_ts`` ingested for a source (sync cursor)."""
        ...

    # ── layer 2: clinical timeline ───────────────────────────────────────────

    def insert_glucose(self, events: list[GlucoseEvent]) -> int: ...
    def insert_insulin(self, events: list[InsulinEvent]) -> int: ...
    def insert_meals(self, events: list[MealEvent]) -> int: ...
    def insert_activity(self, events: list[ActivityEvent]) -> int: ...
    def insert_sleep(self, events: list[SleepEvent]) -> int: ...
    def insert_recovery(self, events: list[RecoveryEvent]) -> int: ...
    def insert_device(self, events: list[DeviceEvent]) -> int: ...
    def insert_predictions(self, events: list[PredictionEvent]) -> int: ...

    def get_glucose(self, start: datetime, end: datetime) -> list[GlucoseEvent]: ...
    def get_insulin(self, start: datetime, end: datetime) -> list[InsulinEvent]: ...
    def get_meals(self, start: datetime, end: datetime) -> list[MealEvent]: ...
    def get_activity(self, start: datetime, end: datetime) -> list[ActivityEvent]: ...
    def get_sleep(self, start: datetime, end: datetime) -> list[SleepEvent]: ...
    def get_recovery(self, start: datetime, end: datetime) -> list[RecoveryEvent]: ...
    def get_predictions(self, start: datetime, end: datetime) -> list[PredictionEvent]: ...

    def coverage(self) -> CoverageStats:
        """Data-sufficiency summary across the whole timeline (cold-start input)."""
        ...

    # ── layer 3: rollups ─────────────────────────────────────────────────────

    def upsert_rollups(self, rollups: list[Rollup]) -> int: ...
    def get_rollups(
        self, period: RollupPeriod, start: datetime, end: datetime
    ) -> list[Rollup]: ...

    # ── layer 4: agent memory ────────────────────────────────────────────────

    def insert_finding(self, finding: Finding) -> int:
        """Persist a finding; returns its id."""
        ...

    def supersede_finding(self, old_id: int, new_id: int) -> None: ...

    def set_finding_status(self, finding_id: int, status: FindingStatus) -> None: ...

    def get_findings(
        self,
        *,
        agent: str | None = None,
        kind: str | None = None,
        status: FindingStatus | None = None,
        limit: int = 50,
    ) -> list[Finding]: ...

    def insert_hypothesis(self, hypothesis: Hypothesis) -> int: ...
    def get_hypotheses(self, *, status: str | None = None) -> list[Hypothesis]: ...

    def insert_goal(self, goal: Goal) -> int: ...
    def get_goals(self, *, status: GoalStatus | None = None) -> list[Goal]: ...
    def set_goal_status(self, goal_id: int, status: GoalStatus) -> None: ...
    def insert_goal_checkpoint(self, checkpoint: GoalCheckpoint) -> int: ...
    def get_goal_checkpoints(self, goal_id: int) -> list[GoalCheckpoint]: ...
