"""SQLiteStore — the zero-setup on-ramp backend for :class:`StoragePort`.

stdlib ``sqlite3`` only. Design decisions (the parts the protocol leaves open):

- **Timestamps** are stored as ISO-8601 UTC TEXT (``YYYY-MM-DDTHH:MM:SS[.ffffff]+00:00``).
  Because every stored value is normalized to UTC with a fixed layout, lexicographic
  comparison equals chronological comparison, so plain TEXT ``<``/``>=``/``MAX`` are
  correct.
- **Window queries** are half-open: ``start <= ts < end``, ordered by timestamp
  ascending. Sleep events are windowed and ordered on ``ts_start``.
- **Dedupe keys** (mirrors the raw-store idempotency philosophy — re-running a
  connector or normalizer never double-inserts):

  ====================  =======================================
  table                 natural identity (UNIQUE index)
  ====================  =======================================
  raw_events            ``(source, source_id)`` (per the port contract)
  glucose_events        ``ts`` — one CGM reading per instant
  insulin_events        ``(ts, kind)``
  meal_events           ``ts``
  activity_events       ``(ts, kind)``
  sleep_events          ``ts_start``
  recovery_events       ``ts``
  device_events         ``(ts, kind)``
  prediction_events     ``(ts, curve_kind)``
  rollups               ``(period, period_start)`` (true upsert: DO UPDATE)
  ====================  =======================================

  All ``insert_*``/``upsert_*`` methods return the number of *new* rows; for
  rollups, existing periods are updated in place but do not count as new.
- **Provenance** ``raw_event_id`` columns are soft references (no FK constraint),
  so typed events can be persisted independently of raw-event insertion order.
- **get_findings** returns newest first (highest id first); ``get_hypotheses``
  returns insertion order.
- Single-writer assumption; one connection, no pooling.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dexta_intelligence.models import (
    ActivityEvent,
    CoverageStats,
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
    RecoveryEvent,
    Rollup,
    RollupPeriod,
    SleepEvent,
)

if TYPE_CHECKING:
    from dexta_intelligence.models import DeviceEvent, RawEvent

__all__ = ["SQLiteStore"]

SCHEMA_VERSION = 2

_SECONDS_PER_DAY = 86400.0
_CGM_SLOT_SECONDS = 300.0  # expected 5-minute CGM cadence

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS raw_events (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_ts TEXT NOT NULL,
    payload TEXT NOT NULL,
    UNIQUE (source, source_id)
);
CREATE INDEX IF NOT EXISTS idx_raw_events_source_ts ON raw_events (source, source_ts);

CREATE TABLE IF NOT EXISTS glucose_events (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL UNIQUE,
    mg_dl INTEGER NOT NULL,
    trend TEXT,
    raw_event_id INTEGER
);

CREATE TABLE IF NOT EXISTS insulin_events (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,
    units REAL,
    duration_min REAL,
    automatic INTEGER,
    raw_event_id INTEGER,
    UNIQUE (ts, kind)
);

CREATE TABLE IF NOT EXISTS meal_events (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL UNIQUE,
    carbs_g REAL,
    protein_g REAL,
    fat_g REAL,
    note TEXT,
    raw_event_id INTEGER
);

CREATE TABLE IF NOT EXISTS activity_events (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,
    duration_min REAL,
    intensity REAL,
    strain REAL,
    raw_event_id INTEGER,
    UNIQUE (ts, kind)
);

CREATE TABLE IF NOT EXISTS sleep_events (
    id INTEGER PRIMARY KEY,
    ts_start TEXT NOT NULL UNIQUE,
    ts_end TEXT NOT NULL,
    duration_min REAL NOT NULL,
    score REAL,
    stages TEXT,
    raw_event_id INTEGER
);

CREATE TABLE IF NOT EXISTS recovery_events (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL UNIQUE,
    score REAL,
    hrv_ms REAL,
    rhr_bpm REAL,
    raw_event_id INTEGER
);

CREATE TABLE IF NOT EXISTS device_events (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,
    note TEXT,
    raw_event_id INTEGER,
    UNIQUE (ts, kind)
);

CREATE TABLE IF NOT EXISTS prediction_events (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    source TEXT NOT NULL,
    curve_kind TEXT NOT NULL,
    horizon_min INTEGER NOT NULL,
    values_mg_dl TEXT NOT NULL,
    raw_event_id INTEGER,
    UNIQUE (ts, curve_kind)
);
CREATE INDEX IF NOT EXISTS idx_prediction_events_ts ON prediction_events (ts);

CREATE TABLE IF NOT EXISTS rollups (
    period TEXT NOT NULL,
    period_start TEXT NOT NULL,
    n INTEGER NOT NULL,
    mean REAL,
    sd REAL,
    cv REAL,
    tir REAL,
    tar REAL,
    tar2 REAL,
    tbr REAL,
    tbr2 REAL,
    gmi REAL,
    excursion_count INTEGER,
    bolus_units REAL,
    basal_units REAL,
    carbs_g REAL,
    PRIMARY KEY (period, period_start)
);

CREATE TABLE IF NOT EXISTS findings (
    id INTEGER PRIMARY KEY,
    agent TEXT NOT NULL,
    kind TEXT NOT NULL,
    scope TEXT NOT NULL,
    headline TEXT NOT NULL,
    body_md TEXT NOT NULL,
    evidence TEXT NOT NULL,
    stats TEXT NOT NULL,
    confidence REAL NOT NULL,
    status TEXT NOT NULL,
    skeptic_notes TEXT,
    window_start TEXT,
    window_end TEXT,
    superseded_by INTEGER
);
CREATE INDEX IF NOT EXISTS idx_findings_agent_kind_status ON findings (agent, kind, status);

CREATE TABLE IF NOT EXISTS hypotheses (
    id INTEGER PRIMARY KEY,
    statement TEXT NOT NULL,
    status TEXT NOT NULL,
    source_finding_id INTEGER,
    tests TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS goals (
    id INTEGER PRIMARY KEY,
    statement TEXT NOT NULL,
    metric TEXT NOT NULL,
    direction TEXT NOT NULL,
    target REAL,
    tools TEXT NOT NULL,
    cadence_days INTEGER NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS goal_checkpoints (
    id INTEGER PRIMARY KEY,
    goal_id INTEGER NOT NULL REFERENCES goals (id),
    ts TEXT NOT NULL,
    metric_value REAL,
    note TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_goal_checkpoints_goal ON goal_checkpoints (goal_id, ts);
"""


def _prediction_horizon_min(values: list[float]) -> int:
    """Minutes from cycle time to the last predicted point (5-minute spacing)."""
    if not values:
        return 0
    return max(0, (len(values) - 1) * 5)


def _row_to_goal(r: tuple[Any, ...]) -> Goal:
    return Goal(
        id=r[0],
        statement=r[1],
        metric=GoalMetric(r[2]),
        direction=r[3],
        target=r[4],
        tools=json.loads(r[5]),
        cadence_days=r[6],
        status=GoalStatus(r[7]),
        created_at=_opt_text_to_dt(r[8]),
    )


def _dt_to_text(value: datetime) -> str:
    """Aware datetimes are normalized to UTC; naive ones are stored verbatim."""
    if value.tzinfo is not None:
        value = value.astimezone(UTC)
    return value.isoformat()


def _text_to_dt(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(UTC)
    return parsed


def _opt_dt_to_text(value: datetime | None) -> str | None:
    return None if value is None else _dt_to_text(value)


def _opt_text_to_dt(value: str | None) -> datetime | None:
    return None if value is None else _text_to_dt(value)


def _opt_bool(value: int | None) -> bool | None:
    return None if value is None else bool(value)


class SQLiteStore:
    """:class:`StoragePort` implementation over a single stdlib sqlite3 connection."""

    def __init__(self, path: Path | str = ":memory:") -> None:
        location = str(path)
        if location != ":memory:":
            resolved = Path(location).expanduser()
            resolved.parent.mkdir(parents=True, exist_ok=True)
            location = str(resolved)
        self._path = location
        self._conn = sqlite3.connect(location)
        self._conn.execute("PRAGMA foreign_keys = ON")
        if location != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = NORMAL")

    def close(self) -> None:
        self._conn.close()

    # ── lifecycle ────────────────────────────────────────────────────────────

    def migrate(self) -> None:
        """Create or upgrade the schema. Idempotent (IF NOT EXISTS throughout)."""
        with self._conn:
            self._conn.executescript(_SCHEMA)
            row = self._conn.execute("SELECT version FROM schema_version").fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,)
                )
            elif row[0] < SCHEMA_VERSION:
                self._conn.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))

    # ── layer 1: raw events ──────────────────────────────────────────────────

    def upsert_raw_events(self, events: list[RawEvent]) -> dict[str, int]:
        if not events:
            return {}
        rows = [
            (e.source, e.source_id, _dt_to_text(e.source_ts), json.dumps(e.payload))
            for e in events
        ]
        with self._conn:
            self._conn.executemany(
                "INSERT INTO raw_events (source, source_id, source_ts, payload) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(source, source_id) DO NOTHING",
                rows,
            )
        return self._raw_ids({(e.source, e.source_id) for e in events})

    def existing_raw_ids(self, events: list[RawEvent]) -> dict[str, int]:
        if not events:
            return {}
        return self._raw_ids({(e.source, e.source_id) for e in events})

    def _raw_ids(self, keys: set[tuple[str, str]]) -> dict[str, int]:
        """Resolve ``source_id -> id`` for the given ``(source, source_id)`` keys.

        Covers both freshly-inserted and pre-existing rows; ``source_id`` is
        unique within a source, so the returned keys never collide for a
        single-source batch.
        """
        result: dict[str, int] = {}
        for source, source_id in keys:
            row = self._conn.execute(
                "SELECT id FROM raw_events WHERE source = ? AND source_id = ?",
                (source, source_id),
            ).fetchone()
            if row is not None:
                result[source_id] = int(row[0])
        return result

    def get_watermark(self, source: str) -> datetime | None:
        row = self._conn.execute(
            "SELECT MAX(source_ts) FROM raw_events WHERE source = ?", (source,)
        ).fetchone()
        return _opt_text_to_dt(row[0])

    # ── layer 2: clinical timeline ───────────────────────────────────────────

    def insert_glucose(self, events: list[GlucoseEvent]) -> int:
        rows = [(_dt_to_text(e.ts), e.mg_dl, e.trend, e.raw_event_id) for e in events]
        return self._write_counted(
            "INSERT OR IGNORE INTO glucose_events (ts, mg_dl, trend, raw_event_id) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )

    def insert_insulin(self, events: list[InsulinEvent]) -> int:
        rows = [
            (
                _dt_to_text(e.ts),
                e.kind.value,
                e.units,
                e.duration_min,
                None if e.automatic is None else int(e.automatic),
                e.raw_event_id,
            )
            for e in events
        ]
        return self._write_counted(
            "INSERT OR IGNORE INTO insulin_events "
            "(ts, kind, units, duration_min, automatic, raw_event_id) VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )

    def insert_meals(self, events: list[MealEvent]) -> int:
        rows = [
            (_dt_to_text(e.ts), e.carbs_g, e.protein_g, e.fat_g, e.note, e.raw_event_id)
            for e in events
        ]
        return self._write_counted(
            "INSERT OR IGNORE INTO meal_events "
            "(ts, carbs_g, protein_g, fat_g, note, raw_event_id) VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )

    def insert_activity(self, events: list[ActivityEvent]) -> int:
        rows = [
            (_dt_to_text(e.ts), e.kind, e.duration_min, e.intensity, e.strain, e.raw_event_id)
            for e in events
        ]
        return self._write_counted(
            "INSERT OR IGNORE INTO activity_events "
            "(ts, kind, duration_min, intensity, strain, raw_event_id) VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )

    def insert_sleep(self, events: list[SleepEvent]) -> int:
        rows = [
            (
                _dt_to_text(e.ts_start),
                _dt_to_text(e.ts_end),
                e.duration_min,
                e.score,
                None if e.stages is None else json.dumps(e.stages),
                e.raw_event_id,
            )
            for e in events
        ]
        return self._write_counted(
            "INSERT OR IGNORE INTO sleep_events "
            "(ts_start, ts_end, duration_min, score, stages, raw_event_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )

    def insert_recovery(self, events: list[RecoveryEvent]) -> int:
        rows = [
            (_dt_to_text(e.ts), e.score, e.hrv_ms, e.rhr_bpm, e.raw_event_id) for e in events
        ]
        return self._write_counted(
            "INSERT OR IGNORE INTO recovery_events "
            "(ts, score, hrv_ms, rhr_bpm, raw_event_id) VALUES (?, ?, ?, ?, ?)",
            rows,
        )

    def insert_device(self, events: list[DeviceEvent]) -> int:
        rows = [(_dt_to_text(e.ts), e.kind, e.note, e.raw_event_id) for e in events]
        return self._write_counted(
            "INSERT OR IGNORE INTO device_events (ts, kind, note, raw_event_id) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )

    def insert_predictions(self, events: list[PredictionEvent]) -> int:
        rows = [
            (
                _dt_to_text(e.ts),
                e.source,
                e.curve_kind,
                _prediction_horizon_min(e.values_mg_dl),
                json.dumps(e.values_mg_dl),
                e.raw_event_id,
            )
            for e in events
        ]
        return self._write_counted(
            "INSERT OR IGNORE INTO prediction_events "
            "(ts, source, curve_kind, horizon_min, values_mg_dl, raw_event_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )

    def get_glucose(self, start: datetime, end: datetime) -> list[GlucoseEvent]:
        rows = self._window(
            "SELECT ts, mg_dl, trend, raw_event_id FROM glucose_events", "ts", start, end
        )
        return [
            GlucoseEvent(ts=_text_to_dt(r[0]), mg_dl=r[1], trend=r[2], raw_event_id=r[3])
            for r in rows
        ]

    def get_insulin(self, start: datetime, end: datetime) -> list[InsulinEvent]:
        rows = self._window(
            "SELECT ts, kind, units, duration_min, automatic, raw_event_id FROM insulin_events",
            "ts",
            start,
            end,
        )
        return [
            InsulinEvent(
                ts=_text_to_dt(r[0]),
                kind=InsulinKind(r[1]),
                units=r[2],
                duration_min=r[3],
                automatic=_opt_bool(r[4]),
                raw_event_id=r[5],
            )
            for r in rows
        ]

    def get_meals(self, start: datetime, end: datetime) -> list[MealEvent]:
        rows = self._window(
            "SELECT ts, carbs_g, protein_g, fat_g, note, raw_event_id FROM meal_events",
            "ts",
            start,
            end,
        )
        return [
            MealEvent(
                ts=_text_to_dt(r[0]),
                carbs_g=r[1],
                protein_g=r[2],
                fat_g=r[3],
                note=r[4],
                raw_event_id=r[5],
            )
            for r in rows
        ]

    def get_activity(self, start: datetime, end: datetime) -> list[ActivityEvent]:
        rows = self._window(
            "SELECT ts, kind, duration_min, intensity, strain, raw_event_id FROM activity_events",
            "ts",
            start,
            end,
        )
        return [
            ActivityEvent(
                ts=_text_to_dt(r[0]),
                kind=r[1],
                duration_min=r[2],
                intensity=r[3],
                strain=r[4],
                raw_event_id=r[5],
            )
            for r in rows
        ]

    def get_sleep(self, start: datetime, end: datetime) -> list[SleepEvent]:
        rows = self._window(
            "SELECT ts_start, ts_end, duration_min, score, stages, raw_event_id "
            "FROM sleep_events",
            "ts_start",
            start,
            end,
        )
        return [
            SleepEvent(
                ts_start=_text_to_dt(r[0]),
                ts_end=_text_to_dt(r[1]),
                duration_min=r[2],
                score=r[3],
                stages=None if r[4] is None else json.loads(r[4]),
                raw_event_id=r[5],
            )
            for r in rows
        ]

    def get_recovery(self, start: datetime, end: datetime) -> list[RecoveryEvent]:
        rows = self._window(
            "SELECT ts, score, hrv_ms, rhr_bpm, raw_event_id FROM recovery_events",
            "ts",
            start,
            end,
        )
        return [
            RecoveryEvent(
                ts=_text_to_dt(r[0]),
                score=r[1],
                hrv_ms=r[2],
                rhr_bpm=r[3],
                raw_event_id=r[4],
            )
            for r in rows
        ]

    def get_predictions(self, start: datetime, end: datetime) -> list[PredictionEvent]:
        rows = self._window(
            "SELECT ts, source, curve_kind, values_mg_dl, raw_event_id FROM prediction_events",
            "ts",
            start,
            end,
        )
        return [
            PredictionEvent(
                ts=_text_to_dt(r[0]),
                source=r[1],
                curve_kind=r[2],
                values_mg_dl=json.loads(r[3]),
                raw_event_id=r[4],
            )
            for r in rows
        ]

    def coverage(self) -> CoverageStats:
        first_ts, last_ts = self._timeline_bounds()
        n_glucose = self._count("glucose_events")
        n_insulin = self._count("insulin_events")
        n_meals = self._count("meal_events")
        n_sleep = self._count("sleep_events")
        n_activity = self._count("activity_events")

        span_days = 0.0
        glucose_coverage_pct = 0.0
        days_with_insulin_pct = 0.0
        if first_ts is not None and last_ts is not None:
            span_seconds = (last_ts - first_ts).total_seconds()
            span_days = span_seconds / _SECONDS_PER_DAY
            if n_glucose:
                expected_slots = span_seconds / _CGM_SLOT_SECONDS + 1
                glucose_coverage_pct = min(100.0, 100.0 * n_glucose / expected_slots)
            if n_insulin:
                row = self._conn.execute(
                    "SELECT COUNT(DISTINCT substr(ts, 1, 10)) FROM insulin_events"
                ).fetchone()
                total_days = (last_ts.date() - first_ts.date()).days + 1
                days_with_insulin_pct = min(100.0, 100.0 * row[0] / total_days)

        return CoverageStats(
            first_ts=first_ts,
            last_ts=last_ts,
            span_days=span_days,
            n_glucose=n_glucose,
            glucose_coverage_pct=glucose_coverage_pct,
            n_insulin=n_insulin,
            days_with_insulin_pct=days_with_insulin_pct,
            n_meals=n_meals,
            n_sleep=n_sleep,
            n_activity=n_activity,
        )

    # ── layer 3: rollups ─────────────────────────────────────────────────────

    def upsert_rollups(self, rollups: list[Rollup]) -> int:
        rows = [
            (
                r.period.value,
                _dt_to_text(r.period_start),
                r.n,
                r.mean,
                r.sd,
                r.cv,
                r.tir,
                r.tar,
                r.tar2,
                r.tbr,
                r.tbr2,
                r.gmi,
                r.excursion_count,
                r.bolus_units,
                r.basal_units,
                r.carbs_g,
            )
            for r in rollups
        ]
        before = self._count("rollups")
        with self._conn:
            self._conn.executemany(
                "INSERT INTO rollups (period, period_start, n, mean, sd, cv, tir, tar, tar2, "
                "tbr, tbr2, gmi, excursion_count, bolus_units, basal_units, carbs_g) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(period, period_start) DO UPDATE SET "
                "n = excluded.n, mean = excluded.mean, sd = excluded.sd, cv = excluded.cv, "
                "tir = excluded.tir, tar = excluded.tar, tar2 = excluded.tar2, "
                "tbr = excluded.tbr, tbr2 = excluded.tbr2, gmi = excluded.gmi, "
                "excursion_count = excluded.excursion_count, "
                "bolus_units = excluded.bolus_units, basal_units = excluded.basal_units, "
                "carbs_g = excluded.carbs_g",
                rows,
            )
        return self._count("rollups") - before

    def get_rollups(self, period: RollupPeriod, start: datetime, end: datetime) -> list[Rollup]:
        cursor = self._conn.execute(
            "SELECT period, period_start, n, mean, sd, cv, tir, tar, tar2, tbr, tbr2, gmi, "
            "excursion_count, bolus_units, basal_units, carbs_g FROM rollups "
            "WHERE period = ? AND period_start >= ? AND period_start < ? "
            "ORDER BY period_start ASC",
            (period.value, _dt_to_text(start), _dt_to_text(end)),
        )
        return [
            Rollup(
                period=RollupPeriod(r[0]),
                period_start=_text_to_dt(r[1]),
                n=r[2],
                mean=r[3],
                sd=r[4],
                cv=r[5],
                tir=r[6],
                tar=r[7],
                tar2=r[8],
                tbr=r[9],
                tbr2=r[10],
                gmi=r[11],
                excursion_count=r[12],
                bolus_units=r[13],
                basal_units=r[14],
                carbs_g=r[15],
            )
            for r in cursor.fetchall()
        ]

    # ── layer 4: agent memory ────────────────────────────────────────────────

    def insert_finding(self, finding: Finding) -> int:
        """Persist a finding with a freshly assigned id (any incoming id is ignored)."""
        with self._conn:
            cursor = self._conn.execute(
                "INSERT INTO findings (agent, kind, scope, headline, body_md, evidence, stats, "
                "confidence, status, skeptic_notes, window_start, window_end, superseded_by) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    finding.agent,
                    finding.kind,
                    finding.scope,
                    finding.headline,
                    finding.body_md,
                    json.dumps(finding.evidence),
                    finding.stats.model_dump_json(),
                    finding.confidence,
                    finding.status.value,
                    finding.skeptic_notes,
                    _opt_dt_to_text(finding.window_start),
                    _opt_dt_to_text(finding.window_end),
                    finding.superseded_by,
                ),
            )
        assert cursor.lastrowid is not None
        return cursor.lastrowid

    def supersede_finding(self, old_id: int, new_id: int) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE findings SET status = ?, superseded_by = ? WHERE id = ?",
                (FindingStatus.SUPERSEDED.value, new_id, old_id),
            )

    def set_finding_status(self, finding_id: int, status: FindingStatus) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE findings SET status = ? WHERE id = ?", (status.value, finding_id)
            )

    def get_findings(
        self,
        *,
        agent: str | None = None,
        kind: str | None = None,
        status: FindingStatus | None = None,
        limit: int = 50,
    ) -> list[Finding]:
        clauses: list[str] = []
        params: list[Any] = []
        if agent is not None:
            clauses.append("agent = ?")
            params.append(agent)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        params.append(limit)
        cursor = self._conn.execute(
            "SELECT id, agent, kind, scope, headline, body_md, evidence, stats, confidence, "
            "status, skeptic_notes, window_start, window_end, superseded_by FROM findings "
            f"{where}ORDER BY id DESC LIMIT ?",
            params,
        )
        return [
            Finding(
                id=r[0],
                agent=r[1],
                kind=r[2],
                scope=r[3],
                headline=r[4],
                body_md=r[5],
                evidence=json.loads(r[6]),
                stats=FindingStats.model_validate_json(r[7]),
                confidence=r[8],
                status=FindingStatus(r[9]),
                skeptic_notes=r[10],
                window_start=_opt_text_to_dt(r[11]),
                window_end=_opt_text_to_dt(r[12]),
                superseded_by=r[13],
            )
            for r in cursor.fetchall()
        ]

    def insert_hypothesis(self, hypothesis: Hypothesis) -> int:
        """Persist a hypothesis with a freshly assigned id (any incoming id is ignored)."""
        with self._conn:
            cursor = self._conn.execute(
                "INSERT INTO hypotheses (statement, status, source_finding_id, tests) "
                "VALUES (?, ?, ?, ?)",
                (
                    hypothesis.statement,
                    hypothesis.status.value,
                    hypothesis.source_finding_id,
                    json.dumps(hypothesis.tests),
                ),
            )
        assert cursor.lastrowid is not None
        return cursor.lastrowid

    def get_hypotheses(self, *, status: str | None = None) -> list[Hypothesis]:
        if status is None:
            cursor = self._conn.execute(
                "SELECT id, statement, status, source_finding_id, tests FROM hypotheses "
                "ORDER BY id ASC"
            )
        else:
            cursor = self._conn.execute(
                "SELECT id, statement, status, source_finding_id, tests FROM hypotheses "
                "WHERE status = ? ORDER BY id ASC",
                (status,),
            )
        return [
            Hypothesis(
                id=r[0],
                statement=r[1],
                status=HypothesisStatus(r[2]),
                source_finding_id=r[3],
                tests=json.loads(r[4]),
            )
            for r in cursor.fetchall()
        ]

    # ── goals ────────────────────────────────────────────────────────────────

    def insert_goal(self, goal: Goal) -> int:
        with self._conn:
            cursor = self._conn.execute(
                "INSERT INTO goals "
                "(statement, metric, direction, target, tools, cadence_days, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    goal.statement,
                    goal.metric.value,
                    goal.direction,
                    goal.target,
                    json.dumps(goal.tools),
                    goal.cadence_days,
                    goal.status.value,
                    None if goal.created_at is None else _dt_to_text(goal.created_at),
                ),
            )
        assert cursor.lastrowid is not None
        return cursor.lastrowid

    def get_goals(self, *, status: GoalStatus | None = None) -> list[Goal]:
        sql = (
            "SELECT id, statement, metric, direction, target, tools, cadence_days, "
            "status, created_at FROM goals"
        )
        params: tuple[Any, ...] = ()
        if status is not None:
            sql += " WHERE status = ?"
            params = (status.value,)
        sql += " ORDER BY id ASC"
        return [_row_to_goal(r) for r in self._conn.execute(sql, params).fetchall()]

    def set_goal_status(self, goal_id: int, status: GoalStatus) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE goals SET status = ? WHERE id = ?", (status.value, goal_id)
            )

    def insert_goal_checkpoint(self, checkpoint: GoalCheckpoint) -> int:
        with self._conn:
            cursor = self._conn.execute(
                "INSERT INTO goal_checkpoints (goal_id, ts, metric_value, note) "
                "VALUES (?, ?, ?, ?)",
                (
                    checkpoint.goal_id,
                    _dt_to_text(checkpoint.ts),
                    checkpoint.metric_value,
                    checkpoint.note,
                ),
            )
        assert cursor.lastrowid is not None
        return cursor.lastrowid

    def get_goal_checkpoints(self, goal_id: int) -> list[GoalCheckpoint]:
        cursor = self._conn.execute(
            "SELECT id, goal_id, ts, metric_value, note FROM goal_checkpoints "
            "WHERE goal_id = ? ORDER BY ts ASC",
            (goal_id,),
        )
        return [
            GoalCheckpoint(
                id=r[0], goal_id=r[1], ts=_text_to_dt(r[2]), metric_value=r[3], note=r[4]
            )
            for r in cursor.fetchall()
        ]

    # ── internals ────────────────────────────────────────────────────────────

    def _write_counted(self, sql: str, rows: list[tuple[Any, ...]]) -> int:
        """Run a batched conflict-ignoring insert; return the number of new rows."""
        before = self._conn.total_changes
        with self._conn:
            self._conn.executemany(sql, rows)
        return self._conn.total_changes - before

    def _window(
        self, select: str, ts_column: str, start: datetime, end: datetime
    ) -> list[tuple[Any, ...]]:
        cursor = self._conn.execute(
            f"{select} WHERE {ts_column} >= ? AND {ts_column} < ? ORDER BY {ts_column} ASC",
            (_dt_to_text(start), _dt_to_text(end)),
        )
        rows: list[tuple[Any, ...]] = cursor.fetchall()
        return rows

    def _count(self, table: str) -> int:
        row = self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return int(row[0])

    def _timeline_bounds(self) -> tuple[datetime | None, datetime | None]:
        """Min/max timestamp across the whole clinical timeline (layer 2)."""
        columns = [
            ("glucose_events", "ts"),
            ("insulin_events", "ts"),
            ("meal_events", "ts"),
            ("activity_events", "ts"),
            ("sleep_events", "ts_start"),
            ("sleep_events", "ts_end"),
            ("recovery_events", "ts"),
            ("device_events", "ts"),
            ("prediction_events", "ts"),
        ]
        lows: list[str] = []
        highs: list[str] = []
        for table, column in columns:
            row = self._conn.execute(f"SELECT MIN({column}), MAX({column}) FROM {table}").fetchone()
            if row[0] is not None:
                lows.append(row[0])
                highs.append(row[1])
        if not lows:
            return None, None
        return _text_to_dt(min(lows)), _text_to_dt(max(highs))
