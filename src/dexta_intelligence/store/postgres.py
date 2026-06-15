"""PostgresStore — the reference backend for :class:`StoragePort`.

psycopg 3 (``psycopg[binary]``). This is the production-grade twin of
``SQLiteStore``: identical semantics, native column types. The only design
choices the protocol leaves open are resolved exactly as sqlite resolves them,
so the two backends are observationally indistinguishable through the port.

- **Timestamps** are stored as ``TIMESTAMPTZ``. Aware datetimes go in verbatim
  (psycopg adapts the tz); reads are normalized to aware UTC, matching sqlite.
- **JSON payloads** (``payload``/``evidence``/``stats``/``tests``/``tools``/
  ``values_mg_dl``/``stages``) are stored as ``JSONB`` and round-trip as native
  Python objects — no manual ``json.loads`` on the way out.
- **Ids** are ``BIGSERIAL``.
- **Window queries** are half-open (``start <= ts < end``), ordered ascending;
  sleep is windowed and ordered on ``ts_start``.
- **Dedupe keys** mirror sqlite exactly: ``ON CONFLICT DO NOTHING`` for events
  (returns the count of *new* rows), ``DO UPDATE`` for rollups (existing periods
  updated in place, not counted as new).
- **get_findings** returns newest first (highest id first); ``get_hypotheses``
  and ``get_goals`` return insertion order.
- Single connection, no pooling.
"""

from __future__ import annotations

from datetime import UTC, datetime
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

__all__ = ["PostgresStore"]

SCHEMA_VERSION = 2

_SECONDS_PER_DAY = 86400.0
_CGM_SLOT_SECONDS = 300.0  # expected 5-minute CGM cadence

_MISSING_DRIVER_MSG = (
    "PostgresStore requires psycopg 3. Install it with: "
    "pip install 'dexta-intelligence[postgres]'"
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS raw_events (
    id BIGSERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_ts TIMESTAMPTZ NOT NULL,
    payload JSONB NOT NULL,
    UNIQUE (source, source_id)
);
CREATE INDEX IF NOT EXISTS idx_raw_events_source_ts ON raw_events (source, source_ts);

CREATE TABLE IF NOT EXISTS glucose_events (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL UNIQUE,
    mg_dl INTEGER NOT NULL,
    trend TEXT,
    raw_event_id BIGINT
);

CREATE TABLE IF NOT EXISTS insulin_events (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL,
    kind TEXT NOT NULL,
    units DOUBLE PRECISION,
    duration_min DOUBLE PRECISION,
    automatic BOOLEAN,
    raw_event_id BIGINT,
    UNIQUE (ts, kind)
);

CREATE TABLE IF NOT EXISTS meal_events (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL UNIQUE,
    carbs_g DOUBLE PRECISION,
    protein_g DOUBLE PRECISION,
    fat_g DOUBLE PRECISION,
    note TEXT,
    raw_event_id BIGINT
);

CREATE TABLE IF NOT EXISTS activity_events (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL,
    kind TEXT NOT NULL,
    duration_min DOUBLE PRECISION,
    intensity DOUBLE PRECISION,
    strain DOUBLE PRECISION,
    raw_event_id BIGINT,
    UNIQUE (ts, kind)
);

CREATE TABLE IF NOT EXISTS sleep_events (
    id BIGSERIAL PRIMARY KEY,
    ts_start TIMESTAMPTZ NOT NULL UNIQUE,
    ts_end TIMESTAMPTZ NOT NULL,
    duration_min DOUBLE PRECISION NOT NULL,
    score DOUBLE PRECISION,
    stages JSONB,
    raw_event_id BIGINT
);

CREATE TABLE IF NOT EXISTS recovery_events (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL UNIQUE,
    score DOUBLE PRECISION,
    hrv_ms DOUBLE PRECISION,
    rhr_bpm DOUBLE PRECISION,
    raw_event_id BIGINT
);

CREATE TABLE IF NOT EXISTS device_events (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL,
    kind TEXT NOT NULL,
    note TEXT,
    raw_event_id BIGINT,
    UNIQUE (ts, kind)
);

CREATE TABLE IF NOT EXISTS prediction_events (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL,
    source TEXT NOT NULL,
    curve_kind TEXT NOT NULL,
    horizon_min INTEGER NOT NULL,
    values_mg_dl JSONB NOT NULL,
    raw_event_id BIGINT,
    UNIQUE (ts, curve_kind)
);
CREATE INDEX IF NOT EXISTS idx_prediction_events_ts ON prediction_events (ts);

CREATE TABLE IF NOT EXISTS rollups (
    period TEXT NOT NULL,
    period_start TIMESTAMPTZ NOT NULL,
    n INTEGER NOT NULL,
    mean DOUBLE PRECISION,
    sd DOUBLE PRECISION,
    cv DOUBLE PRECISION,
    tir DOUBLE PRECISION,
    tar DOUBLE PRECISION,
    tar2 DOUBLE PRECISION,
    tbr DOUBLE PRECISION,
    tbr2 DOUBLE PRECISION,
    gmi DOUBLE PRECISION,
    excursion_count INTEGER,
    bolus_units DOUBLE PRECISION,
    basal_units DOUBLE PRECISION,
    carbs_g DOUBLE PRECISION,
    PRIMARY KEY (period, period_start)
);

CREATE TABLE IF NOT EXISTS findings (
    id BIGSERIAL PRIMARY KEY,
    agent TEXT NOT NULL,
    kind TEXT NOT NULL,
    scope TEXT NOT NULL,
    headline TEXT NOT NULL,
    body_md TEXT NOT NULL,
    evidence JSONB NOT NULL,
    stats JSONB NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    status TEXT NOT NULL,
    skeptic_notes TEXT,
    window_start TIMESTAMPTZ,
    window_end TIMESTAMPTZ,
    superseded_by BIGINT
);
CREATE INDEX IF NOT EXISTS idx_findings_agent_kind_status ON findings (agent, kind, status);

CREATE TABLE IF NOT EXISTS hypotheses (
    id BIGSERIAL PRIMARY KEY,
    statement TEXT NOT NULL,
    status TEXT NOT NULL,
    source_finding_id BIGINT,
    tests JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS goals (
    id BIGSERIAL PRIMARY KEY,
    statement TEXT NOT NULL,
    metric TEXT NOT NULL,
    direction TEXT NOT NULL,
    target DOUBLE PRECISION,
    tools JSONB NOT NULL,
    cadence_days INTEGER NOT NULL,
    status TEXT NOT NULL,
    created_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS goal_checkpoints (
    id BIGSERIAL PRIMARY KEY,
    goal_id BIGINT NOT NULL REFERENCES goals (id),
    ts TIMESTAMPTZ NOT NULL,
    metric_value DOUBLE PRECISION,
    note TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_goal_checkpoints_goal ON goal_checkpoints (goal_id, ts);
"""


def _prediction_horizon_min(values: list[float]) -> int:
    """Minutes from cycle time to the last predicted point (5-minute spacing)."""
    if not values:
        return 0
    return max(0, (len(values) - 1) * 5)


def _to_utc(value: datetime) -> datetime:
    """Aware datetimes are normalized to UTC; naive ones are assumed UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _opt_utc(value: datetime | None) -> datetime | None:
    return None if value is None else _to_utc(value)


def _row_to_goal(r: tuple[Any, ...]) -> Goal:
    return Goal(
        id=r[0],
        statement=r[1],
        metric=GoalMetric(r[2]),
        direction=r[3],
        target=r[4],
        tools=r[5],
        cadence_days=r[6],
        status=GoalStatus(r[7]),
        created_at=_opt_utc(r[8]),
    )


class PostgresStore:
    """:class:`StoragePort` implementation over a single psycopg 3 connection."""

    def __init__(self, dsn: str) -> None:
        try:
            import psycopg  # noqa: PLC0415 - lazy: module must import without the driver
            from psycopg.types.json import Jsonb  # noqa: PLC0415
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised when driver absent
            raise RuntimeError(_MISSING_DRIVER_MSG) from exc

        self._jsonb = Jsonb
        self._conn = psycopg.connect(dsn)

    def close(self) -> None:
        self._conn.close()

    # ── lifecycle ────────────────────────────────────────────────────────────

    def migrate(self) -> None:
        """Create or upgrade the schema. Idempotent (IF NOT EXISTS throughout)."""
        with self._conn, self._conn.cursor() as cur:
            cur.execute(_SCHEMA)
            cur.execute("SELECT version FROM schema_version")
            row = cur.fetchone()
            if row is None:
                cur.execute(
                    "INSERT INTO schema_version (version) VALUES (%s)", (SCHEMA_VERSION,)
                )
            elif row[0] < SCHEMA_VERSION:
                cur.execute("UPDATE schema_version SET version = %s", (SCHEMA_VERSION,))

    # ── layer 1: raw events ──────────────────────────────────────────────────

    def upsert_raw_events(self, events: list[RawEvent]) -> dict[str, int]:
        if not events:
            return {}
        rows = [
            (e.source, e.source_id, _to_utc(e.source_ts), self._jsonb(e.payload))
            for e in events
        ]
        with self._conn, self._conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO raw_events (source, source_id, source_ts, payload) "
                "VALUES (%s, %s, %s, %s) ON CONFLICT (source, source_id) DO NOTHING",
                rows,
            )
        return self._raw_ids({(e.source, e.source_id) for e in events})

    def existing_raw_ids(self, events: list[RawEvent]) -> dict[str, int]:
        if not events:
            return {}
        return self._raw_ids({(e.source, e.source_id) for e in events})

    def _raw_ids(self, keys: set[tuple[str, str]]) -> dict[str, int]:
        """Resolve ``source_id -> id`` for the given ``(source, source_id)`` keys.

        ``ON CONFLICT DO NOTHING ... RETURNING`` skips the conflicting rows, so
        the ids are read back here — covering both freshly-inserted and
        pre-existing rows. ``source_id`` is unique within a source.
        """
        result: dict[str, int] = {}
        with self._conn.cursor() as cur:
            for source, source_id in keys:
                cur.execute(
                    "SELECT id FROM raw_events WHERE source = %s AND source_id = %s",
                    (source, source_id),
                )
                row = cur.fetchone()
                if row is not None:
                    result[source_id] = int(row[0])
        return result

    def get_watermark(self, source: str) -> datetime | None:
        with self._conn.cursor() as cur:
            cur.execute("SELECT MAX(source_ts) FROM raw_events WHERE source = %s", (source,))
            row = cur.fetchone()
        assert row is not None
        return _opt_utc(row[0])

    # ── layer 2: clinical timeline ───────────────────────────────────────────

    def insert_glucose(self, events: list[GlucoseEvent]) -> int:
        rows = [(_to_utc(e.ts), e.mg_dl, e.trend, e.raw_event_id) for e in events]
        return self._write_counted(
            "INSERT INTO glucose_events (ts, mg_dl, trend, raw_event_id) "
            "VALUES (%s, %s, %s, %s) ON CONFLICT (ts) DO NOTHING",
            rows,
        )

    def insert_insulin(self, events: list[InsulinEvent]) -> int:
        rows = [
            (
                _to_utc(e.ts),
                e.kind.value,
                e.units,
                e.duration_min,
                e.automatic,
                e.raw_event_id,
            )
            for e in events
        ]
        return self._write_counted(
            "INSERT INTO insulin_events "
            "(ts, kind, units, duration_min, automatic, raw_event_id) "
            "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (ts, kind) DO NOTHING",
            rows,
        )

    def insert_meals(self, events: list[MealEvent]) -> int:
        rows = [
            (_to_utc(e.ts), e.carbs_g, e.protein_g, e.fat_g, e.note, e.raw_event_id)
            for e in events
        ]
        return self._write_counted(
            "INSERT INTO meal_events "
            "(ts, carbs_g, protein_g, fat_g, note, raw_event_id) "
            "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (ts) DO NOTHING",
            rows,
        )

    def insert_activity(self, events: list[ActivityEvent]) -> int:
        rows = [
            (_to_utc(e.ts), e.kind, e.duration_min, e.intensity, e.strain, e.raw_event_id)
            for e in events
        ]
        return self._write_counted(
            "INSERT INTO activity_events "
            "(ts, kind, duration_min, intensity, strain, raw_event_id) "
            "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (ts, kind) DO NOTHING",
            rows,
        )

    def insert_sleep(self, events: list[SleepEvent]) -> int:
        rows = [
            (
                _to_utc(e.ts_start),
                _to_utc(e.ts_end),
                e.duration_min,
                e.score,
                None if e.stages is None else self._jsonb(e.stages),
                e.raw_event_id,
            )
            for e in events
        ]
        return self._write_counted(
            "INSERT INTO sleep_events "
            "(ts_start, ts_end, duration_min, score, stages, raw_event_id) "
            "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (ts_start) DO NOTHING",
            rows,
        )

    def insert_recovery(self, events: list[RecoveryEvent]) -> int:
        rows = [(_to_utc(e.ts), e.score, e.hrv_ms, e.rhr_bpm, e.raw_event_id) for e in events]
        return self._write_counted(
            "INSERT INTO recovery_events "
            "(ts, score, hrv_ms, rhr_bpm, raw_event_id) "
            "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (ts) DO NOTHING",
            rows,
        )

    def insert_device(self, events: list[DeviceEvent]) -> int:
        rows = [(_to_utc(e.ts), e.kind, e.note, e.raw_event_id) for e in events]
        return self._write_counted(
            "INSERT INTO device_events (ts, kind, note, raw_event_id) "
            "VALUES (%s, %s, %s, %s) ON CONFLICT (ts, kind) DO NOTHING",
            rows,
        )

    def insert_predictions(self, events: list[PredictionEvent]) -> int:
        rows = [
            (
                _to_utc(e.ts),
                e.source,
                e.curve_kind,
                _prediction_horizon_min(e.values_mg_dl),
                self._jsonb(e.values_mg_dl),
                e.raw_event_id,
            )
            for e in events
        ]
        return self._write_counted(
            "INSERT INTO prediction_events "
            "(ts, source, curve_kind, horizon_min, values_mg_dl, raw_event_id) "
            "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (ts, curve_kind) DO NOTHING",
            rows,
        )

    def get_glucose(self, start: datetime, end: datetime) -> list[GlucoseEvent]:
        rows = self._window(
            "SELECT ts, mg_dl, trend, raw_event_id FROM glucose_events", "ts", start, end
        )
        return [
            GlucoseEvent(ts=_to_utc(r[0]), mg_dl=r[1], trend=r[2], raw_event_id=r[3])
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
                ts=_to_utc(r[0]),
                kind=InsulinKind(r[1]),
                units=r[2],
                duration_min=r[3],
                automatic=r[4],
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
                ts=_to_utc(r[0]),
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
                ts=_to_utc(r[0]),
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
                ts_start=_to_utc(r[0]),
                ts_end=_to_utc(r[1]),
                duration_min=r[2],
                score=r[3],
                stages=r[4],
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
                ts=_to_utc(r[0]),
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
                ts=_to_utc(r[0]),
                source=r[1],
                curve_kind=r[2],
                values_mg_dl=r[3],
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
                with self._conn.cursor() as cur:
                    cur.execute(
                        "SELECT COUNT(DISTINCT (ts AT TIME ZONE 'UTC')::date) "
                        "FROM insulin_events"
                    )
                    row = cur.fetchone()
                assert row is not None
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
                _to_utc(r.period_start),
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
        with self._conn, self._conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO rollups (period, period_start, n, mean, sd, cv, tir, tar, tar2, "
                "tbr, tbr2, gmi, excursion_count, bolus_units, basal_units, carbs_g) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (period, period_start) DO UPDATE SET "
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
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT period, period_start, n, mean, sd, cv, tir, tar, tar2, tbr, tbr2, gmi, "
                "excursion_count, bolus_units, basal_units, carbs_g FROM rollups "
                "WHERE period = %s AND period_start >= %s AND period_start < %s "
                "ORDER BY period_start ASC",
                (period.value, _to_utc(start), _to_utc(end)),
            )
            fetched = cur.fetchall()
        return [
            Rollup(
                period=RollupPeriod(r[0]),
                period_start=_to_utc(r[1]),
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
            for r in fetched
        ]

    # ── layer 4: agent memory ────────────────────────────────────────────────

    def insert_finding(self, finding: Finding) -> int:
        """Persist a finding with a freshly assigned id (any incoming id is ignored)."""
        with self._conn, self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO findings (agent, kind, scope, headline, body_md, evidence, stats, "
                "confidence, status, skeptic_notes, window_start, window_end, superseded_by) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
                (
                    finding.agent,
                    finding.kind,
                    finding.scope,
                    finding.headline,
                    finding.body_md,
                    self._jsonb(finding.evidence),
                    self._jsonb(finding.stats.model_dump(mode="json")),
                    finding.confidence,
                    finding.status.value,
                    finding.skeptic_notes,
                    _opt_utc(finding.window_start),
                    _opt_utc(finding.window_end),
                    finding.superseded_by,
                ),
            )
            row = cur.fetchone()
        assert row is not None
        return int(row[0])

    def supersede_finding(self, old_id: int, new_id: int) -> None:
        with self._conn, self._conn.cursor() as cur:
            cur.execute(
                "UPDATE findings SET status = %s, superseded_by = %s WHERE id = %s",
                (FindingStatus.SUPERSEDED.value, new_id, old_id),
            )

    def set_finding_status(self, finding_id: int, status: FindingStatus) -> None:
        with self._conn, self._conn.cursor() as cur:
            cur.execute(
                "UPDATE findings SET status = %s WHERE id = %s", (status.value, finding_id)
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
            clauses.append("agent = %s")
            params.append(agent)
        if kind is not None:
            clauses.append("kind = %s")
            params.append(kind)
        if status is not None:
            clauses.append("status = %s")
            params.append(status.value)
        where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        params.append(limit)
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT id, agent, kind, scope, headline, body_md, evidence, stats, confidence, "
                "status, skeptic_notes, window_start, window_end, superseded_by FROM findings "
                f"{where}ORDER BY id DESC LIMIT %s",
                params,
            )
            fetched = cur.fetchall()
        return [
            Finding(
                id=r[0],
                agent=r[1],
                kind=r[2],
                scope=r[3],
                headline=r[4],
                body_md=r[5],
                evidence=r[6],
                stats=FindingStats.model_validate(r[7]),
                confidence=r[8],
                status=FindingStatus(r[9]),
                skeptic_notes=r[10],
                window_start=_opt_utc(r[11]),
                window_end=_opt_utc(r[12]),
                superseded_by=r[13],
            )
            for r in fetched
        ]

    def insert_hypothesis(self, hypothesis: Hypothesis) -> int:
        """Persist a hypothesis with a freshly assigned id (any incoming id is ignored)."""
        with self._conn, self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO hypotheses (statement, status, source_finding_id, tests) "
                "VALUES (%s, %s, %s, %s) RETURNING id",
                (
                    hypothesis.statement,
                    hypothesis.status.value,
                    hypothesis.source_finding_id,
                    self._jsonb(hypothesis.tests),
                ),
            )
            row = cur.fetchone()
        assert row is not None
        return int(row[0])

    def get_hypotheses(self, *, status: str | None = None) -> list[Hypothesis]:
        with self._conn.cursor() as cur:
            if status is None:
                cur.execute(
                    "SELECT id, statement, status, source_finding_id, tests FROM hypotheses "
                    "ORDER BY id ASC"
                )
            else:
                cur.execute(
                    "SELECT id, statement, status, source_finding_id, tests FROM hypotheses "
                    "WHERE status = %s ORDER BY id ASC",
                    (status,),
                )
            fetched = cur.fetchall()
        return [
            Hypothesis(
                id=r[0],
                statement=r[1],
                status=HypothesisStatus(r[2]),
                source_finding_id=r[3],
                tests=r[4],
            )
            for r in fetched
        ]

    # ── goals ────────────────────────────────────────────────────────────────

    def insert_goal(self, goal: Goal) -> int:
        with self._conn, self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO goals "
                "(statement, metric, direction, target, tools, cadence_days, status, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
                (
                    goal.statement,
                    goal.metric.value,
                    goal.direction,
                    goal.target,
                    self._jsonb(goal.tools),
                    goal.cadence_days,
                    goal.status.value,
                    _opt_utc(goal.created_at),
                ),
            )
            row = cur.fetchone()
        assert row is not None
        return int(row[0])

    def get_goals(self, *, status: GoalStatus | None = None) -> list[Goal]:
        sql = (
            "SELECT id, statement, metric, direction, target, tools, cadence_days, "
            "status, created_at FROM goals"
        )
        params: tuple[Any, ...] = ()
        if status is not None:
            sql += " WHERE status = %s"
            params = (status.value,)
        sql += " ORDER BY id ASC"
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            fetched = cur.fetchall()
        return [_row_to_goal(r) for r in fetched]

    def set_goal_status(self, goal_id: int, status: GoalStatus) -> None:
        with self._conn, self._conn.cursor() as cur:
            cur.execute("UPDATE goals SET status = %s WHERE id = %s", (status.value, goal_id))

    def insert_goal_checkpoint(self, checkpoint: GoalCheckpoint) -> int:
        with self._conn, self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO goal_checkpoints (goal_id, ts, metric_value, note) "
                "VALUES (%s, %s, %s, %s) RETURNING id",
                (
                    checkpoint.goal_id,
                    _to_utc(checkpoint.ts),
                    checkpoint.metric_value,
                    checkpoint.note,
                ),
            )
            row = cur.fetchone()
        assert row is not None
        return int(row[0])

    def get_goal_checkpoints(self, goal_id: int) -> list[GoalCheckpoint]:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT id, goal_id, ts, metric_value, note FROM goal_checkpoints "
                "WHERE goal_id = %s ORDER BY ts ASC",
                (goal_id,),
            )
            fetched = cur.fetchall()
        return [
            GoalCheckpoint(
                id=r[0], goal_id=r[1], ts=_to_utc(r[2]), metric_value=r[3], note=r[4]
            )
            for r in fetched
        ]

    # ── internals ────────────────────────────────────────────────────────────

    def _write_counted(self, sql: str, rows: list[tuple[Any, ...]]) -> int:
        """Run a batched conflict-ignoring insert; return the number of new rows."""
        if not rows:
            return 0
        with self._conn, self._conn.cursor() as cur:
            cur.executemany(sql, rows)
            inserted = cur.rowcount
        return int(inserted)

    def _window(
        self, select: str, ts_column: str, start: datetime, end: datetime
    ) -> list[tuple[Any, ...]]:
        with self._conn.cursor() as cur:
            cur.execute(
                f"{select} WHERE {ts_column} >= %s AND {ts_column} < %s "
                f"ORDER BY {ts_column} ASC",
                (_to_utc(start), _to_utc(end)),
            )
            rows: list[tuple[Any, ...]] = cur.fetchall()
        return rows

    def _count(self, table: str) -> int:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            row = cur.fetchone()
        assert row is not None
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
        lows: list[datetime] = []
        highs: list[datetime] = []
        with self._conn.cursor() as cur:
            for table, column in columns:
                cur.execute(f"SELECT MIN({column}), MAX({column}) FROM {table}")
                row = cur.fetchone()
                assert row is not None
                if row[0] is not None:
                    lows.append(row[0])
                    highs.append(row[1])
        if not lows:
            return None, None
        return _to_utc(min(lows)), _to_utc(max(highs))
