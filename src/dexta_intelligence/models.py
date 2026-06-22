"""Core typed records for the clinical timeline and agent memory.

These models are the lingua franca of the whole system:

- **Connectors** normalize provider payloads into timeline events.
- **Analytics** consume timeline events and produce numbers.
- **Agents** consume analytics + memory and produce :class:`Finding` records.
- **The guard** audits prose against the numbers in ``Finding.evidence``.

Design rules
------------
1. Every timeline event carries ``raw_event_id`` provenance back to the
   immutable raw store. No event exists without a source.
2. Models are frozen. Mutation is a store operation, not an attribute write.
3. All timestamps are timezone-aware UTC. Naive datetimes are rejected at
   validation time - silent local-time bugs are endemic in CGM data and we
   refuse to inherit them.
"""

from __future__ import annotations

import enum
from datetime import UTC, date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "ActivityEvent",
    "ChatSession",
    "ChatTurn",
    "ContextRequest",
    "CoverageStats",
    "DeviceEvent",
    "Finding",
    "FindingStats",
    "FindingStatus",
    "GlucoseEvent",
    "Goal",
    "GoalCheckpoint",
    "GoalMetric",
    "GoalStatus",
    "Hypothesis",
    "HypothesisStatus",
    "InsulinEvent",
    "InsulinKind",
    "InvestigationRun",
    "ManualEvent",
    "MealEvent",
    "OpenInvestigation",
    "PredictionEvent",
    "RawEvent",
    "RecoveryEvent",
    "Rollup",
    "RollupPeriod",
    "RunFinding",
    "SleepEvent",
    "TherapyProfile",
]


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        msg = "naive datetime rejected: all timestamps must be timezone-aware (UTC)"
        raise ValueError(msg)
    return value.astimezone(UTC)


def _require_utc_opt(value: datetime | None) -> datetime | None:
    return None if value is None else _require_utc(value)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 - raw store
# ─────────────────────────────────────────────────────────────────────────────


class RawEvent(_FrozenModel):
    """Immutable, verbatim provider record.

    ``(source, source_id)`` is the idempotency key: re-ingesting the same
    provider record is a no-op. The payload is never interpreted here - only
    stored, so normalization can always be replayed.
    """

    source: str
    source_id: str
    source_ts: datetime
    payload: dict[str, Any]

    _utc = field_validator("source_ts")(_require_utc)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 - clinical timeline
# ─────────────────────────────────────────────────────────────────────────────


class GlucoseEvent(_FrozenModel):
    ts: datetime
    mg_dl: int = Field(ge=10, le=600)
    trend: str | None = None
    raw_event_id: int | None = None

    _utc = field_validator("ts")(_require_utc)


class InsulinKind(enum.StrEnum):
    BOLUS = "bolus"
    BASAL = "basal"
    TEMP_BASAL = "temp_basal"
    SUSPEND = "suspend"


class InsulinEvent(_FrozenModel):
    ts: datetime
    kind: InsulinKind
    units: float | None = Field(default=None, ge=0)
    duration_min: float | None = Field(default=None, ge=0)
    automatic: bool | None = None
    """True for algorithm-issued delivery (SMB / closed-loop temp basal)."""
    raw_event_id: int | None = None

    _utc = field_validator("ts")(_require_utc)


class MealEvent(_FrozenModel):
    ts: datetime
    carbs_g: float | None = Field(default=None, ge=0)
    protein_g: float | None = Field(default=None, ge=0)
    fat_g: float | None = Field(default=None, ge=0)
    note: str | None = None
    raw_event_id: int | None = None

    _utc = field_validator("ts")(_require_utc)


class ManualEvent(_FrozenModel):
    """User-reported context attached to the timeline.

    Unlike connector-derived events, this is logged by the user to record
    real-world context (meals, stress, site changes, illness, travel, free-form
    notes). It can optionally link back to an :class:`InvestigationRun` or a
    specific glucose event so the agent can correlate a report with what it saw.
    """

    event_type: str
    """One of: meal, exercise, sleep, illness, stress, alcohol, site_change,
    sensor_issue, pump_issue, medication, travel, note."""
    event_ts: datetime
    end_ts: datetime | None = None
    title: str | None = None
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    intensity: str | None = None
    confidence: str = "user_reported"
    source: str = "manual"
    linked_run_id: str | None = None
    linked_glucose_event_id: int | None = None
    created_at: datetime
    id: int | None = None

    _utc_event = field_validator("event_ts")(_require_utc)
    _utc_created = field_validator("created_at")(_require_utc)
    _utc_end = field_validator("end_ts")(_require_utc_opt)


class ContextRequest(_FrozenModel):
    """A question dexta asks the user to log missing context for an event.

    Active context acquisition: when a deterministic gap is found (an unexplained
    spike with no logged meal or note nearby), dexta asks the user to record what
    happened. It never fabricates the missing value; it requests it. ``question``
    is observation-only (it passes the dosing-advice gate). ``evidence`` holds the
    deterministic facts behind the request (e.g. ``peak_mg_dl``).
    """

    kind: str
    event_ts: datetime
    question: str
    suggested_event_type: str
    evidence: dict[str, Any] = Field(default_factory=dict)

    _utc_event = field_validator("event_ts")(_require_utc)


class TherapyProfile(_FrozenModel):
    """A versioned snapshot of the user's insulin/therapy settings.

    Devices report only the CURRENT profile, so dexta records a new version
    whenever the content changes (``content_hash`` differs from the latest).
    ``active_from`` is when this version was first seen; ``active_to`` is when
    the next version superseded it (None for the live one). This is what lets an
    investigation of a March event read the March profile, not today's.

    ``content`` is the formatted profile payload (active profile name, segments,
    DIA, etc.) as produced by the connector. Read-only clinical context, never
    dosing advice.
    """

    source: str
    name: str
    content: dict[str, Any]
    content_hash: str
    active_from: datetime
    active_to: datetime | None = None
    created_at: datetime
    id: int | None = None

    _utc_from = field_validator("active_from", "created_at")(_require_utc)
    _utc_to = field_validator("active_to")(_require_utc_opt)


class ActivityEvent(_FrozenModel):
    ts: datetime
    kind: str
    duration_min: float | None = Field(default=None, ge=0)
    intensity: float | None = None
    strain: float | None = None
    raw_event_id: int | None = None

    _utc = field_validator("ts")(_require_utc)


class SleepEvent(_FrozenModel):
    ts_start: datetime
    ts_end: datetime
    duration_min: float = Field(ge=0)
    score: float | None = None
    stages: dict[str, float] | None = None
    raw_event_id: int | None = None

    _utc_start = field_validator("ts_start")(_require_utc)
    _utc_end = field_validator("ts_end")(_require_utc)


class RecoveryEvent(_FrozenModel):
    ts: datetime
    score: float | None = None
    hrv_ms: float | None = None
    rhr_bpm: float | None = None
    raw_event_id: int | None = None

    _utc = field_validator("ts")(_require_utc)


class DeviceEvent(_FrozenModel):
    ts: datetime
    kind: str
    note: str | None = None
    raw_event_id: int | None = None

    _utc = field_validator("ts")(_require_utc)


class PredictionEvent(_FrozenModel):
    """One algorithm-logged glucose forecast curve for a single dosing cycle.

    Looping algorithms publish their own forecast every cycle (oref0/AAPS via
    ``openaps.suggested.predBGs``, Loop via ``loop.predicted`` in Nightscout
    ``devicestatus``). These curves are the logged ground truth of the
    algorithm's belief, which the Prediction Reconciliation agent compares
    against realized CGM.
    """

    ts: datetime
    """Algorithm cycle time - the timestamp of ``values_mg_dl[0]``."""
    source: str
    """Forecasting algorithm, e.g. ``"openaps"`` or ``"loop"``."""
    curve_kind: Literal["iob", "cob", "uam", "zt", "loop"]
    """oref0 scenario curves (insulin-only / carbs-as-announced / unannounced
    meal / zero-temp) or Loop's single blended forecast."""
    values_mg_dl: list[float]
    """Predicted mg/dL at 5-minute spacing starting at ``ts``."""
    raw_event_id: int | None = None

    _utc = field_validator("ts")(_require_utc)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 - rollups
# ─────────────────────────────────────────────────────────────────────────────


class RollupPeriod(enum.StrEnum):
    FIFTEEN_MIN = "15m"
    HOURLY = "1h"
    DAILY = "1d"
    WEEKLY = "1w"


class Rollup(_FrozenModel):
    """Pre-aggregated metrics for one period. A cache over the timeline, never truth."""

    period: RollupPeriod
    period_start: datetime
    n: int = Field(ge=0)
    mean: float | None = None
    sd: float | None = None
    cv: float | None = None
    tir: float | None = None
    tar: float | None = None
    tar2: float | None = None
    tbr: float | None = None
    tbr2: float | None = None
    gmi: float | None = None
    excursion_count: int | None = None
    bolus_units: float | None = None
    basal_units: float | None = None
    carbs_g: float | None = None

    _utc = field_validator("period_start")(_require_utc)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 4 - agent memory
# ─────────────────────────────────────────────────────────────────────────────


class FindingStatus(enum.StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"
    DISMISSED = "dismissed"
    STALE = "stale"


class FindingStats(_FrozenModel):
    """Statistical backing for a quantitative claim.

    Mandatory for any finding that asserts an effect. The skeptic and the
    rigor layer fill this in; the guard refuses prose that outruns it.
    """

    effect_size: float | None = None
    n: int | None = None
    p_perm: float | None = None
    """Permutation p-value (distribution-free)."""
    q_fdr: float | None = None
    """Benjamini-Hochberg adjusted q across the analysis run."""
    replicated: bool | None = None
    """Did the effect hold on a temporally disjoint split?"""


class Finding(_FrozenModel):
    """A single durable unit of agent knowledge.

    ``evidence`` holds every number the prose is allowed to cite.
    """

    agent: str
    kind: str
    scope: str
    headline: str
    body_md: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)
    stats: FindingStats = Field(default_factory=FindingStats)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    status: FindingStatus = FindingStatus.ACTIVE
    skeptic_notes: str | None = None
    window_start: datetime | None = None
    window_end: datetime | None = None
    id: int | None = None
    superseded_by: int | None = None
    #: When this finding was last re-derived. Drives freshness: a finding not
    #: re-confirmed within its TTL is retired to STALE. ``seen_count`` is how many
    #: analyses have produced it (recurrence), which lengthens the TTL.
    last_verified: datetime | None = None
    seen_count: int = 1


class HypothesisStatus(enum.StrEnum):
    OPEN = "open"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    STALE = "stale"


class Hypothesis(_FrozenModel):
    """A candidate pattern that has not cleared the rigor bar (yet)."""

    statement: str
    status: HypothesisStatus = HypothesisStatus.OPEN
    source_finding_id: int | None = None
    tests: list[dict[str, Any]] = Field(default_factory=list)
    id: int | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Goals - user-stated objectives pursued by background agents
# ─────────────────────────────────────────────────────────────────────────────


class GoalStatus(enum.StrEnum):
    ACTIVE = "active"
    ACHIEVED = "achieved"
    PAUSED = "paused"
    ABANDONED = "abandoned"


class GoalMetric(enum.StrEnum):
    """Deterministic success metrics. A goal's progress is never LLM-judged."""

    TIR = "tir"
    NOCTURNAL_TBR = "nocturnal_tbr"
    TBR = "tbr"
    MEAN_GLUCOSE = "mean_glucose"
    CV = "cv"


class Goal(_FrozenModel):
    """A user objective the model composes into a background investigation.

    ``metric`` + ``direction`` define success deterministically (e.g. metric
    ``nocturnal_tbr``, direction ``decrease``). ``tools`` is the model-composed
    plan: each entry is a ``{"tool", "args"}`` call the tick runs to keep the
    goal's evidence fresh. Treatment changes are never a goal output.
    """

    statement: str
    metric: GoalMetric
    direction: Literal["increase", "decrease"]
    target: float | None = None
    tools: list[dict[str, Any]] = Field(default_factory=list)
    cadence_days: int = 7
    status: GoalStatus = GoalStatus.ACTIVE
    created_at: datetime | None = None
    id: int | None = None


class GoalCheckpoint(_FrozenModel):
    """One background tick: the measured metric and a progress note."""

    goal_id: int
    ts: datetime
    metric_value: float | None
    note: str
    id: int | None = None

    _utc = field_validator("ts")(_require_utc)


# ─────────────────────────────────────────────────────────────────────────────
# Chat - durable GUI conversation history
# ─────────────────────────────────────────────────────────────────────────────


class ChatTurn(_FrozenModel):
    """One persisted message in a GUI chat session.

    Durable so a conversation survives a server restart; ``session_id`` scopes a
    conversation, ``role`` is the speaker (e.g. ``"user"`` / ``"assistant"``).
    """

    session_id: str
    role: str
    content: str
    ts: datetime
    id: int | None = None

    _utc = field_validator("ts")(_require_utc)


class ChatSession(_FrozenModel):
    """A summary of one chat conversation - for enumerating past threads.

    ``last_ts`` is the most recent turn's timestamp, ``turn_count`` the number of
    messages, ``preview`` the first user message (a label for the conversation).
    """

    session_id: str
    last_ts: datetime
    turn_count: int
    preview: str

    _utc = field_validator("last_ts")(_require_utc)


# ─────────────────────────────────────────────────────────────────────────────
# Investigation runs (the observable record of one investigation)
# ─────────────────────────────────────────────────────────────────────────────


class RunFinding(_FrozenModel):
    """A snapshot of one finding as it stood when an investigation produced it.

    An investigation run is an immutable historical record, so it stores what it
    found at the time rather than a live link that drifts as findings are
    superseded.
    """

    headline: str
    kind: str
    confidence: float
    status: str


class InvestigationRun(_FrozenModel):
    """A persisted record of one coordinator investigation.

    Captures the observable process behind a set of findings: which producers
    were planned, the step-by-step trace of what ran, the findings produced
    (snapshotted in ``findings``), and the window inspected. This is what turns
    isolated answers into an auditable investigation history.
    """

    run_id: str
    kind: str
    status: str
    question: str | None
    window_start: date
    window_end: date
    plan: list[str]
    trace: list[str]
    findings: list[RunFinding]
    n_findings: int
    started_at: datetime
    finished_at: datetime
    coverage_summary: dict[str, Any] | None = None
    """Data-sufficiency snapshot at run time (glucose coverage, span, counts,
    ``limited`` flag). Drives coverage-aware gating. None on legacy rows."""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    """Instrument log: the tools the run called (name, scope, ok) for the
    orchestrator drill, or one entry per producer for a deep-analysis run."""
    evidence_items: list[dict[str, Any]] = Field(default_factory=list)
    """The guard-audited numbers behind each finding (evidence-drawer source)."""
    answer: str | None = None
    """The drill's prose conclusion (orchestrator question runs). None for
    deep-analysis runs, which produce findings rather than a single answer."""
    id: int | None = None

    _utc = field_validator("started_at", "finished_at")(_require_utc)


class OpenInvestigation(_FrozenModel):
    """An investigation that accrues across daemon cycles until it is sufficient.

    Unlike ``InvestigationRun`` (a finished record), this is a standing intent:
    each cycle updates ``current`` toward ``target`` and, once the deterministic
    sufficiency condition is met, the investigation flips to ``ready`` and is
    eventually promoted into a concrete run.
    """

    question: str
    condition_type: str
    subject: str
    target: float
    current: float
    status: str
    created_at: datetime
    promoted_run_id: str | None = None
    id: int | None = None

    _utc = field_validator("created_at")(_require_utc)


# ─────────────────────────────────────────────────────────────────────────────
# Coverage (input to cold-start gating)
# ─────────────────────────────────────────────────────────────────────────────


class CoverageStats(_FrozenModel):
    """How much data exists - the single input to capability gating."""

    first_ts: datetime | None
    last_ts: datetime | None
    span_days: float
    n_glucose: int
    glucose_coverage_pct: float
    """Fraction of expected 5-minute slots actually present, 0-100."""
    n_insulin: int
    days_with_insulin_pct: float
    n_meals: int
    n_sleep: int
    n_activity: int
