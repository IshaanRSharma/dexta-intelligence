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
   validation time — silent local-time bugs are endemic in CGM data and we
   refuse to inherit them.
"""

from __future__ import annotations

import enum
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "ActivityEvent",
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
    "MealEvent",
    "PredictionEvent",
    "RawEvent",
    "RecoveryEvent",
    "Rollup",
    "RollupPeriod",
    "SleepEvent",
]


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        msg = "naive datetime rejected: all timestamps must be timezone-aware (UTC)"
        raise ValueError(msg)
    return value.astimezone(UTC)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — raw store
# ─────────────────────────────────────────────────────────────────────────────


class RawEvent(_FrozenModel):
    """Immutable, verbatim provider record.

    ``(source, source_id)`` is the idempotency key: re-ingesting the same
    provider record is a no-op. The payload is never interpreted here — only
    stored, so normalization can always be replayed.
    """

    source: str
    source_id: str
    source_ts: datetime
    payload: dict[str, Any]

    _utc = field_validator("source_ts")(_require_utc)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — clinical timeline
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
    against realized CGM (spec §7.1).
    """

    ts: datetime
    """Algorithm cycle time — the timestamp of ``values_mg_dl[0]``."""
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
# Layer 3 — rollups
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
# Layer 4 — agent memory
# ─────────────────────────────────────────────────────────────────────────────


class FindingStatus(enum.StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"
    DISMISSED = "dismissed"


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

    The unification of what the donor codebase kept in four shapes
    (``pod_insights`` rows, ``Insight``, ``CoachFinding``, clinical-brief
    dicts). ``evidence`` holds every number the prose is allowed to cite.
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
# Goals — user-stated objectives pursued by background agents
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
# Coverage (input to cold-start gating)
# ─────────────────────────────────────────────────────────────────────────────


class CoverageStats(_FrozenModel):
    """How much data exists — the single input to capability gating."""

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
