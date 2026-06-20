"""E5 robustness - deterministic agents under data corruption.

Ground truth is non-LLM: the deterministic observation + pattern agents run on
clean :func:`scenario_all` glucose and on corrupted variants of it. A robust
discovery layer surfaces (nearly) the same set of finding ``kind``\\ s
regardless of realistic data defects.

Corruptions applied to the clean series (each independently):

- ``dropout`` - 15% of readings removed at random.
- ``duplicates`` - 5% of readings re-inserted at ±1s offsets (the store dedupes
  exact-timestamp collisions, so duplicates are nudged to survive the UNIQUE
  index and exercise downstream dedup tolerance).
- ``gap`` - a contiguous 3-day block of readings removed.
- ``tz_shift`` - a +1h timestamp shift applied to a contiguous one-week block.

Metrics per corruption: Jaccard similarity of the clean-vs-corrupted finding
``kind`` sets (target ≥ 0.8) and the count of corruption-induced *new* kinds
(target 0).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import timedelta

from dexta_intelligence.agents.base import AgentContext, AgentRegistry
from dexta_intelligence.agents.observation import register_observation
from dexta_intelligence.agents.pattern import register_pattern
from dexta_intelligence.agents.skeptic import register_skeptic
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.models import CoverageStats, GlucoseEvent
from dexta_intelligence.store import SQLiteStore
from dexta_intelligence.testing.synthetic import DEFAULT_START, scenario_all
from dexta_intelligence.workflows.deep_analysis import run_deep_analysis

__all__ = ["E5PerturbationResult", "run_e5"]

#: Spec §14 robustness targets.
_JACCARD_TARGET = 0.8

#: Corruption parameters.
_DROPOUT_FRAC = 0.15
_DUPLICATE_FRAC = 0.05
_GAP_DAYS = 3
_TZ_SHIFT_HOURS = 1
_TZ_SHIFT_WEEK_DAY = 30  # first day of the contiguous shifted week


@dataclass(frozen=True, slots=True)
class E5CorruptionRow:
    """Per-corruption robustness measurement."""

    name: str
    jaccard: float
    new_kinds: tuple[str, ...]
    n_clean_kinds: int
    n_corrupt_kinds: int


@dataclass(frozen=True, slots=True)
class E5PerturbationResult:
    """Outcome of one E5 sweep across all corruption variants."""

    n_days: int
    clean_kinds: tuple[str, ...]
    corruptions: tuple[E5CorruptionRow, ...]
    min_jaccard: float
    total_new_kinds: int
    jaccard_target: float
    passed: bool


def _drop(glucose: list[GlucoseEvent], rng: random.Random) -> list[GlucoseEvent]:
    return [g for g in glucose if rng.random() >= _DROPOUT_FRAC]


def _duplicate(glucose: list[GlucoseEvent], rng: random.Random) -> list[GlucoseEvent]:
    """Re-insert a fraction of readings at ±1s offsets to survive store dedup."""
    out = list(glucose)
    for g in glucose:
        if rng.random() < _DUPLICATE_FRAC:
            offset = timedelta(seconds=rng.choice((-1, 1)))
            out.append(GlucoseEvent(ts=g.ts + offset, mg_dl=g.mg_dl, trend=g.trend))
    return out


def _gap(glucose: list[GlucoseEvent], n_days: int) -> list[GlucoseEvent]:
    """Remove a contiguous 3-day block from the middle of the window."""
    gap_start = DEFAULT_START + timedelta(days=n_days // 2)
    gap_end = gap_start + timedelta(days=_GAP_DAYS)
    return [g for g in glucose if not (gap_start <= g.ts < gap_end)]


def _tz_shift(glucose: list[GlucoseEvent]) -> list[GlucoseEvent]:
    """Shift one contiguous week of readings by +1h (timezone-like defect)."""
    shift_start = DEFAULT_START + timedelta(days=_TZ_SHIFT_WEEK_DAY)
    shift_end = shift_start + timedelta(days=7)
    out: list[GlucoseEvent] = []
    for g in glucose:
        if shift_start <= g.ts < shift_end:
            out.append(
                GlucoseEvent(
                    ts=g.ts + timedelta(hours=_TZ_SHIFT_HOURS),
                    mg_dl=g.mg_dl,
                    trend=g.trend,
                )
            )
        else:
            out.append(g)
    return out


def _gates(n_days: int) -> ColdStartReport:
    """Coverage that unlocks the observation + pattern gates for the run."""
    end = DEFAULT_START + timedelta(days=n_days)
    return ColdStartReport.from_coverage(
        CoverageStats(
            first_ts=DEFAULT_START,
            last_ts=end,
            span_days=float(n_days),
            n_glucose=n_days * 288,
            glucose_coverage_pct=95.0,
            n_insulin=0,
            days_with_insulin_pct=0.0,
            n_meals=0,
            n_sleep=0,
            n_activity=0,
        )
    )


def _run_kinds(
    glucose: list[GlucoseEvent], window: tuple[object, object], gates: ColdStartReport
) -> frozenset[str]:
    """Run observation + pattern agents on ``glucose`` and return finding kinds."""
    store = SQLiteStore(":memory:")
    store.migrate()
    try:
        store.insert_glucose(glucose)
        registry = AgentRegistry()
        register_observation(registry)
        register_pattern(registry)
        register_skeptic(registry)
        ctx = AgentContext(store=store, window=window, gates=gates, run_id="e5")  # type: ignore[arg-type]
        report = run_deep_analysis(registry, ctx, skip_skeptic=False, persist=False)
        return frozenset(f.kind for f in report.findings)
    finally:
        store.close()


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def run_e5(*, seed: int = 5000, n_days: int = 90) -> E5PerturbationResult:
    """Run the deterministic agents on clean vs corrupted ``scenario_all`` data."""
    events, _manifest = scenario_all(seed=seed, n_days=n_days)
    clean = events["glucose"]
    window = (DEFAULT_START.date(), (DEFAULT_START + timedelta(days=n_days - 1)).date())
    gates = _gates(n_days)

    clean_kinds = _run_kinds(clean, window, gates)

    rng = random.Random(seed)
    variants: dict[str, list[GlucoseEvent]] = {
        "dropout": _drop(clean, rng),
        "duplicates": _duplicate(clean, rng),
        "gap": _gap(clean, n_days),
        "tz_shift": _tz_shift(clean),
    }

    rows: list[E5CorruptionRow] = []
    for name, corrupted in variants.items():
        kinds = _run_kinds(corrupted, window, gates)
        new_kinds = tuple(sorted(kinds - clean_kinds))
        rows.append(
            E5CorruptionRow(
                name=name,
                jaccard=_jaccard(clean_kinds, kinds),
                new_kinds=new_kinds,
                n_clean_kinds=len(clean_kinds),
                n_corrupt_kinds=len(kinds),
            )
        )

    min_jaccard = min((r.jaccard for r in rows), default=1.0)
    total_new = sum(len(r.new_kinds) for r in rows)
    passed = min_jaccard >= _JACCARD_TARGET and total_new == 0

    return E5PerturbationResult(
        n_days=n_days,
        clean_kinds=tuple(sorted(clean_kinds)),
        corruptions=tuple(rows),
        min_jaccard=min_jaccard,
        total_new_kinds=total_new,
        jaccard_target=_JACCARD_TARGET,
        passed=passed,
    )
