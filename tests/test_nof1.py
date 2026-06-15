"""Tests for the single-subject (n-of-1) research mode.

Planted real effect -> supported + replicated; null data -> not a false
"supported"; thin data -> underpowered without crashing; runs are deterministic;
the CLI surface prints the pre-registration, the rigor results, and the verdict.
"""

from __future__ import annotations

import io
import random
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.cli.research import cmd_nof1
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.config import Config
from dexta_intelligence.models import FindingStatus, GlucoseEvent
from dexta_intelligence.store import SQLiteStore
from dexta_intelligence.testing.synthetic import EventsByType, generate_null
from dexta_intelligence.workflows.nof1 import (
    COMPARISONS,
    Hypothesis,
    parse_hypothesis,
    result_to_finding,
    run_nof1,
)

if TYPE_CHECKING:
    from pathlib import Path


def _store_from_events(events: EventsByType) -> SQLiteStore:
    store = SQLiteStore(":memory:")
    store.migrate()
    store.insert_glucose(events["glucose"])
    store.insert_meals(events["meal"])
    store.insert_insulin(events["insulin"])
    store.insert_activity(events["activity"])
    store.insert_sleep(events["sleep"])
    return store


def _store_with_weekend_effect(*, gap: int, days: int = 70, seed: int = 2026) -> SQLiteStore:
    """Hand-plant a clear weekend effect: every weekend day runs ``gap`` mg/dL
    above weekdays. ``gap=0`` is a genuine null (same distribution both groups)."""
    store = SQLiteStore(":memory:")
    store.migrate()
    rng = random.Random(seed)
    start = datetime(2026, 1, 5, tzinfo=UTC)  # a Monday
    glucose: list[GlucoseEvent] = []
    for day in range(days):
        base = start + timedelta(days=day)
        weekend = base.weekday() in (5, 6)
        center = 130 + (gap if weekend else 0)
        for slot in range(288):  # full 5-min day so each day clears the readings floor
            ts = base + timedelta(minutes=5 * slot)
            glucose.append(GlucoseEvent(ts=ts, mg_dl=center + rng.randint(-15, 15)))
    store.insert_glucose(glucose)
    return store


def _ctx(store: SQLiteStore) -> AgentContext:
    cov = store.coverage()
    assert cov.first_ts is not None and cov.last_ts is not None
    return AgentContext(
        store=store,
        window=(cov.first_ts.date(), cov.last_ts.date()),
        gates=ColdStartReport.from_coverage(cov),
        run_id="test-run",
    )


# ── planted effect → supported + replicated ──────────────────────────────────


def test_planted_weekend_effect_supported_and_replicated() -> None:
    store = _store_with_weekend_effect(gap=40)
    result = run_nof1(_ctx(store), Hypothesis(comparison="weekend", metric="mean_glucose"))
    store.close()

    assert result.ok
    assert result.verdict == "supported"
    assert result.replicated is True
    assert result.powered is True
    assert result.effect_size is not None and result.effect_size > 25  # ~+40 mg/dL planted
    assert result.p_perm is not None and result.p_perm < 0.10
    assert result.n_a >= 8 and result.n_b >= 8


# ── null data → never a false "supported" ────────────────────────────────────


def test_null_weekend_not_falsely_supported() -> None:
    # gap=0: weekends and weekdays are drawn from the same distribution.
    store = _store_with_weekend_effect(gap=0)
    result = run_nof1(_ctx(store), Hypothesis(comparison="weekend"))
    store.close()

    assert result.verdict in ("not_supported", "underpowered")


def test_null_synthetic_workout_not_falsely_supported() -> None:
    # The synthetic baseline plants no workout-day glucose effect.
    events, _ = generate_null(seed=1, n_days=70)
    result = run_nof1(_ctx(_store_from_events(events)), Hypothesis(comparison="workout"))
    assert result.verdict in ("not_supported", "underpowered")


# ── thin data → underpowered, no crash ───────────────────────────────────────


def test_thin_data_underpowered_no_crash() -> None:
    store = SQLiteStore(":memory:")
    store.migrate()
    base = datetime(2026, 1, 1, tzinfo=UTC)
    # Three days of readings: not enough days to form powered weekend groups.
    store.insert_glucose(
        [GlucoseEvent(ts=base + timedelta(minutes=5 * i), mg_dl=120) for i in range(864)]
    )
    result = run_nof1(_ctx(store), Hypothesis(comparison="weekend"))

    assert result.verdict == "underpowered"
    assert result.powered is False
    assert result.p_perm is None
    assert "data" in result.reason.lower()


def test_missing_stream_underpowered_no_crash() -> None:
    # No sleep stream at all -> sleep comparison can't form -> underpowered.
    events, _ = generate_null(seed=5, n_days=40)
    store = SQLiteStore(":memory:")
    store.migrate()
    store.insert_glucose(events["glucose"])  # glucose only, no sleep
    result = run_nof1(_ctx(store), Hypothesis(comparison="sleep"))
    assert result.verdict == "underpowered"
    assert result.ok is False


def test_unknown_comparison_underpowered_no_crash() -> None:
    events, _ = generate_null(seed=3, n_days=40)
    result = run_nof1(_ctx(_store_from_events(events)), Hypothesis(comparison="nonsense"))
    assert result.verdict == "underpowered"
    assert "unknown comparison" in result.reason


# ── determinism ──────────────────────────────────────────────────────────────


def test_two_runs_identical() -> None:
    ctx = _ctx(_store_with_weekend_effect(gap=40))
    hyp = Hypothesis(comparison="weekend")
    first = run_nof1(ctx, hyp, seed=42)
    second = run_nof1(ctx, hyp, seed=42)

    assert first.p_perm == second.p_perm
    assert first.verdict == second.verdict
    assert first.effect_size == second.effect_size
    assert first.replicated == second.replicated


# ── hypothesis parsing + spec ────────────────────────────────────────────────


def test_parse_hypothesis_keywords() -> None:
    assert parse_hypothesis("weekends run higher").comparison == "weekend"  # type: ignore[union-attr]
    assert parse_hypothesis("poor sleep raises my glucose").comparison == "sleep"  # type: ignore[union-attr]
    tir = parse_hypothesis("workout days have better time in range")
    assert tir is not None and tir.comparison == "workout" and tir.metric == "tir"
    assert parse_hypothesis("the weather is nice today") is None


def test_registered_statement_preserved() -> None:
    hyp = Hypothesis(comparison="weekend", statement="my custom registration")
    assert hyp.registered_statement() == "my custom registration"
    generated = Hypothesis(comparison="weekend").registered_statement()
    assert "weekend" in generated.lower()


def test_all_comparisons_have_specs() -> None:
    for spec in COMPARISONS.values():
        assert "tool" in spec and "labels" in spec and "phrase" in spec


# ── persistence as a Finding ─────────────────────────────────────────────────


def test_result_to_finding_shape() -> None:
    ctx = _ctx(_store_with_weekend_effect(gap=40))
    result = run_nof1(ctx, Hypothesis(comparison="weekend"))
    finding = result_to_finding(result, ctx)

    assert finding.kind == "nof1"
    assert finding.agent == "nof1"
    assert finding.stats.p_perm == result.p_perm
    assert finding.stats.replicated == result.replicated
    assert finding.evidence["verdict"] == result.verdict
    assert "not a randomized trial" in finding.body_md
    # round-trips through the store
    store = SQLiteStore(":memory:")
    store.migrate()
    fid = store.insert_finding(finding)
    persisted = store.get_findings(agent="nof1", status=FindingStatus.ACTIVE)
    assert any(f.id == fid for f in persisted)
    store.close()


# ── CLI surface ──────────────────────────────────────────────────────────────


def _opener(store: SQLiteStore):  # type: ignore[no-untyped-def]
    def _open(_config: Config, _db: Path | None = None) -> SQLiteStore:
        return store

    return _open


def test_cmd_nof1_prints_preregistration_and_verdict() -> None:
    store = _store_with_weekend_effect(gap=40)
    out = io.StringIO()
    code = cmd_nof1(
        config=Config(),
        db_path=None,
        out=out,
        compare="weekend",
        opener=_opener(store),
    )
    text = out.getvalue()

    assert code == 0
    assert "Pre-registered hypothesis" in text
    assert "permutation p" in text
    assert "Verdict:" in text
    assert "not a randomized trial" in text
    # not persisted without --save
    assert store.get_findings(agent="nof1") == []


def test_cmd_nof1_save_persists_finding() -> None:
    store = _store_with_weekend_effect(gap=40)
    out = io.StringIO()
    code = cmd_nof1(
        config=Config(),
        db_path=None,
        out=out,
        compare="weekend",
        save=True,
        opener=_opener(store),
    )
    assert code == 0
    assert "persisted as finding" in out.getvalue()
    assert len(store.get_findings(agent="nof1")) == 1


def test_cmd_nof1_unknown_compare_exits_2() -> None:
    out = io.StringIO()
    store = SQLiteStore(":memory:")
    store.migrate()
    code = cmd_nof1(
        config=Config(),
        db_path=None,
        out=out,
        compare="bogus",
        opener=_opener(store),
    )
    assert code == 2
    assert "Unknown comparison" in out.getvalue()
    store.close()
