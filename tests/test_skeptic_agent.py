"""Tests for the Skeptic Agent."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from dexta_intelligence.agents.base import AgentContext, AgentRegistry
from dexta_intelligence.agents.skeptic import (
    AGENT_NAME,
    confound_hypotheses,
    register_skeptic,
    skeptic_agent,
)
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.models import (
    CoverageStats,
    Finding,
    FindingStats,
    FindingStatus,
)
from dexta_intelligence.store import SQLiteStore

T0 = datetime(2025, 1, 1, tzinfo=UTC)


@pytest.fixture
def store() -> SQLiteStore:
    s = SQLiteStore(":memory:")
    s.migrate()
    return s


def _ctx(store: SQLiteStore) -> AgentContext:
    coverage = CoverageStats(
        first_ts=T0,
        last_ts=T0,
        span_days=90.0,
        n_glucose=1000,
        glucose_coverage_pct=95.0,
        n_insulin=0,
        days_with_insulin_pct=0.0,
        n_meals=0,
        n_sleep=0,
        n_activity=0,
    )
    return AgentContext(
        store=store,
        window=(T0.date(), T0.date()),
        gates=ColdStartReport.from_coverage(coverage),
        run_id="skeptic-test",
    )


def _quantitative_finding(
    *,
    group_a: list[float],
    group_b: list[float],
    effect: float,
    kind: str = "pattern_weekday_weekend",
) -> Finding:
    return Finding(
        agent="pattern",
        kind=kind,
        scope="pattern_analysis",
        headline="test finding",
        body_md="body",
        evidence={
            "skeptic_group_a": group_a,
            "skeptic_group_b": group_b,
            "rigor_verdict": "pass",
        },
        stats=FindingStats(
            effect_size=effect,
            n=len(group_a) + len(group_b),
            p_perm=0.01,
            q_fdr=0.05,
            replicated=True,
        ),
        confidence=0.75,
    )


def test_register_skeptic() -> None:
    registry = AgentRegistry()
    register_skeptic(registry)
    assert AGENT_NAME in {a.name for a in registry}


def test_observation_passes_through(store: SQLiteStore) -> None:
    finding = Finding(
        agent="observation",
        kind="observation_glycemic",
        scope="observation",
        headline="TIR 70%",
        body_md="",
        evidence={"tir_pct": 70.0},
        confidence=1.0,
    )
    reviewed = skeptic_agent.review([finding], _ctx(store))
    assert reviewed[0].status == FindingStatus.ACTIVE
    assert reviewed[0].skeptic_notes is not None
    assert "observation" in reviewed[0].skeptic_notes


def test_rejects_quantitative_without_p_value(store: SQLiteStore) -> None:
    finding = Finding(
        agent="pattern",
        kind="pattern_tod_drift",
        scope="pattern_analysis",
        headline="drift",
        body_md="",
        evidence={},
        stats=FindingStats(effect_size=5.0, n=20),
        confidence=0.7,
    )
    reviewed = skeptic_agent.review([finding], _ctx(store))
    assert reviewed[0].status == FindingStatus.REJECTED


def test_confound_flag_lowers_confidence(store: SQLiteStore) -> None:
    # Two distinct groups with real separation: should pass rigor on both seeds.
    high_a = [200.0] * 12
    low_b = [120.0] * 12
    f1 = _quantitative_finding(
        group_a=high_a, group_b=low_b, effect=80.0, kind="pattern_weekday_weekend"
    )
    f2 = _quantitative_finding(
        group_a=high_a, group_b=low_b, effect=80.0, kind="pattern_sleep_glucose"
    )
    reviewed = skeptic_agent.review([f1, f2], _ctx(store))
    for f in reviewed:
        assert f.skeptic_notes is not None
        assert "confound" in f.skeptic_notes
        assert f.confidence <= 0.5


def test_contradicts_prior_in_memory(store: SQLiteStore) -> None:
    prior = Finding(
        agent="pattern",
        kind="pattern_tod_drift",
        scope="pattern_analysis",
        headline="old",
        body_md="",
        evidence={},
        stats=FindingStats(effect_size=10.0, n=20, p_perm=0.02, q_fdr=0.08, replicated=True),
        confidence=0.75,
    )
    prior_id = store.insert_finding(prior)

    current = _quantitative_finding(
        group_a=[180.0] * 12,
        group_b=[160.0] * 12,
        effect=-10.0,
        kind="pattern_tod_drift",
    )
    reviewed = skeptic_agent.review([current], _ctx(store))
    assert reviewed[0].skeptic_notes is not None
    assert str(prior_id) in reviewed[0].skeptic_notes
    assert reviewed[0].confidence <= 0.35


def test_clean_finding_survives(store: SQLiteStore) -> None:
    finding = _quantitative_finding(
        group_a=[200.0] * 12,
        group_b=[120.0] * 12,
        effect=80.0,
    )
    reviewed = skeptic_agent.review([finding], _ctx(store))
    assert reviewed[0].status == FindingStatus.ACTIVE
    assert reviewed[0].confidence >= 0.5


def test_confound_hypotheses_one_per_pair(store: SQLiteStore) -> None:
    high_a = [200.0] * 12
    low_b = [120.0] * 12
    f1 = _quantitative_finding(
        group_a=high_a, group_b=low_b, effect=80.0, kind="pattern_weekday_weekend"
    )
    f2 = _quantitative_finding(
        group_a=high_a, group_b=low_b, effect=80.0, kind="pattern_sleep_glucose"
    )
    reviewed = skeptic_agent.review([f1, f2], _ctx(store))

    hypotheses = confound_hypotheses(reviewed)
    # The pair surfaces on both findings but yields exactly one hypothesis.
    assert len(hypotheses) == 1
    statement = hypotheses[0].statement
    assert statement.startswith("Disentangle pattern_sleep_glucose vs pattern_weekday_weekend:")
    assert statement.endswith("- stratify when more data allows [skeptic]")


def test_confound_hypotheses_none_without_flag(store: SQLiteStore) -> None:
    finding = _quantitative_finding(
        group_a=[200.0] * 12,
        group_b=[120.0] * 12,
        effect=80.0,
        kind="pattern_weekday_weekend",
    )
    reviewed = skeptic_agent.review([finding], _ctx(store))
    assert confound_hypotheses(reviewed) == []
