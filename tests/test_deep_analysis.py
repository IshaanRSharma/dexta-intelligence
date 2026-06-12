"""Tests for the deep analysis workflow."""

from __future__ import annotations

from datetime import timedelta

import pytest

from dexta_intelligence.agents.base import AgentContext, AgentRegistry, DataRequirement
from dexta_intelligence.agents.observation import register_observation
from dexta_intelligence.agents.skeptic import register_skeptic
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.models import CoverageStats, Finding, FindingStats, FindingStatus
from dexta_intelligence.store import SQLiteStore
from dexta_intelligence.testing.synthetic import DEFAULT_START, generate_null
from dexta_intelligence.workflows.deep_analysis import persist_findings, run_deep_analysis

WINDOW = (DEFAULT_START.date(), (DEFAULT_START + timedelta(days=29)).date())


@pytest.fixture
def store() -> SQLiteStore:
    s = SQLiteStore(":memory:")
    s.migrate()
    return s


def _ctx(store: SQLiteStore) -> AgentContext:
    coverage = store.coverage()
    if coverage.span_days < 30:
        coverage = CoverageStats(
            first_ts=DEFAULT_START,
            last_ts=DEFAULT_START + timedelta(days=30),
            span_days=30.0,
            n_glucose=30 * 288,
            glucose_coverage_pct=95.0,
            n_insulin=0,
            days_with_insulin_pct=0.0,
            n_meals=0,
            n_sleep=0,
            n_activity=0,
        )
    return AgentContext(
        store=store,
        window=WINDOW,
        gates=ColdStartReport.from_coverage(coverage),
        run_id="deep-test",
    )


def test_deep_analysis_runs_observation_and_skeptic(store: SQLiteStore) -> None:
    events, _manifest = generate_null(seed=42, n_days=30, start=DEFAULT_START)
    store.insert_glucose(events["glucose"])

    registry = AgentRegistry()
    register_observation(registry)
    register_skeptic(registry)

    report = run_deep_analysis(registry, _ctx(store), persist=True)
    assert report.findings
    assert len(report.persisted_ids) == len(report.findings)
    for finding in report.findings:
        if finding.agent == "observation":
            assert finding.skeptic_notes is not None


class _StubProducer:
    """Test producer that emits pre-built findings (no data requirement)."""

    name = "stub"
    requires = DataRequirement()

    def __init__(self, findings: list[Finding]) -> None:
        self._findings = findings

    def run(self, ctx: AgentContext) -> list[Finding]:
        return list(self._findings)


def _confound_finding(kind: str) -> Finding:
    groups = [200.0] * 12, [120.0] * 12
    return Finding(
        agent="pattern",
        kind=kind,
        scope="pattern_analysis",
        headline=f"{kind} effect",
        body_md="body",
        evidence={
            "skeptic_group_a": groups[0],
            "skeptic_group_b": groups[1],
            "rigor_verdict": "pass",
        },
        stats=FindingStats(
            effect_size=80.0, n=24, p_perm=0.01, q_fdr=0.05, replicated=True
        ),
        confidence=0.75,
    )


def _confound_registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register(
        _StubProducer(
            [
                _confound_finding("pattern_weekday_weekend"),
                _confound_finding("pattern_sleep_glucose"),
            ]
        )
    )
    register_skeptic(registry)
    return registry


def test_confound_pair_banks_one_hypothesis(store: SQLiteStore) -> None:
    report = run_deep_analysis(_confound_registry(), _ctx(store), persist=True)
    assert len(report.banked_hypotheses) == 1
    open_hyps = store.get_hypotheses(status="open")
    assert len(open_hyps) == 1
    assert open_hyps[0].statement.startswith("Disentangle ")


def test_confound_rerun_does_not_duplicate(store: SQLiteStore) -> None:
    ctx = _ctx(store)
    first = run_deep_analysis(_confound_registry(), ctx, persist=True)
    assert len(first.banked_hypotheses) == 1

    second = run_deep_analysis(_confound_registry(), ctx, persist=True)
    assert second.banked_hypotheses == ()
    assert len(store.get_hypotheses(status="open")) == 1


def test_no_confound_banks_nothing(store: SQLiteStore) -> None:
    registry = AgentRegistry()
    registry.register(_StubProducer([_confound_finding("pattern_weekday_weekend")]))
    register_skeptic(registry)

    report = run_deep_analysis(registry, _ctx(store), persist=True)
    assert report.banked_hypotheses == ()
    assert store.get_hypotheses(status="open") == []


def _plain_finding(kind: str = "obs_glucose", scope: str = "daily") -> Finding:
    return Finding(
        agent="observation",
        kind=kind,
        scope=scope,
        headline=f"{kind} headline",
        body_md="body",
        stats=FindingStats(effect_size=1.0, n=10),
    )


def test_persist_findings_supersedes_prior_active(store: SQLiteStore) -> None:
    first = persist_findings(store, [_plain_finding()])
    second = persist_findings(store, [_plain_finding()])

    active = store.get_findings(status=FindingStatus.ACTIVE, limit=100)
    assert len(active) == 1
    assert active[0].id == second[0]

    superseded = store.get_findings(status=FindingStatus.SUPERSEDED, limit=100)
    assert len(superseded) == 1
    assert superseded[0].id == first[0]
    assert superseded[0].superseded_by == second[0]


def test_persist_findings_distinct_scope_coexists(store: SQLiteStore) -> None:
    persist_findings(store, [_plain_finding(scope="daily")])
    persist_findings(store, [_plain_finding(scope="weekly")])

    active = store.get_findings(status=FindingStatus.ACTIVE, limit=100)
    assert len(active) == 2
    assert {f.scope for f in active} == {"daily", "weekly"}


def test_rerun_analysis_keeps_active_flat_and_grows_graveyard(store: SQLiteStore) -> None:
    registry = _confound_registry()
    ctx = _ctx(store)

    run_deep_analysis(registry, ctx, persist=True)
    active_after_first = store.get_findings(status=FindingStatus.ACTIVE, limit=100)

    run_deep_analysis(registry, ctx, persist=True)
    active_after_second = store.get_findings(status=FindingStatus.ACTIVE, limit=100)
    superseded = store.get_findings(status=FindingStatus.SUPERSEDED, limit=100)

    assert active_after_first  # produced something
    assert len(active_after_second) == len(active_after_first)
    assert len(superseded) == len(active_after_first)


def test_skip_skeptic(store: SQLiteStore) -> None:
    events, _manifest = generate_null(seed=43, n_days=30, start=DEFAULT_START)
    store.insert_glucose(events["glucose"])

    registry = AgentRegistry()
    register_observation(registry)
    register_skeptic(registry)

    report = run_deep_analysis(registry, _ctx(store), skip_skeptic=True, persist=False)
    for finding in report.findings:
        assert finding.skeptic_notes is None
