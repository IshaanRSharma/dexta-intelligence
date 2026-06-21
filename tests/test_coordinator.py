"""Tests for the investigations coordinator.

Two paths: the deterministic fallback (no model) must run the full producer set
and return what deep_analysis would (parity), and a scripted planning model that
selects a subset must run only those producers. The skeptic post-pass applies in
both; thin data returns [] without crashing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import pytest

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.agents.coordinator import CoordinatorAgent
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.config import Config
from dexta_intelligence.models import CoverageStats, FindingStatus
from dexta_intelligence.store import SQLiteStore
from dexta_intelligence.testing.synthetic import DEFAULT_START, generate_null
from dexta_intelligence.workflows.deep_analysis import run_deep_analysis
from dexta_intelligence.workflows.lenses import PRODUCERS, build_registry

if TYPE_CHECKING:
    from collections.abc import Sequence

WINDOW = (DEFAULT_START.date(), (DEFAULT_START + timedelta(days=29)).date())


@dataclass
class _Reply:
    content: str


class _ScriptedModel:
    """Returns queued JSON replies in order; records the prompts it saw."""

    def __init__(self, replies: Sequence[dict[str, Any]]) -> None:
        self._replies = [json.dumps(r) for r in replies]
        self.prompts: list[str] = []

    def invoke(self, messages: list[dict[str, str]]) -> _Reply:
        self.prompts.append(messages[-1]["content"])
        if self._replies:
            return _Reply(self._replies.pop(0))
        return _Reply("{}")


@pytest.fixture
def store() -> SQLiteStore:
    s = SQLiteStore(":memory:")
    s.migrate()
    return s


def _full_coverage_ctx(store: SQLiteStore) -> AgentContext:
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
        run_id="coordinator-test",
    )


def _seed_glucose(store: SQLiteStore, *, seed: int = 42) -> None:
    events, _manifest = generate_null(seed=seed, n_days=30, start=DEFAULT_START)
    store.insert_glucose(events["glucose"])


def test_no_model_parity_with_deep_analysis(store: SQLiteStore) -> None:
    """No model: coordinator runs the full producer set, same findings as deep_analysis."""
    _seed_glucose(store)
    ctx = _full_coverage_ctx(store)

    coordinator_findings = CoordinatorAgent(model=None).investigate(ctx)

    registry, _ = build_registry("analyze", Config())
    report = run_deep_analysis(registry, ctx, persist=False)

    assert coordinator_findings  # produced something on real data
    assert {f.kind for f in coordinator_findings} == {f.kind for f in report.findings}
    assert {f.headline for f in coordinator_findings} == {f.headline for f in report.findings}


def test_skeptic_pass_applied(store: SQLiteStore) -> None:
    _seed_glucose(store)
    findings = CoordinatorAgent(model=None).investigate(_full_coverage_ctx(store))
    observations = [f for f in findings if f.agent == "observation"]
    assert observations
    for finding in observations:
        assert finding.skeptic_notes is not None


def test_run_trace_streams_plan_and_per_producer_events(store: SQLiteStore) -> None:
    """A RunTrace with an on_event sink narrates the investigation live: a plan
    event, then running/producer_done per producer, then step lines."""
    from dexta_intelligence.agents.coordinator import RunTrace  # noqa: PLC0415

    _seed_glucose(store)
    events: list[dict[str, Any]] = []
    rec = RunTrace(on_event=events.append)
    CoordinatorAgent(model=None).investigate(_full_coverage_ctx(store), trace=rec)

    kinds = [e["kind"] for e in events]
    assert "plan" in kinds
    assert "running" in kinds
    assert "producer_done" in kinds

    plan = next(e for e in events if e["kind"] == "plan")
    assert plan["payload"]["steps"] == list(PRODUCERS)
    started = {e["payload"]["producer"] for e in events if e["kind"] == "running"}
    done = {e["payload"]["producer"] for e in events if e["kind"] == "producer_done"}
    assert started == done
    assert all(
        isinstance(e["payload"]["n_findings"], int)
        for e in events
        if e["kind"] == "producer_done"
    )
    assert kinds[0] == "coverage"
    assert "glucose_coverage_pct" in events[0]["payload"]
    assert rec.coverage_summary is not None
    assert rec.tool_calls  # one entry per producer that ran


def test_record_run_persists_coverage_and_evidence(store: SQLiteStore) -> None:
    """The persisted run carries the coverage snapshot and per-finding evidence."""
    _seed_glucose(store)
    findings = CoordinatorAgent(model=None).investigate(_full_coverage_ctx(store))
    run = store.get_investigation_runs(limit=1)[0]
    assert run.coverage_summary is not None
    assert "glucose_coverage_pct" in run.coverage_summary
    assert run.tool_calls  # producer-level instrument log
    assert len(run.evidence_items) == len(findings)
    if findings:
        assert run.evidence_items[0]["finding"] == findings[0].headline
        assert "numbers" in run.evidence_items[0]


def test_poor_coverage_marks_run_limited(store: SQLiteStore) -> None:
    """A run over thin sensor coverage is flagged limited even with findings."""
    from dexta_intelligence.agents.coordinator import (  # noqa: PLC0415
        RunTrace,
        _coverage_summary,
        _final_status,
    )
    from dexta_intelligence.models import Finding  # noqa: PLC0415

    _seed_glucose(store)
    assert _coverage_summary(_full_coverage_ctx(store))["limited"] is False

    one = [Finding(agent="observation", kind="pattern", scope="x", headline="h")]
    limited = RunTrace(coverage_summary={"glucose_coverage_pct": 40.0, "limited": True})
    good = RunTrace(coverage_summary={"glucose_coverage_pct": 90.0, "limited": False})
    # Poor coverage forces "limited" even with findings; good coverage + findings completes.
    assert _final_status(one, limited) == "limited"
    assert _final_status(one, good) == "completed"
    assert _final_status([], good) == "limited"  # no findings is still limited


def test_scripted_plan_selects_subset(store: SQLiteStore) -> None:
    """A planner that picks only 'observation' runs only that producer."""
    _seed_glucose(store)
    model = _ScriptedModel([{"investigations": ["observation"], "reason": "narrow"}])

    findings = CoordinatorAgent(model=model).investigate(
        _full_coverage_ctx(store), goal="summarize my glucose"
    )

    assert findings
    assert {f.agent for f in findings} == {"observation"}
    assert model.prompts  # the planner was actually consulted


def test_replan_runs_a_bounded_followup_round(store: SQLiteStore) -> None:
    """Round 1 picks observation; the re-plan picks pattern; both run, bounded to 2."""
    _seed_glucose(store)
    model = _ScriptedModel(
        [
            {"investigations": ["observation"], "reason": "start narrow"},
            {"investigations": ["pattern"], "reason": "drill into variability"},
        ]
    )

    findings = CoordinatorAgent(model=model, synthesize_connections=False).investigate(
        _full_coverage_ctx(store), goal="why is my variability high"
    )

    # The re-plan fired (a second planning call) and the loop is bounded to 2 -
    # so the planner is consulted exactly twice, never a third time.
    assert len(model.prompts) == 2
    assert "observation" in {f.agent for f in findings}  # round 1 ran (null data → no patterns)


def test_replan_declines_when_satisfied(store: SQLiteStore) -> None:
    """An empty re-plan ends the loop after the first round."""
    _seed_glucose(store)
    model = _ScriptedModel(
        [
            {"investigations": ["observation"], "reason": "enough"},
            {"investigations": [], "reason": "first round covers it"},
        ]
    )
    findings = CoordinatorAgent(model=model).investigate(_full_coverage_ctx(store))
    assert {f.agent for f in findings} == {"observation"}


def test_unknown_producer_falls_back_to_full_set(store: SQLiteStore) -> None:
    """A selection naming no known producer plans the full producer set."""
    _seed_glucose(store)
    model = _ScriptedModel([{"investigations": ["nonsense"], "reason": "oops"}])

    plan = CoordinatorAgent(model=model)._plan(_full_coverage_ctx(store), None)
    assert plan == list(PRODUCERS)


def test_thin_data_returns_empty_without_crash(store: SQLiteStore) -> None:
    """Below every data requirement: producers are gated out, no exception, []."""
    coverage = CoverageStats(
        first_ts=DEFAULT_START,
        last_ts=DEFAULT_START + timedelta(days=1),
        span_days=1.0,
        n_glucose=10,
        glucose_coverage_pct=5.0,
        n_insulin=0,
        days_with_insulin_pct=0.0,
        n_meals=0,
        n_sleep=0,
        n_activity=0,
    )
    ctx = AgentContext(
        store=store,
        window=(DEFAULT_START.date(), (DEFAULT_START + timedelta(days=1)).date()),
        gates=ColdStartReport.from_coverage(coverage),
        run_id="thin",
    )
    assert CoordinatorAgent(model=None).investigate(ctx) == []


def test_planner_failure_falls_back(store: SQLiteStore) -> None:
    """A model whose invoke raises must not abort the run - full set instead."""
    _seed_glucose(store)

    class _BoomModel:
        def invoke(self, _messages: list[dict[str, str]]) -> _Reply:
            raise RuntimeError("api down")

    coordinator = CoordinatorAgent(model=_BoomModel())
    plan = coordinator._plan(_full_coverage_ctx(store), None)
    assert plan == list(PRODUCERS)
    # And the run itself completes without raising.
    assert coordinator.investigate(_full_coverage_ctx(store)) is not None


def test_recall_digest_not_raw_findings_in_prompt(store: SQLiteStore) -> None:
    """Planning context is the compact recall digest, never raw finding bodies."""
    _seed_glucose(store)
    ctx = _full_coverage_ctx(store)
    # Bank a finding so recall has something to summarize.
    first = CoordinatorAgent(model=None).investigate(ctx)
    from dexta_intelligence.workflows.deep_analysis import persist_findings  # noqa: PLC0415

    persist_findings(store, first)

    model = _ScriptedModel([{"investigations": ["observation"], "reason": "ok"}])
    CoordinatorAgent(model=model).investigate(ctx, goal="overnight")
    prompt = model.prompts[0]
    # body_md is the raw finding body; it must never reach the planner.
    assert "body_md" not in prompt
    active = store.get_findings(status=FindingStatus.ACTIVE, limit=10)
    for finding in active:
        if finding.body_md and finding.body_md not in ("body", ""):
            assert finding.body_md not in prompt


def test_investigation_run_persisted_with_plan_and_findings(store: SQLiteStore) -> None:
    """investigate() records one InvestigationRun: plan, trace, and a findings snapshot."""
    _seed_glucose(store)
    ctx = _full_coverage_ctx(store)
    findings = CoordinatorAgent(model=None).investigate(ctx)

    runs = store.get_investigation_runs()
    assert len(runs) == 1
    run = runs[0]
    assert run.run_id == "coordinator-test"
    assert run.kind == "deep_analysis"  # no goal
    assert run.status == "completed"
    assert run.plan == list(PRODUCERS)
    assert run.trace  # at least the plan line + one round line
    assert run.n_findings == len(findings)
    assert {f.headline for f in run.findings} == {f.headline for f in findings}


def test_investigation_run_records_goal_and_feeds_recall(store: SQLiteStore) -> None:
    """A goal-scoped run is kind='question' and is recalled by the planner."""
    from dexta_intelligence.agents.coordinator import _past_investigations  # noqa: PLC0415

    _seed_glucose(store)
    ctx = _full_coverage_ctx(store)
    CoordinatorAgent(model=None).investigate(ctx, goal="overnight lows")

    run = store.get_investigation_runs()[0]
    assert run.kind == "question"
    assert run.question == "overnight lows"

    digest = _past_investigations(ctx)
    assert "overnight lows" in digest
    assert "observation" in digest
