"""Tests for the Pattern Agent (spec §7)."""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING

import pytest

from dexta_intelligence.agents.base import AgentContext, AgentRegistry
from dexta_intelligence.agents.pattern import (
    AGENT_NAME,
    pattern_agent,
    register_pattern,
)
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.models import CoverageStats, GlucoseEvent
from dexta_intelligence.stats.rigor import benjamini_hochberg
from dexta_intelligence.store import SQLiteStore
from dexta_intelligence.testing.synthetic import (
    DEFAULT_START,
    generate_null,
    scenario_sensitivity_shift,
    scenario_sleep_quality,
    scenario_weekday_breakfast,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

WINDOW_90 = (DEFAULT_START.date(), (DEFAULT_START + timedelta(days=89)).date())


@pytest.fixture
def store() -> Iterator[SQLiteStore]:
    s = SQLiteStore(":memory:")
    s.migrate()
    yield s
    s.close()


def _gates_90d() -> ColdStartReport:
    return ColdStartReport.from_coverage(
        CoverageStats(
            first_ts=DEFAULT_START,
            last_ts=DEFAULT_START + timedelta(days=90),
            span_days=90.0,
            n_glucose=90 * 288,
            glucose_coverage_pct=95.0,
            n_insulin=90,
            days_with_insulin_pct=90.0,
            n_meals=270,
            n_sleep=89,
            n_activity=40,
        )
    )


def _ctx(
    store: SQLiteStore,
    *,
    window: tuple[date, date] = WINDOW_90,
    gates: ColdStartReport | None = None,
) -> AgentContext:
    return AgentContext(
        store=store,
        window=window,
        gates=gates or _gates_90d(),
        run_id="test-run",
    )


def _load_events(store: SQLiteStore, events: dict[str, list[object]]) -> None:
    store.insert_glucose(events["glucose"])
    if events.get("insulin"):
        store.insert_insulin(events["insulin"])
    if events.get("meal"):
        store.insert_meals(events["meal"])
    if events.get("sleep"):
        store.insert_sleep(events["sleep"])
    if events.get("activity"):
        store.insert_activity(events["activity"])


class TestPlantedEffects:
    def test_sensitivity_shift_detects_tod_drift(self, store: SQLiteStore) -> None:
        """Planted regime shift → overnight drift between window halves."""
        events, manifest = scenario_sensitivity_shift(seed=11, n_days=90, effect_size=35.0)
        assert manifest.effect("sensitivity_regime_shift") is not None
        _load_events(store, events)

        findings = pattern_agent.run(_ctx(store))
        kinds = {f.kind for f in findings}
        assert "pattern_tod_drift" in kinds
        hit = next(f for f in findings if f.kind == "pattern_tod_drift")
        assert hit.stats.p_perm is not None
        assert hit.stats.q_fdr is not None
        assert hit.evidence["rigor_verdict"] == "pass"

    def test_sleep_quality_detects_sleep_glucose(self, store: SQLiteStore) -> None:
        events, manifest = scenario_sleep_quality(seed=13, n_days=90, effect_size=40.0)
        assert manifest.effect("sleep_quality_association") is not None
        _load_events(store, events)

        findings = pattern_agent.run(_ctx(store))
        assert any(f.kind == "pattern_sleep_glucose" for f in findings)

    def test_weekday_breakfast_may_surface_weekday_pattern(self, store: SQLiteStore) -> None:
        events, manifest = scenario_weekday_breakfast(seed=17, n_days=90, effect_size=45.0)
        assert manifest.effect("weekday_breakfast_spike") is not None
        _load_events(store, events)

        findings = pattern_agent.run(_ctx(store))
        kinds = {f.kind for f in findings}
        assert kinds, "expected at least one rigor-passed pattern from planted weekday effect"


class TestRigorNull:
    def test_null_scenario_emits_no_pattern_findings(self, store: SQLiteStore) -> None:
        events, manifest = generate_null(seed=7, n_days=90, start=DEFAULT_START)
        assert manifest.effects == []
        _load_events(store, events)

        findings = pattern_agent.run(_ctx(store))
        assert findings == []


class TestMissingDataSkips:
    def test_glucose_only_does_not_crash(self, store: SQLiteStore) -> None:
        events, _ = generate_null(seed=3, n_days=30, start=DEFAULT_START)
        store.insert_glucose(events["glucose"])
        window = (DEFAULT_START.date(), (DEFAULT_START + timedelta(days=29)).date())
        gates = ColdStartReport.from_coverage(
            CoverageStats(
                first_ts=DEFAULT_START,
                last_ts=DEFAULT_START + timedelta(days=30),
                span_days=30.0,
                n_glucose=30 * 288,
                glucose_coverage_pct=90.0,
                n_insulin=0,
                days_with_insulin_pct=0.0,
                n_meals=0,
                n_sleep=0,
                n_activity=0,
            )
        )
        findings = pattern_agent.run(_ctx(store, window=window, gates=gates))
        assert isinstance(findings, list)


class TestFDRFamily:
    def test_multiple_pvalues_not_all_rejected_at_lenient_alpha(self) -> None:
        """Borderline p-values: FDR thins simultaneous claims."""
        pvalues = [0.001, 0.002, 0.05, 0.06, 0.07]
        result = benjamini_hochberg(pvalues, alpha=0.05)
        assert 0 < sum(result.reject) < len(pvalues)

    def test_findings_carry_fdr_family_size(self, store: SQLiteStore) -> None:
        events, _ = scenario_sensitivity_shift(seed=11, n_days=90, effect_size=35.0)
        _load_events(store, events)
        findings = pattern_agent.run(_ctx(store))
        for finding in findings:
            assert finding.evidence.get("fdr_family_size", 0) >= 1
            assert finding.stats.q_fdr is not None


class TestDataRequirementGating:
    def test_under_data_agent_skipped_by_registry(self, store: SQLiteStore) -> None:
        store.insert_glucose(
            [
                GlucoseEvent(ts=DEFAULT_START, mg_dl=120),
                GlucoseEvent(ts=DEFAULT_START + timedelta(minutes=5), mg_dl=122),
            ]
        )
        coverage = store.coverage()
        gates = ColdStartReport.from_coverage(coverage)
        ctx = AgentContext(
            store=store,
            window=(DEFAULT_START.date(), (DEFAULT_START + timedelta(days=2)).date()),
            gates=gates,
            run_id="gated",
        )
        skipped: list[str] = []

        def on_skip(name: str, _reasons: list[str]) -> None:
            skipped.append(name)

        registry = AgentRegistry()
        register_pattern(registry)
        findings = registry.run_all(ctx, on_skip=on_skip)
        assert findings == []
        assert skipped == [AGENT_NAME]


class TestAgentContract:
    def test_name_requires_and_registration(self) -> None:
        assert pattern_agent.name == AGENT_NAME
        assert pattern_agent.requires.min_span_days >= 7.0
        registry = AgentRegistry()
        register_pattern(registry)
        assert next(iter(registry)).name == AGENT_NAME
