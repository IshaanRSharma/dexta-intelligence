"""Tests for the Observation Agent."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from dexta_intelligence.agents.base import AgentContext, AgentRegistry
from dexta_intelligence.agents.observation import (
    AGENT_NAME,
    observation_agent,
    register_observation,
)
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.models import (
    CoverageStats,
    GlucoseEvent,
    InsulinEvent,
    InsulinKind,
    SleepEvent,
)
from dexta_intelligence.store import SQLiteStore

if TYPE_CHECKING:
    from collections.abc import Iterator

T0 = datetime(2025, 6, 1, 0, 0, tzinfo=UTC)
WINDOW = (date(2025, 6, 1), date(2025, 6, 7))

_FORBIDDEN_WORDS = ("should", "good", "bad", "recommend", "adequate", "poor")


@pytest.fixture
def store() -> Iterator[SQLiteStore]:
    s = SQLiteStore(":memory:")
    s.migrate()
    yield s
    s.close()


def _gates(*, span_days: float = 7.0, coverage_pct: float = 80.0) -> ColdStartReport:
    return ColdStartReport.from_coverage(
        CoverageStats(
            first_ts=T0,
            last_ts=T0 + timedelta(days=span_days),
            span_days=span_days,
            n_glucose=500,
            glucose_coverage_pct=coverage_pct,
            n_insulin=0,
            days_with_insulin_pct=0.0,
            n_meals=0,
            n_sleep=0,
            n_activity=0,
        )
    )


def _ctx(store: SQLiteStore, *, window: tuple[date, date] = WINDOW) -> AgentContext:
    return AgentContext(
        store=store,
        window=window,
        gates=_gates(),
        run_id="test-run",
    )


def _series(base: datetime, values: list[int], *, step_min: int = 5) -> list[GlucoseEvent]:
    return [
        GlucoseEvent(ts=base + timedelta(minutes=step_min * i), mg_dl=v)
        for i, v in enumerate(values)
    ]


class TestGlycemicFacts:
    def test_known_tir_and_mean_in_evidence(self, store: SQLiteStore) -> None:
        """Hand-built window: all readings in-range → exact TIR/mean."""
        values = [100] * 20
        store.insert_glucose(_series(T0, values))

        findings = observation_agent.run(_ctx(store))
        hit = next(f for f in findings if f.kind == "observation_glycemic")
        assert hit.evidence["tir_pct"] == 100.0
        assert hit.evidence["mean_mg_dl"] == 100.0
        assert hit.evidence["gmi_pct"] == pytest.approx(5.7, abs=0.05)
        assert hit.evidence["n_readings"] == 20

    def test_no_interpretation_words(self, store: SQLiteStore) -> None:
        store.insert_glucose(_series(T0, [90, 110, 130, 150, 120] * 4))
        findings = observation_agent.run(_ctx(store))
        for finding in findings:
            text = f"{finding.headline} {finding.body_md}".lower()
            for word in _FORBIDDEN_WORDS:
                assert word not in text, f"forbidden word {word!r} in {finding.kind}"


class TestHonestEmpties:
    def test_empty_store_returns_no_findings(self, store: SQLiteStore) -> None:
        assert observation_agent.run(_ctx(store)) == []

    def test_insulin_finding_only_when_insulin_present(self, store: SQLiteStore) -> None:
        store.insert_glucose(_series(T0, [120] * 10))
        kinds = {f.kind for f in observation_agent.run(_ctx(store))}
        assert "observation_insulin" not in kinds

    def test_wearables_finding_when_sleep_present(self, store: SQLiteStore) -> None:
        store.insert_glucose(_series(T0, [120] * 10))
        store.insert_sleep(
            [
                SleepEvent(
                    ts_start=T0 + timedelta(hours=22),
                    ts_end=T0 + timedelta(days=1, hours=6),
                    duration_min=480.0,
                    score=80.0,
                )
            ]
        )
        findings = observation_agent.run(_ctx(store))
        assert any(f.kind == "observation_wearables" for f in findings)
        hit = next(f for f in findings if f.kind == "observation_wearables")
        assert hit.evidence["n_sleep_events"] == 1


class TestInsulinSummary:
    def test_mean_daily_bolus_in_evidence(self, store: SQLiteStore) -> None:
        store.insert_glucose(_series(T0, [120] * 10))
        store.insert_insulin(
            [
                InsulinEvent(ts=T0, kind=InsulinKind.BOLUS, units=4.0),
                InsulinEvent(ts=T0 + timedelta(days=1), kind=InsulinKind.BOLUS, units=6.0),
            ]
        )
        findings = observation_agent.run(_ctx(store))
        hit = next(f for f in findings if f.kind == "observation_insulin")
        assert hit.evidence["mean_bolus_units_per_day"] == 5.0
        assert hit.evidence["days_with_bolus"] == 2


class TestDataRequirementGating:
    def test_under_data_agent_skipped_by_registry(self, store: SQLiteStore) -> None:
        store.insert_glucose([GlucoseEvent(ts=T0, mg_dl=120)])
        coverage = store.coverage()
        gates = ColdStartReport.from_coverage(coverage)
        ctx = AgentContext(store=store, window=WINDOW, gates=gates, run_id="gated")
        skipped: list[str] = []

        def on_skip(name: str, _reasons: list[str]) -> None:
            skipped.append(name)

        registry = AgentRegistry()
        register_observation(registry)
        findings = registry.run_all(ctx, on_skip=on_skip)
        assert findings == []
        assert skipped == [AGENT_NAME]


class TestAgentContract:
    def test_name_requires_and_registration(self) -> None:
        assert observation_agent.name == AGENT_NAME
        assert observation_agent.requires.min_span_days >= 7.0
        assert observation_agent.requires.min_glucose_coverage_pct >= 70.0
        registry = AgentRegistry()
        register_observation(registry)
        assert next(iter(registry)).name == AGENT_NAME
