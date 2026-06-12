"""Tests for the Discovery agent — the LLM curiosity loop.

Two paths are covered: the deterministic fallback (no model) and the full
reasoning loop driven by a scripted fake model, so the plan -> judge ->
claim/wonder -> guard machinery runs end to end without an API key.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.agents.discovery import DiscoveryAgent
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.models import GlucoseEvent, HypothesisStatus, SleepEvent
from dexta_intelligence.store import SQLiteStore

if TYPE_CHECKING:
    from collections.abc import Sequence

_WINDOW_DAYS = 40
_END = datetime(2026, 6, 1, tzinfo=UTC)
_START = _END - timedelta(days=_WINDOW_DAYS)


# ── scripted fake model ──────────────────────────────────────────────────────


@dataclass
class _Reply:
    content: str


class _FakeModel:
    """Returns queued JSON strings in order; records every prompt it saw."""

    def __init__(self, replies: Sequence[dict[str, Any]]) -> None:
        self._replies = [json.dumps(r) for r in replies]
        self.prompts: list[str] = []

    def invoke(self, messages: list[dict[str, str]]) -> _Reply:
        self.prompts.append(messages[-1]["content"])
        if self._replies:
            return _Reply(self._replies.pop(0))
        return _Reply("{}")


# ── fixtures ─────────────────────────────────────────────────────────────────


def _store_with_dawn_effect() -> SQLiteStore:
    """Planted: 03-07h glucose ~65 mg/dL higher than 11-15h, with realistic noise."""
    store = SQLiteStore(":memory:")
    store.migrate()
    rng = random.Random(2026)
    glucose: list[GlucoseEvent] = []
    sleep: list[SleepEvent] = []
    for day in range(_WINDOW_DAYS):
        base = _START + timedelta(days=day)
        for hour, mg in ((3, 185), (4, 190), (5, 188), (6, 186), (12, 120), (13, 122), (14, 118)):
            for minute in (0, 15, 30, 45):
                ts = base.replace(hour=hour, minute=minute)
                jittered = mg + rng.randint(-9, 9)  # real CGM is never flat
                glucose.append(GlucoseEvent(ts=ts, mg_dl=jittered))
        night_start = base.replace(hour=23)
        sleep.append(
            SleepEvent(
                ts_start=night_start,
                ts_end=night_start + timedelta(hours=7),
                duration_min=420,
                score=80.0 if day % 2 else 50.0,
            )
        )
    store.insert_glucose(glucose)
    store.insert_sleep(sleep)
    return store


def _ctx(store: SQLiteStore) -> AgentContext:
    coverage = store.coverage()
    return AgentContext(
        store=store,
        window=(_START.date(), _END.date()),
        gates=ColdStartReport.from_coverage(coverage),
        run_id="test-run",
    )


# ── deterministic fallback (no model) ────────────────────────────────────────


def test_fallback_sweep_finds_planted_dawn_effect() -> None:
    store = _store_with_dawn_effect()
    findings = DiscoveryAgent().run(_ctx(store))

    tod = [f for f in findings if f.kind == "discovery_tod_compare"]
    assert tod, "fallback sweep should surface the planted time-of-day effect"
    finding = tod[0]
    assert finding.agent == "discovery"
    assert finding.stats.effect_size is not None and finding.stats.effect_size > 40
    assert finding.stats.q_fdr is not None
    # every cited number traces to evidence (deterministic headline)
    assert finding.evidence["tool"] == "tod_compare"


def test_agent_requires_minimum_span() -> None:
    store = SQLiteStore(":memory:")
    store.migrate()
    store.insert_glucose(
        [GlucoseEvent(ts=_START + timedelta(hours=i), mg_dl=120) for i in range(48)]
    )
    ctx = _ctx(store)
    reasons = DiscoveryAgent().requires.unmet_reasons(ctx.gates)
    assert reasons, "21-day span requirement should be unmet on ~2 days of data"


# ── full reasoning loop (scripted model) ─────────────────────────────────────


def test_llm_loop_plans_judges_and_claims() -> None:
    store = _store_with_dawn_effect()
    model = _FakeModel(
        [
            {  # plan
                "hypotheses": [
                    {
                        "id": "h1",
                        "claim": "Dawn glucose runs higher than midday.",
                        "tool": "tod_compare",
                        "args": {"hours_a": [3, 7], "hours_b": [11, 15]},
                    }
                ]
            },
            {"verdict": "claim", "reason": "large separation"},  # judge
            {"headline": "Mean glucose is higher in the 03-07h window than 11-15h."},  # write
        ]
    )
    agent = DiscoveryAgent(model=model)  # type: ignore[arg-type]
    findings = agent.run(_ctx(store))

    assert len(findings) == 1
    assert "03-07h" in findings[0].headline
    # the planner actually saw the memory + open-question digests
    assert "DATA AVAILABLE" in model.prompts[0]
    assert "QUESTIONS YOU BANKED" in model.prompts[0]


def test_guard_rejects_fabricated_headline_and_falls_back() -> None:
    store = _store_with_dawn_effect()
    model = _FakeModel(
        [
            {
                "hypotheses": [
                    {
                        "id": "h1",
                        "claim": "Dawn glucose runs higher than midday.",
                        "tool": "tod_compare",
                        "args": {"hours_a": [3, 7], "hours_b": [11, 15]},
                    }
                ]
            },
            {"verdict": "claim", "reason": "clear effect"},
            {"headline": "Glucose spikes a wild 999 mg/dL every single morning."},  # fabricated
        ]
    )
    findings = DiscoveryAgent(model=model).run(_ctx(store))  # type: ignore[arg-type]
    assert len(findings) == 1
    assert "999" not in findings[0].headline  # guard caught it, deterministic seed used


def test_wonder_verdict_banks_open_hypothesis() -> None:
    store = _store_with_dawn_effect()
    model = _FakeModel(
        [
            {
                "hypotheses": [
                    {
                        "id": "h1",
                        "claim": "Maybe weekends differ.",
                        "tool": "tod_compare",
                        "args": {"hours_a": [3, 7], "hours_b": [11, 15]},
                    }
                ]
            },
            {"verdict": "wonder", "reason": "suggestive but want more data"},
        ]
    )
    findings = DiscoveryAgent(model=model).run(_ctx(store))  # type: ignore[arg-type]
    assert findings == []
    open_hyps = store.get_hypotheses(status=HypothesisStatus.OPEN.value)
    assert len(open_hyps) == 1
    assert "Maybe weekends differ." in open_hyps[0].statement
