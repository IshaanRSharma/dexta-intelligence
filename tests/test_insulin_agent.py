"""Tests for the Insulin agent: the LLM curiosity loop over insulin/meal tools.

Covers the deterministic fallback (no model) and the scripted-model reasoning
loop, so the plan -> judge -> claim/wonder -> guard machinery runs end to end
without an API key. Scenarios are planted in an in-memory store: big-carb meals
spike far more than small ones, and corrections are followed by lows so
``rebound_low_rate`` is positive.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.agents.insulin import InsulinAgent
from dexta_intelligence.agents.tools.toolkit import DiscoveryToolkit
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.models import (
    GlucoseEvent,
    InsulinEvent,
    InsulinKind,
    MealEvent,
)
from dexta_intelligence.store import SQLiteStore

if TYPE_CHECKING:
    from collections.abc import Sequence

_WINDOW_DAYS = 40
_END = datetime(2026, 6, 1, tzinfo=UTC)
_START = _END - timedelta(days=_WINDOW_DAYS)


# ── scripted fake model (mirrors tests/test_discovery_agent.py) ───────────────


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


def _readings(base: datetime, hour: int, value: float, rng: random.Random) -> list[GlucoseEvent]:
    out: list[GlucoseEvent] = []
    for minute in (0, 15, 30, 45):
        ts = base.replace(hour=hour, minute=minute)
        out.append(GlucoseEvent(ts=ts, mg_dl=int(value) + rng.randint(-8, 8)))
    return out


def _store_with_insulin_scenarios() -> SQLiteStore:
    """Planted per day:

    - a big-carb meal (60 g) at 08h: baseline ~100, peak ~180 (excursion ~+80);
    - a small-carb meal (15 g) at 12h: baseline ~100, peak ~120 (excursion ~+20);
    - a correction bolus at 16h with baseline ~200 then a rebound low (~60).

    Jitter keeps cohen_d finite. Background filler readings give every hour
    coverage so pre/post baselines always have readings.
    """
    store = SQLiteStore(":memory:")
    store.migrate()
    rng = random.Random(2026)

    glucose: list[GlucoseEvent] = []
    meals: list[MealEvent] = []
    insulin: list[InsulinEvent] = []

    for day in range(_WINDOW_DAYS):
        base = _START + timedelta(days=day)
        # background filler: every hour a flat-ish ~100 reading
        for hour in range(24):
            glucose.append(GlucoseEvent(ts=base.replace(hour=hour), mg_dl=100 + rng.randint(-8, 8)))

        # big-carb meal at 08h: baseline 07h ~100, peak 09h ~180
        glucose += _readings(base, 7, 100, rng)
        glucose += _readings(base, 9, 180, rng)
        meals.append(MealEvent(ts=base.replace(hour=8), carbs_g=60.0))

        # small-carb meal at 12h: baseline 11h ~100, peak 13h ~120
        glucose += _readings(base, 11, 100, rng)
        glucose += _readings(base, 13, 120, rng)
        meals.append(MealEvent(ts=base.replace(hour=12), carbs_g=15.0))

        # correction bolus at 16h: baseline 15h ~200, then 17h dips to ~60 (rebound low)
        glucose += _readings(base, 15, 200, rng)
        glucose += _readings(base, 17, 60, rng)
        insulin.append(InsulinEvent(ts=base.replace(hour=16), kind=InsulinKind.BOLUS, units=4.0))

    store.insert_glucose(glucose)
    store.insert_meals(meals)
    store.insert_insulin(insulin)
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


def test_fallback_sweep_claims_meal_excursion() -> None:
    store = _store_with_insulin_scenarios()
    findings = InsulinAgent().run(_ctx(store))

    meal = [f for f in findings if f.kind == "insulin_meal_response"]
    assert meal, "fallback sweep should surface the planted meal-size excursion"
    finding = meal[0]
    assert finding.agent == "insulin"
    assert finding.scope == "insulin"
    # bigger-carb meals peak well above smaller-carb meals (~+60 mg/dL gap)
    assert finding.stats.effect_size is not None and finding.stats.effect_size > 40
    assert finding.stats.q_fdr is not None
    assert finding.evidence["tool"] == "meal_response"
    assert finding.evidence["mean_excursion_a"] > finding.evidence["mean_excursion_b"]


def test_correction_outcome_reports_rebound_lows() -> None:
    toolkit = DiscoveryToolkit(_ctx(_store_with_insulin_scenarios()))
    result = toolkit.run("correction_outcome", {"window_min": 180})
    assert result.ok
    # every planted correction is followed by a <70 reading
    assert result.summary["rebound_low_rate"] > 0
    assert result.summary["n_boluses"] >= 16


def test_agent_requires_insulin() -> None:
    store = SQLiteStore(":memory:")
    store.migrate()
    store.insert_glucose(
        [GlucoseEvent(ts=_START + timedelta(hours=i), mg_dl=120) for i in range(24 * 20)]
    )
    ctx = _ctx(store)
    reasons = InsulinAgent().requires.unmet_reasons(ctx.gates)
    assert reasons, "needs_insulin requirement should be unmet with no insulin logged"


# ── full reasoning loop (scripted model) ─────────────────────────────────────


def test_llm_loop_plans_judges_and_claims() -> None:
    store = _store_with_insulin_scenarios()
    model = _FakeModel(
        [
            {
                "hypotheses": [
                    {
                        "id": "h1",
                        "claim": "Bigger-carb meals drive larger excursions.",
                        "tool": "meal_response",
                        "args": {"window_min": 120},
                    }
                ]
            },
            {"verdict": "claim", "reason": "large separation"},
            {"headline": "Bigger-carb meals peak 60 mg/dL further above baseline than smaller."},
        ]
    )
    agent = InsulinAgent(model=model)
    findings = agent.run(_ctx(store))

    assert len(findings) == 1
    assert findings[0].kind == "insulin_meal_response"
    assert "DATA AVAILABLE" in model.prompts[0]
    assert "QUESTIONS YOU BANKED" in model.prompts[0]


def test_guard_rejects_fabricated_headline_and_falls_back() -> None:
    store = _store_with_insulin_scenarios()
    model = _FakeModel(
        [
            {
                "hypotheses": [
                    {
                        "id": "h1",
                        "claim": "Bigger-carb meals drive larger excursions.",
                        "tool": "meal_response",
                        "args": {"window_min": 120},
                    }
                ]
            },
            {"verdict": "claim", "reason": "clear effect"},
            {"headline": "Meals spike a wild 999 mg/dL every single time, guaranteed."},
        ]
    )
    findings = InsulinAgent(model=model).run(_ctx(store))
    assert len(findings) == 1
    assert "999" not in findings[0].headline  # guard caught it, deterministic seed used
