"""Phase 2 - banked hypotheses seed and steer the live investigation loop.

Prior open hypotheses (the curiosity backlog the investigator banks) re-enter a
new orchestrator run as live competing hypotheses, and they reach the model in
the first-turn prompt so they actually steer probing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from tests.golden import make_store

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.agents.investigation import (
    HypothesisStatus,
    seed_belief_from_store,
)
from dexta_intelligence.agents.orchestrator import OrchestratorAgent
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.models import Hypothesis as StoredHypothesis
from dexta_intelligence.models import HypothesisStatus as StoredStatus

_WINDOW = (date(2025, 12, 15), date(2026, 3, 15))


def _ctx(name: str) -> AgentContext:
    store = make_store(name)
    return AgentContext(
        store=store,
        window=_WINDOW,
        gates=ColdStartReport.from_coverage(store.coverage()),
        run_id="ledger-test",
    )


def test_seed_is_empty_when_store_has_no_open_hypotheses() -> None:
    belief = seed_belief_from_store(_ctx("late_bolus"))
    assert belief.hypotheses == {}


def test_seed_pulls_open_hypotheses_with_stable_ids() -> None:
    ctx = _ctx("late_bolus")
    hid = ctx.store.insert_hypothesis(
        StoredHypothesis(statement="dawn phenomenon drives the 6am rise")
    )
    belief = seed_belief_from_store(ctx)
    assert belief.hypotheses[f"stored-{hid}"].statement == "dawn phenomenon drives the 6am rise"
    assert belief.hypotheses[f"stored-{hid}"].status is HypothesisStatus.OPEN


def test_seed_respects_limit() -> None:
    ctx = _ctx("late_bolus")
    for i in range(7):
        ctx.store.insert_hypothesis(StoredHypothesis(statement=f"hypothesis {i}"))
    belief = seed_belief_from_store(ctx, limit=3)
    assert len(belief.hypotheses) == 3


def test_seed_ignores_resolved_hypotheses() -> None:
    ctx = _ctx("late_bolus")
    ctx.store.insert_hypothesis(
        StoredHypothesis(statement="settled question", status=StoredStatus.SUPPORTED)
    )
    assert seed_belief_from_store(ctx).hypotheses == {}


def test_seed_selects_open_from_a_mixed_store() -> None:
    ctx = _ctx("late_bolus")
    ctx.store.insert_hypothesis(StoredHypothesis(statement="still open"))
    ctx.store.insert_hypothesis(
        StoredHypothesis(statement="already settled", status=StoredStatus.SUPPORTED)
    )
    statements = [h.statement for h in seed_belief_from_store(ctx).hypotheses.values()]
    assert statements == ["still open"]


def test_seed_dedupes_identical_statements() -> None:
    ctx = _ctx("late_bolus")
    ctx.store.insert_hypothesis(StoredHypothesis(statement="dawn rise"))
    ctx.store.insert_hypothesis(StoredHypothesis(statement="dawn rise"))
    assert len(seed_belief_from_store(ctx).hypotheses) == 1


class _StubStore:
    def __init__(self, hypotheses: list[StoredHypothesis]) -> None:
        self._hypotheses = hypotheses

    def get_hypotheses(self, *, status: str | None = None) -> list[StoredHypothesis]:
        if status is None:
            return list(self._hypotheses)
        return [h for h in self._hypotheses if h.status.value == status]


def test_seed_mints_fresh_id_for_unpersisted_hypothesis() -> None:
    ctx = AgentContext(
        store=_StubStore([StoredHypothesis(statement="not yet saved")]),  # type: ignore[arg-type]
        window=_WINDOW,
        gates=ColdStartReport.from_coverage(make_store("late_bolus").coverage()),
        run_id="stub",
    )
    belief = seed_belief_from_store(ctx)
    assert list(belief.hypotheses) == ["h1"]
    assert belief.hypotheses["h1"].statement == "not yet saved"


def test_model_can_refute_a_seeded_hypothesis_in_place() -> None:
    ctx = _ctx("late_bolus")
    hid = ctx.store.insert_hypothesis(StoredHypothesis(statement="basal too low"))
    belief = seed_belief_from_store(ctx)
    key = f"stored-{hid}"
    belief.tool().fn({"hypotheses": [{"id": key, "status": "refuted"}]})
    assert len(belief.hypotheses) == 1  # no duplicate minted
    assert belief.hypotheses[key].status is HypothesisStatus.REFUTED


@dataclass
class _CapturingModel:
    """Records the system prompt it was handed, then answers."""

    seen_system: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    def bind_tools(self, _schemas: list[dict[str, Any]]) -> _CapturingModel:
        return self

    def invoke(self, messages: list[Any]) -> Any:
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "system":
                self.seen_system = str(message.get("content", ""))
                break
        return type("_Msg", (), {"content": "Noted.", "tool_calls": []})()


def test_seeded_hypotheses_reach_the_model_prompt() -> None:
    ctx = _ctx("late_bolus")
    ctx.store.insert_hypothesis(StoredHypothesis(statement="basal is too low overnight"))
    model = _CapturingModel()
    OrchestratorAgent(model=model).ask(ctx, "why am I high in the morning?")
    assert "Open hypotheses carried from prior analysis" in model.seen_system
    assert "basal is too low overnight" in model.seen_system


def test_no_seed_means_no_carried_section() -> None:
    model = _CapturingModel()
    OrchestratorAgent(model=model).ask(_ctx("late_bolus"), "how am I doing?")
    assert "Open hypotheses carried from prior analysis" not in model.seen_system
    assert "Maintain a working belief state" in model.seen_system
