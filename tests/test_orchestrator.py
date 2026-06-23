"""Orchestrator - the LLM is the top-level decider over workflows-as-tools.

Proves: the model can choose a whole investigation workflow; the treatment gate
credits that composite (no fade); numbers from the workflow bundle stay
guard-traceable; and the model can still chain granular tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from tests.golden import make_store

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.agents.orchestrator import OrchestratorAgent, workflow_tool_specs
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.guard.treatment_gate import SAFE_SENTENCE

_WINDOW = (date(2025, 12, 15), date(2026, 3, 15))
_CAUSE_Q = "Why did I spike on March 14?"


def _ctx(name: str) -> AgentContext:
    store = make_store(name)
    return AgentContext(
        store=store,
        window=_WINDOW,
        gates=ColdStartReport.from_coverage(store.coverage()),
        run_id="orch-test",
    )


@dataclass
class _AIMessage:
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class _FakeToolModel:
    def __init__(self, turns: list[Any]) -> None:
        self._turns = turns
        self.seen_tools: list[str] = []
        self.invocations = 0

    def bind_tools(self, schemas: list[dict[str, Any]]) -> _FakeToolModel:
        self.seen_tools = [s["function"]["name"] for s in schemas]
        return self

    def invoke(self, _messages: list[Any]) -> _AIMessage:
        self.invocations += 1
        turn = self._turns.pop(0) if self._turns else "Nothing more."
        if isinstance(turn, str):
            return _AIMessage(content=turn)
        return _AIMessage(tool_calls=list(turn))


def _spike_call() -> dict[str, Any]:
    return {"name": "investigate_spike", "args": {"when": "2026-03-14"}, "id": "w1"}


# ── the workflow is in the belt and the model can choose it ────────────────────


def test_workflow_tool_is_exposed() -> None:
    specs = workflow_tool_specs(_ctx("late_bolus"), target_low=70, target_high=180)
    assert "investigate_spike" in {s.name for s in specs}


def test_orchestrator_belt_includes_workflows_and_instruments() -> None:
    model = _FakeToolModel(["nothing"])
    OrchestratorAgent(model=model).ask(_ctx("late_bolus"), "hello")
    # The model sees both whole workflows and granular instruments - it chooses.
    assert "investigate_spike" in model.seen_tools
    assert "zoom_event" in model.seen_tools
    assert "recall" in model.seen_tools


def test_model_chooses_workflow_then_answers_faithfully() -> None:
    model = _FakeToolModel(
        [
            [_spike_call()],
            "The March 14 spike peaked at 246 mg/dL - more consistent with late "
            "meal insulin context than basal drift.",
        ]
    )
    answer = OrchestratorAgent(model=model).ask(_ctx("late_bolus"), _CAUSE_Q)
    assert "investigate_spike" in answer.tools_used
    assert answer.faithful  # 246 traces to the workflow bundle's pool
    assert "246" in answer.text


# ── the gate credits the composite workflow (no fade) ──────────────────────────


def test_workflow_call_satisfies_the_treatment_gate() -> None:
    model = _FakeToolModel(
        [
            [_spike_call()],
            "The pattern is more consistent with late/insufficient meal insulin "
            "context than basal drift.",
        ]
    )
    answer = OrchestratorAgent(model=model).ask(_ctx("late_bolus"), _CAUSE_Q)
    # investigate_spike covers zoom+carbs+boluses+basal, so the cause claim stands.
    assert answer.text != SAFE_SENTENCE
    assert "late/insufficient meal insulin" in answer.text
    assert model.invocations == 2  # no retry needed


def test_cause_claim_without_any_investigation_still_fades() -> None:
    model = _FakeToolModel(
        ["It was your dinner.", "Still no tools, still a cause claim."]
    )
    answer = OrchestratorAgent(model=model).ask(_ctx("late_bolus"), _CAUSE_Q)
    assert answer.text == SAFE_SENTENCE


# ── the model can chain granular tools after the workflow ──────────────────────


def test_model_can_chain_workflow_then_granular_tool() -> None:
    model = _FakeToolModel(
        [
            [_spike_call()],
            [{"name": "daily_series", "args": {"metric": "tir"}, "id": "g1"}],
            "March 14 looks like late meal insulin; the month's time-in-range "
            "trend gives it context.",
        ]
    )
    answer = OrchestratorAgent(model=model).ask(_ctx("late_bolus"), _CAUSE_Q)
    assert "investigate_spike" in answer.tools_used
    assert "daily_series" in answer.tools_used
    assert answer.faithful


def test_timing_context_is_on_the_agent_belt() -> None:
    # Regression: the deterministic timing_context engine must be a tool the agent
    # can actually call, not just a CLI/report (the "built but not wired" bug).
    model = _FakeToolModel(["All looks steady."])
    OrchestratorAgent(model=model).ask(_ctx("late_bolus"), "what's my dinner pattern?")
    assert "timing_context" in model.seen_tools


def test_use_belief_false_drops_the_reasoning_scaffold() -> None:
    model = _FakeToolModel(["All looks steady."])
    answer = OrchestratorAgent(model=model, use_belief=False).ask(_ctx("late_bolus"), "how am I?")
    assert "update_belief" not in model.seen_tools
    assert "request_context" not in model.seen_tools
    assert answer.synthesis is None
