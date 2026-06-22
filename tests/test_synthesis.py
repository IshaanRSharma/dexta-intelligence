"""Phase 6 - the deliberate synthesis pass.

At the end of an investigation, synthesize() fuses the belief state and tool
trace into one grounded explanation: the leading hypothesis, the alternatives
ruled out, the evidence (re-audited against the tool pool so untraceable figures
are dropped), the cross-modal probes, and the open gaps. The orchestrator
attaches it to a faithful answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from tests.golden import make_store

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.agents.investigation import (
    BeliefState,
    Hypothesis,
    HypothesisStatus,
    synthesize,
)
from dexta_intelligence.agents.orchestrator import OrchestratorAgent
from dexta_intelligence.agents.reason import ToolCall
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.guard.treatment_gate import SAFE_SENTENCE


def _step(name: str) -> ToolCall:
    return ToolCall(name=name, args={}, ok=True, result={})


def _belief() -> BeliefState:
    state = BeliefState()
    state.hypotheses = {
        "h1": Hypothesis("h1", "late bolus", HypothesisStatus.SUPPORTED),
        "h2": Hypothesis("h2", "basal drift", HypothesisStatus.REFUTED),
        "h3": Hypothesis("h3", "dawn effect", HypothesisStatus.OPEN),
    }
    state.confidence = 0.8
    state.summary = "the late bolus explains the spike"
    state.gaps = ["no meal log on weekends"]
    return state


def test_leading_is_the_supported_hypothesis() -> None:
    synth = synthesize(_belief(), [], {})
    assert synth.leading == "late bolus"
    assert synth.confidence == 0.8
    assert synth.explanation == "the late bolus explains the spike"


def test_leading_falls_back_to_open_when_none_supported() -> None:
    state = BeliefState()
    state.hypotheses = {"h1": Hypothesis("h1", "dawn effect", HypothesisStatus.OPEN)}
    assert synthesize(state, [], {}).leading == "dawn effect"


def test_ruled_out_lists_refuted_hypotheses() -> None:
    assert synthesize(_belief(), [], {}).ruled_out == ("basal drift",)


def test_explanation_falls_back_to_leading_without_summary() -> None:
    state = BeliefState()
    state.hypotheses = {"h1": Hypothesis("h1", "late bolus", HypothesisStatus.SUPPORTED)}
    assert synthesize(state, [], {}).explanation == "late bolus"


def test_evidence_is_gated_against_the_pool() -> None:
    state = BeliefState()
    state.evidence = ["glucose peaked at 246 mg/dL", "an untraceable 999 figure", "no numbers here"]
    pool = {"find_spikes_0": {"peak_mg_dl": 246}}
    grounded = synthesize(state, [], pool).evidence
    assert "glucose peaked at 246 mg/dL" in grounded
    assert "no numbers here" in grounded
    assert "an untraceable 999 figure" not in grounded  # dropped: 999 not in the pool


def test_text_fields_drop_untraceable_numbers() -> None:
    state = BeliefState()
    state.hypotheses = {"h1": Hypothesis("h1", "late bolus", HypothesisStatus.SUPPORTED)}
    state.summary = "your average ran 142 mg/dL this week"  # 142 is in no pool
    synth = synthesize(state, [], {})
    assert "142" not in synth.explanation
    assert synth.explanation == "late bolus"  # falls back to the (number-free) leading


def test_probes_exclude_scaffolding_and_dedupe() -> None:
    steps = [
        _step("update_belief"),
        _step("find_spikes"),
        _step("find_spikes"),
        _step("request_context"),
        _step("get_cob"),
    ]
    assert synthesize(_belief(), steps, {}).probes == ("find_spikes", "get_cob")


def test_gaps_pass_through() -> None:
    assert synthesize(_belief(), [], {}).gaps == ("no meal log on weekends",)


def test_as_dict_is_serializable_shape() -> None:
    d = synthesize(_belief(), [_step("find_spikes")], {}).as_dict()
    assert d["leading"] == "late bolus"
    assert d["ruled_out"] == ["basal drift"]
    assert d["probes"] == ["find_spikes"]


@dataclass
class _AIMessage:
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class _FakeToolModel:
    def __init__(self, turns: list[Any]) -> None:
        self._turns = turns

    def bind_tools(self, _schemas: list[dict[str, Any]]) -> _FakeToolModel:
        return self

    def invoke(self, _messages: list[Any]) -> _AIMessage:
        turn = self._turns.pop(0) if self._turns else "Nothing more."
        return _AIMessage(content=turn) if isinstance(turn, str) else _AIMessage(tool_calls=turn)


def _ctx() -> AgentContext:
    store = make_store("late_bolus")
    return AgentContext(
        store=store,
        window=(date(2025, 12, 15), date(2026, 3, 15)),
        gates=ColdStartReport.from_coverage(store.coverage()),
        run_id="synth-test",
    )


def test_orchestrator_attaches_synthesis_to_a_faithful_answer() -> None:
    update = [
        {
            "name": "update_belief",
            "args": {
                "hypotheses": [{"id": "h1", "statement": "late bolus", "status": "supported"}],
                "confidence": 0.8,
                "summary": "the late bolus explains it",
            },
            "id": "1",
        }
    ]
    model = _FakeToolModel([update, "Your mornings look steady overall."])
    answer = OrchestratorAgent(model=model).ask(_ctx(), "how am I doing this month?")
    assert answer.synthesis is not None
    assert answer.synthesis.leading == "late bolus"
    assert answer.synthesis.confidence == 0.8


def test_no_synthesis_when_the_answer_is_not_a_clean_conclusion() -> None:
    # model errors out (no tools, empty) -> fallback answer, no synthesis attached.
    model = _FakeToolModel([""])
    answer = OrchestratorAgent(model=model).ask(_ctx(), "how am I doing this month?")
    assert answer.synthesis is None


def test_no_synthesis_when_the_answer_is_unfaithful() -> None:
    # an answer with an untraceable number is flagged unfaithful; no synthesis attaches.
    model = _FakeToolModel(["Your average was 142 mg/dL this month."])
    answer = OrchestratorAgent(model=model).ask(_ctx(), "how am I doing this month?")
    assert answer.faithful is False
    assert answer.synthesis is None


def test_no_synthesis_when_the_answer_fades_to_safety() -> None:
    # a cause claim with no investigation fades to the safe sentence; nothing to synthesize.
    model = _FakeToolModel(["It was your dinner.", "Still a cause claim, still no tools."])
    answer = OrchestratorAgent(model=model).ask(_ctx(), "Why did I spike on March 14?")
    assert answer.text == SAFE_SENTENCE
    assert answer.synthesis is None
