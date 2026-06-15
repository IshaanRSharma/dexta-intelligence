"""The fade gate (M1/M2/M11) and the seeker's hard limits (M12).

Gate mechanics run through the real ChatAgent with a scripted tool-calling
model over the golden late_bolus dataset — the same path production takes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from tests.golden import make_store

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.agents.chat import ChatAgent
from dexta_intelligence.agents.reason import ToolCall
from dexta_intelligence.agents.seeker import GoalSeekingAgent
from dexta_intelligence.coldstart import CapabilitySet, ColdStartReport
from dexta_intelligence.guard.treatment_gate import (
    NO_TREATMENT_DISCLAIMER,
    SAFE_SENTENCE,
    assess_trace,
    is_cause_question,
)
from dexta_intelligence.investigations import spike as workflow

_WINDOW = (date(2025, 12, 15), date(2026, 3, 15))
_CAUSE_Q = "Why did I spike on March 14?"


def _ctx(name: str) -> AgentContext:
    store = make_store(name)
    return AgentContext(
        store=store,
        window=_WINDOW,
        gates=ColdStartReport.from_coverage(store.coverage()),
        run_id="gate-test",
    )


@dataclass
class _AIMessage:
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class _FakeToolModel:
    def __init__(self, turns: list[Any]) -> None:
        self._turns = turns
        self.invocations = 0

    def bind_tools(self, _schemas: list[dict[str, Any]]) -> _FakeToolModel:
        return self

    def invoke(self, _messages: list[Any]) -> _AIMessage:
        self.invocations += 1
        turn = self._turns.pop(0) if self._turns else "Nothing more."
        if isinstance(turn, str):
            return _AIMessage(content=turn)
        return _AIMessage(tool_calls=list(turn))


def _ok_step(name: str) -> ToolCall:
    return ToolCall(name=name, args={}, ok=True, result={})


_FULL_CAPS = CapabilitySet(
    has_insulin=True, has_meals=True, has_sleep=False, has_activity=False
)
_NO_INSULIN_CAPS = CapabilitySet(
    has_insulin=False, has_meals=False, has_sleep=False, has_activity=False
)
_FULL_PATH = [_ok_step(n) for n in
              ("zoom_event", "get_carb_entries", "get_boluses", "get_basal_timeline")]


# ── classification + verdicts (unit) ──────────────────────────────────────────


def test_cause_questions_classified() -> None:
    for q in (
        "Why did I spike on March 14?",
        "What caused this high?",
        "Explain last night's spike",
        "why do I go high after dinner",
    ):
        assert is_cause_question(q), q
    for q in ("Compare weekends to weekdays", "How was my TIR in March?"):
        assert not is_cause_question(q), q


def test_full_inspection_is_compliant() -> None:
    report = assess_trace(_CAUSE_Q, _FULL_PATH, _FULL_CAPS)
    assert report.applies and report.compliant and not report.missing


def test_uninspected_cause_claim_is_non_compliant() -> None:
    report = assess_trace(_CAUSE_Q, [], _FULL_CAPS)
    assert report.applies and not report.compliant
    assert set(report.missing) == {
        "zoom_event", "get_carb_entries", "get_boluses", "get_basal_timeline"
    }
    assert "get_boluses" in report.retry_hint


def test_find_spikes_satisfies_the_zoom_requirement() -> None:
    steps = [_ok_step(n) for n in
             ("find_spikes", "get_carb_entries", "get_boluses", "get_basal_timeline")]
    assert assess_trace(_CAUSE_Q, steps, _FULL_CAPS).compliant


def test_failed_tool_calls_do_not_count() -> None:
    steps = list(_FULL_PATH[:-1])
    steps.append(ToolCall(name="get_basal_timeline", args={}, ok=False, result={"error": "x"}))
    report = assess_trace(_CAUSE_Q, steps, _FULL_CAPS)
    assert not report.compliant and report.missing == ("get_basal_timeline",)


def test_requirements_adapt_to_capabilities() -> None:
    no_meals = CapabilitySet(
        has_insulin=True, has_meals=False, has_sleep=False, has_activity=False
    )
    steps = [_ok_step(n) for n in ("zoom_event", "get_boluses", "get_basal_timeline")]
    assert assess_trace(_CAUSE_Q, steps, no_meals).compliant


def test_no_insulin_is_compliant_but_flagged() -> None:
    report = assess_trace(_CAUSE_Q, [], _NO_INSULIN_CAPS)
    assert report.applies and report.compliant and not report.insulin_available


def test_research_only_is_non_compliant() -> None:
    report = assess_trace(_CAUSE_Q, [_ok_step("search_evidence")], _FULL_CAPS)
    assert report.research_only and not report.compliant
    assert "data tools first" in report.retry_hint


def test_non_cause_question_is_exempt() -> None:
    report = assess_trace("Compare weekends to weekdays", [], _FULL_CAPS)
    assert not report.applies and report.compliant


def test_gate_sentences_stay_in_sync_with_the_workflow() -> None:
    assert SAFE_SENTENCE == workflow.INSUFFICIENT_SENTENCE
    assert NO_TREATMENT_DISCLAIMER == workflow.NO_TREATMENT_DISCLAIMER


# ── fade behavior through the real chat agent (M2) ────────────────────────────


def test_persistent_non_compliance_fades_to_safe_sentence() -> None:
    model = _FakeToolModel(
        [
            "The spike was caused by your dinner.",   # round 1: cause claim, no tools
            "Still my final answer, no tools used.",  # the one retry: still no tools
        ]
    )
    answer = ChatAgent(model=model).ask(_ctx("late_bolus"), _CAUSE_Q)
    assert answer.text == SAFE_SENTENCE
    assert answer.stopped_reason == "treatment_gate"


def test_retry_with_hint_recovers_and_keeps_the_answer() -> None:
    inspect_calls = [
        {"name": "zoom_event", "args": {"timestamp": "2026-03-14T20:42:00+00:00"}, "id": "1"},
        {"name": "get_carb_entries", "args": {}, "id": "2"},
        {"name": "get_boluses", "args": {}, "id": "3"},
        {"name": "get_basal_timeline", "args": {}, "id": "4"},
    ]
    good = "The pattern is more consistent with late meal insulin context than basal drift."
    model = _FakeToolModel(
        [
            "The spike was caused by your dinner.",  # round 1: non-compliant
            inspect_calls,                           # retry: inspects everything
            good,                                    # retry answer
        ]
    )
    answer = ChatAgent(model=model).ask(_ctx("late_bolus"), _CAUSE_Q)
    assert answer.text == good
    assert {"zoom_event", "get_carb_entries", "get_boluses", "get_basal_timeline"} <= set(
        answer.tools_used
    )


def test_exactly_one_retry_is_allowed() -> None:
    model = _FakeToolModel(["Cause claim.", "Another cause claim.", "A third."])
    ChatAgent(model=model).ask(_ctx("late_bolus"), _CAUSE_Q)
    # initial loop + one retry — the third scripted turn is never consumed.
    assert model.invocations == 2


def test_no_insulin_answer_carries_the_disclaimer() -> None:
    model = _FakeToolModel(["Glucose rose after dinner — the shape suggests a meal."])
    answer = ChatAgent(model=model).ask(_ctx("no_insulin"), _CAUSE_Q)
    assert NO_TREATMENT_DISCLAIMER in answer.text
    assert model.invocations == 1  # no retry needed; disclaimer is appended


def test_non_cause_question_passes_untouched() -> None:
    model = _FakeToolModel(["Weekends look similar to weekdays in your data."])
    answer = ChatAgent(model=model).ask(_ctx("late_bolus"), "Compare weekends to weekdays")
    assert answer.text.startswith("Weekends look similar")
    assert model.invocations == 1


# ── seeker hard limits (M12) ──────────────────────────────────────────────────

_UNSATISFIED = (
    '{"satisfied": false, "missing": "zoom the spike", '
    '"next_tool_hint": "zoom_event", "reason": "never zoomed"}'
)


def test_seeker_defaults_to_two_rounds() -> None:
    assert GoalSeekingAgent.__dataclass_fields__["max_rounds"].default == 2


def test_seeker_never_exceeds_max_rounds() -> None:
    set_window = [{"name": "set_window",
                   "args": {"start": "2026-03-01", "end": "2026-03-15"}, "id": "1"}]
    model = _FakeToolModel(
        [
            set_window, "Partial answer one.", _UNSATISFIED,   # round 1 + reflect
            set_window, "Partial answer two.", _UNSATISFIED,   # round 2 + reflect
            "A third round would start here.",
        ]
    )
    GoalSeekingAgent(model=model).pursue(_ctx("late_bolus"), "How was March overall?")
    assert model.invocations == 6  # 2 rounds x (tool turn + answer + reflection)


def test_seeker_stops_when_no_new_tool_is_called() -> None:
    set_window = [{"name": "set_window",
                   "args": {"start": "2026-03-01", "end": "2026-03-15"}, "id": "1"}]
    model = _FakeToolModel(
        [
            set_window, "Partial one.", _UNSATISFIED,
            set_window, "Partial two.", _UNSATISFIED,  # same tool again → stop
            set_window, "Partial three.", _UNSATISFIED,
        ]
    )
    GoalSeekingAgent(model=model, max_rounds=3).pursue(_ctx("late_bolus"), "How was March?")
    assert model.invocations == 6  # round 3 never runs


def test_seeker_stops_when_reflection_names_no_real_tool() -> None:
    bad_reflection = (
        '{"satisfied": false, "missing": "vibes", '
        '"next_tool_hint": "consult_the_oracle", "reason": "?"}'
    )
    model = _FakeToolModel(
        [
            [{"name": "set_window",
              "args": {"start": "2026-03-01", "end": "2026-03-15"}, "id": "1"}],
            "Partial answer.",
            bad_reflection,
            "Round two would start here.",
        ]
    )
    GoalSeekingAgent(model=model, max_rounds=3).pursue(_ctx("late_bolus"), "How was March?")
    assert model.invocations == 3  # one round + its reflection, then stop
