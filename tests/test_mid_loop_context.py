"""Phase 5 - mid-loop active context acquisition.

When a gap blocks discrimination, the agent can call request_context for the
moment it cannot explain and surface a precise, dosing-gated logging request
instead of guessing. It reuses the batch detector's proximity rule, and finds
nothing to ask when a meal or note is already logged nearby.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

from tests.golden import make_store

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.agents.brief import _ADVICE_RE
from dexta_intelligence.agents.context_acquisition import (
    context_request_at,
    request_context_tool,
)
from dexta_intelligence.agents.orchestrator import _BELIEF_DIRECTIVE, OrchestratorAgent
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.models import ManualEvent, MealEvent

if TYPE_CHECKING:
    from dexta_intelligence.store import SQLiteStore

_MOMENT = "2026-02-01T08:30"
_MOMENT_TS = datetime(2026, 2, 1, 8, 30, tzinfo=UTC)


def _ctx(store: SQLiteStore) -> AgentContext:
    return AgentContext(
        store=store,
        window=(date(2026, 1, 1), date(2026, 3, 1)),
        gates=ColdStartReport.from_coverage(store.coverage()),
        run_id="ctx-test",
    )


def test_request_when_no_meal_or_note_nearby() -> None:
    ctx = _ctx(make_store("late_bolus"))
    req = context_request_at(ctx, _MOMENT)
    assert req is not None
    assert req.kind == "blocked_discrimination"
    assert req.event_ts == _MOMENT_TS


def test_no_request_when_a_meal_is_logged_nearby() -> None:
    store = make_store("late_bolus")
    store.insert_meals([MealEvent(ts=_MOMENT_TS, carbs_g=45.0)])
    assert context_request_at(_ctx(store), _MOMENT) is None


def test_no_request_when_a_note_is_logged_nearby() -> None:
    store = make_store("late_bolus")
    store.add_manual_event(
        ManualEvent(
            event_type="stress",
            event_ts=_MOMENT_TS,
            description="big presentation",
            created_at=_MOMENT_TS,
        )
    )
    assert context_request_at(_ctx(store), _MOMENT) is None


def test_unparseable_moment_yields_no_request() -> None:
    assert context_request_at(_ctx(make_store("late_bolus")), "last tuesday") is None


def test_tz_aware_moment_is_normalized_to_utc() -> None:
    # 08:30+05:00 is 03:30 UTC; the request's event_ts must be UTC-normalized.
    req = context_request_at(_ctx(make_store("late_bolus")), "2026-02-01T08:30+05:00")
    assert req is not None
    assert req.event_ts == datetime(2026, 2, 1, 3, 30, tzinfo=UTC)


def test_request_question_passes_the_dosing_gate() -> None:
    req = context_request_at(_ctx(make_store("late_bolus")), _MOMENT)
    assert req is not None
    assert not _ADVICE_RE.search(req.question)


def test_tool_returns_missing_when_blind() -> None:
    tool = request_context_tool(_ctx(make_store("late_bolus")))
    result, numbers = tool.fn({"when": _MOMENT})
    assert result["context_missing"] is True
    assert "log" in result["ask_user"].lower()
    assert numbers == {}  # the ask is meta, never the faithfulness evidence pool


def test_tool_returns_present_when_context_logged() -> None:
    store = make_store("late_bolus")
    store.insert_meals([MealEvent(ts=_MOMENT_TS, carbs_g=30.0)])
    tool = request_context_tool(_ctx(store))
    result, _ = tool.fn({"when": _MOMENT})
    assert result.get("context_present") is True


def test_tool_errors_on_missing_when() -> None:
    tool = request_context_tool(_ctx(make_store("late_bolus")))
    result, _ = tool.fn({})
    assert "error" in result


def test_orchestrator_belt_includes_request_context() -> None:
    class _Model:
        def __init__(self) -> None:
            self.seen_tools: list[str] = []

        def bind_tools(self, schemas: list[dict[str, object]]) -> _Model:
            self.seen_tools = [s["function"]["name"] for s in schemas]  # type: ignore[index]
            return self

        def invoke(self, _messages: list[object]) -> object:
            return type("_Msg", (), {"content": "ok", "tool_calls": []})()

    model = _Model()
    OrchestratorAgent(model=model).ask(_ctx(make_store("late_bolus")), "how am I doing?")
    assert "request_context" in model.seen_tools


def test_directive_mentions_request_context() -> None:
    assert "request_context" in _BELIEF_DIRECTIVE
