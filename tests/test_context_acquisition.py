"""Active context acquisition: detect unexplained spikes, ask the user to log them.

The thesis under test: determinism detects a gap (a spike with no meal and no
note logged nearby) and dexta asks for context; it never fabricates the cause.
A logged meal or note within the proximity window suppresses the request. Every
question is observation-only (passes the dosing-advice gate), and a model that
tries to inject dosing advice cannot leak past it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.agents.brief import _ADVICE_RE
from dexta_intelligence.agents.context_acquisition import ContextAcquisitionAgent
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.models import GlucoseEvent, ManualEvent, MealEvent
from dexta_intelligence.store import SQLiteStore

_END = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


def _ctx(store: SQLiteStore) -> AgentContext:
    coverage = store.coverage()
    return AgentContext(
        store=store,
        window=(_END.date() - timedelta(days=30), _END.date()),
        gates=ColdStartReport.from_coverage(coverage),
        run_id="test-run",
        timezone="UTC",
    )


def _store(glucose: list[GlucoseEvent]) -> SQLiteStore:
    store = SQLiteStore(":memory:")
    store.migrate()
    store.insert_glucose(glucose)
    return store


def _flat_window(value: int, *, hours: int = 24, end: datetime = _END) -> list[GlucoseEvent]:
    start = end - timedelta(hours=hours)
    n = hours * 12
    return [GlucoseEvent(ts=start + timedelta(minutes=5 * i), mg_dl=value) for i in range(n)]


def _plant_spike(glucose: list[GlucoseEvent], at: int, *, n: int = 8, value: int = 300) -> datetime:
    """Set ``n`` consecutive readings from index ``at`` to ``value``; return the peak ts."""
    for i in range(at, at + n):
        glucose[i] = GlucoseEvent(ts=glucose[i].ts, mg_dl=value)
    return glucose[at].ts


# ── the gap fires ──────────────────────────────────────────────────────────────


def test_unexplained_spike_produces_one_request() -> None:
    glucose = _flat_window(110)
    peak_ts = _plant_spike(glucose, 100)
    requests = ContextAcquisitionAgent().build(_ctx(_store(glucose)))

    assert len(requests) == 1
    req = requests[0]
    assert req.kind == "unexplained_spike"
    assert req.suggested_event_type == "meal"
    assert "300 mg/dL" in req.question
    assert "log" in req.question.lower()
    assert not _ADVICE_RE.search(req.question)
    assert req.evidence["peak_mg_dl"] == 300
    assert abs((req.event_ts - peak_ts).total_seconds()) < 1


# ── proximity suppresses ─────────────────────────────────────────────────────────


def test_nearby_meal_suppresses_request() -> None:
    glucose = _flat_window(110)
    peak_ts = _plant_spike(glucose, 100)
    store = _store(glucose)
    store.insert_meals([MealEvent(ts=peak_ts + timedelta(minutes=30), carbs_g=60.0)])

    assert ContextAcquisitionAgent().build(_ctx(store)) == []


def test_nearby_manual_note_suppresses_request() -> None:
    glucose = _flat_window(110)
    peak_ts = _plant_spike(glucose, 100)
    store = _store(glucose)
    store.add_manual_event(
        ManualEvent(
            event_type="note",
            event_ts=peak_ts - timedelta(minutes=45),
            description="forgot to bolus",
            created_at=peak_ts,
        )
    )

    assert ContextAcquisitionAgent().build(_ctx(store)) == []


def test_meal_outside_proximity_does_not_suppress() -> None:
    glucose = _flat_window(110)
    peak_ts = _plant_spike(glucose, 100)
    store = _store(glucose)
    # ~3h before the spike: well outside the +/- 90 min window.
    store.insert_meals([MealEvent(ts=peak_ts - timedelta(hours=3), carbs_g=60.0)])

    assert len(ContextAcquisitionAgent().build(_ctx(store))) == 1


# ── capping ─────────────────────────────────────────────────────────────────────


def test_many_spikes_capped_at_max_requests() -> None:
    glucose = _flat_window(110, hours=24 * 14, end=_END)
    # Eight well-separated spikes (each ~2h apart at minimum); plant far apart.
    for k in range(8):
        _plant_spike(glucose, 100 + k * 200)
    requests = ContextAcquisitionAgent(max_requests=3).build(_ctx(_store(glucose)))

    assert len(requests) == 3
    assert all(r.kind == "unexplained_spike" for r in requests)


def test_every_question_passes_safety_gate() -> None:
    glucose = _flat_window(110, hours=24 * 14, end=_END)
    for k in range(5):
        _plant_spike(glucose, 100 + k * 200)
    requests = ContextAcquisitionAgent().build(_ctx(_store(glucose)))

    assert requests
    for r in requests:
        assert not _ADVICE_RE.search(r.question)


# ── model path ──────────────────────────────────────────────────────────────────


class _SafeRephraseModel:
    """Returns a natural, advice-free rewording."""

    def invoke(self, _messages: list[dict[str, Any]]) -> str:
        return "Something pushed your glucose up here; logging what happened would help."


class _DosingModel:
    """Returns dosing advice that must never reach the user."""

    def invoke(self, _messages: list[dict[str, Any]]) -> str:
        return "Increase your basal insulin to flatten this spike."


class _BrokenModel:
    def invoke(self, _messages: list[dict[str, Any]]) -> str:
        raise RuntimeError("model down")


def test_model_none_is_deterministic_template() -> None:
    glucose = _flat_window(110)
    _plant_spike(glucose, 100)
    req = ContextAcquisitionAgent(model=None).build(_ctx(_store(glucose)))[0]
    assert "300 mg/dL" in req.question
    assert "recurring pattern" in req.question


def test_safe_model_rephrases_question() -> None:
    glucose = _flat_window(110)
    _plant_spike(glucose, 100)
    req = ContextAcquisitionAgent(model=_SafeRephraseModel()).build(_ctx(_store(glucose)))[0]
    assert req.question == (
        "Something pushed your glucose up here; logging what happened would help."
    )


def test_dosing_rephrase_does_not_leak() -> None:
    glucose = _flat_window(110)
    _plant_spike(glucose, 100)
    req = ContextAcquisitionAgent(model=_DosingModel()).build(_ctx(_store(glucose)))[0]
    assert not _ADVICE_RE.search(req.question)
    assert "basal" not in req.question.lower()
    assert "300 mg/dL" in req.question  # the safe template stood


def test_broken_model_falls_back_to_template() -> None:
    glucose = _flat_window(110)
    _plant_spike(glucose, 100)
    req = ContextAcquisitionAgent(model=_BrokenModel()).build(_ctx(_store(glucose)))[0]
    assert "300 mg/dL" in req.question


# ── degraded store ──────────────────────────────────────────────────────────────


class _NoManualStore:
    """Wraps a real store but lacks get_manual_events (minimal/partial store)."""

    def __init__(self, inner: SQLiteStore) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        if name == "get_manual_events":
            raise AttributeError(name)
        return getattr(self._inner, name)


def test_store_without_manual_events_degrades_gracefully() -> None:
    glucose = _flat_window(110)
    _plant_spike(glucose, 100)
    inner = _store(glucose)
    ctx = AgentContext(
        store=_NoManualStore(inner),
        window=(_END.date() - timedelta(days=30), _END.date()),
        gates=ColdStartReport.from_coverage(inner.coverage()),
        run_id="test-run",
        timezone="UTC",
    )
    requests = ContextAcquisitionAgent().build(ctx)
    assert len(requests) == 1
