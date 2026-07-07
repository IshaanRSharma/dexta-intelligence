"""Curator integration on the chat ask path.

The deterministic librarian (memory/curator.select_context) feeds a typed
context block into the ask system prompt, and its drop receipts land in the
answer trace. No LLM writes any of it: a fake tool-calling model captures the
system prompt so the block's construction is asserted from a seeded store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.agents.chat import ChatAgent
from dexta_intelligence.agents.curation import curated_context
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.models import Finding, FindingStats, GlucoseEvent
from dexta_intelligence.store import SQLiteStore

_END = datetime(2026, 6, 1, tzinfo=UTC)
_START = _END - timedelta(days=14)
_NOW = _END + timedelta(hours=1)


@dataclass
class _AIMessage:
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class _FakeToolModel:
    """Replays scripted turns and records every prompt it was invoked with."""

    def __init__(self, turns: list[Any]) -> None:
        self._turns = turns
        self.systems: list[str] = []

    def bind_tools(self, schemas: list[dict[str, Any]]) -> _FakeToolModel:
        return self

    def invoke(self, messages: list[Any]) -> _AIMessage:
        first = messages[0]
        if isinstance(first, dict) and first.get("role") == "system":
            self.systems.append(str(first.get("content", "")))
        turn = self._turns.pop(0) if self._turns else "Nothing more."
        if isinstance(turn, str):
            return _AIMessage(content=turn)
        return _AIMessage(tool_calls=list(turn))


def _store(*, with_glucose: bool = True, severe_low: bool = False) -> SQLiteStore:
    store = SQLiteStore(":memory:")
    store.migrate()
    if not with_glucose:
        return store
    glucose: list[GlucoseEvent] = []
    for day in range(14):
        base = _START + timedelta(days=day)
        for hour, mg in ((3, 220), (4, 225), (12, 120), (13, 122)):
            for minute in (0, 15, 30, 45):
                glucose.append(GlucoseEvent(ts=base.replace(hour=hour, minute=minute), mg_dl=mg))
    if severe_low:
        base = _START + timedelta(days=2)
        for minute in (0, 10, 20, 30):
            glucose.append(GlucoseEvent(ts=base.replace(hour=20, minute=minute), mg_dl=48))
    store.insert_glucose(glucose)
    return store


def _ctx(store: SQLiteStore) -> AgentContext:
    return AgentContext(
        store=store,
        window=(_START.date(), _END.date()),
        gates=ColdStartReport.from_coverage(store.coverage()),
        run_id="curation-test",
    )


def _finding(headline: str = "Overnight highs recur around 03:00") -> Finding:
    return Finding(
        agent="pattern",
        kind="overnight_high",
        scope="overnight",
        headline=headline,
        stats=FindingStats(effect_size=0.5, n=14),
        confidence=0.8,
        window_start=_START,
        window_end=_END,
    )


# ── the block itself (direct, fixed now) ─────────────────────────────────────


def test_block_built_from_seeded_store_with_typed_labels() -> None:
    store = _store()
    store.insert_finding(_finding())
    block, receipts = curated_context(
        _ctx(store), "why are my nights high?", now=_NOW, budget_tokens=4000
    )

    assert "CURATED CONTEXT" in block
    assert "FACTS:" in block
    assert "BELIEFS:" in block
    assert "Overnight highs recur around 03:00" in block
    assert "hyper" in block
    assert receipts == ()  # a budget covering every item prunes nothing


def test_block_is_deterministic() -> None:
    store = _store()
    store.insert_finding(_finding())
    ctx = _ctx(store)
    first = curated_context(ctx, "why are my nights high?", now=_NOW)
    second = curated_context(ctx, "why are my nights high?", now=_NOW)
    assert first == second


def test_noop_when_curator_selects_nothing() -> None:
    store = _store(with_glucose=False)
    block, receipts = curated_context(_ctx(store), "hello there", now=_NOW)
    assert block == ""
    assert receipts == ()


def test_tiny_budget_emits_drop_receipts() -> None:
    store = _store()
    for i in range(6):
        store.insert_finding(_finding(headline=f"Pattern number {i} repeats across many nights"))
    block, receipts = curated_context(
        _ctx(store), "why are my nights high?", now=_NOW, budget_tokens=40
    )
    assert block  # something survives the floor phases
    assert receipts  # and the pruning left receipts
    assert all(line.text.startswith("curator drop") for line in receipts)
    assert all("over budget" in line.text for line in receipts)


def test_safety_floor_severe_episode_never_dropped() -> None:
    store = _store(severe_low=True)
    for i in range(6):
        store.insert_finding(_finding(headline=f"Pattern number {i} repeats across many nights"))
    block, receipts = curated_context(
        _ctx(store), "why are my nights high?", now=_NOW, budget_tokens=10
    )
    assert "severe" in block  # the severe hypo is in the window even over budget
    assert not any("hypo:" in line.text for line in receipts)


# ── wired through ChatAgent ──────────────────────────────────────────────────


def test_chat_system_prompt_carries_the_block() -> None:
    store = _store()
    store.insert_finding(_finding())
    model = _FakeToolModel(["Your nights run high; verify with tools."])
    ChatAgent(model=model).ask(_ctx(store), "why are my nights high?")  # type: ignore[arg-type]

    assert model.systems
    system = model.systems[0]
    assert "CURATED CONTEXT" in system
    assert "BELIEFS:" in system
    assert "Overnight highs recur around 03:00" in system


def test_chat_receipts_land_in_answer_trace() -> None:
    store = _store()
    for i in range(6):
        store.insert_finding(_finding(headline=f"Pattern number {i} repeats across many nights"))
    model = _FakeToolModel(["Several patterns are on record."])
    answer = ChatAgent(model=model, context_budget_tokens=40).ask(  # type: ignore[arg-type]
        _ctx(store), "why are my nights high?"
    )
    assert any(line.text.startswith("curator drop") for line in answer.trace)


def test_chat_noop_leaves_system_prompt_untouched() -> None:
    store = _store(with_glucose=False)
    model = _FakeToolModel(["Hello."])
    answer = ChatAgent(model=model).ask(_ctx(store), "hello there")  # type: ignore[arg-type]

    assert model.systems
    assert "CURATED CONTEXT" not in model.systems[0]
    assert not any("curator drop" in line.text for line in answer.trace)
