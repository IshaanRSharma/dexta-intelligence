"""Conformance tests for the extension points documented in EXTENDING.md.

Each test is the minimal real example from the guide: a connector, a producer
agent, and a belt tool. If the seams ever drift, these fail and the guide is
wrong. Nothing here needs a model or a network.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from dexta_intelligence.agents.base import (
    AgentContext,
    AgentRegistry,
    DataRequirement,
    DextaAgent,
)
from dexta_intelligence.agents.reason import ToolSpec
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.connectors.base import Connector, HealthReport, NormalizedBatch
from dexta_intelligence.models import Finding, GlucoseEvent, RawEvent
from dexta_intelligence.store import SQLiteStore
from dexta_intelligence.workflows.sync import sync

_TS = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


# ── 1. A connector in a few lines ──────────────────────────────────────────────


class _ToyConnector:
    """One source: implement ``source``, ``check``, and ``pull``."""

    source = "toy"

    def check(self) -> HealthReport:
        return HealthReport(ok=True, source=self.source)

    def pull(self, _since: datetime) -> NormalizedBatch:
        raw = RawEvent(source="toy", source_id="g1", source_ts=_TS, payload={"mg_dl": 120})
        return NormalizedBatch(raw=[raw], glucose=[GlucoseEvent(ts=_TS, mg_dl=120)])


def test_a_connector_satisfies_the_protocol_and_ingests() -> None:
    assert isinstance(_ToyConnector(), Connector)
    store = SQLiteStore(":memory:")
    store.migrate()
    try:
        sync(_ToyConnector(), store, now=datetime(2026, 6, 2, tzinfo=UTC))
        got = store.get_glucose(
            datetime(2026, 5, 1, tzinfo=UTC), datetime(2026, 7, 1, tzinfo=UTC)
        )
        assert [g.mg_dl for g in got] == [120]
    finally:
        store.close()


# ── 2. A producer agent in a few lines ─────────────────────────────────────────


class _ToyAgent:
    """One agent: a ``name``, a data ``requires``, and ``run`` returning findings."""

    name = "toy_agent"
    requires = DataRequirement()

    def run(self, _ctx: AgentContext) -> list[Finding]:
        return [Finding(agent=self.name, kind="toy", scope="all", headline="Toy finding")]


def test_an_agent_runs_in_the_registry() -> None:
    assert isinstance(_ToyAgent(), DextaAgent)
    store = SQLiteStore(":memory:")
    store.migrate()
    try:
        ctx = AgentContext(
            store=store,
            window=(date(2026, 6, 1), date(2026, 6, 2)),
            gates=ColdStartReport.from_coverage(store.coverage()),
            run_id="conformance",
        )
        registry = AgentRegistry()
        registry.register(_ToyAgent())
        findings = registry.run_all(ctx)
        assert any(f.headline == "Toy finding" for f in findings)
    finally:
        store.close()


# ── 3. A belt tool in a few lines ───────────────────────────────────────────────


def test_a_tool_spec_has_the_calling_contract() -> None:
    spec = ToolSpec(
        name="toy_tool",
        description="A toy instrument.",
        parameters={"type": "object", "properties": {}},
        fn=lambda _args: ({"answer": 42}, {"answer": 42}),
    )
    result, numbers = spec.fn({})
    assert result == {"answer": 42}
    assert numbers == {"answer": 42}  # the guard-auditable numbers
    schema = spec.schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "toy_tool"
