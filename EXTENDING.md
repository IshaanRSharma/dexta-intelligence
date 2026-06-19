# Extending dexta

dexta is built from a few narrow seams. Adding a data source, an analysis agent,
or a tool the reasoning loop can call is a small, local change. Every recipe
below is exercised by `tests/test_extensibility.py`, so the code here is real,
not aspirational.

The one rule that never bends: **analytics compute facts, the LLM reasons over
them, and two rails (numeric faithfulness, treatment gate) bound the output.**
New code lives on one side of that line. No component emits dosing advice.

---

## Add a connector (a new data source)

A connector pulls provider records and returns them as immutable `RawEvent`
rows plus their normalized timeline projections. Idempotency is structural
(`(source, source_id)` dedup), so re-running a sync is always safe.

Implement the `Connector` protocol — `source`, `check`, `pull`:

```python
from dexta_intelligence.connectors.base import Connector, HealthReport, NormalizedBatch
from dexta_intelligence.models import GlucoseEvent, RawEvent

class MySourceConnector:
    source = "mysource"

    def check(self) -> HealthReport:
        return HealthReport(ok=True, source=self.source)

    def pull(self, since):
        # Fetch everything newer than `since`, return raws + normalized events.
        raw = RawEvent(source="mysource", source_id="g1", source_ts=ts, payload={...})
        return NormalizedBatch(raw=[raw], glucose=[GlucoseEvent(ts=ts, mg_dl=120)])
```

Ingest it through the same path every source uses:

```python
from dexta_intelligence.workflows.sync import sync
sync(MySourceConnector(), store)   # watermark-incremental, idempotent
```

To wire it into config-driven sync (`dexta sync`, the Connectors page), register
it in `build_connectors` (`cli/_common.py`) behind its config check. If the
source has a live "what is my glucose right now?" API, implement
`RealtimeConnector.current` too.

---

## Add an analysis agent (a producer)

An agent declares its data requirement up front (the registry refuses to run it
under-data — cold start is explicit), does read-only work over an
`AgentContext`, and returns `Finding` records. Rigor lives inside the agent; the
skeptic re-checks every finding downstream.

```python
from dexta_intelligence.agents.base import AgentContext, DataRequirement
from dexta_intelligence.models import Finding

class MyAgent:
    name = "myagent"
    requires = DataRequirement(min_span_days=3.0, min_glucose_coverage_pct=50.0)

    def run(self, ctx: AgentContext) -> list[Finding]:
        # Read windows via ctx.store; compute; never fabricate a number.
        return [Finding(agent=self.name, kind="my_pattern", scope="overnight",
                        headline="...", body_md="...")]
```

Register it on a registry and it runs under gating + exception isolation:

```python
registry.register(MyAgent())
findings = registry.run_all(ctx)
```

To make it a coordinator producer (selectable by the planner and `dexta
investigate`), add a `_register_myagent` and an entry in `PRODUCERS`
(`workflows/lenses.py`).

---

## Add a tool (an instrument the reasoning loop can call)

A tool is a `ToolSpec`: a name, a description the model reads, a JSON-Schema for
its arguments, and a function returning `(public_result, evidence_numbers)`. The
second element is the guard-auditable numbers — anything the prose later cites
must appear here.

```python
from dexta_intelligence.agents.reason import ToolSpec

ToolSpec(
    name="my_tool",
    description="What it returns and when the model should call it.",
    parameters={"type": "object", "properties": {"start": {"type": "string"}}},
    fn=lambda args: ({"answer": 42}, {"answer": 42}),
)
```

Add it to the belt in `agents/discovery_tools.py` (`tool_specs`), and add a
one-line renderer in `agents/trace.py` so the call shows up as a readable trace
line. If the result is independent of glucose coverage, append it after the
capability filter so it is always available.

---

## Verifying your extension

```
.venv/bin/ruff check src/ tests/
.venv/bin/mypy src/dexta_intelligence/
.venv/bin/pytest
```

`tests/test_extensibility.py` is the conformance harness for the three seams
above; mirror its toy examples when you add real ones.
