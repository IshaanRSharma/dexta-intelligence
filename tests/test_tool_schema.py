"""The investigator's advertised tool schema must match what it can dispatch.

The producers (discovery, insulin) plan against TOOL_SCHEMA_FOR_LLM and execute
via DiscoveryToolkit.run. If the schema advertises a tool run cannot dispatch,
the model wastes turns on "unknown tool". This locks advertised == dispatchable.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.agents.tools.toolkit import (
    TOOL_SCHEMA_FOR_LLM,
    DiscoveryToolkit,
)
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.models import GlucoseEvent
from dexta_intelligence.store import SQLiteStore

_ADVERTISED = re.compile(r"^\s*\d+\.\s+([a-z_]+)\(", re.MULTILINE)


def _toolkit() -> DiscoveryToolkit:
    store = SQLiteStore(":memory:")
    store.migrate()
    base = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    store.insert_glucose(
        [GlucoseEvent(ts=base + timedelta(minutes=5 * i), mg_dl=120) for i in range(60)]
    )
    ctx = AgentContext(
        store=store,
        window=(date(2026, 6, 1), date(2026, 6, 2)),
        gates=ColdStartReport.from_coverage(store.coverage()),
        run_id="schema-test",
    )
    return DiscoveryToolkit(ctx)


def test_every_advertised_tool_dispatches() -> None:
    advertised = _ADVERTISED.findall(TOOL_SCHEMA_FOR_LLM)
    assert advertised, "schema advertised no tools"
    toolkit = _toolkit()
    for name in advertised:
        result = toolkit.run(name, {})
        # A dispatchable tool may fail on empty args, but never as "unknown tool".
        assert not (result.error or "").startswith("unknown tool"), (
            f"schema advertises {name!r} but run cannot dispatch it"
        )


def test_advertised_set_is_exactly_the_dispatchable_comparison_tools() -> None:
    assert set(_ADVERTISED.findall(TOOL_SCHEMA_FOR_LLM)) == {
        "groupby_compare",
        "tod_compare",
        "event_proximity",
        "basal_overnight",
        "meal_response",
        "correction_outcome",
    }


def test_non_dispatchable_tools_are_not_advertised() -> None:
    # These are belt tools (orchestrator surface), never dispatchable by the
    # investigator; they must not appear in its schema.
    advertised = set(_ADVERTISED.findall(TOOL_SCHEMA_FOR_LLM))
    toolkit = _toolkit()
    for name in ("set_window", "zoom_event", "get_boluses", "find_spikes", "get_current_time"):
        assert name not in advertised
        assert (toolkit.run(name, {}).error or "").startswith("unknown tool")
