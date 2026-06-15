"""Tests for the `dexta demo` zero-config on-ramp."""

from __future__ import annotations

import io
import uuid
from datetime import UTC, datetime

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.cli import cmd_demo, main
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.demo import DEMO_SPIKE_DATE, build_demo_store
from dexta_intelligence.investigations.spike import SAFETY_LINE, explain_spike


def _glucose(store: object) -> list:
    return store.get_glucose(  # type: ignore[attr-defined]
        datetime(2000, 1, 1, tzinfo=UTC), datetime(2100, 1, 1, tzinfo=UTC)
    )


def test_build_demo_store_is_deterministic_and_in_memory() -> None:
    a = build_demo_store()
    b = build_demo_store()
    try:
        ga, gb = _glucose(a), _glucose(b)
        assert len(ga) == len(gb)
        assert [(e.ts, e.mg_dl) for e in ga] == [(e.ts, e.mg_dl) for e in gb]
        assert a._path == ":memory:"  # type: ignore[attr-defined]
    finally:
        a.close()  # type: ignore[attr-defined]
        b.close()  # type: ignore[attr-defined]


def test_demo_store_explains_canonical_spike() -> None:
    store = build_demo_store()
    try:
        gates = ColdStartReport.from_coverage(store.coverage())
        cov = store.coverage()
        window = (cov.first_ts.date(), cov.last_ts.date())  # type: ignore[union-attr]
        ctx = AgentContext(store=store, window=window, gates=gates, run_id=str(uuid.uuid4()))
        report = explain_spike(ctx, DEMO_SPIKE_DATE.isoformat(), model=None)
    finally:
        store.close()
    assert report["confidence"] in ("high", "moderate")
    assert report["trace"]
    assert "late" in report["headline"].lower()


def test_cmd_demo_output() -> None:
    out = io.StringIO()
    rc = cmd_demo(out=out)
    text = out.getvalue()
    assert rc == 0
    assert "synthetic patient" in text
    assert "Investigation trace" in text
    assert "246" in text
    assert "late" in text.lower() and "meal" in text.lower()
    assert SAFETY_LINE in text


def test_main_demo_subcommand() -> None:
    assert main(["demo"]) == 0
