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


def test_demo_store_populates_every_surface() -> None:
    """The demo carries all streams so each page (and differentiator) has data."""
    store = build_demo_store()
    try:
        cov = store.coverage()
        wide = (datetime(2000, 1, 1, tzinfo=UTC), datetime(2100, 1, 1, tzinfo=UTC))
        assert cov.glucose_coverage_pct > 90.0  # no false "limited" banner
        assert cov.n_sleep > 0
        assert cov.n_activity > 0
        assert len(store.get_predictions(*wide)) > 0
        assert len(store.get_profile_versions()) == 2
        assert len(store.get_manual_events(*wide)) >= 3
        # The right profile version is active at the hero spike.
        active = store.get_active_profile(datetime(2026, 3, 14, 20, 0, tzinfo=UTC))
        assert active is not None and active.name == "Spring"
    finally:
        store.close()


def test_demo_has_comprehensive_tandem_treatment() -> None:
    """The demo carries a full t:slim X2 / Control-IQ record: a multi-segment
    profile (basal/CR/ISF schedules), temp basals, corrections, suspends, and
    three meals a day across at least 30 days."""
    from collections import Counter  # noqa: PLC0415

    from dexta_intelligence.models import InsulinKind  # noqa: PLC0415

    store = build_demo_store()
    try:
        wide = (datetime(2000, 1, 1, tzinfo=UTC), datetime(2100, 1, 1, tzinfo=UTC))
        cov = store.coverage()
        assert cov.span_days >= 30
        kinds = Counter(i.kind for i in store.get_insulin(*wide))
        assert kinds[InsulinKind.TEMP_BASAL] > 0  # Control-IQ adjustments
        assert kinds[InsulinKind.SUSPEND] > 0  # low-glucose suspends
        assert kinds[InsulinKind.BOLUS] > 90  # meal + correction boluses
        notes = {m.note for m in store.get_meals(*wide)}
        assert {"breakfast", "lunch", "dinner"} <= notes
        active = store.get_active_profile(datetime(2026, 3, 14, 20, 0, tzinfo=UTC))
        segments = active.content["profiles"][0]["segments"]
        assert len(segments) >= 3  # time-of-day basal/CR/ISF schedule
        assert all("carb_ratio_g_u" in s and "isf_mg_dl_u" in s for s in segments)
        assert active.content["pump_model"] == "Tandem t:slim X2"
    finally:
        store.close()


def test_demo_reconciliation_finds_the_planted_miss() -> None:
    """The logged forecast curves diverge from realized CGM by design, so the
    reconciliation agent surfaces a carb-underestimate forecast miss."""
    import uuid as _uuid  # noqa: PLC0415

    from dexta_intelligence.agents.reconciliation import (  # noqa: PLC0415
        PredictionReconciliationAgent,
    )

    store = build_demo_store()
    try:
        cov = store.coverage()
        ctx = AgentContext(
            store=store,
            window=(cov.first_ts.date(), cov.last_ts.date()),  # type: ignore[union-attr]
            gates=ColdStartReport.from_coverage(cov),
            run_id=str(_uuid.uuid4()),
        )
        findings = PredictionReconciliationAgent().run(ctx)
    finally:
        store.close()
    assert findings
    assert any("carb underestimate" in f.headline.lower() for f in findings)


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
