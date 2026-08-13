"""Tests for the `dexta demo` zero-config on-ramp."""

from __future__ import annotations

import io
import uuid
from datetime import UTC, date, datetime

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


def test_demo_glycemia_is_in_the_range_a_real_record_occupies() -> None:
    """The synthetic patient has to be a plausible one.

    A trace generated independently of its own treatment record sits at ~99% time
    in range with a CV near 11, which is not a person: it makes every meal-versus-
    glucose correlation null by construction and leaves "why did I go high?" with
    nothing to find. These bounds are the 2019 consensus targets, met but not
    trivially, so the demo reads as a well-controlled record rather than a flat one.
    """
    import statistics  # noqa: PLC0415

    store = build_demo_store()
    try:
        cov = store.coverage()
        assert cov.first_ts is not None and cov.last_ts is not None
        values = [g.mg_dl for g in store.get_glucose(cov.first_ts, cov.last_ts)]
    finally:
        store.close()

    mean = statistics.fmean(values)

    def share(lo: float, hi: float) -> float:
        return 100.0 * sum(1 for v in values if lo <= v <= hi) / len(values)

    assert 60.0 <= share(70, 180) <= 88.0          # time in range
    assert 0.5 <= share(0, 69) <= 4.0              # below range
    assert share(0, 53) <= 1.0                     # very low
    assert 10.0 <= share(181, 1000) <= 30.0        # above range
    assert 0.5 <= share(251, 1000) <= 6.0          # very high
    assert 22.0 <= statistics.pstdev(values) / mean * 100 <= 38.0  # coefficient of variation
    assert 135.0 <= mean <= 175.0


def test_demo_glucose_answers_to_its_own_treatment_record() -> None:
    """Carbs move the curve. Without this the record is 185 days of meals and
    boluses that change nothing, every meal-versus-glucose test an agent runs is
    null by construction, and "why did I go high?" has no answer to find."""
    import bisect  # noqa: PLC0415
    import statistics  # noqa: PLC0415
    from datetime import timedelta  # noqa: PLC0415

    store = build_demo_store()
    try:
        wide = (datetime(2000, 1, 1, tzinfo=UTC), datetime(2100, 1, 1, tzinfo=UTC))
        readings = sorted(_glucose(store), key=lambda g: g.ts)
        lunches = [m for m in store.get_meals(*wide) if m.note == "lunch"]
    finally:
        store.close()

    stamps = [g.ts for g in readings]

    def near(ts: datetime) -> int:
        """The reading closest to ``ts``; carb entries do not land on the grid."""
        i = min(bisect.bisect_left(stamps, ts), len(readings) - 1)
        best = min((max(0, i - 1), i), key=lambda j: abs(stamps[j] - ts))
        return readings[best].mg_dl

    rises = [near(m.ts + timedelta(minutes=60)) - near(m.ts) for m in lunches]
    assert len(rises) > 100
    assert statistics.fmean(rises) > 20.0  # a postprandial excursion, not noise


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


def test_demo_extends_to_june_with_every_severity_band() -> None:
    """The record runs into mid-June 2026 and the Mar-Jun extension carries at
    least one of each: regular low, very low, regular high, very high."""
    from dexta_intelligence.analytics.episodes import build_graph  # noqa: PLC0415

    store = build_demo_store()
    try:
        cov = store.coverage()
        assert cov.last_ts is not None and cov.last_ts.date() >= date(2026, 6, 1)
        # the extended window only (after the story/hero period)
        ext = build_graph(
            store, datetime(2026, 3, 20, tzinfo=UTC), datetime(2026, 6, 20, tzinfo=UTC)
        ).summary()
    finally:
        store.close()
    assert ext["num_hypo"] >= 3 and ext["num_hyper"] >= 3   # regular lows and highs
    assert ext["n_severe_hypo"] >= 1                        # a very low (<54)
    assert ext["n_severe_hyper"] >= 1                       # a very high (>250)


def test_demo_episode_graph_tells_the_story() -> None:
    """The seeded patient exercises every episode-graph feature: rebound chains
    bridged by rescue carbs, a correction-bridged night low, a paired treatment
    edge, a severe low, and a sensor gap."""
    from collections import Counter  # noqa: PLC0415

    from dexta_intelligence.analytics.episodes import build_graph  # noqa: PLC0415

    store = build_demo_store()
    try:
        graph = build_graph(
            store,
            datetime(2025, 12, 15, tzinfo=UTC),
            datetime(2026, 3, 15, tzinfo=UTC),
        )
    finally:
        store.close()
    summary = graph.summary()
    assert summary["num_hypo"] >= 8
    assert summary["n_severe_hypo"] >= 1
    assert summary["n_sensor_gaps"] >= 1

    relations = Counter(e.relation for e in graph.edges)
    assert relations["rebound_after_low"] >= 4
    assert relations["low_after_high"] >= 1
    # The planted chains are asserted by their bridge, not by position: the trace
    # now responds to the whole treatment record, so organically occurring rebounds
    # sit among them and either may come first.
    assert any(
        e.relation == "rebound_after_low"
        and e.bridge is not None
        and e.bridge.detail.get("note") == "rescue carbs"
        for e in graph.edges
    )
    assert any(
        e.relation == "low_after_high" and e.bridge is not None and e.bridge.kind == "bolus"
        for e in graph.edges
    )

    link_kinds = Counter(link.kind for ep in graph.episodes for link in ep.links)
    assert link_kinds["treatment"] >= 1
    assert link_kinds["activity"] >= 3  # the post-workout lows have their runs
