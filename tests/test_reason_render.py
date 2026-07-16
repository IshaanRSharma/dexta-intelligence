"""Structure-aware serialization of tool results for the model's context.

The reasoning loop must never hand the model malformed JSON or silently drop a
field. These tests pin :func:`_fit_tool_result`: valid JSON at every budget,
small high-priority fields preserved, large arrays reduced with a marker.
"""

from __future__ import annotations

import json
from typing import Any

from dexta_intelligence.agents.reason import _fit_tool_result


def _parse(result: Any, budget: int) -> Any:
    text = _fit_tool_result(result, budget)
    assert len(text) <= budget or budget < 60, (len(text), budget)
    return json.loads(text)  # must always be valid JSON


def test_small_result_passes_through_verbatim() -> None:
    obj = {"summary": "a high episode", "extreme_mg_dl": 206, "chain": {"in": [], "out": []}}
    parsed = _parse(obj, 4000)
    assert parsed == obj


def test_oversize_result_is_always_valid_json() -> None:
    obj = {"summary": "s", "links": [{"kind": "meal", "carbs_g": i} for i in range(500)]}
    parsed = _parse(obj, 500)
    assert isinstance(parsed, dict)  # parsed, so it was valid JSON


def test_reduction_preserves_summary_and_chain_over_links() -> None:
    # A pathologically dense episode: 400 context links, but the summary and the
    # chain (small, high-value) must survive the budget cut.
    obj = {
        "summary": "A high episode (206 mg/dL peak). It followed a low, rebound after low.",
        "id": "hyper:2026-03-08T17:10:00+00:00",
        "kind": "hyper",
        "links": [{"kind": "meal", "carbs_g": i, "note": "x" * 20} for i in range(400)],
        "chain": {
            "in": [{"src_id": "hypo:2026-03-08T15:50:00+00:00", "relation": "rebound_after_low",
                    "gap_min": 55, "bridge": {"kind": "meal", "detail": {"carbs_g": 16}}}],
            "out": [],
        },
    }
    parsed = _parse(obj, 800)
    assert parsed["summary"] == obj["summary"]
    assert parsed["chain"]["in"][0]["relation"] == "rebound_after_low"
    assert parsed["chain"]["in"][0]["bridge"]["detail"]["carbs_g"] == 16
    # links were the biggest field, so they were trimmed and the drop recorded.
    assert parsed["links_elided"] > 0
    assert len(parsed["links"]) < 400


def test_reduction_terminates_on_tiny_budget_and_small_lists() -> None:
    # Regression: a two-element list under a tiny budget used to spin forever
    # because the in-list marker never shrank the length. Must terminate and
    # return valid JSON.
    obj = {"summary": "s" * 100, "links": [{"a": 1}, {"b": 2}], "chain": {"in": [], "out": []}}
    parsed = _parse(obj, 40)  # smaller than the scalar fields alone
    assert isinstance(parsed, dict)  # returned at all == it terminated


def test_reduction_terminates_when_only_scalars_remain() -> None:
    obj = {"note": "x" * 5000, "items": [1, 2, 3, 4, 5]}
    parsed = _parse(obj, 200)
    assert parsed.get("_truncated") is True  # lists emptied, scalar still too big


def test_reduction_is_deterministic() -> None:
    obj = {"summary": "s", "links": [{"k": i} for i in range(300)],
           "chain": {"in": [{"relation": "follows"}], "out": []}}
    assert _fit_tool_result(obj, 600) == _fit_tool_result(obj, 600)


def test_single_huge_string_field_falls_back_to_valid_preview() -> None:
    obj = {"blob": "z" * 10000}
    parsed = _parse(obj, 400)
    assert parsed.get("_truncated") is True
    assert isinstance(parsed.get("preview"), str)


def test_non_dict_result_stays_valid_json() -> None:
    parsed = _parse(["a" * 100 for _ in range(200)], 500)
    assert parsed.get("_truncated") is True


def test_error_result_passes_through() -> None:
    obj = {"error": "no episode matched"}
    assert json.loads(_fit_tool_result(obj, 4000)) == obj


def test_real_explain_episode_message_keeps_summary_and_chain_under_a_tight_budget() -> None:
    """End to end on a real (small) store: the exact bytes the model would
    receive for a chained episode still carry the narrative and the chain when
    the budget forces reduction. Closes the model/guard asymmetry: what the model
    reads and what the guard audits agree on the causal facts."""
    from datetime import UTC, datetime, timedelta  # noqa: PLC0415

    from dexta_intelligence.agents.base import AgentContext  # noqa: PLC0415
    from dexta_intelligence.agents.tools.episodes import episode_specs  # noqa: PLC0415
    from dexta_intelligence.agents.tools.toolkit import DiscoveryToolkit  # noqa: PLC0415
    from dexta_intelligence.coldstart import ColdStartReport  # noqa: PLC0415
    from dexta_intelligence.models import GlucoseEvent, MealEvent  # noqa: PLC0415
    from dexta_intelligence.store.sqlite import SQLiteStore  # noqa: PLC0415

    start = datetime(2026, 1, 6, tzinfo=UTC)

    def ts(minute: int) -> datetime:
        return start + timedelta(minutes=minute)

    store = SQLiteStore(":memory:")
    store.migrate()
    # low -> rescue carbs -> rebound high, plus filler context to inflate the result.
    store.insert_glucose(
        [GlucoseEvent(ts=ts(m), mg_dl=v) for m, v in
         [(0, 60), (5, 55), (10, 65), (40, 200), (45, 220), (50, 120)]]
    )
    store.insert_meals(
        [MealEvent(ts=ts(20), carbs_g=15.0, note="rescue")]
        + [MealEvent(ts=ts(30 + i), carbs_g=5.0 + i, note="snack " * 4) for i in range(40)]
    )
    try:
        cov = store.coverage()
        assert cov.first_ts is not None and cov.last_ts is not None
        ctx = AgentContext(
            store=store,
            window=(cov.first_ts.date(), cov.last_ts.date()),
            gates=ColdStartReport.from_coverage(cov),
            run_id="test",
            timezone="UTC",
        )
        toolkit = DiscoveryToolkit(ctx, target_low=70, target_high=180)
        specs = {s.name: s for s in episode_specs(ctx, toolkit)}
        high_id = next(
            n["id"] for n in specs["episodes"].fn({})[0]["episodes"] if n["kind"] == "hyper"
        )
        result, _ = specs["explain_episode"].fn({"episode_id": high_id})
    finally:
        store.close()

    # The raw result (many context links) is far over budget; the summary +
    # chain core is small enough to keep.
    assert len(json.dumps(result)) > 3000
    parsed = json.loads(_fit_tool_result(result, 1000))  # force link reduction
    assert len(json.dumps(parsed)) <= 1000
    assert parsed["summary"].startswith("A high episode")
    assert "rebound after low" in parsed["summary"]
    assert parsed["chain"]["in"][0]["relation"] == "rebound_after_low"
    assert parsed["links_elided"] > 0  # links were dropped, not the causal core
