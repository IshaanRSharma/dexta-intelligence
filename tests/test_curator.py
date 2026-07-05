"""Deterministic context curator: selection, floors, receipts, safety, determinism.

The four hard invariants under test:
1. pruning reduces tokens, never ground-truth availability (pure function, no store);
2. severe episodes and treatment-gate inputs are never droppable;
3. every drop carries a trace-ready reason;
4. same inputs yield the same selection and drop list (``now`` is a parameter).
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta

from dexta_intelligence.analytics.episodes import Episode
from dexta_intelligence.memory.curator import (
    ContextSelection,
    ContextType,
    HistoryItem,
    estimate_tokens,
    select_context,
)
from dexta_intelligence.models import EdgeRelation, Finding, FindingEdge, FindingStats

T0 = datetime(2026, 6, 1, tzinfo=UTC)
NOW = datetime(2026, 6, 10, tzinfo=UTC)
WINDOW = (T0, T0 + timedelta(days=7))


def _episode(
    kind: str = "hypo",
    start: datetime = T0 + timedelta(days=1),
    *,
    severe: bool = False,
    significant: bool = True,
    extreme: float | None = 55.0,
) -> Episode:
    return Episode(
        id=f"{kind}:{start.isoformat()}",
        kind=kind,
        start=start,
        end=start + timedelta(minutes=40),
        duration_min=40.0,
        n_readings=8,
        severe=severe,
        clinically_significant=significant,
        extreme_mg_dl=extreme,
        extreme_ts=start if extreme is not None else None,
    )


def _finding(headline: str, *, fid: int | None = 1, confidence: float = 0.7) -> Finding:
    return Finding(
        agent="pattern",
        kind="overnight_low",
        scope="overnight",
        headline=headline,
        id=fid,
        stats=FindingStats(effect_size=0.4),
        confidence=confidence,
        window_start=T0,
        window_end=T0 + timedelta(days=5),
        last_verified=NOW,
    )


def _keys(selection: ContextSelection) -> set[str]:
    return {f"{i.type.value}:{i.key}" for i in selection.selected} | {
        f"{d.item.type.value}:{d.item.key}" for d in selection.dropped
    }


# ── token heuristic ───────────────────────────────────────────────────────────


def test_estimate_tokens_chars_over_four() -> None:
    assert estimate_tokens("abcd" * 10) == 10
    assert estimate_tokens("abcde") == 2
    assert estimate_tokens("") == 1


# ── invariant 1: pure view, nothing destroyed ─────────────────────────────────


def test_curator_takes_no_store() -> None:
    params = inspect.signature(select_context).parameters
    assert "store" not in params, "the curator must be a pure view, not a store client"


def test_every_input_item_is_selected_or_dropped() -> None:
    selection = select_context(
        "overnight lows",
        budget_tokens=30,
        now=NOW,
        window=WINDOW,
        episodes=[_episode(), _episode("hyper", T0 + timedelta(days=2), extreme=260.0)],
        findings=[_finding("Overnight lows recur")],
        history=[HistoryItem(text="prior tool output about lows")],
        metric_bundle={"tbr": 6.5},
    )
    assert selection.dropped, "budget of 30 must force drops"
    n_inputs = 5  # 2 episodes + 1 bundle + 1 finding + 1 history
    assert len(selection.selected) + len(selection.dropped) == n_inputs
    assert len(_keys(selection)) == n_inputs


def test_dropped_items_reappear_under_a_larger_budget() -> None:
    kwargs: dict[str, object] = {
        "now": NOW,
        "window": WINDOW,
        "episodes": [_episode(), _episode("hyper", T0 + timedelta(days=2), extreme=260.0)],
        "findings": [_finding("Overnight lows recur")],
    }
    tight = select_context("overnight lows", budget_tokens=25, **kwargs)  # type: ignore[arg-type]
    roomy = select_context("overnight lows", budget_tokens=10_000, **kwargs)  # type: ignore[arg-type]
    assert tight.dropped
    assert roomy.dropped == ()
    assert _keys(tight) == _keys(roomy), "pruning must never remove ground truth"


# ── invariant 2: safety floor ─────────────────────────────────────────────────


def test_severe_episode_survives_any_budget() -> None:
    severe = _episode(severe=True)
    selection = select_context(
        "anything at all", budget_tokens=1, now=NOW, window=WINDOW, episodes=[severe]
    )
    (item,) = selection.selected
    assert item.key == severe.id
    assert item.protected
    assert selection.dropped == ()
    assert selection.used_tokens > selection.budget_tokens


def test_treatment_gate_history_survives_any_budget() -> None:
    gate_input = HistoryItem(text="get_iob: 2.1 U on board at the event", protected=True)
    filler = HistoryItem(text="unrelated chatter " * 50)
    selection = select_context(
        "why the low", budget_tokens=1, now=NOW, history=[gate_input, filler]
    )
    assert [i.text for i in selection.selected] == [gate_input.text]
    (drop,) = selection.dropped
    assert drop.item.text == filler.text


def test_non_severe_items_are_droppable() -> None:
    selection = select_context(
        "lows", budget_tokens=1, now=NOW, window=WINDOW, episodes=[_episode()]
    )
    assert selection.selected == ()
    assert len(selection.dropped) == 1


# ── invariant 3: every drop has a receipt ─────────────────────────────────────


def test_every_drop_carries_a_reason() -> None:
    selection = select_context(
        "overnight lows and cv",
        budget_tokens=20,
        now=NOW,
        window=WINDOW,
        episodes=[_episode(), _episode("hyper", T0 + timedelta(days=2), extreme=260.0)],
        findings=[_finding("Overnight lows recur")],
        history=[HistoryItem(text="prior tool output " * 20)],
    )
    assert selection.dropped
    for drop in selection.dropped:
        assert drop.reason
        assert "over budget" in drop.reason
        assert "score=" in drop.reason
    lines = selection.trace_lines()
    assert len(lines) == len(selection.dropped)
    for line in lines:
        assert line.startswith("curator drop ")
        assert " tok): " in line


# ── invariant 4: determinism, no wall clock ───────────────────────────────────


def test_same_inputs_same_selection_and_drops() -> None:
    def run() -> ContextSelection:
        return select_context(
            "why did I go low overnight? cv seems high",
            budget_tokens=80,
            now=NOW,
            window=WINDOW,
            episodes=[
                _episode(severe=True),
                _episode("hyper", T0 + timedelta(days=2), extreme=260.0),
                _episode("sensor_gap", T0 + timedelta(days=3), significant=False, extreme=None),
            ],
            findings=[_finding("Overnight lows after late boluses")],
            edges=[
                FindingEdge(
                    src_id=2, dst_id=1, relation=EdgeRelation.CONTRADICTS,
                    knowledge_time=NOW, evidence="opposite effect",
                )
            ],
            history=[HistoryItem(text="get_iob returned 1.2 U", ts=T0, protected=True)],
            metric_bundle={"cv": 33.1, "mean": 142},
        )

    assert run() == run()


# ── scoring behavior ──────────────────────────────────────────────────────────


def test_type_prior_facts_beat_beliefs_all_else_equal() -> None:
    """Relevance-free query, equal salience/freshness: the fact must outrank."""
    episode = _episode(significant=False)  # salience 0.55
    finding = _finding("irrelevant belief", confidence=0.55)
    roomy = select_context(
        "", budget_tokens=10_000, now=NOW, episodes=[episode], findings=[finding]
    )
    scores = {i.type: i.score for i in roomy.selected}
    assert scores[ContextType.FACTS] > scores[ContextType.BELIEFS]


def test_in_window_episode_beats_out_of_window_twin() -> None:
    inside = _episode(start=T0 + timedelta(days=1))
    outside = _episode(start=T0 + timedelta(days=40))
    selection = select_context(
        "", budget_tokens=10_000, now=NOW, window=WINDOW, episodes=[inside, outside]
    )
    by_key = {i.key: i.score for i in selection.selected}
    assert by_key[inside.id] > by_key[outside.id]


def test_contradicted_finding_is_surfaced_and_annotated() -> None:
    quiet = _finding("belief one", fid=1, confidence=0.4)
    contested = _finding("belief two", fid=2, confidence=0.4)
    edge = FindingEdge(
        src_id=9, dst_id=2, relation=EdgeRelation.CONTRADICTS,
        knowledge_time=NOW, evidence="opposite effect: prior +0.4 vs current -0.4",
    )
    selection = select_context(
        "", budget_tokens=10_000, now=NOW, findings=[quiet, contested], edges=[edge]
    )
    by_key = {i.key: i for i in selection.selected}
    assert by_key["finding:2"].score > by_key["finding:1"].score
    assert "contradicted by finding #9" in by_key["finding:2"].text
    assert "opposite effect" in by_key["finding:2"].text


def test_superseded_ancestry_is_annotated() -> None:
    old = _finding("old belief", fid=1)
    edge = FindingEdge(
        src_id=2, dst_id=1, relation=EdgeRelation.SUPERSEDES,
        knowledge_time=NOW, evidence="re-derived",
    )
    selection = select_context(
        "", budget_tokens=10_000, now=NOW, findings=[old], edges=[edge]
    )
    (item,) = selection.selected
    assert "superseded by finding #2" in item.text


def test_conventions_included_only_when_query_touches_the_ontology() -> None:
    with_cv = select_context("is my cv too high", budget_tokens=10_000, now=NOW)
    (item,) = with_cv.selected
    assert item.type is ContextType.CONVENTIONS
    assert item.key == "metric:cv"
    assert "coefficient of variation" in item.text
    assert "percent metric" in item.text

    without = select_context("how was last night", budget_tokens=10_000, now=NOW)
    assert without.selected == ()


def test_per_type_floor_admits_beliefs_against_verbose_history() -> None:
    """History that would fill the whole budget cannot starve the beliefs floor."""
    finding = _finding("carbs late at night", fid=1)
    chatter = [
        HistoryItem(text=f"carbs carbs carbs tool result {i} " * 4, ts=NOW)
        for i in range(30)
    ]
    selection = select_context(
        "carbs", budget_tokens=200, now=NOW, findings=[finding], history=chatter
    )
    selected_types = {i.type for i in selection.selected}
    assert ContextType.BELIEFS in selected_types, "the beliefs floor must reserve room"
    assert selection.used_tokens <= selection.budget_tokens
    assert selection.dropped, "the verbose history must overflow"


def test_selection_fits_budget_when_nothing_is_protected() -> None:
    selection = select_context(
        "overnight lows",
        budget_tokens=40,
        now=NOW,
        window=WINDOW,
        episodes=[_episode(), _episode("hyper", T0 + timedelta(days=2), extreme=260.0)],
        findings=[_finding("Overnight lows recur")],
        history=[HistoryItem(text="prior output " * 10)],
    )
    assert selection.used_tokens <= selection.budget_tokens
