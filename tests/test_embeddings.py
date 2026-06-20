"""Tests for the dependency-free lexical-vector retrieval layer."""

from __future__ import annotations

from dexta_intelligence.memory import embeddings


def test_embed_empty_and_signal_free_is_empty() -> None:
    assert embeddings.embed("") == {}
    assert embeddings.embed("a of to") == {}  # all words below the 3-char floor


def test_embed_is_l2_normalized() -> None:
    vec = embeddings.embed("overnight drift overnight")
    norm = sum(v * v for v in vec.values()) ** 0.5
    assert abs(norm - 1.0) < 1e-9
    assert vec  # non-empty


def test_cosine_bounds_and_empty() -> None:
    a = embeddings.embed("overnight drift")
    assert abs(embeddings.cosine(a, a) - 1.0) < 1e-9  # self-similarity is 1
    assert 0.0 <= embeddings.cosine(a, embeddings.embed("post-meal spike")) <= 1.0
    assert embeddings.cosine(a, {}) == 0.0
    assert embeddings.cosine({}, a) == 0.0


def test_rank_orders_relevant_above_irrelevant() -> None:
    docs = [
        ("Overnight drift +28 mg/dL on weeknights", "drift"),
        ("Post-meal spike after dinner", "spike"),
    ]
    ranked = embeddings.rank("overnight", docs, top_k=2)
    assert ranked[0][1] == "drift"
    assert ranked[0][0] > ranked[1][0]


def test_rank_trigrams_catch_morphological_near_match() -> None:
    # "nocturnal" never appears as a whole word, but shares trigrams with
    # "nocturn" - the fuzzy bridge must surface it above an unrelated doc.
    docs = [
        ("nocturnal hypoglycemia overnight", "nocturnal"),
        ("weekend mealtime carbohydrate ratios", "weekend"),
    ]
    ranked = embeddings.rank("nocturn", docs, top_k=2)
    assert ranked[0][1] == "nocturnal"
    assert ranked[0][0] > 0.0


def test_rank_empty_inputs() -> None:
    assert embeddings.rank("", [("doc", 1)], top_k=3) == []
    assert embeddings.rank("query", [], top_k=3) == []
    assert embeddings.rank("query", [("doc", 1)], top_k=0) == []


def test_rank_stable_on_ties() -> None:
    docs = [("unrelated alpha", "a"), ("unrelated beta", "b")]
    ranked = embeddings.rank("zzzzz", docs, top_k=2)
    # Neither matches the query; both score 0 and keep input order.
    assert [payload for _score, payload in ranked] == ["a", "b"]
    assert all(score == 0.0 for score, _ in ranked)


# ── synonym expansion + finding-aware ranking ────────────────────────────────

from datetime import UTC, datetime, timedelta  # noqa: E402

from dexta_intelligence.models import Finding, FindingStatus  # noqa: E402


def _finding(
    headline: str,
    *,
    status: FindingStatus = FindingStatus.ACTIVE,
    window_end: datetime | None = None,
) -> Finding:
    return Finding(
        agent="discovery",
        kind="pattern",
        scope="overnight",
        headline=headline,
        status=status,
        window_end=window_end,
    )


def test_expand_pulls_in_domain_synonyms() -> None:
    expanded = embeddings.expand("why am I high after lifting").lower()
    assert "workout" in expanded  # lifting -> workout group
    assert "spike" in expanded  # high -> hyper/spike group


def test_expand_handles_multiword_phrases() -> None:
    assert "tir" in embeddings.expand("my time in range dropped").lower()


def test_expand_noop_when_no_synonyms() -> None:
    assert embeddings.expand("plain unmatched prose") == "plain unmatched prose"


def test_synonym_expansion_bridges_vocabulary() -> None:
    findings = [
        _finding("Post-workout glucose drops on training days"),
        _finding("Dinner carbohydrate ratio mismatch"),
    ]
    ranked = embeddings.rank_findings("why am I high after lifting", findings, top_k=2)
    assert ranked[0][1].headline.startswith("Post-workout")
    assert ranked[0][0] > ranked[1][0]


def test_status_weighting_prefers_active_over_superseded() -> None:
    text = "Overnight drift on weeknights"
    findings = [
        _finding(text, status=FindingStatus.SUPERSEDED),
        _finding(text, status=FindingStatus.ACTIVE),
    ]
    ranked = embeddings.rank_findings("overnight drift", findings, top_k=2)
    assert ranked[0][1].status is FindingStatus.ACTIVE
    assert ranked[0][0] > ranked[1][0]


def test_recency_boost_orders_newest_first() -> None:
    text = "Overnight drift on weeknights"
    now = datetime.now(UTC)
    old = _finding(text, window_end=now - timedelta(days=60))
    fresh = _finding(text, window_end=now - timedelta(days=1))
    ranked = embeddings.rank_findings("overnight drift", [old, fresh], top_k=2)
    assert ranked[0][1] is fresh
    assert ranked[0][0] > ranked[1][0]


def test_rank_findings_degrades_on_empty_and_garbage() -> None:
    findings = [_finding("Overnight drift on weeknights")]
    assert embeddings.rank_findings("", findings, top_k=3) == []
    assert embeddings.rank_findings("a of to", findings, top_k=3) == []  # below floor
    assert embeddings.rank_findings("overnight", [], top_k=3) == []
    assert embeddings.rank_findings("overnight", findings, top_k=0) == []
    # an unrelated query drops zero-relevance findings rather than inventing rank
    assert embeddings.rank_findings("xylophone bicycle", findings, top_k=3) == []
