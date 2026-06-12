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
    # "nocturn" — the fuzzy bridge must surface it above an unrelated doc.
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
