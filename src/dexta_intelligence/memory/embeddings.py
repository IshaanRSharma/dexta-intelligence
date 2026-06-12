"""Dependency-free lexical-vector retrieval — honest fuzzy ranking, no model.

This is **not** semantic embedding: there is no learned model and no notion of
meaning. It is a sparse lexical vector — sublinear-TF word features plus
character-trigram features — scored by cosine. That upgrades brittle substring
matching to fuzzy *ranked* retrieval ("nocturnal" still pulls a "nocturn" doc)
with zero dependencies. The :func:`rank` API is the seam: a real vector backend
(pgvector, sentence-transformers) can swap in behind it later without callers
changing. The docstrings stay honest about what this layer is and is not.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["cosine", "embed", "rank"]

T = TypeVar("T")

#: Words shorter than this carry little signal and are dropped.
_MIN_WORD_LEN = 3
#: Character-trigram width — fuzzy bridge across spelling/morphology.
_TRIGRAM_N = 3
#: Trigram features are down-weighted relative to whole-word matches.
_TRIGRAM_WEIGHT = 0.5
_WORD_RE = re.compile(r"[a-z0-9]+")


def embed(text: str) -> dict[str, float]:
    """Sparse lexical vector: sublinear-TF words + char trigrams, L2-normalized.

    Lowercases, tokenizes words of length >= 3, weights each by ``1 + log(tf)``
    (sublinear so a term repeated ten times does not dominate), and adds
    character trigrams (down-weighted) over each word padded with spaces so word
    boundaries become fuzzy-matchable. Returns an L2-normalized ``feature ->
    weight`` dict; an empty or signal-free string yields ``{}``.
    """
    words = [w for w in _WORD_RE.findall(text.lower()) if len(w) >= _MIN_WORD_LEN]
    if not words:
        return {}

    raw: dict[str, float] = {}
    for word, count in Counter(words).items():
        raw[f"w:{word}"] = 1.0 + math.log(count)
    for word in words:
        for tri in _trigrams(word):
            raw[f"t:{tri}"] = raw.get(f"t:{tri}", 0.0) + _TRIGRAM_WEIGHT

    norm = math.sqrt(sum(v * v for v in raw.values()))
    if norm == 0.0:
        return {}
    return {k: v / norm for k, v in raw.items()}


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity over two sparse vectors; 0.0 if either is empty.

    Inputs from :func:`embed` are already L2-normalized, so this is their dot
    product over the smaller key set. Bounded in ``[0.0, 1.0]`` for the
    non-negative weights this module produces.
    """
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    return sum(weight * b.get(key, 0.0) for key, weight in a.items())


def rank(
    query: str, documents: Sequence[tuple[str, T]], *, top_k: int
) -> list[tuple[float, T]]:
    """Rank ``(text, payload)`` documents by lexical cosine to ``query``.

    Embeds the query once, scores each document text against it, and returns the
    ``top_k`` highest-scoring ``(score, payload)`` pairs in descending order.
    Ties keep input order (stable sort). An empty query or no documents yields an
    empty list; zero-scoring documents are still returned (callers threshold).
    """
    if top_k <= 0 or not documents:
        return []
    q = embed(query)
    if not q:
        return []
    scored = [(cosine(q, embed(text)), payload) for text, payload in documents]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[:top_k]


def _trigrams(word: str) -> list[str]:
    padded = f" {word} "
    return [padded[i : i + _TRIGRAM_N] for i in range(len(padded) - _TRIGRAM_N + 1)]
