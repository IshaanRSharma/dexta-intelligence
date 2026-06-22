"""Dependency-free lexical-vector retrieval - honest fuzzy ranking, no model.

This is **not** semantic embedding: there is no learned model and no notion of
meaning. It is a sparse lexical vector - sublinear-TF word features plus
character-trigram features - scored by cosine. That upgrades brittle substring
matching to fuzzy *ranked* retrieval ("nocturnal" still pulls a "nocturn" doc)
with zero dependencies. The :func:`rank` API is the seam: a real vector backend
(pgvector, sentence-transformers) can swap in behind it later without callers
changing. The docstrings stay honest about what this layer is and is not.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dexta_intelligence.models import Finding

__all__ = ["cosine", "embed", "expand", "rank", "rank_findings"]

T = TypeVar("T")

#: Words shorter than this carry little signal and are dropped.
_MIN_WORD_LEN = 3
#: Character-trigram width - fuzzy bridge across spelling/morphology.
_TRIGRAM_N = 3
#: Trigram features are down-weighted relative to whole-word matches.
_TRIGRAM_WEIGHT = 0.5
_WORD_RE = re.compile(r"[a-z0-9]+")

#: Curated diabetes/CGM vocabulary groups. Each group is a set of terms that mean
#: the same thing in this domain; :func:`expand` appends every other member of a
#: group whenever any member appears, so a "workout" finding still matches a
#: "lifting" query. This is hand-maintained domain knowledge, NOT learned
#: synonymy - keep it small and obviously-correct. Multi-word phrases are matched
#: as substrings before tokenization, single words as whole tokens.
_SYNONYM_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"workout", "exercise", "activity", "strain", "training", "lifting", "cardio"}),
    frozenset({"dinner", "evening meal", "supper"}),
    frozenset({"breakfast", "morning meal"}),
    frozenset({"high", "hyper", "spike", "excursion", "elevated"}),
    frozenset({"low", "hypo", "hypoglycemia", "hypoglycaemia"}),
    frozenset({"overnight", "nocturnal", "night", "nighttime"}),
    frozenset({"bolus", "insulin dose", "mealtime insulin"}),
    frozenset({"basal", "background insulin"}),
    frozenset({"tir", "time in range"}),
    frozenset({"tbr", "time below range"}),
    frozenset({"cv", "variability", "glycemic variability"}),
    frozenset({"carb", "carbs", "carbohydrate", "carbohydrates"}),
    frozenset({"correction", "correction bolus", "rescue"}),
    frozenset({"sleep", "rest"}),
    frozenset({"morning", "am", "wake", "waking"}),
)

#: term -> the full set of its group members (term included), for O(1) lookup.
_SYNONYM_INDEX: dict[str, frozenset[str]] = {
    term: group for group in _SYNONYM_GROUPS for term in group
}
#: multi-word phrases (matched as substrings before tokenization), longest first.
_SYNONYM_PHRASES: tuple[str, ...] = tuple(
    sorted((t for t in _SYNONYM_INDEX if " " in t), key=len, reverse=True)
)


def expand(text: str) -> str:
    """Append diabetes-domain synonyms so cross-vocabulary text overlaps.

    Lexical cosine only matches shared tokens, so "lifting" never touches a
    "workout" finding. :data:`_SYNONYM_GROUPS` encodes the equivalences this
    domain needs; this folds every group member of any matched term back into the
    text. Applied to BOTH query and document before :func:`embed`. This is curated
    domain knowledge, not learned synonymy - and not semantic embedding.
    """
    lowered = text.lower()
    extra: list[str] = []
    for phrase in _SYNONYM_PHRASES:
        if phrase in lowered:
            extra.extend(_SYNONYM_INDEX[phrase])
    for word in _WORD_RE.findall(lowered):
        group = _SYNONYM_INDEX.get(word)
        if group is not None:
            extra.extend(group)
    if not extra:
        return text
    return f"{text} {' '.join(extra)}"


def embed(text: str, *, synonyms: bool = False) -> dict[str, float]:
    """Sparse lexical vector: sublinear-TF words + char trigrams, L2-normalized.

    Lowercases, tokenizes words of length >= 3, weights each by ``1 + log(tf)``
    (sublinear so a term repeated ten times does not dominate), and adds
    character trigrams (down-weighted) over each word padded with spaces so word
    boundaries become fuzzy-matchable. Returns an L2-normalized ``feature ->
    weight`` dict; an empty or signal-free string yields ``{}``. With
    ``synonyms=True`` the text is :func:`expand`-ed first (off by default so the
    base vector stays stable for callers that do their own expansion).
    """
    if synonyms:
        text = expand(text)
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
    query: str, documents: Sequence[tuple[str, T]], *, top_k: int, synonyms: bool = False
) -> list[tuple[float, T]]:
    """Rank ``(text, payload)`` documents by lexical cosine to ``query``.

    Embeds the query once, scores each document text against it, and returns the
    ``top_k`` highest-scoring ``(score, payload)`` pairs in descending order.
    Ties keep input order (stable sort). An empty query or no documents yields an
    empty list; zero-scoring documents are still returned (callers threshold).
    With ``synonyms=True`` both query and documents are :func:`expand`-ed first.
    """
    if top_k <= 0 or not documents:
        return []
    q = embed(query, synonyms=synonyms)
    if not q:
        return []
    scored = [
        (cosine(q, embed(text, synonyms=synonyms)), payload) for text, payload in documents
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[:top_k]


def _trigrams(word: str) -> list[str]:
    padded = f" {word} "
    return [padded[i : i + _TRIGRAM_N] for i in range(len(padded) - _TRIGRAM_N + 1)]


# ── finding-aware ranking ────────────────────────────────────────────────────

#: Multiplicative weight by finding status - active beliefs outrank retired ones
#: at equal text relevance. There is no "hypothesis" FindingStatus; the lowest
#: tier covers superseded/rejected/dismissed (still recallable, just down-ranked).
_STATUS_WEIGHT: dict[str, float] = {
    "active": 1.0,
    "superseded": 0.6,
    "rejected": 0.4,
    "dismissed": 0.4,
    "contradicted": 0.3,
    "stale": 0.2,
}
_STATUS_WEIGHT_DEFAULT = 0.7

#: Recency is a mild multiplicative boost, NOT a sort key: a stale-but-relevant
#: finding should still beat a fresh-but-irrelevant one. A finding at the
#: reference time gets the full boost; it decays linearly to 1.0 over the
#: half-life window and never drops below it.
_RECENCY_MAX_BOOST = 1.25
_RECENCY_HALFLIFE = timedelta(days=90)


def rank_findings(
    query: str, findings: Sequence[Finding], *, top_k: int
) -> list[tuple[float, Finding]]:
    """Rank findings by synonym-expanded lexical relevance, status, and recency.

    Scores each finding's ``headline + kind + scope`` by synonym-aware lexical
    cosine to ``query`` (see :func:`expand`/:func:`embed` - lexical + curated
    synonyms, not semantic embeddings), then multiplies by a status weight
    (:data:`_STATUS_WEIGHT`) and a mild recency boost off the finding's
    ``window_end`` (falling back to ``window_start``). Returns the ``top_k``
    highest ``(score, finding)`` pairs, descending; ties keep input order.

    An empty/garbage query (no usable tokens) or empty input yields ``[]`` - the
    weights only ever scale a real text match, so they never invent relevance.
    """
    if top_k <= 0 or not findings:
        return []
    q = embed(query, synonyms=True)
    if not q:
        return []
    now = datetime.now(UTC)
    scored: list[tuple[float, Finding]] = []
    for f in findings:
        text = f"{f.headline} {f.kind} {f.scope}"
        base = cosine(q, embed(text, synonyms=True))
        if base <= 0.0:
            continue
        score = base * _status_weight(f) * _recency_boost(f, now)
        scored.append((score, f))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[:top_k]


def _status_weight(finding: Finding) -> float:
    status = getattr(finding.status, "value", finding.status)
    return _STATUS_WEIGHT.get(str(status), _STATUS_WEIGHT_DEFAULT)


def _recency_boost(finding: Finding, now: datetime) -> float:
    ts = finding.window_end or finding.window_start
    if ts is None:
        return 1.0
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    age = now - ts
    if age <= timedelta(0):
        return _RECENCY_MAX_BOOST
    if age >= _RECENCY_HALFLIFE:
        return 1.0
    fraction = 1.0 - age / _RECENCY_HALFLIFE
    return 1.0 + (_RECENCY_MAX_BOOST - 1.0) * fraction
