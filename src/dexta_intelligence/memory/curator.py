"""Deterministic context curator - a librarian over the episode graph and findings.

``select_context`` assembles the context window for one question: it collects
typed items (FACTS: episode nodes and metric bundles; BELIEFS: findings with
their temporal edges; CONVENTIONS: metric-ontology entries the query touches;
HISTORY: caller-provided prior tool results), scores them deterministically, and
fills a token budget with per-type floors. It returns the selection PLUS a drop
list where every drop carries a reason string suitable for a trace line.

Invariants (each tested):

1. Pruning reduces tokens, never ground-truth availability. The curator is a
   pure function - it takes collections, not a store, so it cannot mutate
   persistence, and every input item comes back as either selected or dropped.
2. Safety floor: severe episodes and treatment-gate inputs (caller-flagged
   ``HistoryItem.protected``) are never droppable, even over budget.
3. Every drop emits a reason string (see :meth:`ContextSelection.trace_lines`).
4. Same inputs produce the same selection and drop list: ``now`` is a
   parameter, scoring reads no wall clock, and all orderings are total.

No LLM anywhere: scoring is lexical cosine (:mod:`.embeddings`, offline and
deterministic) plus fixed weights. Token cost is the chars/4 heuristic
(:func:`estimate_tokens`) - cheap, deterministic, and close enough for budget
arithmetic; the budget is a cap on prompt space, not an exact tokenizer.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dexta_intelligence.guard.metrics import METRICS, Metric, metrics_in_context
from dexta_intelligence.memory.embeddings import cosine, embed
from dexta_intelligence.memory.freshness import finding_ttl_days
from dexta_intelligence.models import EdgeRelation

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import datetime

    from dexta_intelligence.analytics.episodes import Episode
    from dexta_intelligence.models import Finding, FindingEdge

__all__ = [
    "ContextItem",
    "ContextSelection",
    "ContextType",
    "Drop",
    "HistoryItem",
    "estimate_tokens",
    "select_context",
]

_SECONDS_PER_DAY = 86_400.0

#: Scoring weights. Relevance dominates (selection is query-driven), temporal
#: overlap next (the asked window is the view), salience keeps clinical severity
#: and surfaced contradictions ahead of trivia, freshness is a tiebreaker.
_W_RELEVANCE = 0.35
_W_TEMPORAL = 0.25
_W_SALIENCE = 0.25
_W_FRESHNESS = 0.15

#: Freshness horizon for items without their own TTL (findings reuse
#: :func:`~dexta_intelligence.memory.freshness.finding_ttl_days`).
_FRESHNESS_DAYS = 90.0
#: Out-of-window temporal decay horizon (days from the asked window).
_TEMPORAL_DECAY_DAYS = 7.0
#: Neutral score component when the signal is undefined for an item.
_NEUTRAL = 0.5


class ContextType(enum.StrEnum):
    FACTS = "facts"
    BELIEFS = "beliefs"
    CONVENTIONS = "conventions"
    HISTORY = "history"


#: Type prior, multiplicative: deterministic facts outrank model-era beliefs,
#: conventions are definitions (cheap, but only useful when touched), history is
#: re-derivable so it yields first.
_TYPE_PRIOR: dict[ContextType, float] = {
    ContextType.FACTS: 1.0,
    ContextType.BELIEFS: 0.85,
    ContextType.CONVENTIONS: 0.75,
    ContextType.HISTORY: 0.6,
}

#: Fraction of the post-safety-floor budget reserved per type so no type is
#: starved by one verbose neighbour; the unreserved 35% is filled purely by score.
_TYPE_FLOOR: dict[ContextType, float] = {
    ContextType.FACTS: 0.25,
    ContextType.BELIEFS: 0.20,
    ContextType.CONVENTIONS: 0.10,
    ContextType.HISTORY: 0.10,
}

_TYPE_ORDER: dict[ContextType, int] = {
    ContextType.FACTS: 0,
    ContextType.BELIEFS: 1,
    ContextType.CONVENTIONS: 2,
    ContextType.HISTORY: 3,
}


def estimate_tokens(text: str) -> int:
    """Deterministic token estimate: ceil(chars / 4), minimum 1.

    A fixed heuristic (roughly right for English prose and JSON-ish renders)
    keeps the curator model-free; exactness is not required to enforce a budget.
    """
    return max(1, (len(text) + 3) // 4)


@dataclass(frozen=True, slots=True)
class HistoryItem:
    """One caller-provided prior tool result.

    ``protected=True`` marks a treatment-gate input (treatment-context the
    safety layer relies on): it is never droppable.
    """

    text: str
    ts: datetime | None = None
    protected: bool = False


@dataclass(frozen=True, slots=True)
class ContextItem:
    """One scored candidate for the context window."""

    type: ContextType
    key: str
    text: str
    tokens: int
    score: float
    protected: bool = False


@dataclass(frozen=True, slots=True)
class Drop:
    """A pruned item and the trace-ready reason it was pruned."""

    item: ContextItem
    reason: str


@dataclass(frozen=True, slots=True)
class ContextSelection:
    """The curated window: what goes in, what was pruned, and why."""

    selected: tuple[ContextItem, ...]
    dropped: tuple[Drop, ...]
    budget_tokens: int
    used_tokens: int

    def trace_lines(self) -> tuple[str, ...]:
        """One trace line per drop - the receipt for every pruning decision."""
        return tuple(
            f"curator drop {d.item.type.value}:{d.item.key} ({d.item.tokens} tok): {d.reason}"
            for d in self.dropped
        )


def select_context(
    query: str,
    *,
    budget_tokens: int,
    now: datetime,
    window: tuple[datetime, datetime] | None = None,
    episodes: Sequence[Episode] = (),
    metric_bundle: Mapping[str, Any] | None = None,
    findings: Sequence[Finding] = (),
    edges: Sequence[FindingEdge] = (),
    history: Sequence[HistoryItem] = (),
) -> ContextSelection:
    """Deterministically curate a context window for ``query`` under a budget.

    ``window`` is the timeline span the question is about (temporal-overlap
    scoring); ``now`` anchors freshness (no wall-clock read happens here).
    ``edges`` are the findings' temporal edges: a finding party to a CONTRADICTS
    edge is surfaced (salience floor), and SUPERSEDES ancestry is annotated into
    the item text. The store is never touched: pass what the question may need,
    the curator only chooses a view over it.
    """
    q_vec = embed(query, synonyms=True)
    items: list[ContextItem] = []
    items.extend(
        _episode_item(ep, q_vec, window=window, now=now) for ep in episodes
    )
    if metric_bundle is not None:
        items.append(_bundle_item(metric_bundle, q_vec, window=window))
    contradicted, annotations = _edge_index(edges)
    items.extend(
        _finding_item(
            f, i, q_vec, window=window, now=now,
            contradicted=contradicted, annotations=annotations,
        )
        for i, f in enumerate(findings)
    )
    items.extend(_convention_item(m) for m in _touched_metrics(query))
    items.extend(
        _history_item(h, i, q_vec, window=window, now=now)
        for i, h in enumerate(history)
    )
    return _fill(items, budget_tokens)


# ── item builders ────────────────────────────────────────────────────────────


def _score(
    type_: ContextType,
    *,
    relevance: float,
    temporal: float,
    salience: float,
    freshness: float,
) -> float:
    weighted = (
        _W_RELEVANCE * relevance
        + _W_TEMPORAL * temporal
        + _W_SALIENCE * salience
        + _W_FRESHNESS * freshness
    )
    return _TYPE_PRIOR[type_] * weighted


def _relevance(q_vec: dict[str, float], text: str) -> float:
    return cosine(q_vec, embed(text, synonyms=True))


def _temporal(
    start: datetime | None,
    end: datetime | None,
    window: tuple[datetime, datetime] | None,
) -> float:
    """Overlap of the item's span with the asked window, in [0, 1].

    No window or no item time: neutral. Inside the window: overlap fraction of
    the shorter span. Outside: decays linearly from the neutral score to zero
    over :data:`_TEMPORAL_DECAY_DAYS`.
    """
    if window is None:
        return _NEUTRAL
    if start is None and end is None:
        return _NEUTRAL
    lo, hi = window
    s = start if start is not None else end
    e = end if end is not None else start
    assert s is not None and e is not None
    if e < s:
        s, e = e, s
    overlap = (min(e, hi) - max(s, lo)).total_seconds()
    if overlap >= 0:
        shorter = min((e - s).total_seconds(), (hi - lo).total_seconds())
        return 1.0 if shorter <= 0 else min(1.0, overlap / shorter)
    distance_days = -overlap / _SECONDS_PER_DAY
    return _NEUTRAL * max(0.0, 1.0 - distance_days / _TEMPORAL_DECAY_DAYS)


def _freshness(anchor: datetime | None, now: datetime, *, ttl_days: float) -> float:
    if anchor is None:
        return _NEUTRAL
    age_days = (now - anchor).total_seconds() / _SECONDS_PER_DAY
    if age_days <= 0:
        return 1.0
    return max(0.0, 1.0 - age_days / ttl_days)


def _episode_item(
    ep: Episode,
    q_vec: dict[str, float],
    *,
    window: tuple[datetime, datetime] | None,
    now: datetime,
) -> ContextItem:
    parts = [f"{ep.kind} {ep.start.isoformat()}..{ep.end.isoformat()}"]
    parts.append(f"{ep.duration_min} min")
    if ep.extreme_mg_dl is not None:
        parts.append(f"extreme {ep.extreme_mg_dl:g} mg/dL")
    if ep.severe:
        parts.append("severe")
    if ep.clinically_significant:
        parts.append("clinically significant")
    for link in ep.links:
        parts.append(f"context {link.kind} at {link.offset_min:+g} min")
    text = "; ".join(parts)
    if ep.severe:
        salience = 1.0
    elif ep.clinically_significant:
        salience = 0.85
    elif ep.kind == "sensor_gap":
        salience = 0.4
    else:
        salience = 0.55
    score = _score(
        ContextType.FACTS,
        relevance=_relevance(q_vec, text),
        temporal=_temporal(ep.start, ep.end, window),
        salience=salience,
        freshness=_freshness(ep.end, now, ttl_days=_FRESHNESS_DAYS),
    )
    return ContextItem(
        type=ContextType.FACTS,
        key=ep.id,
        text=text,
        tokens=estimate_tokens(text),
        score=score,
        protected=ep.severe,
    )


def _bundle_item(
    bundle: Mapping[str, Any],
    q_vec: dict[str, float],
    *,
    window: tuple[datetime, datetime] | None,
) -> ContextItem:
    text = "metrics: " + ", ".join(f"{k}={bundle[k]}" for k in sorted(bundle))
    score = _score(
        ContextType.FACTS,
        relevance=max(_relevance(q_vec, text), _bundle_key_overlap(q_vec, bundle)),
        temporal=_NEUTRAL if window is None else 1.0,
        salience=0.6,
        freshness=1.0,
    )
    return ContextItem(
        type=ContextType.FACTS,
        key="metric_bundle",
        text=text,
        tokens=estimate_tokens(text),
        score=score,
    )


def _bundle_key_overlap(q_vec: dict[str, float], bundle: Mapping[str, Any]) -> float:
    """Direct metric-name overlap: a query naming a bundled metric is relevant."""
    hits = sum(1 for k in bundle if f"w:{k.lower()}" in q_vec)
    return min(1.0, hits / max(1, min(3, len(bundle))))


def _edge_index(
    edges: Sequence[FindingEdge],
) -> tuple[frozenset[int], dict[int, tuple[str, ...]]]:
    """Finding ids party to a contradiction, and per-id edge annotations."""
    contradicted: set[int] = set()
    notes: dict[int, list[str]] = {}
    for e in edges:
        if e.relation is EdgeRelation.CONTRADICTS:
            contradicted.update((e.src_id, e.dst_id))
            notes.setdefault(e.src_id, []).append(
                f"contradicts finding #{e.dst_id} ({e.evidence})"
            )
            notes.setdefault(e.dst_id, []).append(
                f"contradicted by finding #{e.src_id} ({e.evidence})"
            )
        elif e.relation is EdgeRelation.SUPERSEDES:
            notes.setdefault(e.dst_id, []).append(f"superseded by finding #{e.src_id}")
    return frozenset(contradicted), {k: tuple(v) for k, v in notes.items()}


def _finding_item(
    f: Finding,
    index: int,
    q_vec: dict[str, float],
    *,
    window: tuple[datetime, datetime] | None,
    now: datetime,
    contradicted: frozenset[int],
    annotations: dict[int, tuple[str, ...]],
) -> ContextItem:
    parts = [f"[{f.status.value}] {f.headline} ({f.agent}/{f.kind}/{f.scope})"]
    if f.id is not None:
        for note in annotations.get(f.id, ()):
            parts.append(note)
    text = "; ".join(parts)
    salience = f.confidence
    if f.id is not None and f.id in contradicted:
        salience = max(salience, 0.9)  # surface conflicts, never bury them
    score = _score(
        ContextType.BELIEFS,
        relevance=_relevance(q_vec, text),
        temporal=_temporal(f.window_start, f.window_end, window),
        salience=salience,
        freshness=_freshness(
            f.last_verified or f.window_end, now, ttl_days=finding_ttl_days(f)
        ),
    )
    key = f"finding:{f.id}" if f.id is not None else f"finding:idx{index}"
    return ContextItem(
        type=ContextType.BELIEFS,
        key=key,
        text=text,
        tokens=estimate_tokens(text),
        score=score,
    )


def _touched_metrics(query: str) -> list[Metric]:
    """Ontology entries the query names, in deterministic ontology order."""
    touched = metrics_in_context(query)
    return [m for m in METRICS if m.canonical in touched]


def _convention_item(metric: Metric) -> ContextItem:
    text = (
        f"{metric.canonical}: names {', '.join(metric.prose_aliases)}; "
        f"evidence keys {', '.join(sorted(metric.key_aliases))}"
        + ("; percent metric" if metric.is_percent else "")
    )
    # Inclusion is already query-gated by the ontology match, so relevance is 1.
    score = _score(
        ContextType.CONVENTIONS,
        relevance=1.0,
        temporal=_NEUTRAL,
        salience=0.6,
        freshness=_NEUTRAL,
    )
    return ContextItem(
        type=ContextType.CONVENTIONS,
        key=f"metric:{metric.canonical}",
        text=text,
        tokens=estimate_tokens(text),
        score=score,
    )


def _history_item(
    h: HistoryItem,
    index: int,
    q_vec: dict[str, float],
    *,
    window: tuple[datetime, datetime] | None,
    now: datetime,
) -> ContextItem:
    score = _score(
        ContextType.HISTORY,
        relevance=_relevance(q_vec, h.text),
        temporal=_temporal(h.ts, h.ts, window),
        salience=0.5,
        freshness=_freshness(h.ts, now, ttl_days=_FRESHNESS_DAYS),
    )
    return ContextItem(
        type=ContextType.HISTORY,
        key=f"turn:{index}",
        text=h.text,
        tokens=estimate_tokens(h.text),
        score=score,
        protected=h.protected,
    )


# ── budget fill ──────────────────────────────────────────────────────────────


def _fill(items: list[ContextItem], budget: int) -> ContextSelection:
    """Protected first (safety floor, even over budget), then per-type floors,
    then best-score-first until the budget is spent. Total order everywhere.

    Floors are true reservations: each type gets a sub-budget carved from what
    the safety floor left (shares sum below 1), so a verbose type can never
    starve another type's floor. An item larger than its type's whole floor
    competes in the free phase instead.
    """
    ranked = sorted(items, key=lambda i: (-i.score, _TYPE_ORDER[i.type], i.key))
    chosen: dict[str, ContextItem] = {}
    used = 0
    for item in ranked:
        if item.protected:
            chosen[_ckey(item)] = item
            used += item.tokens

    reservable = max(0, budget - used)
    floor_used: dict[ContextType, int] = dict.fromkeys(_TYPE_ORDER, 0)
    for type_ in _TYPE_ORDER:
        floor = int(reservable * _TYPE_FLOOR[type_])
        for item in ranked:
            if item.type is not type_ or _ckey(item) in chosen:
                continue
            if floor_used[type_] + item.tokens > floor:
                continue
            chosen[_ckey(item)] = item
            used += item.tokens
            floor_used[type_] += item.tokens

    drops: list[Drop] = []
    for item in ranked:
        if _ckey(item) in chosen:
            continue
        if used + item.tokens <= budget:
            chosen[_ckey(item)] = item
            used += item.tokens
        else:
            drops.append(
                Drop(
                    item=item,
                    reason=(
                        f"over budget: {item.tokens} tokens > "
                        f"{max(0, budget - used)} remaining (score={item.score:.3f})"
                    ),
                )
            )

    selected = sorted(
        chosen.values(), key=lambda i: (_TYPE_ORDER[i.type], -i.score, i.key)
    )
    return ContextSelection(
        selected=tuple(selected),
        dropped=tuple(drops),
        budget_tokens=budget,
        used_tokens=used,
    )


def _ckey(item: ContextItem) -> str:
    return f"{item.type.value}:{item.key}"
