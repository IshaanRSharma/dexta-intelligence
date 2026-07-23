"""Autonomous curiosity: scan the deterministic episode graph for salient,
recurring structure and bank it as open hypotheses for later investigation.

Deterministic and model-free: a librarian of "what is worth looking into",
grounded in the episode graph and its causal chains (not ad-hoc detectors). The
daemon runs it on a cadence; each wonder becomes an OPEN
:class:`~dexta_intelligence.models.Hypothesis` the investigator / deep pass
resumes across cycles. Wonders are deduped by a stable ``kind`` against the
existing open set, so a standing pattern is banked once, not every cycle.

Every wonder is an observational question ("is there a common trigger?"), never a
treatment suggestion - the dosing gate is upstream of any answer, but curiosity
does not phrase a dose either.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from dexta_intelligence.models import Hypothesis, HypothesisStatus

if TYPE_CHECKING:
    from dexta_intelligence.analytics.episodes import EpisodeGraph
    from dexta_intelligence.store.port import StoragePort

__all__ = ["bank_curiosities", "scan_curiosities"]

#: A pattern needs at least this many instances to be worth banking a wonder.
_MIN_RECURRENCE = 3
#: Time-of-day clustering only wonders when one window holds this share of events.
_TOD_MAJORITY = 0.6

_CHAIN_PHRASE = {
    "rebound_after_low": "rebounded high after treating a low",
    "low_after_high": "dropped low after a high",
}

_TOD_BUCKETS = ((0, 6, "overnight"), (6, 12, "morning"), (12, 18, "afternoon"), (18, 24, "evening"))


def _bucket(hour: int) -> str:
    for lo, hi, name in _TOD_BUCKETS:
        if lo <= hour < hi:
            return name
    return "overnight"


def _chain_wonder(relation: str, n: int) -> str:
    phrase = _CHAIN_PHRASE.get(relation, relation.replace("_", " "))
    return (
        f"You {phrase} {n} times. Is this a recurring pattern worth investigating "
        "(its timing, context, or a common trigger)?"
    )


def _tod_wonders(graph: EpisodeGraph) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for kind, label in (("hyper", "highs"), ("hypo", "lows")):
        eps = [e for e in graph.episodes if e.kind == kind]
        if len(eps) < _MIN_RECURRENCE:
            continue
        counts = Counter(_bucket(e.start.hour) for e in eps)
        bucket, n = counts.most_common(1)[0]
        if n >= _MIN_RECURRENCE and n >= _TOD_MAJORITY * len(eps):
            out.append((
                f"tod_cluster:{kind}:{bucket}",
                f"Most of your {label} ({n} of {len(eps)}) happen in the {bucket} (UTC). "
                f"What is driving the {bucket} pattern?",
            ))
    return out


def scan_curiosities(graph: EpisodeGraph) -> list[tuple[str, str]]:
    """Deterministic ``(kind, statement)`` wonders about the graph's salient
    recurring structure: recurring causal chains, clusters of severe episodes, and
    time-of-day concentration. ``kind`` is a stable key for deduping."""
    out: list[tuple[str, str]] = []

    relations = Counter(e.relation for e in graph.edges if e.relation != "follows")
    for relation in sorted(relations):
        count = relations[relation]
        if count >= _MIN_RECURRENCE:
            out.append((f"recurring_chain:{relation}", _chain_wonder(relation, count)))

    severe_low = sum(1 for e in graph.episodes if e.kind == "hypo" and e.severe)
    severe_high = sum(1 for e in graph.episodes if e.kind == "hyper" and e.severe)
    if severe_low >= _MIN_RECURRENCE:
        out.append((
            "severe_hypo_cluster",
            f"You had {severe_low} severe low episodes (below 54 mg/dL). Is there a "
            "common trigger worth investigating?",
        ))
    if severe_high >= _MIN_RECURRENCE:
        out.append((
            "severe_hyper_cluster",
            f"You had {severe_high} severe high episodes (above 250 mg/dL). Is there a "
            "common cause worth investigating?",
        ))

    out.extend(_tod_wonders(graph))
    return out


def _curiosity_kind(hypothesis: Hypothesis) -> str | None:
    for test in hypothesis.tests or ():
        if isinstance(test, dict) and "curiosity_kind" in test:
            return str(test["curiosity_kind"])
    return None


def bank_curiosities(store: StoragePort, graph: EpisodeGraph) -> list[Hypothesis]:
    """Bank new wonders as OPEN hypotheses, deduped by ``kind`` against the existing
    open set so a standing pattern is banked once. Returns the newly banked ones,
    each tagged with its ``curiosity_kind`` so the dedup and the source are traceable."""
    open_kinds = {
        _curiosity_kind(h)
        for h in store.get_hypotheses(status=HypothesisStatus.OPEN.value)
    }
    banked: list[Hypothesis] = []
    for kind, statement in scan_curiosities(graph):
        if kind in open_kinds:
            continue
        hypothesis = Hypothesis(
            statement=statement,
            status=HypothesisStatus.OPEN,
            tests=[{"curiosity_kind": kind}],
        )
        new_id = store.insert_hypothesis(hypothesis)
        banked.append(hypothesis.model_copy(update={"id": new_id}))
        open_kinds.add(kind)
    return banked
