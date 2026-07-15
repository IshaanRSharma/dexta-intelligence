"""Deterministic change and contradiction surfacing over the findings graph.

Two readers of the bitemporal edges, no model anywhere:

- :func:`what_changed`: SUPERSEDES edges whose new finding actually says
  something different (a re-derive with the same headline is a re-verification,
  not a change), each with when it happened on the patient timeline, what
  stopped, what started, the edge's deterministic reason string, and the
  episode-summary counts in the weeks either side of the change.
- :func:`contradicted_beliefs`: CONTRADICTS edges, each pairing the disproved
  belief with the finding that disproved it and both evidence windows.

Both read only what deterministic code already wrote (edges are never
model-authored), so their output is safe to hand to an LLM as evidence.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from dexta_intelligence.analytics.episodes import detect_episodes, summarize
from dexta_intelligence.models import EdgeRelation

if TYPE_CHECKING:
    from dexta_intelligence.models import Finding
    from dexta_intelligence.store.port import StoragePort

__all__ = ["contradicted_beliefs", "what_changed"]

_FINDINGS_SCAN_LIMIT = 500
_MAX_ITEMS = 10
#: Days of episode context summarized on each side of a change point.
_EPISODE_CONTEXT_DAYS = 14
_EPISODE_KEYS = ("num_hypo", "num_hyper", "n_severe_hypo", "n_severe_hyper")


def what_changed(
    store: StoragePort, *, now: datetime, within_days: int = 90
) -> list[dict[str, Any]]:
    """Recent regime changes: {when, what_stopped, what_started, evidence, ...}.

    A SUPERSEDES edge marks a change only when the superseding finding says
    something different; identical headlines are re-verifications and skipped.
    Each entry carries the episode-summary counts for the
    :data:`_EPISODE_CONTEXT_DAYS` before and after the change point, so "the
    old pattern stopped" is checkable against the excursion record.
    """
    cutoff = now - timedelta(days=within_days)
    by_id = _findings_by_id(store)
    changes: list[dict[str, Any]] = []
    for edge in reversed(store.get_finding_edges(relation=EdgeRelation.SUPERSEDES)):
        when = edge.event_time or edge.knowledge_time
        if when < cutoff or when > now:
            continue
        new = by_id.get(edge.src_id)
        old = by_id.get(edge.dst_id)
        if new is None or old is None:
            continue
        if new.headline.strip() == old.headline.strip():
            continue
        context = timedelta(days=_EPISODE_CONTEXT_DAYS)
        changes.append(
            {
                "when": when.isoformat(),
                "what_stopped": old.headline,
                "what_started": new.headline,
                "evidence": edge.evidence,
                "old_window": _window(old),
                "new_window": _window(new),
                "episodes_before": _episode_counts(store, when - context, when),
                "episodes_after": _episode_counts(store, when, when + context),
            }
        )
        if len(changes) >= _MAX_ITEMS:
            break
    return changes


def contradicted_beliefs(store: StoragePort) -> list[dict[str, Any]]:
    """Disproved beliefs, newest first: the belief, its contradicting finding,
    the deterministic opposition evidence, and both evidence windows."""
    by_id = _findings_by_id(store)
    out: list[dict[str, Any]] = []
    for edge in reversed(store.get_finding_edges(relation=EdgeRelation.CONTRADICTS)):
        new = by_id.get(edge.src_id)
        old = by_id.get(edge.dst_id)
        if new is None or old is None:
            continue
        when = edge.event_time or edge.knowledge_time
        out.append(
            {
                "belief": old.headline,
                "belief_status": old.status.value,
                "belief_window": _window(old),
                "contradicted_by": new.headline,
                "contradicting_window": _window(new),
                "evidence": edge.evidence,
                "when": when.isoformat(),
            }
        )
        if len(out) >= _MAX_ITEMS:
            break
    return out


def _findings_by_id(store: StoragePort) -> dict[int, Finding]:
    return {
        f.id: f
        for f in store.get_findings(status=None, limit=_FINDINGS_SCAN_LIMIT)
        if f.id is not None
    }


def _window(finding: Finding) -> dict[str, str | None]:
    return {
        "start": finding.window_start.isoformat() if finding.window_start else None,
        "end": finding.window_end.isoformat() if finding.window_end else None,
    }


def _episode_counts(store: StoragePort, start: datetime, end: datetime) -> dict[str, Any]:
    summary = summarize(detect_episodes(store, start, end))
    return {k: summary[k] for k in _EPISODE_KEYS}
