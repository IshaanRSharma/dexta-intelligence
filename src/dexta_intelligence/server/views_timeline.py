"""View-model logic for the Timeline (temporal episode graph) page.

Pure data shaping over the deterministic episode graph
(:mod:`dexta_intelligence.analytics.episodes`): this module builds the graph for
the active window and returns plain dicts. No model, no re-derivation, no HTML.
The interactive drawing lives in ``static/timeline.js``; it fetches the graph
dict this module produces from the ``/episodes.json`` endpoint.
"""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from typing import TYPE_CHECKING, Any

from dexta_intelligence.analytics.episodes import build_graph

if TYPE_CHECKING:
    from dexta_intelligence.config import Config
    from dexta_intelligence.store.port import StoragePort

__all__ = ["episode_graph_payload", "timeline_page_view"]


def _window_bounds(store: StoragePort, config: Config) -> tuple[datetime, datetime]:
    """The active analysis window as UTC datetimes spanning whole days.

    Ends on the last day with data (else today) and reaches back the configured
    deep-analysis window. Mirrors ``cli._common._analysis_window`` but resolves to
    day-spanning datetimes, which is what the episode detector and the axis want.
    """
    coverage = store.coverage()
    end_date = (
        coverage.last_ts.date() if coverage.last_ts is not None else datetime.now(tz=UTC).date()
    )
    start_date = end_date - timedelta(days=config.analysis.deep_analysis_window_days)
    if coverage.first_ts is not None:
        start_date = max(start_date, coverage.first_ts.date())
    start = datetime.combine(start_date, time.min, tzinfo=UTC)
    end = datetime.combine(end_date, time.max, tzinfo=UTC)
    return start, end


def episode_graph_payload(store: StoragePort, config: Config) -> dict[str, Any]:
    """The deterministic episode graph for the active window as a JSON-ready dict.

    Returns ``EpisodeGraph.to_dict()`` (``summary`` + ``nodes``) augmented with the
    axis ``window`` bounds and the analysis timezone, so the front-end can draw a
    stable time axis even when the window holds no episodes.
    """
    start, end = _window_bounds(store, config)
    graph = build_graph(
        store,
        start,
        end,
        target_low=config.analysis.target_low,
        target_high=config.analysis.target_high,
    )
    payload = graph.to_dict()
    payload["window"] = {"start": start.isoformat(), "end": end.isoformat()}
    payload["timezone"] = config.analysis.timezone
    payload["target"] = {
        "low": config.analysis.target_low,
        "high": config.analysis.target_high,
    }
    return payload


def timeline_page_view(store: StoragePort, config: Config) -> dict[str, Any]:
    """Shape the Timeline page: summary tiles, window label, and the graph payload.

    The page renders the summary and an empty-state server-side (so it degrades
    without JavaScript); ``timeline.js`` draws the interactive graph from the same
    payload embedded here, avoiding a second graph build on first paint.
    """
    payload = episode_graph_payload(store, config)
    summary = payload["summary"]
    n_episodes = sum(
        summary.get(k, 0) for k in ("num_hypo", "num_hyper", "n_sensor_gaps")
    )
    start = datetime.fromisoformat(payload["window"]["start"])
    end = datetime.fromisoformat(payload["window"]["end"])
    window_label = f"{start.strftime('%b %d, %Y')} to {end.strftime('%b %d, %Y')}"
    tiles = [
        {
            "label": "Low episodes",
            "value": summary.get("num_hypo", 0),
            "note": _severe_note(summary.get("n_severe_hypo", 0)),
        },
        {
            "label": "High episodes",
            "value": summary.get("num_hyper", 0),
            "note": _severe_note(summary.get("n_severe_hyper", 0)),
        },
        {
            "label": "Longest high",
            "value": _minutes_label(summary.get("longest_hyper_min", 0.0)),
            "note": "",
        },
        {
            "label": "Sensor gaps",
            "value": summary.get("n_sensor_gaps", 0),
            "note": "",
        },
    ]
    return {
        "payload": payload,
        "summary": summary,
        "tiles": tiles,
        "n_episodes": n_episodes,
        "has_episodes": n_episodes > 0,
        "window_label": window_label,
    }


def _severe_note(n_severe: int) -> str:
    if not n_severe:
        return ""
    return f"{n_severe} severe" if n_severe > 1 else "1 severe"


def _minutes_label(minutes: float) -> str:
    if not minutes:
        return "0 min"
    if minutes < 90:
        return f"{minutes:g} min"
    hours = minutes / 60.0
    return f"{hours:.1f} h"
