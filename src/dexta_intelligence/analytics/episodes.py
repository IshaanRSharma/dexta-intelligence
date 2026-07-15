"""Temporal episode graph: glycemic excursions and sensor gaps as first-class,
segmented objects with typed edges to the treatment/behaviour context around them.

The LLM-CGM benchmark (Healey & Kohane, PSB 2025) found a code-execution agent's
worst scores were all temporal-segmentation tasks: longest time in hyperglycemia,
counting hypoglycemic episodes, meal-window extremes, sensor disconnects. The
tool belt computes those on demand and throws the segments away. This module
makes them durable, deterministic nodes:

- an :class:`Episode` per contiguous excursion (hypo / hyper) or sensor gap, with
  its span, extreme, duration, and clinical-significance flag;
- typed :class:`ContextLink` edges to the meals, boluses, activity, and sleep in
  the window around it, each with a signed time offset.

It is a pure function over the store: no model, no numpy, byte-deterministic given
the same events. Excursion thresholds follow the benchmark's own definitions
(strict ``> 180`` / ``< 70``) so episode facts line up with the scored ground
truth. Durations are endpoint-elapsed (last minus first in-run reading), matching
the ``find_lows`` / ``find_spikes`` tool convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import pairwise
from typing import TYPE_CHECKING, Any

from dexta_intelligence.models import InsulinKind

if TYPE_CHECKING:
    from dexta_intelligence.store.port import StoragePort

__all__ = [
    "CLINICAL_MIN_MINUTES",
    "ContextLink",
    "Episode",
    "EpisodeGraph",
    "build_graph",
    "detect_episodes",
    "summarize",
]

#: Consensus glucose thresholds (mg/dL). Strict comparisons match the LLM-CGM
#: ground-truth definitions used by the benchmark harness.
TARGET_LOW = 70
TARGET_HIGH = 180
SEVERE_LOW = 54
SEVERE_HIGH = 250

#: A contiguous excursion of at least this many minutes is a clinically
#: significant event (the consensus hypo/hyper-event definition).
CLINICAL_MIN_MINUTES = 15
#: A reading gap longer than this is a sensor gap (mirrors workflows.monitor).
GAP_MIN_MINUTES = 30

#: How far before an episode a context event may sit and still be linked, by kind
#: (minutes). Activity reaches furthest back: post-exercise insulin sensitization
#: drives lows hours later. Sleep is linked by interval overlap, not a window.
_PRE_MIN: dict[str, float] = {"meal": 180.0, "bolus": 180.0, "activity": 360.0}
#: How far after an episode's end a context event may sit and still be linked.
_POST_MIN = 60.0


@dataclass(frozen=True, slots=True)
class ContextLink:
    """A typed edge from an episode to a nearby context event.

    ``offset_min`` is signed minutes from the episode start (negative = before it),
    so "a 45 g breakfast 20 min before this high" is legible without re-deriving.
    """

    kind: str  # "meal" | "bolus" | "activity" | "sleep"
    ts: datetime
    offset_min: float
    detail: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "ts": self.ts.isoformat(),
                "offset_min": self.offset_min, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class Episode:
    """One contiguous glycemic excursion or sensor gap, with its context edges.

    ``id`` is a stable, human-legible node handle (``hyper:2025-01-16T03:10:00+00:00``)
    so the graph is addressable: an agent can name an episode and traverse to the
    context around it rather than re-deriving it from a trace.
    """

    id: str
    kind: str  # "hypo" | "hyper" | "sensor_gap"
    start: datetime
    end: datetime
    duration_min: float
    n_readings: int
    severe: bool
    clinically_significant: bool
    extreme_mg_dl: float | None
    extreme_ts: datetime | None
    links: tuple[ContextLink, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "kind": self.kind, "start": self.start.isoformat(),
            "end": self.end.isoformat(), "duration_min": self.duration_min,
            "n_readings": self.n_readings, "severe": self.severe,
            "clinically_significant": self.clinically_significant,
            "extreme_mg_dl": self.extreme_mg_dl,
            "extreme_ts": self.extreme_ts.isoformat() if self.extreme_ts else None,
            "links": [link.to_dict() for link in self.links],
        }


def _episode_id(kind: str, start: datetime) -> str:
    return f"{kind}:{start.isoformat()}"


def _minutes(a: datetime, b: datetime) -> float:
    return (a - b).total_seconds() / 60.0


def _excursions(
    readings: list[tuple[datetime, int]], *, low: int, high: int,
) -> list[Episode]:
    """Contiguous hypo (< low) and hyper (> high) runs as episodes (no links yet)."""
    episodes: list[Episode] = []
    run: list[tuple[datetime, int]] = []
    run_kind: str | None = None

    def close() -> None:
        if not run or run_kind is None:
            return
        start, end = run[0][0], run[-1][0]
        dur = _minutes(end, start)
        if run_kind == "hypo":
            ext_ts, ext = min(run, key=lambda r: r[1])
            severe = ext < SEVERE_LOW
        else:
            ext_ts, ext = max(run, key=lambda r: r[1])
            severe = ext > SEVERE_HIGH
        episodes.append(Episode(
            id=_episode_id(run_kind, start), kind=run_kind, start=start, end=end,
            duration_min=round(dur, 1), n_readings=len(run), severe=severe,
            clinically_significant=dur >= CLINICAL_MIN_MINUTES,
            extreme_mg_dl=float(ext), extreme_ts=ext_ts,
        ))

    for ts, v in readings:
        kind = "hypo" if v < low else "hyper" if v > high else None
        if kind != run_kind:
            close()
            run, run_kind = [], kind
        if kind is not None:
            run.append((ts, v))
    close()
    return episodes


def _sensor_gaps(readings: list[tuple[datetime, int]], gap_min: float) -> list[Episode]:
    """Runs between consecutive readings farther apart than ``gap_min`` minutes."""
    gaps: list[Episode] = []
    for (t0, _), (t1, _) in pairwise(readings):
        dur = _minutes(t1, t0)
        if dur > gap_min:
            gaps.append(Episode(
                id=_episode_id("sensor_gap", t0), kind="sensor_gap", start=t0, end=t1,
                duration_min=round(dur, 1), n_readings=0, severe=False,
                clinically_significant=False, extreme_mg_dl=None, extreme_ts=None,
            ))
    return gaps


def _link(ep: Episode, kind: str, ts: datetime, detail: dict[str, Any]) -> ContextLink:
    return ContextLink(kind=kind, ts=ts, offset_min=round(_minutes(ts, ep.start), 1),
                       detail=detail)


def _attach_context(episodes: list[Episode], store: StoragePort,
                    start: datetime, end: datetime) -> list[Episode]:
    """Bind meals/boluses/activity/sleep to each excursion episode as edges.

    Context is fetched over a window widened by the largest pre-episode reach, so
    an event that precedes the analysis window but still bears on an early episode
    is available to link.
    """
    lo = start - timedelta(minutes=max(_PRE_MIN.values()))
    hi = end + timedelta(minutes=_POST_MIN)
    meals = store.get_meals(lo, hi)
    boluses = [i for i in store.get_insulin(lo, hi) if i.kind == InsulinKind.BOLUS]
    activity = store.get_activity(lo, hi)
    sleep = store.get_sleep(lo, hi)

    def in_window(ts: datetime, ep: Episode, kind: str) -> bool:
        return (ep.start - timedelta(minutes=_PRE_MIN[kind]) <= ts
                <= ep.end + timedelta(minutes=_POST_MIN))

    out: list[Episode] = []
    for ep in episodes:
        if ep.kind == "sensor_gap":
            out.append(ep)
            continue
        links: list[ContextLink] = []
        for m in meals:
            if in_window(m.ts, ep, "meal"):
                links.append(_link(ep, "meal", m.ts, {"carbs_g": m.carbs_g, "note": m.note}))
        for b in boluses:
            if in_window(b.ts, ep, "bolus"):
                links.append(_link(ep, "bolus", b.ts, {"units": b.units, "automatic": b.automatic}))
        for a in activity:
            if in_window(a.ts, ep, "activity"):
                links.append(_link(ep, "activity", a.ts,
                                   {"kind": a.kind, "intensity": a.intensity}))
        for s in sleep:
            if s.ts_start <= ep.end and s.ts_end >= ep.start:  # interval overlap
                links.append(_link(ep, "sleep", s.ts_start, {"score": s.score}))
        links.sort(key=lambda link: link.offset_min)
        out.append(Episode(
            id=ep.id, kind=ep.kind, start=ep.start, end=ep.end,
            duration_min=ep.duration_min, n_readings=ep.n_readings, severe=ep.severe,
            clinically_significant=ep.clinically_significant,
            extreme_mg_dl=ep.extreme_mg_dl, extreme_ts=ep.extreme_ts, links=tuple(links),
        ))
    return out


def detect_episodes(
    store: StoragePort, start: datetime, end: datetime, *,
    target_low: int = TARGET_LOW, target_high: int = TARGET_HIGH,
    gap_min: float = GAP_MIN_MINUTES,
) -> list[Episode]:
    """Segment ``[start, end]`` into hypo/hyper excursions and sensor gaps, each an
    :class:`Episode` with its context edges, ordered by start time.

    Deterministic and model-free. Returns an empty list when there are no readings.
    """
    readings = [(g.ts, g.mg_dl) for g in store.get_glucose(start, end)]
    readings.sort(key=lambda r: r[0])
    if not readings:
        return []
    episodes = _excursions(readings, low=target_low, high=target_high)
    episodes = _attach_context(episodes, store, start, end)
    episodes += _sensor_gaps(readings, gap_min)
    episodes.sort(key=lambda e: e.start)
    return episodes


@dataclass(frozen=True, slots=True)
class EpisodeGraph:
    """An addressable, traversable view over a window's episodes.

    Nodes are :class:`Episode` objects keyed by ``id``; edges are their
    :class:`ContextLink`\\ s. ``node`` and ``at`` are the two entry points an agent
    uses: name a node, or find the one covering a moment, then read its edges.
    """

    episodes: tuple[Episode, ...]

    def node(self, episode_id: str) -> Episode | None:
        return next((e for e in self.episodes if e.id == episode_id), None)

    def at(self, ts: datetime) -> Episode | None:
        """The excursion covering ``ts``, else the nearest excursion by start time.

        Reverse traversal: from a moment (or a context event's time) to the episode
        it belongs to. Sensor gaps are skipped; they are not excursions to explain.
        """
        excursions = [e for e in self.episodes if e.kind != "sensor_gap"]
        covering = [e for e in excursions if e.start <= ts <= e.end]
        if covering:
            return covering[0]
        return min(excursions, key=lambda e: abs(_minutes(e.start, ts)), default=None)

    def summary(self) -> dict[str, Any]:
        return summarize(list(self.episodes))

    def to_dict(self) -> dict[str, Any]:
        return {"summary": self.summary(), "nodes": [e.to_dict() for e in self.episodes]}


def build_graph(
    store: StoragePort, start: datetime, end: datetime, *,
    target_low: int = TARGET_LOW, target_high: int = TARGET_HIGH,
    gap_min: float = GAP_MIN_MINUTES,
) -> EpisodeGraph:
    """Detect episodes over ``[start, end]`` and wrap them as a traversable graph."""
    episodes = detect_episodes(
        store, start, end, target_low=target_low, target_high=target_high, gap_min=gap_min
    )
    return EpisodeGraph(episodes=tuple(episodes))


def summarize(episodes: list[Episode]) -> dict[str, Any]:
    """Roll episodes up to the ontology-aligned counts the guard and UI consume.

    Keys line up with the metric ontology (``num_hypo``, ``longest_hyper_min``) so
    this doubles as a faithfulness-guard evidence bundle for episode-scoped prose.
    """
    hypo = [e for e in episodes if e.kind == "hypo"]
    hyper = [e for e in episodes if e.kind == "hyper"]
    gaps = [e for e in episodes if e.kind == "sensor_gap"]
    return {
        "num_hypo": len(hypo),
        "num_hyper": len(hyper),
        "n_sensor_gaps": len(gaps),
        "n_clinically_significant_hypo": sum(1 for e in hypo if e.clinically_significant),
        "n_severe_hypo": sum(1 for e in hypo if e.severe),
        "n_severe_hyper": sum(1 for e in hyper if e.severe),
        "longest_hyper_min": max((e.duration_min for e in hyper), default=0.0),
        "longest_hypo_min": max((e.duration_min for e in hypo), default=0.0),
    }
