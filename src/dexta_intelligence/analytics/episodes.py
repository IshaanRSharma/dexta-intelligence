"""Temporal episode graph: glycemic excursions and sensor gaps as first-class,
segmented objects with typed edges to the treatment/behaviour context around them.

In the LLM-CGM benchmark (Healey & Kohane, PSB 2025) GPT-4 scored lowest on the
anomaly-detection and pattern-recognition categories, with several individual
temporal-extreme questions also low (longest hyperglycemia, counting hypoglycemic
episodes, meal-window extremes, sensor disconnects). Those are all
episode-boundary computations: contiguous excursions and gaps the tool belt
computes on demand and throws away. This module makes them durable, deterministic
nodes so the segmentation is held fixed:

- an :class:`Episode` per contiguous excursion (hypo / hyper) or sensor gap, with
  its span, extreme, duration, and clinical-significance flag;
- typed :class:`ContextLink` edges to the meals, boluses, activity, and sleep in
  the window around it, each with a signed time offset. A meal and the manual
  bolus that recorded the same action pair into one "treatment" edge
  (:func:`pair_treatments`), biased toward NOT merging: a false "separate" is
  safe, a false "merged" hides the missed-bolus signal.

It is a pure function over the store: no model, no numpy, byte-deterministic given
the same events. Excursion thresholds are the 2019 international consensus cut
points (Battelino et al., Diabetes Care 2019: strict ``> 180`` / ``< 70``, severe
``> 250`` / ``< 54``); the LLM-CGM benchmark adopts the same definitions, so
episode facts line up with its scored ground truth. Durations are endpoint-elapsed
(last minus first in-run reading), matching the ``find_lows`` / ``find_spikes``
tool convention. A run never spans a sensor gap: two excursions either side of a
dark sensor are separate observed episodes, and a chain across a gap is only the
weak ``follows`` since the trajectory through the hole is unobserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import pairwise
from typing import TYPE_CHECKING, Any

from dexta_intelligence.models import InsulinKind

if TYPE_CHECKING:
    from dexta_intelligence.models import InsulinEvent, MealEvent
    from dexta_intelligence.store.port import StoragePort

__all__ = [
    "CLINICAL_MIN_MINUTES",
    "ContextLink",
    "Episode",
    "EpisodeEdge",
    "EpisodeGraph",
    "build_graph",
    "detect_episodes",
    "pair_treatments",
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
_PRE_MIN: dict[str, float] = {
    "meal": 180.0, "bolus": 180.0, "treatment": 180.0, "activity": 360.0,
}
#: How far after an episode's end a context event may sit and still be linked.
_POST_MIN = 60.0

#: A meal and a manual bolus this close (minutes) are candidates for one
#: treatment (a single bolus-wizard action recorded as two events).
TREATMENT_PAIR_MAX_MIN = 15.0

#: Consecutive excursions further apart than this (first end to next start,
#: minutes) are independent events, not a chain.
CHAIN_MAX_GAP_MIN = 180.0


@dataclass(frozen=True, slots=True)
class ContextLink:
    """A typed edge from an episode to a nearby context event.

    ``offset_min`` is signed minutes from the episode start (negative = before it),
    so "a 45 g breakfast 20 min before this high" is legible without re-deriving.
    """

    kind: str  # "treatment" | "meal" | "bolus" | "activity" | "sleep"
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


@dataclass(frozen=True, slots=True)
class EpisodeEdge:
    """A typed edge between consecutive excursions (first END to next START).

    Relation names are descriptive geometry, never blame: ``rebound_after_low``
    is a low, then a carb-bearing bridge event in the gap, then a high;
    ``low_after_high`` is the insulin-bridged mirror; anything else within the
    chain window is the weak ``follows``. ``bridge`` is the load-bearing event
    in the gap, its ``offset_min`` measured from the first episode's end.
    """

    src_id: str
    dst_id: str
    relation: str  # "rebound_after_low" | "low_after_high" | "follows"
    gap_min: float
    bridge: ContextLink | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "src_id": self.src_id, "dst_id": self.dst_id, "relation": self.relation,
            "gap_min": self.gap_min,
            "bridge": self.bridge.to_dict() if self.bridge else None,
        }


def _episode_id(kind: str, start: datetime) -> str:
    return f"{kind}:{start.isoformat()}"


def _minutes(a: datetime, b: datetime) -> float:
    return (a - b).total_seconds() / 60.0


def _excursions(
    readings: list[tuple[datetime, int]], *, low: int, high: int, gap_min: float,
) -> list[Episode]:
    """Contiguous hypo (< low) and hyper (> high) runs as episodes (no links yet).

    A run is also broken by a sensor gap longer than ``gap_min``: two lows either
    side of a dark sensor are separate observed episodes, not one whose duration
    silently spans the hole and falsely trips clinical significance.
    """
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
        gap_break = bool(run) and kind == run_kind and _minutes(ts, run[-1][0]) > gap_min
        if kind != run_kind or gap_break:
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


def pair_treatments(
    meals: list[MealEvent], boluses: list[InsulinEvent],
) -> tuple[
    list[tuple[MealEvent, InsulinEvent]], list[MealEvent], list[InsulinEvent]
]:
    """Pair meals with the manual boluses that recorded the same treatment.

    A bolus-wizard action lands as two events (carbs and units); downstream the
    pair should read as one "treatment 58 g + 5.2 U" node. Two rules, both
    deliberately conservative because a false "separate" is safe while a false
    "merged" erases the missed-bolus signal:

    - a shared ``raw_event_id`` means the two events came from one device record;
    - otherwise a mutual, unambiguous nearest match within
      ``TREATMENT_PAIR_MAX_MIN`` minutes, skipping automatic boluses. Any
      ambiguity (two candidate boluses for a meal, two candidate meals for a
      bolus) pairs nothing.

    Returns ``(pairs, unpaired_meals, unpaired_boluses)``, all ordered by time.
    """
    ms = sorted(meals, key=lambda m: m.ts)
    bs = sorted(boluses, key=lambda b: b.ts)
    used_m: set[int] = set()
    used_b: set[int] = set()
    pairs: list[tuple[MealEvent, InsulinEvent]] = []

    by_raw: dict[int, list[int]] = {}
    for j, b in enumerate(bs):
        if b.raw_event_id is not None:
            by_raw.setdefault(b.raw_event_id, []).append(j)
    for i, m in enumerate(ms):
        if m.raw_event_id is None:
            continue
        js = [j for j in by_raw.get(m.raw_event_id, []) if j not in used_b]
        if len(js) == 1:
            pairs.append((m, bs[js[0]]))
            used_m.add(i)
            used_b.add(js[0])

    def close(m: MealEvent, b: InsulinEvent) -> bool:
        return abs(_minutes(b.ts, m.ts)) <= TREATMENT_PAIR_MAX_MIN

    meal_cands = {
        i: [j for j, b in enumerate(bs)
            if j not in used_b and not b.automatic and close(ms[i], b)]
        for i in range(len(ms)) if i not in used_m
    }
    bolus_cands = {
        j: [i for i in meal_cands if j in meal_cands[i]]
        for j in range(len(bs)) if j not in used_b
    }
    for i, js in meal_cands.items():
        if len(js) == 1 and bolus_cands.get(js[0]) == [i]:
            pairs.append((ms[i], bs[js[0]]))
            used_m.add(i)
            used_b.add(js[0])

    pairs.sort(key=lambda p: p[0].ts)
    rest_m = [m for i, m in enumerate(ms) if i not in used_m]
    rest_b = [b for j, b in enumerate(bs) if j not in used_b]
    return pairs, rest_m, rest_b


@dataclass(frozen=True, slots=True)
class _WindowContext:
    """Every context event in (a widened) analysis window, treatments resolved."""

    treatments: list[tuple[MealEvent, InsulinEvent]]
    meals: list[MealEvent]
    boluses: list[InsulinEvent]
    activity: list[Any]
    sleep: list[Any]


def _fetch_context(store: StoragePort, start: datetime, end: datetime) -> _WindowContext:
    """Fetch over a window widened by the largest pre-episode reach, so an event
    preceding the analysis window but bearing on an early episode is available."""
    lo = start - timedelta(minutes=max(_PRE_MIN.values()))
    hi = end + timedelta(minutes=_POST_MIN)
    all_meals = store.get_meals(lo, hi)
    all_boluses = [i for i in store.get_insulin(lo, hi) if i.kind == InsulinKind.BOLUS]
    # Resolve carb+bolus pairs into treatments before attaching, so an episode
    # links to one "treatment 58 g + 5.2 U" node instead of two halves.
    treatments, meals, boluses = pair_treatments(all_meals, all_boluses)
    return _WindowContext(
        treatments=treatments, meals=meals, boluses=boluses,
        activity=store.get_activity(lo, hi), sleep=store.get_sleep(lo, hi),
    )


def _attach_context(episodes: list[Episode], ctx: _WindowContext) -> list[Episode]:
    """Bind treatments/meals/boluses/activity/sleep to each excursion as edges."""
    treatments, meals, boluses = ctx.treatments, ctx.meals, ctx.boluses
    activity, sleep = ctx.activity, ctx.sleep

    def in_window(ts: datetime, ep: Episode, kind: str) -> bool:
        return (ep.start - timedelta(minutes=_PRE_MIN[kind]) <= ts
                <= ep.end + timedelta(minutes=_POST_MIN))

    out: list[Episode] = []
    for ep in episodes:
        if ep.kind == "sensor_gap":
            out.append(ep)
            continue
        links: list[ContextLink] = []
        for m, b in treatments:
            if in_window(m.ts, ep, "treatment"):
                links.append(_link(ep, "treatment", m.ts, {
                    "carbs_g": m.carbs_g, "units": b.units, "automatic": b.automatic,
                    "note": m.note, "bolus_dt_min": round(_minutes(b.ts, m.ts), 1),
                }))
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


def _detect(
    store: StoragePort, start: datetime, end: datetime, *,
    target_low: int, target_high: int, gap_min: float,
) -> tuple[list[Episode], _WindowContext | None]:
    readings = [(g.ts, g.mg_dl) for g in store.get_glucose(start, end)]
    readings.sort(key=lambda r: r[0])
    if not readings:
        return [], None
    ctx = _fetch_context(store, start, end)
    episodes = _excursions(readings, low=target_low, high=target_high, gap_min=gap_min)
    episodes = _attach_context(episodes, ctx)
    episodes += _sensor_gaps(readings, gap_min)
    episodes.sort(key=lambda e: e.start)
    return episodes, ctx


def detect_episodes(
    store: StoragePort, start: datetime, end: datetime, *,
    target_low: int = TARGET_LOW, target_high: int = TARGET_HIGH,
    gap_min: float = GAP_MIN_MINUTES,
) -> list[Episode]:
    """Segment ``[start, end]`` into hypo/hyper excursions and sensor gaps, each an
    :class:`Episode` with its context edges, ordered by start time.

    Deterministic and model-free. Returns an empty list when there are no readings.
    """
    episodes, _ = _detect(
        store, start, end, target_low=target_low, target_high=target_high, gap_min=gap_min
    )
    return episodes


def _gap_events(
    ctx: _WindowContext, lo: datetime, hi: datetime,
) -> list[tuple[str, datetime, dict[str, Any]]]:
    """Every carb- or insulin-bearing event inside ``[lo, hi]`` as (kind, ts, detail)."""
    out: list[tuple[str, datetime, dict[str, Any]]] = []
    for m, b in ctx.treatments:
        if lo <= m.ts <= hi:
            out.append(("treatment", m.ts, {
                "carbs_g": m.carbs_g, "units": b.units, "automatic": b.automatic,
                "note": m.note,
            }))
    for m in ctx.meals:
        if lo <= m.ts <= hi:
            out.append(("meal", m.ts, {"carbs_g": m.carbs_g, "note": m.note}))
    for b in ctx.boluses:
        if lo <= b.ts <= hi:
            out.append(("bolus", b.ts, {"units": b.units, "automatic": b.automatic}))
    return out


def _pick_bridge(
    events: list[tuple[str, datetime, dict[str, Any]]], key: str, anchor: datetime,
) -> ContextLink | None:
    """The most load-bearing event (largest ``key`` amount, earliest on a tie),
    offset from the first episode's end."""
    loaded = [
        (kind, ts, detail) for kind, ts, detail in events
        if isinstance(detail.get(key), (int, float)) and detail[key] > 0
    ]
    if not loaded:
        return None
    kind, ts, detail = max(loaded, key=lambda e: (e[2][key], -e[1].timestamp()))
    return ContextLink(kind=kind, ts=ts,
                       offset_min=round(_minutes(ts, anchor), 1), detail=detail)


def _chain_episodes(episodes: list[Episode], ctx: _WindowContext) -> list[EpisodeEdge]:
    """Edges between consecutive excursions no more than CHAIN_MAX_GAP_MIN apart.

    A confident relation needs a load-bearing bridge event in the gap (carbs for
    low-then-high, insulin for high-then-low); everything else stays the weak
    "follows". Names describe the geometry and never assign blame.
    """
    excursions = [e for e in episodes if e.kind != "sensor_gap"]
    sensor_gaps = [e for e in episodes if e.kind == "sensor_gap"]
    edges: list[EpisodeEdge] = []
    for a, b in pairwise(excursions):
        gap = round(_minutes(b.start, a.end), 1)
        if gap < 0 or gap > CHAIN_MAX_GAP_MIN:
            continue
        # A dark sensor between the two excursions leaves the trajectory through
        # the hole unobserved, so keep the weak "follows" and never claim a
        # confident rebound.
        crosses_gap = any(g.start < b.start and g.end > a.end for g in sensor_gaps)
        events = _gap_events(ctx, a.end, b.start)
        relation = "follows"
        bridge: ContextLink | None = None
        if not crosses_gap and a.kind == "hypo" and b.kind == "hyper":
            bridge = _pick_bridge(events, "carbs_g", a.end)
            if bridge is not None:
                relation = "rebound_after_low"
        elif not crosses_gap and a.kind == "hyper" and b.kind == "hypo":
            bridge = _pick_bridge(events, "units", a.end)
            if bridge is not None:
                relation = "low_after_high"
        edges.append(EpisodeEdge(
            src_id=a.id, dst_id=b.id, relation=relation, gap_min=gap, bridge=bridge,
        ))
    return edges


@dataclass(frozen=True, slots=True)
class EpisodeGraph:
    """An addressable, traversable view over a window's episodes.

    Nodes are :class:`Episode` objects keyed by ``id``; context edges are their
    :class:`ContextLink`\\ s and ``edges`` holds the episode-to-episode
    :class:`EpisodeEdge` chains. ``node`` and ``at`` are the two entry points an
    agent uses: name a node, or find the one covering a moment, then read its
    edges; ``edges_for`` walks the chain either direction.
    """

    episodes: tuple[Episode, ...]
    edges: tuple[EpisodeEdge, ...] = ()

    def node(self, episode_id: str) -> Episode | None:
        return next((e for e in self.episodes if e.id == episode_id), None)

    def edges_for(self, episode_id: str) -> dict[str, list[EpisodeEdge]]:
        """Chain edges touching an episode: ``in`` arrives at it, ``out`` leaves it."""
        return {
            "in": [e for e in self.edges if e.dst_id == episode_id],
            "out": [e for e in self.edges if e.src_id == episode_id],
        }

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
        return {
            "summary": self.summary(),
            "nodes": [e.to_dict() for e in self.episodes],
            "edges": [e.to_dict() for e in self.edges],
        }


def build_graph(
    store: StoragePort, start: datetime, end: datetime, *,
    target_low: int = TARGET_LOW, target_high: int = TARGET_HIGH,
    gap_min: float = GAP_MIN_MINUTES,
) -> EpisodeGraph:
    """Detect episodes over ``[start, end]`` and wrap them as a traversable graph,
    chained episode to episode where consecutive excursions sit close enough."""
    episodes, ctx = _detect(
        store, start, end, target_low=target_low, target_high=target_high, gap_min=gap_min
    )
    edges = _chain_episodes(episodes, ctx) if ctx is not None else []
    return EpisodeGraph(episodes=tuple(episodes), edges=tuple(edges))


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
        "n_clinically_significant_hyper": sum(1 for e in hyper if e.clinically_significant),
        "n_severe_hypo": sum(1 for e in hypo if e.severe),
        "n_severe_hyper": sum(1 for e in hyper if e.severe),
        "longest_hyper_min": max((e.duration_min for e in hyper), default=0.0),
        "longest_hypo_min": max((e.duration_min for e in hypo), default=0.0),
    }
