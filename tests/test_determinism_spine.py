"""End-to-end determinism proof for the deterministic spine.

Same store state + same question-window must yield byte-identical output across
fresh processes whose hash randomization differs. This is the strong version of
the "same inputs, same selection" invariant: it runs the whole spine (episode
graph -> curator selection + drop list -> provenance bindings -> edge authoring)
in subprocesses with ``PYTHONHASHSEED`` varied, and compares the canonical
serialization byte for byte. Any dict/set iteration that leaked into an output
order would diverge here; a stable result means every ordering has a total key.
"""

from __future__ import annotations

import os
import subprocess
import sys

# The worker: build synthetic store state, run the spine, print a canonical dump.
# Kept as an inline program so each run is a genuinely fresh interpreter (the
# point of the PYTHONHASHSEED variation).
_WORKER = r"""
import json, tempfile, os
from datetime import UTC, datetime, timedelta

from dexta_intelligence.store import SQLiteStore
from dexta_intelligence.models import (
    GlucoseEvent, InsulinEvent, InsulinKind, MealEvent,
    Finding, FindingStats, FindingStatus, FindingEdge, EdgeRelation,
)
from dexta_intelligence.analytics.episodes import detect_episodes
from dexta_intelligence.memory.curator import select_context, HistoryItem
from dexta_intelligence.guard.faithfulness import build_provenance

T0 = datetime(2026, 6, 1, tzinfo=UTC)
NOW = datetime(2026, 6, 10, tzinfo=UTC)
START, END = T0, T0 + timedelta(days=5)

db = os.path.join(tempfile.mkdtemp(), "spine.db")
store = SQLiteStore(db)
store.migrate()

# A synthetic 5-day trace with an overnight low and a post-meal high, plus
# meals/boluses that bind as context edges. Insertion order is deliberately
# jumbled so any hash-order sensitivity in the pipeline has room to surface.
glucose = []
def add_run(start, minutes, value):
    for i in range(minutes // 5):
        glucose.append(GlucoseEvent(ts=start + timedelta(minutes=5 * i), mg_dl=value))
add_run(T0 + timedelta(hours=2), 60, 55)    # overnight low
add_run(T0 + timedelta(hours=1), 60, 120)   # baseline
add_run(T0 + timedelta(days=1, hours=13), 90, 260)  # post-lunch high
add_run(T0 + timedelta(days=2, hours=8), 45, 120)
# insert in a scrambled order
import random
random.Random(0).shuffle(glucose)
store.insert_glucose(glucose)

store.insert_meals([
    MealEvent(ts=T0 + timedelta(days=1, hours=12, minutes=45), carbs_g=75.0, note="lunch"),
    MealEvent(ts=T0 + timedelta(hours=1, minutes=30), carbs_g=30.0, note="snack"),
])
def bolus(ts, units):
    return InsulinEvent(ts=ts, kind=InsulinKind.BOLUS, units=units)
store.insert_insulin([
    bolus(T0 + timedelta(days=1, hours=12, minutes=50), 6.0),
    bolus(T0 + timedelta(hours=1, minutes=35), 2.0),
])

episodes = detect_episodes(store, START, END)

findings = [
    Finding(agent="pattern", kind="overnight_low", scope="overnight",
            headline="Overnight lows after late boluses", id=1,
            stats=FindingStats(effect_size=0.4), evidence={"cv_pct": 33.1, "mean": 142.0},
            confidence=0.7, status=FindingStatus.ACTIVE,
            window_start=START, window_end=END, seen_count=3),
    Finding(agent="pattern", kind="post_meal_high", scope="lunch",
            headline="Lunch spikes above 250", id=2,
            stats=FindingStats(effect_size=0.6), evidence={"tir_pct": 55.0},
            confidence=0.8, status=FindingStatus.ACTIVE,
            window_start=START, window_end=END, seen_count=2),
]
edges = [
    FindingEdge(src_id=2, dst_id=1, relation=EdgeRelation.CONTRADICTS,
                knowledge_time=NOW, evidence="opposite direction"),
    FindingEdge(src_id=1, dst_id=2, relation=EdgeRelation.SUPERSEDES,
                knowledge_time=NOW, evidence="newer analysis"),
]

selection = select_context(
    "why did I go low overnight and spike after lunch? cv and tir seem off",
    budget_tokens=200, now=NOW, window=(START, END),
    episodes=episodes, findings=findings, edges=edges,
    metric_bundle={"cv": 33.1, "mean": 142, "tir": 0.55},
    history=[
        HistoryItem(text="get_iob returned 1.2 U", ts=T0, protected=True),
        HistoryItem(text="get_carb_entries returned 2 entries", ts=T0 + timedelta(hours=2)),
    ],
)

evidence = {
    "glucose_stats": {"mean": 142.0, "cv_pct": 33.1, "tir_pct": 55.0},
    "daily_means": [130.0, 140.0, 156.0],
    "episodes": [{"extreme_mg_dl": 55.0}, {"extreme_mg_dl": 260.0}],
}

dump = {
    "episodes": [
        {"id": e.id, "kind": e.kind, "start": e.start.isoformat(),
         "duration_min": e.duration_min, "severe": e.severe,
         "links": [(l.kind, l.offset_min) for l in e.links]}
        for e in episodes
    ],
    "selected": [
        {"key": i.key, "type": i.type.value, "tokens": i.tokens,
         "score": round(i.score, 9), "text": i.text}
        for i in selection.selected
    ],
    "drops": list(selection.trace_lines()),
    "used_tokens": selection.used_tokens,
    "provenance": build_provenance(evidence),
}
print(json.dumps(dump, sort_keys=False))
store.close()
"""


def _run(seed: str) -> str:
    env = dict(os.environ, PYTHONHASHSEED=seed)
    proc = subprocess.run(
        [sys.executable, "-c", _WORKER],
        env=env, capture_output=True, text=True, check=True,
    )
    return proc.stdout


def test_spine_is_byte_identical_across_hash_seeds() -> None:
    """The full spine output is identical across three different hash seeds. If
    any dict/set iteration order leaked into the episode graph, curator selection,
    drop receipts, or provenance bindings, these would diverge."""
    outputs = [_run(seed) for seed in ("0", "1", "524287")]
    assert outputs[0].strip(), "worker produced no output"
    assert outputs[0] == outputs[1] == outputs[2]
