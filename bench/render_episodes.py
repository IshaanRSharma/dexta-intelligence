"""Render the temporal episode graph (analytics.episodes) for one synthetic patient.

Default visualization stub for ISSUES #14: a timeline where hypo/hyper excursions
are shaded episode nodes, sensor gaps are hatched bands, context events (meals,
boluses, activity, sleep) are typed markers, and thin connectors are the graph
edges binding an episode to the context around it. A slice of a few days is drawn
so nodes and edges stay legible.

Usage: python bench/render_episodes.py  ->  bench/figures/episodes.{png,svg}
(needs matplotlib; no API key.)
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import matplotlib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from dexta_intelligence.analytics.episodes import detect_episodes, summarize
from dexta_intelligence.store.sqlite import SQLiteStore
from dexta_intelligence.testing.synthetic import SensitivityRegimeShift, generate_dataset

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
TRACE = "#3a3a38"
HYPER = "#e2894e"
HYPER_SEVERE = "#c9531f"
HYPO = "#2a78d6"
HYPO_SEVERE = "#1b4f96"
GAP = "#b8b6ad"
MEAL = "#1baf7a"
BOLUS = "#2a78d6"
ACTIVITY = "#d6a72a"
EDGE = "#9a988f"

MARKER = {"meal": ("^", MEAL), "bolus": ("v", BOLUS), "activity": ("o", ACTIVITY)}


def _build_store() -> SQLiteStore:
    """A reference patient with a mid-window sensitivity shift plus an injected gap.

    The gap is dropped just after the sensitivity shift (which starts on day 10),
    so it lands in the same post-shift slice as the record's richest hyper and
    hypo episodes.
    """
    events, _ = generate_dataset(
        seed=5, n_days=21,
        effects=(SensitivityRegimeShift(effect_size=60.0, after_day=10),), name="episodes-demo",
    )
    g = events["glucose"]
    start = g[0].ts
    gap_lo = start + timedelta(days=10, hours=12)
    gap_hi = start + timedelta(days=10, hours=14)
    kept = [e for e in g if not (gap_lo <= e.ts <= gap_hi)]
    store = SQLiteStore(":memory:")
    store.migrate()
    store.insert_glucose(kept)
    store.insert_insulin(events["insulin"])
    store.insert_meals(events["meal"])
    store.insert_activity(events["activity"])
    store.insert_sleep(events["sleep"])
    return store


def main() -> int:
    store = _build_store()
    cov = store.coverage()
    assert cov.first_ts is not None and cov.last_ts is not None
    episodes = detect_episodes(store, cov.first_ts, cov.last_ts)

    # centre a 2-day slice on the longest hyper episode (spans the injected gap)
    hyper = [e for e in episodes if e.kind == "hyper"]
    anchor = max(hyper, key=lambda e: e.duration_min).start if hyper else cov.first_ts
    slice_lo = anchor - timedelta(hours=14)
    slice_hi = slice_lo + timedelta(days=2)
    readings = [(g.ts, g.mg_dl) for g in store.get_glucose(slice_lo, slice_hi)]
    sliced = [e for e in episodes if slice_lo <= e.start <= slice_hi]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(11.5, 5.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.axhspan(70, 180, color=GRID, alpha=0.35, zorder=0)
    if readings:
        ax.plot([t for t, _ in readings], [v for _, v in readings],
                color=TRACE, linewidth=1.1, zorder=3)

    for ep in sliced:
        if ep.kind == "sensor_gap":
            ax.axvspan(ep.start, ep.end, facecolor=GAP, alpha=0.5, hatch="////",
                       edgecolor="white", zorder=1)
            continue
        color = (
            (HYPER_SEVERE if ep.severe else HYPER) if ep.kind == "hyper"
            else (HYPO_SEVERE if ep.severe else HYPO)
        )
        ax.axvspan(ep.start, ep.end, color=color, alpha=0.16, zorder=1)
        if ep.extreme_ts is not None and ep.extreme_mg_dl is not None:
            ax.scatter([ep.extreme_ts], [ep.extreme_mg_dl], s=22, color=color, zorder=5)
            for link in ep.links:
                mk = MARKER.get(link.kind)
                if mk is None:
                    continue
                y = ep.extreme_mg_dl
                ax.plot([link.ts, ep.extreme_ts], [y, ep.extreme_mg_dl],
                        color=EDGE, linewidth=0.7, alpha=0.55, zorder=2)
                ax.scatter([link.ts], [y], marker=mk[0], s=34, color=mk[1],
                           edgecolors="white", linewidths=0.4, zorder=6)

    s = summarize(episodes)
    ax.set_xlim(slice_lo, slice_hi)
    ax.set_ylim(40, 300)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.set_ylabel("glucose (mg/dL)", fontsize=9, color=MUTED)
    ax.tick_params(labelsize=8, colors=MUTED, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)

    ax.set_title(
        "Temporal episode graph: excursions as nodes, context as typed edges\n",
        fontsize=12.5, color=INK, loc="left", pad=18, fontweight="bold",
    )
    ax.text(0, 1.04,
            f"synthetic reference patient, 2-day slice. Full record: {s['num_hyper']} hyper "
            f"/ {s['num_hypo']} hypo episodes ({s['n_clinically_significant_hypo']} clinically "
            f"significant), longest hyper {s['longest_hyper_min']:.0f} min, "
            f"{s['n_sensor_gaps']} sensor gap(s).",
            transform=ax.transAxes, fontsize=8.5, color=INK_2)

    handles = [
        plt.Line2D([], [], color=HYPER, marker="s", linestyle="", markersize=9, label="hyper"),
        plt.Line2D([], [], color=HYPO, marker="s", linestyle="", markersize=9, label="hypo"),
        plt.Line2D([], [], color=GAP, marker="s", linestyle="", markersize=9, label="sensor gap"),
        plt.Line2D([], [], color=MEAL, marker="^", linestyle="", markersize=8, label="meal"),
        plt.Line2D([], [], color=BOLUS, marker="v", linestyle="", markersize=8, label="bolus"),
        plt.Line2D([], [], color=ACTIVITY, marker="o", linestyle="", markersize=8,
                   label="activity"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=False,
              labelcolor=INK_2, ncol=3)

    dest = Path(__file__).resolve().parent / "figures"
    dest.mkdir(exist_ok=True)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(dest / "episodes.png", facecolor=SURFACE)
    fig.savefig(dest / "episodes.svg", facecolor=SURFACE)
    print(f"WROTE {dest / 'episodes.png'} and .svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
