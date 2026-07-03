"""Render the LLM-CGM head-to-head error figure (plain model vs dexta harness).

Values are the hand-verified absolute errors from bench/LLMCGM_RESULTS.md
(the auto-scorer is a floor; the hand-verified table is the source of truth).
Each metric's error is in its native unit, so bars compare per-question error
magnitude, not a single unit.

Usage: python bench/render_figure.py  ->  bench/figures/llmcgm_error.{png,svg}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
PLAIN = "#2a78d6"
DEXTA = "#1baf7a"

# (metric label, unit, plain abs error, dexta abs error), hand-verified.
ROWS = [
    ("Longest high episode", "min", 108.0, 0.0),
    ("Overnight mean", "mg/dL", 25.0, 0.04),
    ("Time in range", "pp", 5.9, 1.2),
    ("Coefficient of variation", "pp", 4.2, 0.01),
    ("Mean glucose", "mg/dL", 3.3, 0.04),
    ("Max glucose", "mg/dL", 0.0, 0.0),
    ("Min glucose", "mg/dL", 0.0, 0.0),
    ("Dinner max glucose", "mg/dL", 0.0, 0.0),
]


def _fmt(v: float) -> str:
    if v == 0:
        return "0 (exact)"
    return f"{v:g}"


def main() -> int:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "svg.fonttype": "none",
    })
    labels = [f"{name}  ({unit})" for name, unit, _, _ in ROWS]
    plain = [r[2] for r in ROWS]
    dexta = [r[3] for r in ROWS]
    y = list(range(len(ROWS)))
    h = 0.34

    fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    b1 = ax.barh([i - h / 2 for i in y], plain, height=h, color=PLAIN,
                 label="plain model (raw CSV in-context)", zorder=3)
    b2 = ax.barh([i + h / 2 for i in y], dexta, height=h, color=DEXTA,
                 label="dexta harness (tools + rails)", zorder=3)

    xmax = max(plain) * 1.22
    for bars, vals in ((b1, plain), (b2, dexta)):
        for bar, v in zip(bars, vals, strict=True):
            ax.text(bar.get_width() + xmax * 0.008, bar.get_y() + bar.get_height() / 2,
                    _fmt(v), va="center", ha="left", fontsize=8.5, color=INK_2)

    ax.set_yticks(y, labels=labels, fontsize=9.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax)
    ax.set_xlabel("absolute error (native unit per metric)", fontsize=9, color=MUTED)
    ax.tick_params(axis="x", labelsize=8.5, colors=MUTED, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)

    ax.set_title(
        "Same model, same data, same questions: absolute error per LLM-CGM task\n",
        fontsize=12.5, color=INK, loc="left", pad=18, fontweight="bold",
    )
    ax.text(0, 1.04,
            "claude-sonnet-4-6, 21 days synthetic CGM (6,048 readings), "
            "LLM-CGM benchmark (Healey & Kohane, PSB 2025)",
            transform=ax.transAxes, fontsize=9, color=INK_2)
    ax.legend(loc="lower right", fontsize=9, frameon=False, labelcolor=INK_2)
    fig.text(0.012, 0.012,
             "Mean absolute error: plain 14.7 vs dexta 0.15. dexta's only non-zero errors are "
             "definitional (TIR boundary convention). Hand-verified: bench/LLMCGM_RESULTS.md.",
             fontsize=7.5, color=MUTED)

    dest = Path(__file__).resolve().parent / "figures"
    dest.mkdir(exist_ok=True)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(dest / "llmcgm_error.png", facecolor=SURFACE)
    fig.savefig(dest / "llmcgm_error.svg", facecolor=SURFACE)
    print(f"WROTE {dest / 'llmcgm_error.png'} and .svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
