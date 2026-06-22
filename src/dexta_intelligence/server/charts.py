"""Server-rendered SVG charts - pure functions (series in, SVG out).

Hand-rolled polylines, consistent with :func:`render.sparkline_svg`. No client
framework or charting dependency.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

__all__ = [
    "TraceMarker",
    "dual_trace_svg",
    "glucose_trace_svg",
]


@dataclass(frozen=True, slots=True)
class TraceMarker:
    """A point event on a glucose trace (bolus, carb entry, etc.)."""

    ts: datetime
    kind: str
    label: str


def glucose_trace_svg(
    readings: Sequence[tuple[datetime, float]],
    *,
    target_low: float = 70.0,
    target_high: float = 180.0,
    highlight_start: datetime | None = None,
    highlight_end: datetime | None = None,
    markers: Sequence[TraceMarker] = (),
    annotation: str | None = None,
    width: int = 640,
    height: int = 220,
) -> str:
    """Minute-level CGM trace with target band, spike shading, and event markers."""
    pts = [(ts, v) for ts, v in readings if v is not None]
    if len(pts) < 2:
        return _empty_chart(width, height, "not enough readings")

    t0, t1 = pts[0][0], pts[-1][0]
    span_s = max(1.0, (t1 - t0).total_seconds())
    vals = [v for _, v in pts]
    y_min = max(40.0, min(vals) - 25.0)
    y_max = min(400.0, max(vals) + 25.0)
    if y_max - y_min < 40:
        mid = (y_max + y_min) / 2
        y_min, y_max = mid - 40, mid + 40

    pad_l, pad_r, pad_t, pad_b = 44, 12, 18, 28
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    def x_of(ts: datetime) -> float:
        return pad_l + plot_w * (ts - t0).total_seconds() / span_s

    def y_of(val: float) -> float:
        return pad_t + plot_h * (1.0 - (val - y_min) / (y_max - y_min))

    parts: list[str] = [
        f'<svg class="chart chart-glucose" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" '
        f'aria-label="glucose trace">',
        f'<rect class="chart-bg" x="0" y="0" width="{width}" height="{height}" '
        f'rx="8" ry="8" />',
    ]

    band_y0 = y_of(target_high)
    band_y1 = y_of(target_low)
    parts.append(
        f'<rect class="chart-band-target" x="{pad_l:.1f}" y="{band_y0:.1f}" '
        f'width="{plot_w:.1f}" height="{max(1.0, band_y1 - band_y0):.1f}" />'
    )

    if highlight_start is not None and highlight_end is not None:
        hx0 = x_of(highlight_start)
        hx1 = x_of(highlight_end)
        if hx1 > hx0:
            parts.append(
                f'<rect class="chart-band-spike" x="{hx0:.1f}" y="{pad_t:.1f}" '
                f'width="{hx1 - hx0:.1f}" height="{plot_h:.1f}" />'
            )

    coords = " ".join(f"{x_of(ts):.1f},{y_of(v):.1f}" for ts, v in pts)
    parts.append(f'<polyline class="chart-line" points="{coords}" fill="none" />')

    peak_ts, peak_v = max(pts, key=lambda p: p[1])
    parts.append(
        f'<circle class="chart-peak" cx="{x_of(peak_ts):.1f}" cy="{y_of(peak_v):.1f}" r="4" />'
    )

    for marker in markers:
        if not (t0 <= marker.ts <= t1):
            continue
        mx = x_of(marker.ts)
        parts.append(
            f'<line class="chart-marker-line chart-marker-{html.escape(marker.kind)}" '
            f'x1="{mx:.1f}" y1="{pad_t:.1f}" x2="{mx:.1f}" y2="{pad_t + plot_h:.1f}" />'
        )
        parts.append(
            f'<circle class="chart-marker-dot chart-marker-{html.escape(marker.kind)}" '
            f'cx="{mx:.1f}" cy="{y_of(_nearest_value(pts, marker.ts)):.1f}" r="5" />'
        )
        parts.append(
            f'<text class="chart-marker-label" x="{mx:.1f}" y="{pad_t + plot_h + 16:.1f}" '
            f'text-anchor="middle">{html.escape(marker.label)}</text>'
        )

    parts.append(
        f'<text class="chart-axis-label" x="{pad_l - 6:.1f}" y="{y_of(target_high):.1f}" '
        f'text-anchor="end" dominant-baseline="middle">{int(target_high)}</text>'
    )
    parts.append(
        f'<text class="chart-axis-label" x="{pad_l - 6:.1f}" y="{y_of(target_low):.1f}" '
        f'text-anchor="end" dominant-baseline="middle">{int(target_low)}</text>'
    )

    if annotation:
        parts.append(
            f'<text class="chart-annotation" x="{pad_l:.1f}" y="{pad_t - 4:.1f}">'
            f'{html.escape(annotation)}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def dual_trace_svg(
    expected: Sequence[float],
    actual: Sequence[float],
    *,
    step_min: int = 5,
    width: int = 480,
    height: int = 120,
    expected_label: str = "expected",
    actual_label: str = "actual",
) -> str:
    """Overlay expected vs realized curves with divergence shading."""
    n = min(len(expected), len(actual))
    if n < 2:
        return _empty_chart(width, height, "not enough points")

    exp = [float(v) for v in expected[:n]]
    act = [float(v) for v in actual[:n]]
    lo = min(*exp, *act) - 10.0
    hi = max(*exp, *act) + 10.0
    span = (hi - lo) or 1.0

    pad_l, pad_r, pad_t, pad_b = 8, 8, 20, 8
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    def x_of(i: int) -> float:
        return pad_l + plot_w * i / (n - 1)

    def y_of(val: float) -> float:
        return pad_t + plot_h * (1.0 - (val - lo) / span)

    parts: list[str] = [
        f'<svg class="chart chart-dual" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" '
        f'aria-label="expected vs actual glucose">',
    ]

    diverge_polys: list[str] = []
    for i in range(n - 1):
        if abs(exp[i] - act[i]) >= 8 or abs(exp[i + 1] - act[i + 1]) >= 8:
            x0, x1 = x_of(i), x_of(i + 1)
            diverge_polys.append(
                f"{x0:.1f},{y_of(exp[i]):.1f} {x1:.1f},{y_of(exp[i + 1]):.1f} "
                f"{x1:.1f},{y_of(act[i + 1]):.1f} {x0:.1f},{y_of(act[i]):.1f}"
            )
    for poly in diverge_polys:
        parts.append(f'<polygon class="chart-divergence" points="{poly}" />')

    exp_coords = " ".join(f"{x_of(i):.1f},{y_of(v):.1f}" for i, v in enumerate(exp))
    act_coords = " ".join(f"{x_of(i):.1f},{y_of(v):.1f}" for i, v in enumerate(act))
    parts.append(f'<polyline class="chart-expected" points="{exp_coords}" fill="none" />')
    parts.append(f'<polyline class="chart-actual" points="{act_coords}" fill="none" />')

    parts.append(
        f'<text class="chart-legend" x="{pad_l:.1f}" y="{pad_t - 6:.1f}">'
        f'<tspan class="chart-legend-expected">{html.escape(expected_label)}</tspan>'
        f'<tspan dx="12" class="chart-legend-actual">{html.escape(actual_label)}</tspan>'
        f"</text>"
    )
    parts.append(
        f'<text class="chart-axis-label" x="{width - pad_r:.1f}" y="{height - 2:.1f}" '
        f'text-anchor="end">+{step_min * (n - 1)} min</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _nearest_value(pts: Sequence[tuple[datetime, float]], ts: datetime) -> float:
    return min(pts, key=lambda p: abs((p[0] - ts).total_seconds()))[1]


def _empty_chart(width: int, height: int, label: str) -> str:
    mid = height / 2
    return (
        f'<svg class="chart chart-empty" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" '
        f'aria-label="{html.escape(label)}">'
        f'<line class="spark-flat" x1="8" y1="{mid:.1f}" x2="{width - 8}" y2="{mid:.1f}" />'
        f"</svg>"
    )
