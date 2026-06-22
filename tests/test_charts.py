"""Unit tests for server-rendered SVG chart builders."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from dexta_intelligence.server.charts import TraceMarker, dual_trace_svg, glucose_trace_svg


def _series(n: int = 12, base: float = 120.0) -> list[tuple[datetime, float]]:
    t0 = datetime(2026, 3, 14, 18, 0, tzinfo=UTC)
    out: list[tuple[datetime, float]] = []
    for i in range(n):
        val = base + (80 if 4 <= i <= 8 else 0)
        out.append((t0 + timedelta(minutes=5 * i), val))
    return out


def test_glucose_trace_renders_target_band_and_line() -> None:
    svg = glucose_trace_svg(_series(), target_low=70, target_high=180)
    assert "chart-band-target" in svg
    assert "chart-line" in svg
    assert "polyline" in svg
    assert 'aria-label="glucose trace"' in svg


def test_glucose_trace_marks_spike_and_annotation() -> None:
    readings = _series()
    t0 = readings[0][0]
    svg = glucose_trace_svg(
        readings,
        highlight_start=readings[4][0],
        highlight_end=readings[8][0],
        annotation="late bolus, +22 min",
        markers=[
            TraceMarker(t0 + timedelta(minutes=30), "carb", "60g"),
            TraceMarker(t0 + timedelta(minutes=52), "bolus", "4.5U"),
        ],
    )
    assert "chart-band-spike" in svg
    assert "late bolus, +22 min" in svg
    assert "chart-marker-carb" in svg
    assert "chart-marker-bolus" in svg
    assert "60g" in svg


def test_glucose_trace_empty_for_sparse_data() -> None:
    svg = glucose_trace_svg([(datetime.now(tz=UTC), 100.0)])
    assert "chart-empty" in svg or "spark-flat" in svg


def test_dual_trace_overlays_expected_and_actual() -> None:
    expected = [110.0, 115.0, 120.0, 125.0, 130.0]
    actual = [110.0, 118.0, 135.0, 160.0, 175.0]
    svg = dual_trace_svg(expected, actual)
    assert "chart-expected" in svg
    assert "chart-actual" in svg
    assert "chart-divergence" in svg


def test_dual_trace_empty_for_short_series() -> None:
    svg = dual_trace_svg([100.0], [101.0])
    assert "chart-empty" in svg or "spark-flat" in svg
