"""Tests for the pure agent-trace formatter."""

from __future__ import annotations

from dexta_intelligence.agents.reason import ToolCall
from dexta_intelligence.agents.trace import TraceLine, render_trace


def _typical_path() -> list[ToolCall]:
    """A realistic orient → narrow → drill → compare path."""
    return [
        ToolCall(
            name="list_segments",
            args={},
            ok=True,
            result={
                "granularity": "month",
                "segments": [
                    {"period": "2026-03", "n_days": 31, "mean_glucose": 138.2,
                     "tir_pct": 64.1, "n_lows": 3},
                    {"period": "2026-04", "n_days": 30, "mean_glucose": 151.0,
                     "tir_pct": 55.7, "n_lows": 1},
                ],
            },
        ),
        ToolCall(
            name="set_window",
            args={"start": "2026-03-01", "end": "2026-03-31"},
            ok=True,
            result={
                "active_start": "2026-03-01",
                "active_end": "2026-03-31",
                "n_days": 31,
                "n_readings": 372,
            },
        ),
        ToolCall(
            name="zoom_event",
            args={"timestamp": "2026-03-14T14:00:00+00:00", "pad_hours": 6},
            ok=True,
            result={
                "center": "2026-03-14T14:00:00+00:00",
                "pad_hours": 6,
                "n_readings": 48,
                "readings": [],
                "pre_mean": 120.0,
                "post_mean": 210.5,
                "peak": 244.0,
                "nadir": 88.0,
            },
        ),
        ToolCall(
            name="tod_compare",
            args={"hours_a": [3, 7], "hours_b": [11, 15]},
            ok=True,
            result={
                "label_a": "03-07h",
                "label_b": "11-15h",
                "n_a": 20,
                "n_b": 21,
                "mean_a": 142.3,
                "mean_b": 118.9,
                "delta": 23.4,
                "cohen_d": 0.91,
                "interpretation": "large",
            },
        ),
    ]


def test_renders_full_path_with_right_facts() -> None:
    lines = render_trace(_typical_path())
    assert len(lines) == 4
    assert all(isinstance(ln, TraceLine) for ln in lines)

    scan, scope, zoom, compare = lines

    assert scan.icon == "scan"
    assert "2 segments" in scan.text

    assert scope.icon == "scope"
    assert "2026-03-01 → 2026-03-31" in scope.text
    assert "31 days" in scope.text
    assert "372 readings" in scope.text

    assert zoom.icon == "zoom"
    assert "2026-03-14T14:00:00+00:00" in zoom.text
    assert "±6h" in zoom.text
    assert "peak 244.0" in zoom.text
    assert "nadir 88.0" in zoom.text

    assert compare.icon == "compare"
    assert "03-07h vs 11-15h" in compare.text
    assert "large difference" in compare.text
    assert "+23.4" in compare.text


def test_set_window_clamp_note() -> None:
    (line,) = render_trace([
        ToolCall(
            name="set_window",
            args={"start": "2020-01-01", "end": "2030-01-01"},
            ok=True,
            result={
                "active_start": "2026-03-01",
                "active_end": "2026-04-30",
                "n_days": 61,
                "n_readings": 700,
                "note": "clamped to available data",
            },
        )
    ])
    assert "clamped to available data" in line.text


def test_daily_series_and_singular_day() -> None:
    (line,) = render_trace([
        ToolCall(
            name="daily_series",
            args={"metric": "tir"},
            ok=True,
            result={"metric": "tir", "n_days": 1, "series": [], "mean_value": 60.0},
        )
    ])
    assert line.icon == "trend"
    assert "read the tir day-by-day" in line.text
    assert "1 day" in line.text  # singular, not "1 days"


def test_recall_with_count() -> None:
    (line,) = render_trace([
        ToolCall(
            name="recall",
            args={"query": "overnight"},
            ok=True,
            result={"findings": [{"headline": "x"}, {"headline": "y"}], "open_questions": []},
        )
    ])
    assert line.icon == "recall"
    assert "checked what I already know" in line.text
    assert "2 findings" in line.text


def test_not_ok_renders_failure_line_no_crash() -> None:
    (line,) = render_trace([
        ToolCall(
            name="zoom_event",
            args={"timestamp": "nonsense"},
            ok=False,
            result={"error": "bad timestamp: invalid isoformat"},
        )
    ])
    assert "couldn't zoom_event" in line.text
    assert "bad timestamp: invalid isoformat" in line.text


def test_empty_list_yields_empty() -> None:
    assert render_trace([]) == []


def test_unknown_tool_generic_line() -> None:
    (line,) = render_trace([
        ToolCall(name="mystery_tool", args={}, ok=True, result={"whatever": 1})
    ])
    assert line.text == "ran mystery_tool"


def test_missing_key_degrades_gracefully() -> None:
    # A set_window result missing every expected key must not KeyError.
    (window,) = render_trace([
        ToolCall(name="set_window", args={}, ok=True, result={})
    ])
    assert window.icon == "scope"
    assert "?" in window.text  # placeholders, no crash

    # A tod_compare result missing delta still renders.
    (compare,) = render_trace([
        ToolCall(
            name="tod_compare",
            args={},
            ok=True,
            result={"label_a": "a", "label_b": "b", "interpretation": "small"},
        )
    ])
    assert "a vs b" in compare.text
    assert "small difference" in compare.text
    assert "n/a" in compare.text

    # A non-dict result (defensive) must not crash either.
    (weird,) = render_trace([
        ToolCall(name="daily_series", args={}, ok=True, result="oops")
    ])
    assert weird.icon == "trend"
