"""Glucose-orientation tools: coverage, structure, windowing, descriptive stats."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dexta_intelligence.agents.reason import ToolSpec
from dexta_intelligence.agents.tools.toolkit import (
    _HOURS_PAIR,
    _SPIKE_THRESHOLD,
    _item_numbers,
    _numbers,
)

if TYPE_CHECKING:
    from dexta_intelligence.agents.base import AgentContext
    from dexta_intelligence.agents.tools.toolkit import DiscoveryToolkit


def glucose_specs(ctx: AgentContext, toolkit: DiscoveryToolkit) -> list[ToolSpec]:
    """Coverage, list_segments, set_window, zoom_event, daily_series, glucose_stats,
    find_spikes - the orient-and-drill primitives over the CGM series."""

    def coverage(_args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        cov = ctx.store.coverage()
        numbers = {
            "span_days": cov.span_days,
            "glucose_coverage_pct": cov.glucose_coverage_pct,
            "n_meals": cov.n_meals,
            "n_insulin": cov.n_insulin,
            "n_sleep": cov.n_sleep,
            "n_activity": cov.n_activity,
        }
        out: dict[str, Any] = {"summary": toolkit.data_summary(), **numbers}
        if toolkit._last_glucose_ts:
            out["last_glucose_ts"] = toolkit._last_glucose_ts.isoformat()
        if toolkit._last_insulin_ts:
            out["last_insulin_ts"] = toolkit._last_insulin_ts.isoformat()
        gap = toolkit._treatment_gap_note()
        if gap:
            out["treatment_gap"] = gap
        missing = toolkit.capabilities().missing_notes()
        if missing:
            out["unavailable"] = missing
        return out, numbers

    def list_segments(_args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        result = toolkit.list_segments()
        numbers: dict[str, Any] = {}
        for seg in result.get("segments", []):
            numbers[seg["period"]] = {
                "n_days": seg["n_days"],
                "mean_glucose": seg["mean_glucose"],
                "tir_pct": seg["tir_pct"],
                "n_lows": seg["n_lows"],
            }
        return result, numbers

    def set_window(args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        result = toolkit.set_window(str(args.get("start", "")), str(args.get("end", "")))
        return result, _numbers(result, ("n_days", "n_readings"))

    def zoom_event(args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        result = toolkit.zoom_event(
            str(args.get("timestamp", "")), int(args.get("pad_hours", 12))
        )
        return result, _numbers(result, ("pre_mean", "post_mean", "peak", "nadir"))

    def daily_series(args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        result = toolkit.daily_series(str(args.get("metric", "")))
        numbers: dict[str, Any] = {}
        for row in result.get("series", []):
            numbers[row["date"]] = row["value"]
        if result.get("mean_value") is not None:
            numbers["mean_value"] = result["mean_value"]
        return result, numbers

    def glucose_stats(args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        raw = args.get("hours")
        hours = (
            (int(raw[0]), int(raw[1]))
            if isinstance(raw, (list, tuple)) and len(raw) == 2
            else None
        )
        result = toolkit.glucose_stats(day=str(args.get("day", "")), hours=hours)
        return result, _numbers(
            result,
            (
                "n", "mean", "sd", "variance", "cv_pct", "median", "p10", "p25",
                "p75", "p90", "minimum", "maximum", "tir_pct", "tbr_pct", "tar_pct",
                "tbr54_pct", "tar250_pct", "gmi_pct",
            ),
        )

    def find_spikes(args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        result = toolkit.find_spikes(
            float(args.get("threshold", _SPIKE_THRESHOLD)), int(args.get("top_n", 10))
        )
        numbers = _numbers(result, ("threshold", "n_spikes"))
        numbers.update(_item_numbers(result.get("spikes", []), "spike"))
        return result, numbers

    return [
        ToolSpec(
            name="coverage",
            description="How much data exists (days, coverage %, streams).",
            parameters={"type": "object", "properties": {}},
            fn=coverage,
        ),
        ToolSpec(
            name="list_segments",
            description=(
                "Coarse structure of the whole record: one row per month (per week "
                "if span < 60d) with n_days, mean_glucose, tir_pct, n_lows. Call FIRST "
                "to orient before set_window."
            ),
            parameters={"type": "object", "properties": {}},
            fn=list_segments,
        ),
        ToolSpec(
            name="set_window",
            description=(
                "Narrow the ACTIVE window all later tools read from to [start, end] "
                "ISO dates. Clamps to available data; returns active_start/end, "
                "n_days, n_readings. Re-call with the full span to widen back out."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "start": {"type": "string", "description": "ISO date, e.g. 2026-03-01"},
                    "end": {"type": "string", "description": "ISO date, e.g. 2026-03-31"},
                },
                "required": ["start", "end"],
            },
            fn=set_window,
        ),
        ToolSpec(
            name="zoom_event",
            description=(
                "Drill a spike: window tight around an ISO timestamp (+/- pad_hours, "
                "default 12); returns the minute-level trace with pre_mean, post_mean, "
                "peak, nadir."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string", "description": "ISO datetime of the event"},
                    "pad_hours": {"type": "integer", "minimum": 1, "maximum": 72},
                },
                "required": ["timestamp"],
            },
            fn=zoom_event,
        ),
        ToolSpec(
            name="daily_series",
            description=(
                "Per-day time series over the ACTIVE window as [{date, value}] to spot "
                "trends/change-points. metric: tir|mean_glucose|tbr|cv."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "enum": ["tir", "mean_glucose", "tbr", "cv"],
                    },
                },
                "required": ["metric"],
            },
            fn=daily_series,
        ),
        ToolSpec(
            name="glucose_stats",
            description=(
                "Descriptive stats over one window: n, mean, SD, variance, CV, median, "
                "p10/25/75/90, min, max, TIR/TBR/TAR %, GMI. Use this for any "
                "'what was my mean/variance/SD/TIR/GMI for <period>' question - never "
                "estimate these from a raw trace. Defaults to the ACTIVE window; pass "
                "day (ISO date) for one local day and/or hours [start,end) for a "
                "time-of-day band (evening=[17,24], overnight=[0,6], morning=[6,11])."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "day": {
                        "type": "string",
                        "description": "ISO date for one local day, e.g. 2026-06-15",
                    },
                    "hours": _HOURS_PAIR,
                },
            },
            fn=glucose_stats,
        ),
        ToolSpec(
            name="find_spikes",
            description=(
                "Excursion peaks above threshold in the ACTIVE window ({ts, peak_mg_dl, "
                "duration_min}, largest first). Locates the spike to zoom_event when the "
                "user names a day but not a time."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "threshold": {"type": "number", "minimum": 140, "maximum": 400},
                    "top_n": {"type": "integer", "minimum": 1, "maximum": 25},
                },
            },
            fn=find_spikes,
        ),
    ]
