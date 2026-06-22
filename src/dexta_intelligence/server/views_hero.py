"""Dashboard hero chart - the demo centerpiece glucose trace with attribution."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.agents.tools.toolkit import DiscoveryToolkit
from dexta_intelligence.investigations.spike import LATE_BOLUS_MIN
from dexta_intelligence.server.charts import TraceMarker, glucose_trace_svg

if TYPE_CHECKING:
    from dexta_intelligence.coldstart import ColdStartReport
    from dexta_intelligence.config import Config
    from dexta_intelligence.store.port import StoragePort

__all__ = ["hero_chart_view"]

_LOOKBACK_DAYS = 14
_ZOOM_PAD_HOURS = 3


def hero_chart_view(
    store: StoragePort,
    config: Config,
    gates: ColdStartReport,
) -> dict[str, Any]:
    """Build the dashboard hero glucose trace, or ``has_chart=False`` when unavailable."""
    if gates.below_hard_floor:
        return {"has_chart": False}

    coverage = store.coverage()
    if coverage.last_ts is None or coverage.first_ts is None:
        return {"has_chart": False}

    end = coverage.last_ts.date()
    start = max(coverage.first_ts.date(), end - timedelta(days=_LOOKBACK_DAYS))
    ctx = AgentContext(
        store=store,
        window=(start, end),
        gates=gates,
        run_id="hero-chart",
        timezone=config.analysis.timezone,
    )
    tk = DiscoveryToolkit(
        ctx,
        target_low=config.analysis.target_low,
        target_high=config.analysis.target_high,
    )
    tk.set_window(start.isoformat(), end.isoformat())
    spikes = tk.find_spikes()
    if not spikes.get("n_spikes"):
        return {"has_chart": False}

    spike = spikes["spikes"][0]
    zoom = tk.zoom_event(str(spike["ts"]), pad_hours=_ZOOM_PAD_HOURS)
    raw = zoom.get("readings") or []
    if zoom.get("error") or len(raw) < 2:
        return {"has_chart": False}

    readings = _parse_readings(raw)
    if len(readings) < 2:
        return {"has_chart": False}

    threshold = float(spikes.get("threshold", 200.0))
    highlight_start, highlight_end = _excursion_bounds(readings, threshold)
    markers, delays = _collect_markers(tk)
    annotation = _attribution(delays)

    spike_center = _aware_ts(spike.get("ts")) or readings[len(readings) // 2][0]
    peak = float(spike.get("peak_mg_dl", max(v for _, v in readings)))
    svg = glucose_trace_svg(
        readings,
        target_low=float(config.analysis.target_low),
        target_high=float(config.analysis.target_high),
        highlight_start=highlight_start,
        highlight_end=highlight_end,
        markers=markers,
        annotation=annotation,
    )
    return {
        "has_chart": True,
        "svg": svg,
        "title": "Latest excursion",
        "subtitle": f"Peak {peak:.0f} mg/dL · {spike_center.strftime('%b %d %H:%M')} UTC",
        "annotation": annotation,
    }


def _aware_ts(value: object) -> datetime | None:
    """Parse an ISO timestamp to an aware UTC datetime, or None on garbage."""
    try:
        ts = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts


def _parse_readings(raw: list[dict[str, Any]]) -> list[tuple[datetime, float]]:
    out: list[tuple[datetime, float]] = []
    for row in raw:
        ts = _aware_ts(row.get("ts"))
        val = row.get("mg_dl")
        if ts is not None and val is not None:
            out.append((ts, float(val)))
    return out


def _collect_markers(tk: DiscoveryToolkit) -> tuple[list[TraceMarker], list[int]]:
    """Carb and bolus markers for the trace, plus the bolus-vs-carb delays."""
    markers: list[TraceMarker] = []
    for entry in tk.get_carb_entries().get("entries") or []:
        ts = _aware_ts(entry.get("ts"))
        if ts is not None:
            markers.append(TraceMarker(ts=ts, kind="carb", label=f"{entry.get('carbs_g', '?')}g"))
    delays: list[int] = []
    for bolus in tk.get_boluses().get("boluses") or []:
        ts = _aware_ts(bolus.get("ts"))
        if ts is None:
            continue
        units = bolus.get("units")
        delay = bolus.get("minutes_after_carb_entry")
        if isinstance(delay, (int, float)):
            delays.append(int(delay))
        markers.append(
            TraceMarker(ts=ts, kind="bolus", label=f"{units}U" if units is not None else "bolus")
        )
    return markers, delays


def _excursion_bounds(
    readings: list[tuple[datetime, float]], threshold: float
) -> tuple[datetime | None, datetime | None]:
    """First contiguous above-threshold run in the trace."""
    start: datetime | None = None
    end: datetime | None = None
    for ts, val in readings:
        if val >= threshold:
            if start is None:
                start = ts
            end = ts
        elif start is not None:
            break
    return start, end


def _attribution(delays: list[int]) -> str | None:
    if not delays:
        return None
    delay = delays[0]
    if delay >= LATE_BOLUS_MIN:
        return f"late bolus, +{delay} min"
    return f"bolus {delay:+d} min vs carb entry"
