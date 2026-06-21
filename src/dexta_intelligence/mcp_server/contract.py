"""Pure MCP tool implementations - StoragePort + optional RealtimeConnector.

No fastmcp imports here: deterministic, JSON-serializable dict outputs only.
Computed numbers trace to stored or live readings; undefined metrics are
``None`` or omitted, never fabricated ``0.0``.
"""

from __future__ import annotations

import csv
import io
import json
import math
import statistics
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

from dexta_intelligence.analytics.oref import insulin_totals
from dexta_intelligence.analytics.rollups import (
    EXPECTED_READINGS_PER_DAY,
    TARGET_HIGH_MG_DL,
    TARGET_LOW_MG_DL,
    VERY_HIGH_MG_DL,
    VERY_LOW_MG_DL,
    coverage_fraction,
)
from dexta_intelligence.models import InsulinKind
from dexta_intelligence.stats.core import summarize

if TYPE_CHECKING:
    from dexta_intelligence.connectors.base import RealtimeConnector
    from dexta_intelligence.models import GlucoseEvent
    from dexta_intelligence.store.port import StoragePort

__all__ = [
    "ALERT_DISCLAIMER",
    "EPISODE_MIN_DURATION_MINUTES",
    "IOB_DISCLAIMER",
    "MAX_EVENT_ITEMS",
    "STALE_THRESHOLD_MINUTES",
    "TIME_BLOCKS_UTC",
    "analyze_time_blocks",
    "check_alerts",
    "detect_episodes",
    "export_data",
    "get_agp_report",
    "get_basal_timeline",
    "get_boluses",
    "get_carb_entries",
    "get_current_glucose",
    "get_episode_details",
    "get_glucose_readings",
    "get_iob",
    "get_statistics",
    "get_status_summary",
]

#: Freshness window aligned with RealtimeConnector conventions (10 minutes).
STALE_THRESHOLD_MINUTES = 10
#: Minimum contiguous out-of-range duration to count as an episode.
EPISODE_MIN_DURATION_MINUTES = 15
#: Informational-only disclaimer - never dosing advice.
ALERT_DISCLAIMER = (
    "Informational only. Not medical advice. Do not use for insulin dosing "
    "or treatment decisions. Consult your diabetes care team."
)
#: IOB is Tier B analysis context computed from logged boluses - never dosing input.
IOB_DISCLAIMER = (
    "Analysis context only. Computed from logged boluses, not pump state. "
    "Never use for insulin dosing or treatment decisions."
)
#: Maximum events returned per treatment list (callers narrow the window).
MAX_EVENT_ITEMS = 100
#: Bolus lookback for IOB; comfortably past the oref0 5 h DIA floor.
_IOB_LOOKBACK_HOURS = 8

#: UTC time-of-day blocks for :func:`analyze_time_blocks`.
TIME_BLOCKS_UTC: dict[str, tuple[int, int]] = {
    "overnight": (0, 6),
    "morning": (6, 12),
    "afternoon": (12, 18),
    "evening": (18, 24),
}

_GMI_INTERCEPT = 3.31
_GMI_SLOPE = 0.02392
_AGP_BINS_PER_DAY = EXPECTED_READINGS_PER_DAY  # 288 five-minute slots
_CONTEXT_READINGS = 6  # 30 minutes at 5-minute cadence

# Trend → projected mg/dL change per 5 minutes (Dexcom vocabulary).
_TREND_RATE_PER_5MIN: dict[str, float] = {
    "DoubleUp": 15.0,
    "SingleUp": 12.5,
    "FortyFiveUp": 7.5,
    "Flat": 0.0,
    "FortyFiveDown": -7.5,
    "SingleDown": -12.5,
    "DoubleDown": -15.0,
}


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        msg = "naive datetime rejected: all timestamps must be timezone-aware (UTC)"
        raise ValueError(msg)
    return value.astimezone(UTC)


def _parse_iso(value: str) -> datetime:
    return _require_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))


def _iso(value: datetime) -> str:
    return _require_utc(value).isoformat()


def _reading_dict(event: GlucoseEvent) -> dict[str, Any]:
    return {
        "timestamp": _iso(event.ts),
        "mg_dl": event.mg_dl,
        "trend": event.trend,
    }


def _validate_window(start: datetime, end: datetime) -> tuple[datetime, datetime]:
    start_utc = _require_utc(start)
    end_utc = _require_utc(end)
    if start_utc >= end_utc:
        msg = f"invalid window: start ({start_utc}) must be before end ({end_utc})"
        raise ValueError(msg)
    return start_utc, end_utc


def _window_readings(
    store: StoragePort, start: datetime, end: datetime
) -> list[GlucoseEvent]:
    start_utc, end_utc = _validate_window(start, end)
    return store.get_glucose(start_utc, end_utc)


def _cap_events(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str | None]:
    if len(items) <= MAX_EVENT_ITEMS:
        return items, None
    note = f"showing first {MAX_EVENT_ITEMS} of {len(items)} events - narrow the window"
    return items[:MAX_EVENT_ITEMS], note


def _expected_readings(start: datetime, end: datetime) -> int:
    minutes = (_require_utc(end) - _require_utc(start)).total_seconds() / 60.0
    return max(1, int(minutes / 5))


def _episode_id(kind: str, start: datetime) -> str:
    return f"{kind}:{_iso(start)}"


def _duration_minutes(start: datetime, end: datetime) -> int:
    return max(0, int((_require_utc(end) - _require_utc(start)).total_seconds() / 60))


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    sorted_vals = sorted(values)
    pos = q * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _tir_bands(
    values: list[int],
    *,
    target_low: int = TARGET_LOW_MG_DL,
    target_high: int = TARGET_HIGH_MG_DL,
) -> dict[str, float | None]:
    n = len(values)
    if n == 0:
        return {
            "tir_pct": None,
            "tbr_pct": None,
            "tbr2_pct": None,
            "tar_pct": None,
            "tar2_pct": None,
        }

    def pct(count: int) -> float:
        return 100.0 * count / n

    tbr_count = sum(1 for v in values if v < target_low)
    tar_count = sum(1 for v in values if v > target_high)
    tbr2_count = sum(1 for v in values if v < VERY_LOW_MG_DL)
    tar2_count = sum(1 for v in values if v > VERY_HIGH_MG_DL)
    return {
        "tir_pct": pct(n - tbr_count - tar_count),
        "tbr_pct": pct(tbr_count),
        "tbr2_pct": pct(tbr2_count),
        "tar_pct": pct(tar_count),
        "tar2_pct": pct(tar2_count),
    }


def _statistics_payload(
    readings: list[GlucoseEvent],
    *,
    start: datetime,
    end: datetime,
    target_low: int = TARGET_LOW_MG_DL,
    target_high: int = TARGET_HIGH_MG_DL,
) -> dict[str, Any]:
    values = [float(g.mg_dl) for g in readings]
    stats = summarize(values)
    bands = _tir_bands([g.mg_dl for g in readings], target_low=target_low, target_high=target_high)
    expected = _expected_readings(start, end)
    payload: dict[str, Any] = {
        "start": _iso(start),
        "end": _iso(end),
        "reading_count": stats.n,
        "coverage_pct": round(coverage_fraction(stats.n, expected=expected) * 100.0, 1)
        if stats.n > 0
        else None,
        "thresholds": {"low": target_low, "high": target_high},
    }
    if stats.n == 0:
        return payload

    payload["mean_mg_dl"] = stats.mean
    payload["median_mg_dl"] = stats.median
    payload["sd_mg_dl"] = stats.sd
    payload["cv_pct"] = stats.cv_pct
    payload["min_mg_dl"] = stats.minimum
    payload["max_mg_dl"] = stats.maximum
    if stats.mean is not None:
        payload["gmi_pct"] = round(_GMI_INTERCEPT + _GMI_SLOPE * stats.mean, 1)
    payload.update(bands)
    return payload


def _latest_stored(store: StoragePort, *, now: datetime) -> GlucoseEvent | None:
    readings = store.get_glucose(now - timedelta(days=30), now + timedelta(minutes=1))
    return readings[-1] if readings else None


def get_current_glucose(
    store: StoragePort,
    realtime: RealtimeConnector | None = None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Current glucose from live connector or latest stored reading."""
    clock = _require_utc(now or datetime.now(tz=UTC))
    source = "realtime" if realtime is not None else "store"
    reading: GlucoseEvent | None = None

    if realtime is not None:
        reading = realtime.current()

    if reading is None:
        source = "store"
        reading = _latest_stored(store, now=clock)

    if reading is None:
        return {
            "status": "no_data",
            "source": None,
            "stale": None,
            "reading": None,
        }

    age_minutes = (_require_utc(clock) - reading.ts).total_seconds() / 60.0
    stale = age_minutes > STALE_THRESHOLD_MINUTES
    return {
        "status": "ok",
        "source": source,
        "stale": stale,
        "age_minutes": round(age_minutes, 1),
        "reading": _reading_dict(reading),
    }


def get_glucose_readings(
    store: StoragePort,
    start: datetime,
    end: datetime,
    *,
    max_count: int | None = None,
) -> dict[str, Any]:
    """Windowed glucose readings from the store."""
    readings = _window_readings(store, start, end)
    if max_count is not None and max_count > 0:
        readings = readings[:max_count]
    return {
        "start": _iso(start),
        "end": _iso(end),
        "count": len(readings),
        "readings": [_reading_dict(g) for g in readings],
    }


def get_statistics(
    store: StoragePort,
    start: datetime,
    end: datetime,
    *,
    target_low: int = TARGET_LOW_MG_DL,
    target_high: int = TARGET_HIGH_MG_DL,
) -> dict[str, Any]:
    """Descriptive glucose statistics for a UTC window."""
    readings = _window_readings(store, start, end)
    return _statistics_payload(
        readings,
        start=start,
        end=end,
        target_low=target_low,
        target_high=target_high,
    )


def _classify_reading(
    mg_dl: int,
    *,
    target_low: int,
    target_high: int,
) -> Literal["severe_hypo", "hypo", "severe_hyper", "hyper"] | None:
    if mg_dl < VERY_LOW_MG_DL:
        return "severe_hypo"
    if mg_dl < target_low:
        return "hypo"
    if mg_dl > VERY_HIGH_MG_DL:
        return "severe_hyper"
    if mg_dl > target_high:
        return "hyper"
    return None


def _detect_episode_runs(
    readings: list[GlucoseEvent],
    *,
    target_low: int = TARGET_LOW_MG_DL,
    target_high: int = TARGET_HIGH_MG_DL,
) -> list[dict[str, Any]]:
    if not readings:
        return []

    episodes: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for reading in readings:
        kind = _classify_reading(reading.mg_dl, target_low=target_low, target_high=target_high)
        if kind is None:
            if current is not None:
                episodes.append(current)
                current = None
            continue

        is_low = kind in ("hypo", "severe_hypo")
        if current is not None and current["is_low"] == is_low:
            current["end"] = reading.ts
            current["values"].append(reading.mg_dl)
            if kind in ("severe_hypo", "severe_hyper"):
                current["kind"] = kind
        else:
            if current is not None:
                episodes.append(current)
            current = {
                "kind": kind,
                "is_low": is_low,
                "start": reading.ts,
                "end": reading.ts,
                "values": [reading.mg_dl],
            }

    if current is not None:
        current["ongoing"] = True
        episodes.append(current)

    formatted: list[dict[str, Any]] = []
    for ep in episodes:
        duration = _duration_minutes(ep["start"], ep["end"])
        if duration < EPISODE_MIN_DURATION_MINUTES:
            continue
        values: list[int] = ep["values"]
        extreme = min(values) if ep["is_low"] else max(values)
        formatted.append(
            {
                "id": _episode_id(ep["kind"], ep["start"]),
                "kind": ep["kind"],
                "start": _iso(ep["start"]),
                "end": _iso(ep["end"]),
                "duration_minutes": duration,
                "extreme_mg_dl": extreme,
                "mean_mg_dl": round(statistics.fmean(values), 1),
                "reading_count": len(values),
                "ongoing": ep.get("ongoing", False),
            }
        )
    return formatted


def detect_episodes(
    store: StoragePort,
    start: datetime,
    end: datetime,
    *,
    target_low: int = TARGET_LOW_MG_DL,
    target_high: int = TARGET_HIGH_MG_DL,
) -> dict[str, Any]:
    """Detect hypo/hyper episodes with minimum duration in a UTC window."""
    readings = _window_readings(store, start, end)
    episodes = _detect_episode_runs(
        readings, target_low=target_low, target_high=target_high
    )
    low_kinds = ("hypo", "severe_hypo")
    high_kinds = ("hyper", "severe_hyper")
    low_eps = [e for e in episodes if e["kind"] in low_kinds]
    high_eps = [e for e in episodes if e["kind"] in high_kinds]
    return {
        "start": _iso(start),
        "end": _iso(end),
        "reading_count": len(readings),
        "episodes": episodes,
        "summary": {
            "total": len(episodes),
            "hypo_count": len(low_eps),
            "hyper_count": len(high_eps),
            "severe_hypo_count": sum(1 for e in low_eps if e["kind"] == "severe_hypo"),
            "severe_hyper_count": sum(1 for e in high_eps if e["kind"] == "severe_hyper"),
            "total_hypo_minutes": sum(e["duration_minutes"] for e in low_eps) or None,
            "total_hyper_minutes": sum(e["duration_minutes"] for e in high_eps) or None,
        },
    }


def get_episode_details(
    store: StoragePort,
    episode_id: str,
    *,
    context_minutes: int = 30,
) -> dict[str, Any]:
    """One episode by id plus surrounding context readings."""
    if ":" not in episode_id:
        msg = f"invalid episode_id {episode_id!r}: expected 'kind:ISO-8601-start'"
        raise ValueError(msg)
    _kind, start_iso = episode_id.split(":", 1)
    episode_start = _parse_iso(start_iso)

    search_start = episode_start - timedelta(days=7)
    search_end = episode_start + timedelta(days=7)
    readings = _window_readings(store, search_start, search_end)
    episodes = _detect_episode_runs(readings)
    match = next((e for e in episodes if e["id"] == episode_id), None)
    if match is None:
        return {"episode_id": episode_id, "status": "not_found", "episode": None}

    ep_start = _parse_iso(match["start"])
    ep_end = _parse_iso(match["end"])
    context_start = ep_start - timedelta(minutes=context_minutes)
    context_end = ep_end + timedelta(minutes=context_minutes)
    context = _window_readings(store, context_start, context_end)

    return {
        "episode_id": episode_id,
        "status": "ok",
        "episode": match,
        "context": {
            "start": _iso(context_start),
            "end": _iso(context_end),
            "readings": [_reading_dict(g) for g in context],
        },
    }


def analyze_time_blocks(
    store: StoragePort,
    start: datetime,
    end: datetime,
    *,
    target_low: int = TARGET_LOW_MG_DL,
    target_high: int = TARGET_HIGH_MG_DL,
) -> dict[str, Any]:
    """Per-block statistics using fixed UTC day segments."""
    readings = _window_readings(store, start, end)
    buckets: dict[str, list[GlucoseEvent]] = {name: [] for name in TIME_BLOCKS_UTC}

    for reading in readings:
        hour = reading.ts.hour
        for name, (lo, hi) in TIME_BLOCKS_UTC.items():
            if lo <= hour < hi:
                buckets[name].append(reading)
                break

    blocks: dict[str, Any] = {}
    for name, (lo, hi) in TIME_BLOCKS_UTC.items():
        block_readings = buckets[name]
        label = f"{lo:02d}:00-{hi:02d}:00"
        if not block_readings:
            blocks[name] = {
                "utc_range": label,
                "reading_count": 0,
            }
            continue
        blocks[name] = _statistics_payload(
            block_readings,
            start=start,
            end=end,
            target_low=target_low,
            target_high=target_high,
        )
        blocks[name]["utc_range"] = label

    return {
        "start": _iso(start),
        "end": _iso(end),
        "timezone": "UTC",
        "blocks": blocks,
    }


def _project_glucose(mg_dl: int, trend: str | None, *, intervals: int) -> int | None:
    if trend is None or trend not in _TREND_RATE_PER_5MIN:
        return None
    rate = _TREND_RATE_PER_5MIN[trend]
    projected = mg_dl + rate * intervals
    return max(10, min(600, round(projected)))


def check_alerts(
    store: StoragePort,
    realtime: RealtimeConnector | None = None,
    *,
    now: datetime | None = None,
    urgent_low: int = VERY_LOW_MG_DL,
    low: int = TARGET_LOW_MG_DL,
    high: int = TARGET_HIGH_MG_DL,
    urgent_high: int = VERY_HIGH_MG_DL,
) -> dict[str, Any]:
    """Threshold + trend projection alerts - informational only."""
    current = get_current_glucose(store, realtime, now=now)
    payload: dict[str, Any] = {
        "disclaimer": ALERT_DISCLAIMER,
        "current": current,
        "alerts": [],
        "projections": None,
    }

    reading = current.get("reading")
    if reading is None:
        payload["status"] = "no_data"
        return payload

    mg_dl = int(reading["mg_dl"])
    trend = reading.get("trend")
    alerts: list[dict[str, Any]] = []

    if mg_dl < urgent_low:
        alerts.append({"level": "urgent", "kind": "severe_hypo", "mg_dl": mg_dl})
    elif mg_dl < low:
        alerts.append({"level": "warning", "kind": "hypo", "mg_dl": mg_dl})
    elif mg_dl > urgent_high:
        alerts.append({"level": "urgent", "kind": "severe_hyper", "mg_dl": mg_dl})
    elif mg_dl > high:
        alerts.append({"level": "warning", "kind": "hyper", "mg_dl": mg_dl})

    if trend in ("SingleDown", "DoubleDown") and mg_dl < 100:
        alerts.append({"level": "warning", "kind": "falling_fast", "mg_dl": mg_dl, "trend": trend})
    elif trend in ("SingleUp", "DoubleUp") and mg_dl > 150:
        alerts.append({"level": "warning", "kind": "rising_fast", "mg_dl": mg_dl, "trend": trend})

    proj_15 = _project_glucose(mg_dl, trend, intervals=3)
    proj_30 = _project_glucose(mg_dl, trend, intervals=6)
    if proj_15 is not None or proj_30 is not None:
        payload["projections"] = {
            "minutes_15": proj_15,
            "minutes_30": proj_30,
            "trend": trend,
        }

    payload["alerts"] = alerts
    payload["status"] = "alert" if alerts else "ok"
    return payload


def export_data(
    store: StoragePort,
    start: datetime,
    end: datetime,
    *,
    format: Literal["json", "csv"] = "json",
) -> dict[str, Any]:
    """Export readings as a serialized string payload."""
    readings = _window_readings(store, start, end)
    records = [_reading_dict(g) for g in readings]
    result: dict[str, Any] = {
        "start": _iso(start),
        "end": _iso(end),
        "format": format,
        "count": len(records),
        "data": "",
    }
    if not records:
        return result

    if format == "json":
        result["data"] = json.dumps(records, separators=(",", ":"))
    else:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=["timestamp", "mg_dl", "trend"])
        writer.writeheader()
        for row in records:
            writer.writerow(row)
        result["data"] = buffer.getvalue()
    return result


def get_agp_report(
    store: StoragePort,
    start: datetime,
    end: datetime,
) -> dict[str, Any]:
    """AGP-style percentile profile - 5-minute-of-day bins across days."""
    readings = _window_readings(store, start, end)
    if not readings:
        return {
            "start": _iso(start),
            "end": _iso(end),
            "reading_count": 0,
            "bins": [],
        }

    by_bin: dict[int, list[float]] = {i: [] for i in range(_AGP_BINS_PER_DAY)}
    for reading in readings:
        ts = _require_utc(reading.ts)
        minute_of_day = ts.hour * 60 + ts.minute
        bin_index = minute_of_day // 5
        if 0 <= bin_index < _AGP_BINS_PER_DAY:
            by_bin[bin_index].append(float(reading.mg_dl))

    bins: list[dict[str, Any]] = []
    for bin_index in range(_AGP_BINS_PER_DAY):
        values = by_bin[bin_index]
        minute_of_day = bin_index * 5
        entry: dict[str, Any] = {
            "minute_of_day": minute_of_day,
            "time_utc": f"{minute_of_day // 60:02d}:{minute_of_day % 60:02d}",
            "reading_count": len(values),
        }
        if values:
            entry["p5"] = _percentile(values, 0.05)
            entry["p25"] = _percentile(values, 0.25)
            entry["p50"] = _percentile(values, 0.50)
            entry["p75"] = _percentile(values, 0.75)
            entry["p95"] = _percentile(values, 0.95)
        bins.append(entry)

    return {
        "start": _iso(start),
        "end": _iso(end),
        "reading_count": len(readings),
        "day_count": len({g.ts.date() for g in readings}),
        "bins": bins,
    }


def get_status_summary(
    store: StoragePort,
    realtime: RealtimeConnector | None = None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Current reading, last-24h stats, active episodes, and alerts."""
    clock = _require_utc(now or datetime.now(tz=UTC))
    window_start = clock - timedelta(hours=24)
    current = get_current_glucose(store, realtime, now=clock)
    stats_24h = get_statistics(store, window_start, clock)
    episodes = detect_episodes(store, window_start, clock)
    active_episodes = [e for e in episodes["episodes"] if e.get("ongoing")]
    alerts = check_alerts(store, realtime, now=clock)

    return {
        "as_of": _iso(clock),
        "current": current,
        "last_24h": stats_24h,
        "active_episodes": active_episodes,
        "alerts": {
            "status": alerts["status"],
            "count": len(alerts["alerts"]),
            "items": alerts["alerts"],
            "disclaimer": ALERT_DISCLAIMER,
        },
    }


# ── insulin extension (read-only, capability-gated) ──────────────────────────


def get_boluses(
    store: StoragePort,
    start: datetime,
    end: datetime,
) -> dict[str, Any]:
    """Bolus insulin events in a UTC window."""
    start_utc, end_utc = _validate_window(start, end)
    boluses = [
        e for e in store.get_insulin(start_utc, end_utc) if e.kind is InsulinKind.BOLUS
    ]
    items = [
        {"ts": _iso(e.ts), "units": e.units, "automatic": e.automatic} for e in boluses
    ]
    capped, note = _cap_events(items)
    payload: dict[str, Any] = {
        "start": _iso(start_utc),
        "end": _iso(end_utc),
        "n_boluses": len(boluses),
        "total_units": round(sum(e.units for e in boluses if e.units is not None), 2),
        "boluses": capped,
    }
    if note is not None:
        payload["truncation_note"] = note
    return payload


def get_carb_entries(
    store: StoragePort,
    start: datetime,
    end: datetime,
) -> dict[str, Any]:
    """Meal events carrying carb counts in a UTC window."""
    start_utc, end_utc = _validate_window(start, end)
    meals = [m for m in store.get_meals(start_utc, end_utc) if m.carbs_g is not None]
    items: list[dict[str, Any]] = []
    for meal in meals:
        entry: dict[str, Any] = {"ts": _iso(meal.ts), "carbs_g": meal.carbs_g}
        if meal.protein_g is not None:
            entry["protein_g"] = meal.protein_g
        if meal.fat_g is not None:
            entry["fat_g"] = meal.fat_g
        if meal.note is not None:
            entry["note"] = meal.note
        items.append(entry)
    capped, note = _cap_events(items)
    payload: dict[str, Any] = {
        "start": _iso(start_utc),
        "end": _iso(end_utc),
        "n_entries": len(meals),
        "total_carbs_g": round(sum(m.carbs_g for m in meals if m.carbs_g is not None), 1),
        "entries": capped,
    }
    if note is not None:
        payload["truncation_note"] = note
    return payload


def get_basal_timeline(
    store: StoragePort,
    start: datetime,
    end: datetime,
) -> dict[str, Any]:
    """Basal, temp-basal, and suspend events in a UTC window."""
    start_utc, end_utc = _validate_window(start, end)
    basal_kinds = (InsulinKind.BASAL, InsulinKind.TEMP_BASAL, InsulinKind.SUSPEND)
    events = [
        e for e in store.get_insulin(start_utc, end_utc) if e.kind in basal_kinds
    ]
    items: list[dict[str, Any]] = []
    for event in events:
        item: dict[str, Any] = {"ts": _iso(event.ts), "kind": event.kind.value}
        if event.units is not None:
            item["units"] = event.units
        if event.duration_min is not None:
            item["duration_min"] = event.duration_min
        items.append(item)
    capped, note = _cap_events(items)
    n_temp_basal = sum(1 for e in events if e.kind is InsulinKind.TEMP_BASAL)
    n_suspend = sum(1 for e in events if e.kind is InsulinKind.SUSPEND)
    payload: dict[str, Any] = {
        "start": _iso(start_utc),
        "end": _iso(end_utc),
        "n_basal": sum(1 for e in events if e.kind is InsulinKind.BASAL),
        "n_temp_basal": n_temp_basal,
        "n_suspend": n_suspend,
        "basal_stable": n_temp_basal == 0 and n_suspend == 0,
        "events": capped,
    }
    if note is not None:
        payload["truncation_note"] = note
    return payload


def get_iob(store: StoragePort, timestamp: datetime) -> dict[str, Any]:
    """Tier B insulin-on-board at a moment, from logged boluses (oref0 curve)."""
    at = _require_utc(timestamp)
    lookback_start = at - timedelta(hours=_IOB_LOOKBACK_HOURS)
    doses: list[tuple[datetime, float]] = []
    n_recent = 0
    for event in store.get_insulin(lookback_start, at + timedelta(minutes=1)):
        if event.kind is not InsulinKind.BOLUS or event.ts > at:
            continue
        n_recent += 1
        if event.units is not None:
            doses.append((event.ts, event.units))
    totals = insulin_totals(doses, at)
    return {
        "timestamp": _iso(at),
        "iob_units": round(totals.iob, 3),
        "n_recent_boluses": n_recent,
        "tier": "B",
        "method": "computed from logged boluses (oref0 exponential curve)",
        "note": IOB_DISCLAIMER,
    }
