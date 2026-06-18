"""Pure formatter: an agent's tool-call path → human-readable narrative.

The "show your thinking" surface. ``render_trace`` turns a sequence of executed
``ToolCall``s (``agents/reason.py``) into one ``TraceLine`` each, restating only
fields the tool result already guarantees. It NEVER makes a model call and NEVER
introduces a number absent from the result — it cannot fabricate, so it needs no
guard. Every
field access tolerates a missing key, so a partial result degrades to a plainer
line instead of raising.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from dexta_intelligence.agents.reason import ToolCall

__all__ = [
    "TraceLine",
    "render_trace",
]


@dataclass(frozen=True, slots=True)
class TraceLine:
    """One narrated step of the agent's path.

    ``icon`` is a stable category the surface maps to a glyph/colour:
    ``scope`` | ``zoom`` | ``compare`` | ``recall`` | ``scan`` | ``trend``
    | ``treatment`` | ``time``.
    ``text`` restates the tool result — no number not already in it.
    """

    icon: str
    text: str


def render_trace(steps: Sequence[ToolCall]) -> list[TraceLine]:
    """Map each executed ``ToolCall`` to one ``TraceLine``.

    A not-ok step renders a graceful "couldn't …" line; an unknown tool renders
    a generic "ran …" line. Never raises on a missing result key.
    """
    return [_render_one(step) for step in steps]


def _render_one(step: ToolCall) -> TraceLine:
    if not step.ok:
        return TraceLine("scope", f"couldn't {step.name} ({_error_of(step.result)})")
    renderer = _RENDERERS.get(step.name)
    if renderer is None:
        return TraceLine("scope", f"ran {step.name}")
    result = step.result if isinstance(step.result, dict) else {}
    return renderer(result)


# ── per-tool renderers ───────────────────────────────────────────────────────


def _set_window(r: dict[str, Any]) -> TraceLine:
    start = _get(r, "active_start", "?")
    end = _get(r, "active_end", "?")
    days = _count(r, "n_days", "day")
    readings = _count(r, "n_readings", "reading")
    text = f"narrowed to {start} → {end} ({days}, {readings})"
    note = r.get("note")
    if note:
        text = f"{text} — {note}"
    return TraceLine("scope", text)


def _zoom_event(r: dict[str, Any]) -> TraceLine:
    center = _get(r, "center", "?")
    pad = _get(r, "pad_hours", "?")
    peak = _get(r, "peak", "?")
    nadir = _get(r, "nadir", "?")
    return TraceLine("zoom", f"zoomed to {center} ±{pad}h — peak {peak}, nadir {nadir}")


def _daily_series(r: dict[str, Any]) -> TraceLine:
    metric = _get(r, "metric", "metric")
    return TraceLine("trend", f"read the {metric} day-by-day over {_count(r, 'n_days', 'day')}")


def _list_segments(r: dict[str, Any]) -> TraceLine:
    n = len(r.get("segments") or [])
    return TraceLine("scan", f"scanned {_plural(n, 'segment')} to orient")


def _compare(r: dict[str, Any]) -> TraceLine:
    label_a = _get(r, "label_a", "group A")
    label_b = _get(r, "label_b", "group B")
    interp = _get(r, "interpretation", "no")
    delta = r.get("delta")
    delta_str = _signed(delta) if isinstance(delta, (int, float)) else "n/a"
    return TraceLine("compare", f"{label_a} vs {label_b}: {interp} difference ({delta_str})")


def _recall(r: dict[str, Any]) -> TraceLine:
    findings = r.get("findings")
    if isinstance(findings, list) and findings:
        count = _plural(len(findings), "finding")
        return TraceLine("recall", f"checked what I already know ({count})")
    return TraceLine("recall", "checked what I already know")


def _coverage(r: dict[str, Any]) -> TraceLine:
    days = r.get("span_days")
    if isinstance(days, (int, float)):
        return TraceLine("scan", f"checked how much data exists ({_plural(int(days), 'day')})")
    return TraceLine("scan", "checked how much data exists")


def _search_evidence(r: dict[str, Any]) -> TraceLine:
    hits = r.get("hits")
    n = len(hits) if isinstance(hits, list) else 0
    return TraceLine("scan", f"searched the literature ({_plural(n, 'hit')})")


def _get_carb_entries(r: dict[str, Any]) -> TraceLine:
    n = r.get("n_entries")
    if n == 0:
        return TraceLine("treatment", "checked carb entries — none in the window")
    total = r.get("total_carbs_g")
    extra = f", {total}g total" if isinstance(total, (int, float)) else ""
    return TraceLine(
        "treatment", f"checked carb entries ({_count(r, 'n_entries', 'entry')}{extra})"
    )


def _get_boluses(r: dict[str, Any]) -> TraceLine:
    n = r.get("n_boluses")
    if n == 0:
        return TraceLine("treatment", "checked bolus timing — no boluses in the window")
    delays = [
        b["minutes_after_carb_entry"]
        for b in r.get("boluses") or []
        if isinstance(b, dict) and isinstance(b.get("minutes_after_carb_entry"), (int, float))
    ]
    extra = f"; nearest {delays[0]:+.0f} min vs carb entry" if delays else ""
    return TraceLine(
        "treatment", f"checked bolus timing ({_count(r, 'n_boluses', 'bolus')}{extra})"
    )


def _get_basal_timeline(r: dict[str, Any]) -> TraceLine:
    if r.get("basal_stable"):
        return TraceLine("treatment", "checked basal/temp-basal context — stable")
    n_temp = r.get("n_temp_basal", "?")
    n_susp = r.get("n_suspend", "?")
    return TraceLine(
        "treatment",
        f"checked basal/temp-basal context — {n_temp} temp-basal, {n_susp} suspend",
    )


def _get_iob(r: dict[str, Any]) -> TraceLine:
    return TraceLine(
        "treatment",
        f"checked insulin on board ({_get(r, 'iob_units', '?')} U, tier {_get(r, 'tier', '?')})",
    )


def _get_insulin_profile(r: dict[str, Any]) -> TraceLine:
    if r.get("error"):
        return TraceLine("treatment", "checked insulin profile — not synced")
    name = _get(r, "active_profile", "?")
    n_seg = len(r.get("active_segments") or [])
    return TraceLine("treatment", f"checked insulin profile — {name!r} ({n_seg} segments)")


def _get_active_profile(r: dict[str, Any]) -> TraceLine:
    if r.get("error"):
        return TraceLine("treatment", "checked therapy profile — none available")
    name = _get(r, "version_name", _get(r, "active_profile", "?"))
    if r.get("versioned"):
        return TraceLine("treatment", f"loaded the profile in effect then — {name!r}")
    return TraceLine("treatment", f"loaded the current profile (no history yet) — {name!r}")


def _get_cob(r: dict[str, Any]) -> TraceLine:
    return TraceLine(
        "treatment",
        f"checked carbs on board ({_get(r, 'cob_g', '?')} g, tier {_get(r, 'tier', '?')})",
    )


def _find_spikes(r: dict[str, Any]) -> TraceLine:
    threshold = _get(r, "threshold", "?")
    return TraceLine(
        "zoom", f"scanned for excursions ≥ {threshold} ({_count(r, 'n_spikes', 'spike')})"
    )


def _find_similar_events(r: dict[str, Any]) -> TraceLine:
    n = r.get("n_similar")
    if not n:
        return TraceLine("compare", "looked for similar events — none found")
    spiking = r.get("n_spiking", "?")
    return TraceLine(
        "compare", f"compared {_plural(int(n), 'similar event')} ({spiking} spiked)"
    )


def _manual_summary(r: dict[str, Any]) -> str:
    rows = r.get("events") or []
    types = [e.get("event_type") for e in rows if isinstance(e, dict) and e.get("event_type")]
    head = f": {types[0]}" if len(types) == 1 else ""
    return f"{_count(r, 'n_events', 'user-reported note')}{head}"


def _get_manual_events(r: dict[str, Any]) -> TraceLine:
    if not r.get("n_events"):
        return TraceLine("recall", "checked manual context — no user-reported context found")
    return TraceLine("recall", f"checked manual context ({_manual_summary(r)})")


def _search_manual_events(r: dict[str, Any]) -> TraceLine:
    q = _get(r, "query", "")
    label = f" for {q!r}" if q else ""
    if not r.get("n_events"):
        return TraceLine("recall", f"searched manual context{label} — no matches")
    return TraceLine("recall", f"searched manual context{label} ({_manual_summary(r)})")


def _get_context_around_event(r: dict[str, Any]) -> TraceLine:
    if not r.get("n_events"):
        return TraceLine("recall", "checked manual context — no user-reported context found")
    return TraceLine("recall", f"checked manual context near the event ({_manual_summary(r)})")


def _current_time(r: dict[str, Any]) -> TraceLine:
    return TraceLine("time", f"anchored 'now' ({_get(r, 'date', '?')}, {_get(r, 'weekday', '?')})")


def _weekday(r: dict[str, Any]) -> TraceLine:
    return TraceLine("time", f"resolved {_get(r, 'date', '?')} → {_get(r, 'weekday', '?')}")


def _relative_date(r: dict[str, Any]) -> TraceLine:
    return TraceLine("time", f"resolved relative date → {_get(r, 'date', '?')}")


_RENDERERS: dict[str, Callable[[dict[str, Any]], TraceLine]] = {
    "get_carb_entries": _get_carb_entries,
    "get_boluses": _get_boluses,
    "get_basal_timeline": _get_basal_timeline,
    "get_iob": _get_iob,
    "get_insulin_profile": _get_insulin_profile,
    "get_active_profile": _get_active_profile,
    "get_cob": _get_cob,
    "find_spikes": _find_spikes,
    "find_similar_events": _find_similar_events,
    "get_manual_events": _get_manual_events,
    "search_manual_events": _search_manual_events,
    "get_context_around_event": _get_context_around_event,
    "get_current_time": _current_time,
    "get_weekday": _weekday,
    "parse_relative_date": _relative_date,
    "set_window": _set_window,
    "zoom_event": _zoom_event,
    "daily_series": _daily_series,
    "list_segments": _list_segments,
    "tod_compare": _compare,
    "groupby_compare": _compare,
    "event_proximity": _compare,
    "basal_overnight": _compare,
    "meal_response": _compare,
    "correction_outcome": _compare,
    "recall": _recall,
    "coverage": _coverage,
    "search_evidence": _search_evidence,
}


# ── formatting helpers ───────────────────────────────────────────────────────


def _get(r: dict[str, Any], key: str, default: str) -> str:
    val = r.get(key)
    return str(val) if val is not None else default


def _count(r: dict[str, Any], key: str, noun: str) -> str:
    n = r.get(key)
    if isinstance(n, (int, float)):
        return _plural(int(n), noun)
    return f"? {noun}s"


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _signed(value: float) -> str:
    return f"+{value}" if value > 0 else str(value)


def _error_of(result: Any) -> str:
    if isinstance(result, dict):
        err = result.get("error")
        if err:
            return str(err)
    return "unknown error"
