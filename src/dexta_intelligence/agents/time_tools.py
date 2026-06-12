"""LLM-callable date/time tools — deterministic calendar grounding.

The model must call these whenever it interprets ANY date, including dates
from other tools' output. Computing date properties from a string in-head
produces silent errors (the donor repo recorded the LLM calling 2026-05-02
"Wednesday"; it's a Saturday); these tools answer deterministically.

Always-on, like ``recall``/``coverage``:
  - get_current_time(timezone)   — what's "now"
  - get_weekday(date)            — Monday/.../Sunday for a given date
  - parse_relative_date(expr)    — "last Tuesday" / "3 days ago" → ISO date

Each tool's evidence dict carries the resolved date components (year/month/
day, plus hour/minute for "now") because the faithfulness guard's number
extractor parses bare years and day-of-month integers out of prose — a date
the model cites must trace to the pool like any other figure.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, tzinfo
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from dexta_intelligence.agents import _calendar as cal
from dexta_intelligence.agents.reason import ToolSpec

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

__all__ = ["CALENDAR_TOOL_NAMES", "time_tool_specs"]

CALENDAR_TOOL_NAMES: tuple[str, ...] = ("get_current_time", "get_weekday", "parse_relative_date")

_TIMEZONE_PARAM = {
    "type": "string",
    "description": "IANA timezone name, e.g. 'America/New_York' (defaults to the user's timezone)",
}


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _resolve_tz(name: str) -> tuple[tzinfo, str]:
    """ZoneInfo for ``name``, falling back to UTC on anything invalid."""
    try:
        return ZoneInfo(name), name
    except Exception:
        logger.debug("invalid timezone %r — falling back to UTC", name)
        return UTC, "UTC"


def _date_evidence(d: date, **extra: int) -> dict[str, Any]:
    return {"year": d.year, "month": d.month, "day": d.day, **extra}


def time_tool_specs(
    default_timezone: str = "UTC",
    *,
    now_fn: Callable[[], datetime] | None = None,
) -> list[ToolSpec]:
    """The three calendar ToolSpecs. ``now_fn`` lets tests freeze "now"."""
    now = now_fn or _utc_now

    def get_current_time(args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        tz, tz_used = _resolve_tz(str(args.get("timezone") or default_timezone))
        local = now().astimezone(tz)
        result = {
            "datetime": local.isoformat(),
            "date": local.strftime("%Y-%m-%d"),
            "weekday": cal.weekday_name(local.date()),
            "is_weekend": cal.is_weekend(local.date()),
            "timezone": tz_used,
            "utc_offset": local.strftime("%z"),
        }
        return result, _date_evidence(local.date(), hour=local.hour, minute=local.minute)

    def get_weekday(args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        raw = str(args.get("date") or "")
        d = cal.parse_iso_date(raw)
        if d is None:
            return {"ok": False, "error": f"could not parse date {raw!r} — pass ISO YYYY-MM-DD"}, {}
        result = {
            "ok": True,
            "date": d.isoformat(),
            "weekday": cal.weekday_name(d),
            "weekday_short": cal.weekday_name(d, short=True),
            "is_weekend": cal.is_weekend(d),
        }
        return result, _date_evidence(d)

    def parse_relative_date(args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        expr = str(args.get("expression") or "")
        tz, _ = _resolve_tz(str(args.get("timezone") or default_timezone))
        today = now().astimezone(tz).date()
        resolved = cal.parse_relative_date(expr, today=today)
        if resolved is None:
            return {
                "ok": False,
                "error": (
                    f"could not parse {expr!r} — try 'yesterday', 'last tuesday', "
                    "'N days ago', or a concrete ISO date"
                ),
            }, {}
        result = {
            "ok": True,
            "date": resolved.isoformat(),
            "weekday": cal.weekday_name(resolved),
            "is_weekend": cal.is_weekend(resolved),
        }
        return result, _date_evidence(resolved)

    return [
        ToolSpec(
            name="get_current_time",
            description=(
                "Get the current date, time, and weekday in the user's local timezone. "
                "Call this BEFORE interpreting any relative time expression: 'today', "
                "'yesterday', 'last night', '3 days ago', 'this week', 'right now'. "
                "Use the returned date and utc_offset to build the ISO date arguments "
                "for data tools (set_window start/end, zoom_event timestamp)."
            ),
            parameters={
                "type": "object",
                "properties": {"timezone": _TIMEZONE_PARAM},
            },
            fn=get_current_time,
        ),
        ToolSpec(
            name="get_weekday",
            description=(
                "Return the weekday name for a given ISO date. Call this whenever you "
                "need the day of the week a date falls on — NEVER compute the weekday "
                "yourself from a date string; you will get it wrong."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "ISO date YYYY-MM-DD (a trailing time component is ignored)",
                    },
                },
                "required": ["date"],
            },
            fn=get_weekday,
        ),
        ToolSpec(
            name="parse_relative_date",
            description=(
                "Resolve a natural-language date phrase to a concrete ISO date you can "
                "pass to data tools like set_window. Recognized: 'today', 'yesterday', "
                "'tomorrow', 'N days ago', 'in N days', 'last/this/next monday'..'sunday', "
                "'last week', 'last month'. Returns ok=false with an error when the "
                "phrase isn't understood — ask the user to be more specific, don't guess."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "the natural-language phrase, e.g. 'last tuesday'",
                    },
                    "timezone": _TIMEZONE_PARAM,
                },
                "required": ["expression"],
            },
            fn=parse_relative_date,
        ),
    ]
