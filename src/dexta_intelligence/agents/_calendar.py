"""Calendar / date-math primitives - the single source of truth for date interpretation.

LLMs compute date properties from strings unreliably (e.g. calling 2026-05-02
"Wednesday" when it is a Saturday), so anything that interprets a date - LLM-facing
tools, prompt formatters - must call these helpers instead. Pure functions, no I/O.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

__all__ = [
    "is_weekend",
    "parse_iso_date",
    "parse_relative_date",
    "weekday_name",
]

# Long form ("Saturday") is what goes in LLM context - clearer to the model
# than "Sat" and unambiguous next to 3-letter month abbreviations.
_LONG_WEEKDAY = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_SHORT_WEEKDAY = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def parse_iso_date(value: str | date | datetime | None) -> date | None:
    """Coerce common date inputs to a ``date``. Returns None on failure.

    Accepts ``date`` / ``datetime`` directly, ISO strings ("2026-05-02" or
    "2026-05-02T20:34:00+00:00"), or anything with a 10-char ISO date prefix.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    s = str(value).strip()
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except (ValueError, TypeError):
        return None


def weekday_name(value: str | date | datetime | None, *, short: bool = False) -> str | None:
    """Return the weekday name for any date-like input, or None if unparseable."""
    d = parse_iso_date(value)
    if d is None:
        return None
    return (_SHORT_WEEKDAY if short else _LONG_WEEKDAY)[d.weekday()]


def is_weekend(value: str | date | datetime | None) -> bool | None:
    d = parse_iso_date(value)
    if d is None:
        return None
    return d.weekday() >= 5


def parse_relative_date(  # noqa: PLR0911 - a phrase table; each branch is one rule
    expr: str, today: date | None = None
) -> date | None:
    """Resolve a natural-language relative-date phrase to a concrete ``date``.

    Recognized:
      - "today", "yesterday", "tomorrow"
      - "N days ago", "in N days"
      - "last/this/next monday".."sunday"
      - "last week" → 7 days ago, "last month" → 30 days ago

    Returns None for anything unparseable - caller can ask the user to
    clarify rather than guess.
    """
    if not expr:
        return None
    today = today or date.today()
    s = expr.strip().lower()

    if s in ("today", "now"):
        return today
    if s == "yesterday":
        return today - timedelta(days=1)
    if s == "tomorrow":
        return today + timedelta(days=1)

    parts = s.split()
    if len(parts) == 3 and parts[1] == "days" and parts[2] == "ago" and parts[0].isdigit():
        return today - timedelta(days=int(parts[0]))
    if len(parts) == 3 and parts[0] == "in" and parts[1].isdigit() and parts[2] in ("day", "days"):
        return today + timedelta(days=int(parts[1]))

    # Coarse but useful anchors.
    if s == "last week":
        return today - timedelta(days=7)
    if s == "last month":
        return today - timedelta(days=30)

    weekday_map = {n.lower(): i for i, n in enumerate(_LONG_WEEKDAY)}
    weekday_map.update({n.lower(): i for i, n in enumerate(_SHORT_WEEKDAY)})
    if len(parts) == 2 and parts[0] in ("last", "this", "next") and parts[1] in weekday_map:
        target_wd = weekday_map[parts[1]]
        days_diff = (target_wd - today.weekday()) % 7
        if parts[0] == "last":
            # "last Tuesday" when today is Wed → 1 day ago, not 6 days from now.
            days_diff -= 7
        elif parts[0] == "next":
            # Next future occurrence - skip today even if same weekday.
            days_diff = days_diff if days_diff > 0 else 7
        return today + timedelta(days=days_diff)

    return None
