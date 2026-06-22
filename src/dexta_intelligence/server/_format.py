"""Small presentation helpers shared across the server views."""

from __future__ import annotations

from datetime import UTC, datetime

__all__ = ["_relative_time"]


def _relative_time(ts: datetime | None, now: datetime) -> str:
    """Human relative time of ``ts`` against ``now``; ``None`` renders 'never'.
    A naive ``ts`` is read as UTC."""
    if ts is None:
        return "never"
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    secs = int((now - ts.astimezone(UTC)).total_seconds())
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"
