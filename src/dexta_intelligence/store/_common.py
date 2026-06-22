"""Pure helpers shared by the SQLite and Postgres stores.

Only backend-agnostic functions live here. Row mappers differ between the
backends (TEXT-JSON vs JSONB) and stay in their own modules.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = ["_opt_json", "_prediction_horizon_min"]


def _opt_json(value: str | None, default: Any) -> Any:
    """Decode a nullable JSON text column, falling back for legacy NULL rows."""
    return default if value is None else json.loads(value)


def _prediction_horizon_min(values: list[float]) -> int:
    """Minutes from cycle time to the last predicted point (5-minute spacing)."""
    if not values:
        return 0
    return max(0, (len(values) - 1) * 5)
