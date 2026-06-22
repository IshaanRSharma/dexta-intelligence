"""View-model for the Memory Inspector page.

Surfaces what memory the retrieval guard would reuse versus withhold, and why.
``_recall`` returns the ACTIVE, non-dosing beliefs that an answer may reason
from (``findings``), plus the memory it withholds with a reason
(``excluded``): stale, rejected, superseded, contradicted, or safety-blocked
because it reads as dosing advice. Runs on page load and degrades to empty
lists on any failure so the page never 500s.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any

from dexta_intelligence.coldstart import ColdStartReport

if TYPE_CHECKING:
    from datetime import datetime

    from dexta_intelligence.config import Config
    from dexta_intelligence.store.port import StoragePort

__all__ = ["memory_page_view"]


def memory_page_view(store: StoragePort, config: Config, now: datetime) -> dict[str, Any]:
    """Run the retrieval guard and shape its reused / withheld memory for the page."""
    from dexta_intelligence.agents.base import AgentContext  # noqa: PLC0415
    from dexta_intelligence.agents.tools.toolkit import _recall  # noqa: PLC0415

    try:
        coverage = store.coverage()
        gates = ColdStartReport.from_coverage(coverage)
        end = coverage.last_ts.date() if coverage.last_ts is not None else now.date()
        days = config.analysis.deep_analysis_window_days
        ctx = AgentContext(
            store=store,
            window=(end - timedelta(days=days), end),
            gates=gates,
            run_id="gui-memory",
            timezone=config.analysis.timezone,
        )
        payload, _numbers = _recall(ctx, "")
    except Exception:
        return {"used": [], "excluded": [], "n_used": 0, "n_excluded": 0}

    used = list(payload.get("findings", []))
    excluded = list(payload.get("excluded", []))
    return {
        "used": used,
        "excluded": excluded,
        "n_used": len(used),
        "n_excluded": len(excluded),
    }
