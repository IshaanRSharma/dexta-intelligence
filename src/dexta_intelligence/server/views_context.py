"""View-model for the Active Context Acquisition page.

Surfaces the questions dexta asks the user to log: unexplained glucose spikes
with no meal or note nearby. dexta asks rather than guessing the cause. Runs the
deterministic agent (no model) on page load, and degrades to an empty list on
any failure so the page never 500s.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from dexta_intelligence.coldstart import ColdStartReport

if TYPE_CHECKING:
    from dexta_intelligence.config import Config
    from dexta_intelligence.store.port import StoragePort

__all__ = ["context_page_view"]


def context_page_view(store: StoragePort, config: Config, now: datetime) -> dict[str, Any]:
    """Run the context-acquisition agent and shape its requests into page rows."""
    from dexta_intelligence.agents.base import AgentContext  # noqa: PLC0415
    from dexta_intelligence.agents.context_acquisition import (  # noqa: PLC0415
        ContextAcquisitionAgent,
    )

    try:
        coverage = store.coverage()
        gates = ColdStartReport.from_coverage(coverage)
        end = coverage.last_ts.date() if coverage.last_ts is not None else now.date()
        days = config.analysis.deep_analysis_window_days
        ctx = AgentContext(
            store=store,
            window=(end - timedelta(days=days), end),
            gates=gates,
            run_id="gui-context",
            timezone=config.analysis.timezone,
        )
        requests = ContextAcquisitionAgent().build(ctx)
    except Exception:
        return {"requests": [], "n": 0}

    rows = [
        {
            "question": r.question,
            "suggested_event_type": r.suggested_event_type,
            "when": r.event_ts.isoformat(),
            "peak": r.evidence.get("peak_mg_dl"),
        }
        for r in requests
    ]
    return {"requests": rows, "n": len(rows)}
