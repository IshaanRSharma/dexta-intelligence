"""View-model logic for the System observability page.

Pure data shaping: pipeline health, agent runs, the instrument log aggregated
across runs, rigor signals (skeptic rejections, coverage-limited runs), and the
configured model. Reads the store and config; returns plain dicts the template
renders. No HTML, no side effects.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from dexta_intelligence.models import FindingStatus
from dexta_intelligence.server._format import _relative_time

if TYPE_CHECKING:
    from dexta_intelligence.config import Config
    from dexta_intelligence.store.port import StoragePort

__all__ = ["system_page_view"]

#: A window wide enough to count all-time rows for the pipeline summary.
_ALL_TIME_START = datetime(1970, 1, 1, tzinfo=UTC)


def _safe(call: Any, default: Any) -> Any:
    """Run a store call, returning ``default`` on any failure (older DBs)."""
    try:
        return call()
    except Exception:
        return default


def system_page_view(config: Config, store: StoragePort, now: datetime) -> dict[str, Any]:
    """Assemble the System page sections from the store and config."""
    end = now + timedelta(days=1)
    coverage = _safe(store.coverage, None)
    source_counts = _safe(store.source_event_counts, {})
    manual = _safe(lambda: store.get_manual_events(_ALL_TIME_START, end), [])
    predictions = _safe(lambda: store.get_predictions(_ALL_TIME_START, end), [])
    profile_versions = _safe(store.get_profile_versions, [])
    runs = _safe(lambda: store.get_investigation_runs(limit=200), [])

    last_sync = None
    for src in source_counts:
        ts = _safe(lambda s=src: store.get_watermark(s), None)
        if ts is not None and (last_sync is None or ts > last_sync):
            last_sync = ts

    pipeline = {
        "raw_events": sum(source_counts.values()),
        "sources": len(source_counts),
        "glucose": coverage.n_glucose if coverage else 0,
        "insulin": coverage.n_insulin if coverage else 0,
        "meals": coverage.n_meals if coverage else 0,
        "sleep": coverage.n_sleep if coverage else 0,
        "activity": coverage.n_activity if coverage else 0,
        "manual_logs": len(manual),
        "predictions": len(predictions),
        "profile_versions": len(profile_versions),
        "coverage_pct": round(coverage.glucose_coverage_pct, 1) if coverage else 0.0,
        "span_days": round(coverage.span_days, 1) if coverage else 0.0,
        "last_sync": _relative_time(last_sync, now),
    }

    by_status: dict[str, int] = {}
    instruments: dict[str, dict[str, int]] = {}
    for run in runs:
        by_status[run.status] = by_status.get(run.status, 0) + 1
        for call in run.tool_calls:
            name = str(call.get("producer", "unknown"))
            agg = instruments.setdefault(name, {"runs": 0, "findings": 0})
            agg["runs"] += 1
            agg["findings"] += int(call.get("n_findings", 0) or 0)

    agent_runs = {
        "total": len(runs),
        "by_status": by_status,
        "last_run": _relative_time(runs[0].finished_at, now) if runs else "never",
        "recent": [
            {
                "question": r.question or "Whole-record investigation",
                "status": r.status,
                "n_findings": r.n_findings,
                "when": _relative_time(r.finished_at, now),
            }
            for r in runs[:10]
        ],
    }

    instrument_rows = [
        {"name": name, "runs": agg["runs"], "findings": agg["findings"]}
        for name, agg in sorted(instruments.items(), key=lambda kv: -kv[1]["runs"])
    ]

    rejected = _safe(lambda: store.get_findings(status=FindingStatus.REJECTED, limit=1000), [])
    active = _safe(lambda: store.get_findings(status=FindingStatus.ACTIVE, limit=1000), [])
    stale = _safe(lambda: store.get_findings(status=FindingStatus.STALE, limit=1000), [])
    rigor = {
        "active_findings": len(active),
        "skeptic_rejected": len(rejected),
        "stale_findings": len(stale),
        "limited_runs": by_status.get("limited", 0),
        "failed_runs": by_status.get("failed", 0),
    }

    roles = config.llm.roles or {}
    model = {
        "provider": config.llm.provider,
        "model": config.llm.model,
        "discovery": roles.get("discovery", config.llm.model),
        "chat": roles.get("chat", config.llm.model),
    }

    return {
        "pipeline": pipeline,
        "agent_runs": agent_runs,
        "instruments": instrument_rows,
        "rigor": rigor,
        "model": model,
    }
