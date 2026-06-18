"""Open investigations: questions that accrue evidence across daemon cycles.

A watch trigger opens an investigation with a DETERMINISTIC sufficiency
condition, so the daemon can re-check it each cycle without an LLM:

- ``event_count``: wait until a pattern (an anomaly name) has recurred N times.
- ``days_elapsed``: wait until M days of data have accrued since it was opened.

Each cycle recomputes progress. When the condition is met, the coordinator runs
the real investigation (producing an InvestigationRun + findings) and the entry
is marked promoted. This closes the loop from real-time watch to durable finding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dexta_intelligence.models import OpenInvestigation

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    from langchain_core.language_models.chat_models import BaseChatModel

    from dexta_intelligence.agents.base import AgentContext
    from dexta_intelligence.config import Config
    from dexta_intelligence.store.port import StoragePort

logger = logging.getLogger(__name__)

CONDITION_EVENT_COUNT = "event_count"
CONDITION_DAYS_ELAPSED = "days_elapsed"
STATUS_COLLECTING = "collecting"
STATUS_PROMOTED = "promoted"
STATUS_DISMISSED = "dismissed"

#: How many recurrences of an anomaly before it is worth a deep investigation.
DEFAULT_EVENT_TARGET = 3.0

__all__ = [
    "CONDITION_DAYS_ELAPSED",
    "CONDITION_EVENT_COUNT",
    "OpenInvestigationReport",
    "ensure_open_investigation",
    "evaluate_open_investigations",
    "open_from_anomalies",
    "progress",
]


@dataclass(frozen=True, slots=True)
class OpenInvestigationReport:
    """Outcome of one evaluation pass over the open-investigations queue."""

    evaluated: int = 0
    promoted: int = 0


def _anomaly_count(store: StoragePort, subject: str) -> float:
    """How many times the anomaly named ``subject`` has been recorded."""
    try:
        rows = store.get_findings(agent="monitor", kind="anomaly", status=None, limit=1_000_000)
    except Exception:
        return 0.0
    return float(sum(1 for f in rows if f.scope == subject))


def progress(store: StoragePort, inv: OpenInvestigation, now: datetime) -> float:
    """Current progress toward the sufficiency target (deterministic, no LLM)."""
    if inv.condition_type == CONDITION_EVENT_COUNT:
        return _anomaly_count(store, inv.subject)
    if inv.condition_type == CONDITION_DAYS_ELAPSED:
        return max(0.0, (now - inv.created_at).total_seconds() / 86_400.0)
    return inv.current


def ensure_open_investigation(
    store: StoragePort,
    *,
    question: str,
    condition_type: str,
    subject: str,
    target: float,
    now: datetime,
) -> OpenInvestigation | None:
    """Open an investigation if an equivalent one is not already pending.

    Deduped on ``(condition_type, subject)`` against collecting/promoted entries,
    so a recurring trigger never stacks duplicates. Returns the new row or None.
    """
    try:
        existing = store.get_open_investigations()
    except Exception:
        existing = []
    for e in existing:
        if (
            e.condition_type == condition_type
            and e.subject == subject
            and e.status in (STATUS_COLLECTING, STATUS_PROMOTED)
        ):
            return None
    inv = OpenInvestigation(
        question=question,
        condition_type=condition_type,
        subject=subject,
        target=target,
        current=0.0,
        status=STATUS_COLLECTING,
        created_at=now,
    )
    try:
        store.insert_open_investigation(inv)
    except Exception:
        logger.warning("open_investigations: failed to open %r", question, exc_info=True)
        return None
    return inv


def open_from_anomalies(
    store: StoragePort, anomaly_names: list[str], *, now: datetime
) -> int:
    """Open an event-count investigation per distinct anomaly name. Returns opened."""
    opened = 0
    for name in dict.fromkeys(anomaly_names):  # distinct, order-preserving
        inv = ensure_open_investigation(
            store,
            question=f"Why does {name.replace('_', ' ')} keep happening?",
            condition_type=CONDITION_EVENT_COUNT,
            subject=name,
            target=DEFAULT_EVENT_TARGET,
            now=now,
        )
        if inv is not None:
            opened += 1
    return opened


def evaluate_open_investigations(
    ctx: AgentContext,
    config: Config,
    model: BaseChatModel | None,
    *,
    now: datetime,
    investigate_fn: Callable[[OpenInvestigation], str | None] | None = None,
) -> OpenInvestigationReport:
    """Recompute progress for collecting investigations; promote those that are met.

    ``investigate_fn`` runs the investigation for a met entry and returns a run id
    (injectable for tests). The default runs the coordinator and persists findings.
    Never raises: a failed promotion leaves the entry collecting.
    """
    try:
        pending = ctx.store.get_open_investigations(status=STATUS_COLLECTING)
    except Exception:
        return OpenInvestigationReport()

    runner = (
        investigate_fn
        if investigate_fn is not None
        else _default_investigate(ctx, config, model)
    )
    promoted = 0
    for inv in pending:
        if inv.id is None:
            continue
        current = progress(ctx.store, inv, now)
        if current >= inv.target:
            run_id = runner(inv)
            try:
                ctx.store.update_open_investigation(
                    inv.id, current=current, status=STATUS_PROMOTED, promoted_run_id=run_id
                )
            except Exception:
                logger.warning("open_investigations: promote update failed", exc_info=True)
                continue
            promoted += 1
        else:
            try:
                ctx.store.update_open_investigation(
                    inv.id, current=current, status=STATUS_COLLECTING
                )
            except Exception:
                logger.debug("open_investigations: progress update failed", exc_info=True)
    return OpenInvestigationReport(evaluated=len(pending), promoted=promoted)


def _default_investigate(
    ctx: AgentContext, config: Config, model: BaseChatModel | None
) -> Callable[[OpenInvestigation], str | None]:
    def run(inv: OpenInvestigation) -> str | None:
        from dexta_intelligence.agents.coordinator import CoordinatorAgent  # noqa: PLC0415
        from dexta_intelligence.workflows.deep_analysis import persist_findings  # noqa: PLC0415

        try:
            findings = CoordinatorAgent(model=model, config=config).investigate(
                ctx, goal=inv.question
            )
            persist_findings(ctx.store, findings)
        except Exception:
            logger.warning("open_investigations: investigation failed", exc_info=True)
            return None
        return ctx.run_id

    return run
