"""The open-investigations queue: deterministic progress + promote-when-met.

No model or network: progress is computed from stored anomaly findings and the
clock, and promotion uses an injected investigate function.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.config import Config
from dexta_intelligence.models import Finding, FindingStatus, OpenInvestigation
from dexta_intelligence.store import SQLiteStore
from dexta_intelligence.workflows.open_investigations import (
    CONDITION_DAYS_ELAPSED,
    CONDITION_EVENT_COUNT,
    ensure_open_investigation,
    evaluate_open_investigations,
    open_from_anomalies,
    progress,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

NOW = datetime(2026, 6, 17, 12, 0, tzinfo=UTC)


@pytest.fixture
def store() -> Iterator[SQLiteStore]:
    s = SQLiteStore(":memory:")
    s.migrate()
    yield s
    s.close()


def _ctx(store: SQLiteStore) -> AgentContext:
    return AgentContext(
        store=store,
        window=(date(2026, 6, 1), date(2026, 6, 17)),
        gates=ColdStartReport.from_coverage(store.coverage()),
        run_id="oi-test",
    )


def _anomaly(name: str) -> Finding:
    return Finding(
        agent="monitor",
        kind="anomaly",
        scope=name,
        headline=f"{name} fired",
        status=FindingStatus.ACTIVE,
    )


def _open(subject: str, target: float, *, ct: str = CONDITION_EVENT_COUNT) -> OpenInvestigation:
    return OpenInvestigation(
        question=f"Why does {subject} keep happening?",
        condition_type=ct,
        subject=subject,
        target=target,
        current=0.0,
        status="collecting",
        created_at=NOW - timedelta(days=3),
    )


def test_event_count_progress_counts_anomalies(store: SQLiteStore) -> None:
    store.insert_finding(_anomaly("severe_high"))
    store.insert_finding(_anomaly("severe_high"))
    store.insert_finding(_anomaly("rapid_rise"))
    assert progress(store, _open("severe_high", 3), NOW) == 2.0
    assert progress(store, _open("rapid_rise", 3), NOW) == 1.0


def test_days_elapsed_progress(store: SQLiteStore) -> None:
    inv = _open("x", 7, ct=CONDITION_DAYS_ELAPSED)  # created 3 days before NOW
    assert progress(store, inv, NOW) == pytest.approx(3.0, abs=0.01)


def test_ensure_dedupes_by_subject(store: SQLiteStore) -> None:
    a = ensure_open_investigation(
        store, question="q", condition_type=CONDITION_EVENT_COUNT,
        subject="severe_high", target=3, now=NOW,
    )
    b = ensure_open_investigation(
        store, question="q", condition_type=CONDITION_EVENT_COUNT,
        subject="severe_high", target=3, now=NOW,
    )
    assert a is not None
    assert b is None  # already pending
    assert len(store.get_open_investigations()) == 1


def test_open_from_anomalies_distinct(store: SQLiteStore) -> None:
    opened = open_from_anomalies(store, ["severe_high", "severe_high", "rapid_rise"], now=NOW)
    assert opened == 2
    assert {o.subject for o in store.get_open_investigations()} == {"severe_high", "rapid_rise"}


def test_evaluate_promotes_when_target_met(store: SQLiteStore) -> None:
    store.insert_open_investigation(_open("severe_high", 2))
    store.insert_finding(_anomaly("severe_high"))
    store.insert_finding(_anomaly("severe_high"))

    report = evaluate_open_investigations(
        _ctx(store), Config(), None, now=NOW, investigate_fn=lambda _inv: "run-123"
    )
    assert report.promoted == 1
    (inv,) = store.get_open_investigations()
    assert inv.status == "promoted"
    assert inv.promoted_run_id == "run-123"
    assert inv.current == 2.0


def test_evaluate_updates_progress_when_not_met(store: SQLiteStore) -> None:
    store.insert_open_investigation(_open("severe_high", 5))
    store.insert_finding(_anomaly("severe_high"))

    report = evaluate_open_investigations(
        _ctx(store), Config(), None, now=NOW, investigate_fn=lambda _inv: "x"
    )
    assert report.promoted == 0
    (inv,) = store.get_open_investigations()
    assert inv.status == "collecting"
    assert inv.current == 1.0
