"""Round-trip tests for the ``investigation_runs`` StoragePort surface (SQLite).

Runs are returned newest-first and capped to ``limit``; the findings snapshot
and plan/trace JSON round-trip intact.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

import pytest

from dexta_intelligence.models import InvestigationRun, RunFinding
from dexta_intelligence.store import SQLiteStore

if TYPE_CHECKING:
    from collections.abc import Iterator

T0 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)


@pytest.fixture
def store() -> Iterator[SQLiteStore]:
    s = SQLiteStore(":memory:")
    s.migrate()
    yield s
    s.close()


def _run(run_id: str, *, question: str | None = None, n: int = 1) -> InvestigationRun:
    return InvestigationRun(
        run_id=run_id,
        kind="question" if question else "deep_analysis",
        status="completed",
        question=question,
        window_start=date(2026, 5, 1),
        window_end=date(2026, 6, 1),
        plan=["observation", "pattern"],
        trace=["Planned: observation, pattern", "Round 1: ran observation, pattern -> 1"],
        findings=[
            RunFinding(
                headline=f"finding {i}",
                kind="overnight-lows",
                confidence=0.7,
                status="active",
            )
            for i in range(n)
        ],
        n_findings=n,
        started_at=T0,
        finished_at=T0,
    )


def test_round_trip_preserves_plan_trace_and_findings(store: SQLiteStore) -> None:
    store.insert_investigation_run(_run("r1", question="overnight lows", n=2))
    (got,) = store.get_investigation_runs()
    assert got.run_id == "r1"
    assert got.kind == "question"
    assert got.question == "overnight lows"
    assert got.window_start == date(2026, 5, 1)
    assert got.plan == ["observation", "pattern"]
    assert len(got.trace) == 2
    assert [f.headline for f in got.findings] == ["finding 0", "finding 1"]
    assert got.findings[0].confidence == 0.7


def test_runs_returned_newest_first(store: SQLiteStore) -> None:
    store.insert_investigation_run(_run("old"))
    store.insert_investigation_run(_run("new"))
    assert [r.run_id for r in store.get_investigation_runs()] == ["new", "old"]


def test_limit_caps_runs(store: SQLiteStore) -> None:
    for i in range(4):
        store.insert_investigation_run(_run(f"r{i}"))
    assert len(store.get_investigation_runs(limit=2)) == 2


def test_get_one_run_by_id(store: SQLiteStore) -> None:
    row_id = store.insert_investigation_run(_run("r1"))
    got = store.get_investigation_run(row_id)
    assert got is not None and got.run_id == "r1"
    assert store.get_investigation_run(9999) is None
