"""Reasoning commands: ask, goals, brief, wiki."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, TextIO

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.cli._common import (
    MEDICAL_DISCLAIMER,
    StoreOpener,
    _analysis_window,
    _ctx_for,
    _maybe_close_store,
    model_for_role,
    open_sqlite_store,
)
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.memory.wiki import generate_wiki

if TYPE_CHECKING:
    from pathlib import Path

    from dexta_intelligence.config import Config
    from dexta_intelligence.models import Finding
    from dexta_intelligence.store.port import StoragePort


def cmd_ask(
    *,
    question: str,
    config: Config,
    db_path: Path | None,
    out: TextIO,
    opener: StoreOpener = open_sqlite_store,
    model: Any = None,
    seek: bool = False,
) -> int:
    """Answer a question: the router picks the tool family, the loop reasons.

    Default uses the orchestrator: the model decides the approach (a whole
    investigation workflow, granular tools, or a chain) over the full belt.
    ``--seek`` runs the goal-seeking agent that reflects and re-scopes across
    rounds until the question is actually answered. Both print the traversal trace.
    """
    from dexta_intelligence.agents.orchestrator import OrchestratorAgent  # noqa: PLC0415
    from dexta_intelligence.agents.seeker import GoalSeekingAgent  # noqa: PLC0415

    chat_model = model if model is not None else model_for_role(config, "explain")
    if chat_model is None:
        out.write(
            "Chat needs a language model. Install the extra and set a provider key:\n"
            "  pip install 'dexta-intelligence[llm]'  and  export OPENROUTER_API_KEY=...\n"
        )
        return 1

    store = opener(config, db_path)
    try:
        coverage = store.coverage()
        gates = ColdStartReport.from_coverage(coverage)
        if gates.below_hard_floor:
            out.write(f"Only {coverage.span_days:.1f} days of data — too little to reason over.\n")
            return 1
        end_date = coverage.last_ts.date() if coverage.last_ts is not None else None
        window = _analysis_window(config, end_date)
        ctx = AgentContext(
            store=store, window=window, gates=gates, run_id=str(uuid.uuid4()),
            timezone=config.analysis.timezone,
        )
        low, high = config.analysis.target_low, config.analysis.target_high
        if seek:
            answer = GoalSeekingAgent(
                model=chat_model, target_low=low, target_high=high
            ).pursue(ctx, question)
        else:
            answer = OrchestratorAgent(
                model=chat_model, target_low=low, target_high=high
            ).ask(ctx, question)
    finally:
        _maybe_close_store(store, opener)

    for line in answer.trace:
        out.write(f"  · {line.text}\n")
    out.write(f"\n{answer.text}\n")
    if answer.tools_used:
        out.write(f"\n  (tools: {', '.join(answer.tools_used)})\n")
    out.write(f"\n{MEDICAL_DISCLAIMER}\n")
    return 0


def cmd_monitor(
    *,
    config: Config,
    db_path: Path | None,
    out: TextIO,
    opener: StoreOpener = open_sqlite_store,
) -> int:
    """Scan recent data for anomalies (severe lows/highs, TIR cliffs, sensor gaps).

    Deterministic — no model. Writes ``kind="anomaly"`` findings and logs each via
    the default notifier; the dashboard and ``dexta brief`` surface them."""
    from dexta_intelligence.notifications import LogNotifier  # noqa: PLC0415
    from dexta_intelligence.workflows.monitor import run_monitor  # noqa: PLC0415

    store = opener(config, db_path)
    try:
        ctx = _ctx_for(config, store)
        anomalies = run_monitor(ctx, notify=LogNotifier(), persist=True)
    finally:
        _maybe_close_store(store, opener)

    if not anomalies:
        out.write("No anomalies in the recent window.\n")
        return 0
    out.write(f"{len(anomalies)} anomaly(ies) detected:\n")
    for a in anomalies:
        out.write(f"  [{a.severity}] {a.headline}\n")
    out.write(f"\n{MEDICAL_DISCLAIMER}\n")
    return 0


def cmd_explain(
    *,
    when: str,
    config: Config,
    db_path: Path | None,
    out: TextIO,
    opener: StoreOpener = open_sqlite_store,
    model: Any = None,
) -> int:
    """Explain a spike: deterministic investigation, optional LLM synthesis.

    ``when`` is an ISO date (auto-locates the day's largest excursion) or an
    ISO datetime. Works without a model; with one, the ``explain`` role synthesizes
    the finding from the evidence bundle (guard-audited).
    """
    from dexta_intelligence.investigations.spike import explain_spike  # noqa: PLC0415

    store = opener(config, db_path)
    try:
        coverage = store.coverage()
        gates = ColdStartReport.from_coverage(coverage)
        if gates.below_hard_floor:
            out.write(f"Only {coverage.span_days:.1f} days of data — too little to reason over.\n")
            return 1
        # The whole record: similar-event recurrence wants all history.
        window = (
            coverage.first_ts.date() if coverage.first_ts else datetime.now(tz=UTC).date(),
            coverage.last_ts.date() if coverage.last_ts else datetime.now(tz=UTC).date(),
        )
        ctx = AgentContext(
            store=store, window=window, gates=gates, run_id=str(uuid.uuid4()),
            timezone=config.analysis.timezone,
        )
        polish_model = model if model is not None else model_for_role(config, "explain")
        report = explain_spike(
            ctx,
            when,
            model=polish_model,
            target_low=config.analysis.target_low,
            target_high=config.analysis.target_high,
        )
    finally:
        _maybe_close_store(store, opener)

    out.write("Investigation trace:\n")
    for i, line in enumerate(report["trace"], 1):
        out.write(f"  {i}. {line}\n")
    out.write(f"\nFinding:\n  {report['headline']}\n")
    if report["evidence"]:
        out.write("\nEvidence:\n")
        for item in report["evidence"]:
            out.write(f"  - {item}\n")
    out.write(f"\nConfidence: {report['confidence']}\n")
    if report["limitations"]:
        out.write("\nLimitations:\n")
        for item in report["limitations"]:
            out.write(f"  - {item}\n")
    out.write(f"\nSafety: {report['safety']}\n")
    return 0


def cmd_demo(*, out: TextIO, model: Any = None) -> int:
    """Run dexta end-to-end on a synthetic patient — no data or API key needed.

    Builds an in-memory store with a planted recurring late-bolus dinner spike,
    then explains the canonical spike exactly like ``dexta explain`` does
    (deterministic when ``model`` is ``None``)."""
    from dexta_intelligence.demo import DEMO_SPIKE_DATE, build_demo_store  # noqa: PLC0415
    from dexta_intelligence.investigations.spike import explain_spike  # noqa: PLC0415

    out.write("Running dexta on a synthetic patient — no data or API key needed.\n\n")
    store = build_demo_store()
    try:
        coverage = store.coverage()
        gates = ColdStartReport.from_coverage(coverage)
        window = (
            coverage.first_ts.date() if coverage.first_ts else DEMO_SPIKE_DATE,
            coverage.last_ts.date() if coverage.last_ts else DEMO_SPIKE_DATE,
        )
        ctx = AgentContext(store=store, window=window, gates=gates, run_id=str(uuid.uuid4()))
        report = explain_spike(ctx, DEMO_SPIKE_DATE.isoformat(), model=model)
    finally:
        store.close()

    out.write("Investigation trace:\n")
    for i, line in enumerate(report["trace"], 1):
        out.write(f"  {i}. {line}\n")
    out.write(f"\nFinding:\n  {report['headline']}\n")
    if report["evidence"]:
        out.write("\nEvidence:\n")
        for item in report["evidence"]:
            out.write(f"  - {item}\n")
    out.write(f"\nConfidence: {report['confidence']}\n")
    if report["limitations"]:
        out.write("\nLimitations:\n")
        for item in report["limitations"]:
            out.write(f"  - {item}\n")
    out.write(f"\nSafety: {report['safety']}\n")
    return 0


def cmd_goals(
    *,
    action: str,
    statement: str | None,
    config: Config,
    db_path: Path | None,
    out: TextIO,
    opener: StoreOpener = open_sqlite_store,
    model: Any = None,
    now: datetime | None = None,
    target: float | None = None,
) -> int:
    moment = now or datetime.now(tz=UTC)
    store = opener(config, db_path)
    try:
        handlers = {
            "add": lambda: _goals_add(store, config, statement, out, model, moment, target),
            "list": lambda: _goals_list(store, out),
            "tick": lambda: _goals_tick(store, config, out, moment),
        }
        handler = handlers.get(action)
        if handler is None:
            out.write(f"Unknown goals action: {action}\n")
            return 2
        return handler()
    finally:
        _maybe_close_store(store, opener)


def _goals_add(
    store: StoragePort,
    config: Config,
    statement: str | None,
    out: TextIO,
    model: Any,
    now: datetime,
    target: float | None = None,
) -> int:
    from dexta_intelligence.workflows.goals import METRIC_LABELS, compose_goal  # noqa: PLC0415

    if not statement:
        out.write('Provide a goal, e.g. dexta goals add "reduce my overnight lows"\n')
        return 2
    goal = compose_goal(
        statement, model=model or model_for_role(config, "plan"), now=now, target=target
    )
    goal_id = store.insert_goal(goal)
    target_clause = f", target {goal.target}" if goal.target is not None else ""
    out.write(
        f"Goal #{goal_id}: {goal.statement}\n"
        f"  tracking {METRIC_LABELS[goal.metric]} ({goal.direction}){target_clause}, "
        f"every {goal.cadence_days}d\n"
    )
    return 0


def _goals_list(store: StoragePort, out: TextIO) -> int:
    from dexta_intelligence.workflows.goals import METRIC_LABELS  # noqa: PLC0415

    goals = store.get_goals()
    if not goals:
        out.write('No goals yet. Add one with: dexta goals add "..."\n')
        return 0
    for goal in goals:
        checkpoints = store.get_goal_checkpoints(goal.id) if goal.id else []
        latest = checkpoints[-1].note if checkpoints else "no checkpoints yet"
        out.write(
            f"#{goal.id} [{goal.status.value}] {goal.statement}\n"
            f"   {METRIC_LABELS[goal.metric]} · {latest}\n"
        )
    return 0


def _goals_tick(store: StoragePort, config: Config, out: TextIO, now: datetime) -> int:
    from dexta_intelligence.models import GoalStatus  # noqa: PLC0415
    from dexta_intelligence.workflows.goals import goal_due, tick_goal  # noqa: PLC0415

    active = store.get_goals(status=GoalStatus.ACTIVE)
    if not active:
        out.write("No active goals to advance.\n")
        return 0
    ctx = _ctx_for(config, store)
    model = model_for_role(config, "discovery")
    for goal in active:
        if goal.id is None:
            continue
        if not goal_due(goal, store.get_goal_checkpoints(goal.id), now=now):
            out.write(f"#{goal.id}: not due (every {goal.cadence_days}d)\n")
            continue
        result = tick_goal(goal, ctx, now=now, model=model)
        store.insert_goal_checkpoint(result.checkpoint)
        if result.achieved:
            store.set_goal_status(goal.id, GoalStatus.ACHIEVED)
        flag = " ✓ achieved" if result.achieved else ""
        out.write(f"#{goal.id}{flag}: {result.checkpoint.note}\n")
    return 0


def cmd_brief(
    *,
    config: Config,
    db_path: Path | None,
    out: TextIO,
    opener: StoreOpener = open_sqlite_store,
    model: Any = None,
) -> int:
    """Render a physician-visit brief from accumulated findings."""
    from dexta_intelligence.agents.brief import build_brief, render_markdown  # noqa: PLC0415
    from dexta_intelligence.models import FindingStatus  # noqa: PLC0415

    store = opener(config, db_path)
    try:
        findings = store.get_findings(status=FindingStatus.ACTIVE, limit=100)
        brief = build_brief(
            findings,
            store.coverage(),
            model=model if model is not None else model_for_role(config, "brief"),
            today=datetime.now(tz=UTC).date(),
        )
    finally:
        _maybe_close_store(store, opener)
    out.write(render_markdown(brief))
    out.write("\n")
    return 0


def cmd_wiki(
    *,
    config: Config,
    db_path: Path | None,
    out: TextIO,
    opener: StoreOpener = open_sqlite_store,
    new_findings: tuple[Finding, ...] = (),
) -> int:
    from dexta_intelligence.memory.synthesis import synthesize  # noqa: PLC0415
    from dexta_intelligence.models import FindingStatus  # noqa: PLC0415

    store = opener(config, db_path)
    try:
        synthesis = None
        if (model := model_for_role(config, "brief")) is not None:
            from dexta_intelligence.memory.synthesis import save  # noqa: PLC0415

            active = store.get_findings(status=FindingStatus.ACTIVE, limit=100)
            synthesis = synthesize(active, model)
            if not synthesis.is_empty():
                save(store, synthesis, today=datetime.now(tz=UTC).date())
        report = generate_wiki(
            store,
            root=config.wiki.path.expanduser(),
            today=datetime.now(tz=UTC).date(),
            new_findings=new_findings,
            git=config.wiki.git,
            synthesis=synthesis,
        )
    finally:
        _maybe_close_store(store, opener)
    committed = ", committed" if report.committed else ""
    out.write(f"wiki: {report.root} ({len(report.pages)} pages{committed})\n")
    return 0
