"""The analyze command and its agent-harness helpers."""

from __future__ import annotations

import functools
import uuid
from typing import TYPE_CHECKING, TextIO

from dexta_intelligence.agents.base import AgentContext, AgentRegistry
from dexta_intelligence.cli._common import (
    StoreOpener,
    _analysis_window,
    _maybe_close_store,
    _print_coverage,
    discovery_model,
    open_sqlite_store,
)
from dexta_intelligence.coldstart import HARD_FLOOR_DAYS, ColdStartReport
from dexta_intelligence.config import Config
from dexta_intelligence.workflows import lenses
from dexta_intelligence.workflows.deep_analysis import persist_findings, run_deep_analysis

if TYPE_CHECKING:
    from pathlib import Path

    from dexta_intelligence.models import Finding


@functools.lru_cache(maxsize=1)
def get_registry() -> AgentRegistry:
    """Cached default registry (the ``analyze`` lens, no LLM) for back-compat."""
    try:
        registry, _ = lenses.build_registry("analyze", Config())
    except Exception:  # best-effort registration; analyze reports emptiness plainly
        return AgentRegistry()
    return registry


def _run_agents(
    registry: AgentRegistry,
    ctx: AgentContext,
    out: TextIO,
) -> list[Finding]:
    """Run the deep-analysis workflow and print skip/error lines."""
    def on_skip(name: str, reasons: list[str]) -> None:
        out.write(f"  skipped {name}: {'; '.join(reasons)}\n")

    report = run_deep_analysis(registry, ctx, persist=False, on_skip=on_skip)
    for agent_name, msg in report.errors:
        out.write(f"  ✗ {agent_name}: {msg}\n")
    return list(report.findings)


def _print_finding(out: TextIO, finding: Finding, *, persisted_id: int | None = None) -> None:
    out.write(f"\n  agent: {finding.agent}\n")
    out.write(f"  kind: {finding.kind}\n")
    out.write(f"  status: {finding.status.value}\n")
    out.write(f"  summary: {finding.headline}\n")
    stats = finding.stats
    stat_bits: list[str] = []
    if stats.n is not None:
        stat_bits.append(f"n={stats.n}")
    if stats.effect_size is not None:
        stat_bits.append(f"effect={stats.effect_size}")
    if stats.p_perm is not None:
        stat_bits.append(f"p={stats.p_perm}")
    if stat_bits:
        out.write(f"  evidence stats: {', '.join(stat_bits)}\n")
    if persisted_id is not None:
        out.write(f"  persisted id: {persisted_id}\n")
    if finding.skeptic_notes:
        out.write(f"  skeptic: {finding.skeptic_notes}\n")


def cmd_investigate(
    *,
    goal: str | None,
    config: Config,
    db_path: Path | None,
    out: TextIO,
    opener: StoreOpener = open_sqlite_store,
) -> int:
    """Deep investigation: the coordinator plans which investigations to run for
    a goal (or the whole record when ``goal`` is None), then banks findings."""
    from dexta_intelligence.agents.coordinator import CoordinatorAgent  # noqa: PLC0415

    store = opener(config, db_path)
    try:
        coverage = store.coverage()
        gates = ColdStartReport.from_coverage(coverage)
        _print_coverage(out, gates)
        if gates.below_hard_floor:
            out.write(
                f"\nNeed at least {HARD_FLOOR_DAYS:.0f} days of data before analysis "
                f"(have {coverage.span_days:.1f}).\n"
            )
            return 1
        end_date = coverage.last_ts.date() if coverage.last_ts is not None else None
        window = _analysis_window(config, end_date)
        ctx = AgentContext(store=store, window=window, gates=gates, run_id=str(uuid.uuid4()))

        target = f'goal: "{goal}"' if goal else "the whole record"
        out.write(f"\nInvestigating {target} (run {ctx.run_id})…\n")
        coordinator = CoordinatorAgent(model=discovery_model(config), config=config)
        findings = coordinator.investigate(ctx, goal=goal)

        if not findings:
            out.write("\nNo findings produced.\n")
            return 0
        out.write("\nFindings\n")
        persisted_ids = persist_findings(store, findings)
        for finding, finding_id in zip(findings, persisted_ids, strict=True):
            if finding.status.value == "rejected":
                out.write("  [skeptic rejected]\n")
            _print_finding(out, finding, persisted_id=finding_id)
    finally:
        _maybe_close_store(store, opener)
    return 0


def cmd_analyze(
    *,
    config: Config,
    db_path: Path | None,
    out: TextIO,
    opener: StoreOpener = open_sqlite_store,
    registry: AgentRegistry | None = None,
    lens: str = "analyze",
) -> int:
    window_days: int | None = None
    if registry is not None:
        active_registry = registry
    else:
        active_registry, window_days = lenses.build_registry(
            lens, config, model=discovery_model(config)
        )
    agents = list(active_registry)
    if not agents:
        out.write("No agents registered — nothing to analyze.\n")
        return 0

    store = opener(config, db_path)
    try:
        coverage = store.coverage()
        gates = ColdStartReport.from_coverage(coverage)
        _print_coverage(out, gates)

        if gates.below_hard_floor:
            out.write(
                f"\nNeed at least {HARD_FLOOR_DAYS:.0f} days of data before analysis "
                f"(have {coverage.span_days:.1f}).\n"
            )
            return 1

        end_date = coverage.last_ts.date() if coverage.last_ts is not None else None
        window = _analysis_window(config, end_date, window_days)
        ctx = AgentContext(
            store=store,
            window=window,
            gates=gates,
            run_id=str(uuid.uuid4()),
        )

        out.write(f"\nRunning {len(agents)} agent(s) (run {ctx.run_id})…\n")
        findings = _run_agents(active_registry, ctx, out)

        if not findings:
            out.write("\nNo findings produced.\n")
            return 0

        out.write("\nFindings\n")
        persisted_ids = persist_findings(store, findings)
        for finding, finding_id in zip(findings, persisted_ids, strict=True):
            if finding.status.value == "rejected":
                out.write("  [skeptic rejected]\n")
            _print_finding(out, finding, persisted_id=finding_id)
    finally:
        _maybe_close_store(store, opener)

    return 0
