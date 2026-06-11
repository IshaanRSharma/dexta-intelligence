"""Agent contract and registry — the blackboard architecture.

Agents never invoke each other. Each one reads the store + memory through
:class:`AgentContext`, does its work, and returns :class:`Finding` records.
The registry fans out all registered agents with per-agent exception
isolation (one crashing agent never takes down an analysis run) — a direct
port of the battle-tested detector registry from the donor codebase.

Two hard rules, enforced structurally:

1. **Declared data requirements.** An agent states its minimum data up
   front; the registry refuses to run it under-data. Agents cannot fabricate
   confidence from thin data because they never see thin data.
2. **Deterministic agents take no model.** Only agents constructed with an
   LLM role may produce prose, and that prose must clear the faithfulness
   guard before the finding is accepted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from datetime import date

    from dexta_intelligence.coldstart import ColdStartReport
    from dexta_intelligence.models import Finding
    from dexta_intelligence.store.port import StoragePort

logger = logging.getLogger(__name__)

__all__ = ["AgentContext", "AgentRegistry", "DataRequirement", "DextaAgent"]


@dataclass(frozen=True, slots=True)
class DataRequirement:
    """Minimum data an agent needs to produce honest output."""

    min_span_days: float = 0.0
    min_glucose_coverage_pct: float = 0.0
    needs_insulin: bool = False
    needs_sleep: bool = False
    needs_activity: bool = False

    def unmet_reasons(self, report: ColdStartReport) -> list[str]:
        """Human-readable reasons this requirement is not satisfied (empty = OK)."""
        return report.unmet(self)


@dataclass(frozen=True, slots=True)
class AgentContext:
    """Everything an agent may see. Read-only by convention and by design."""

    store: StoragePort
    window: tuple[date, date]
    gates: ColdStartReport
    run_id: str


@runtime_checkable
class DextaAgent(Protocol):
    """The plugin surface. Community agents implement exactly this."""

    name: str
    requires: DataRequirement

    def run(self, ctx: AgentContext) -> list[Finding]: ...


@dataclass
class AgentRegistry:
    """Ordered agent collection with gated, isolated fan-out."""

    _agents: dict[str, DextaAgent] = field(default_factory=dict)

    def register(self, agent: DextaAgent) -> DextaAgent:
        """Register an agent (usable as a decorator on instances via ``__call__``)."""
        if agent.name in self._agents:
            msg = f"duplicate agent name: {agent.name!r}"
            raise ValueError(msg)
        self._agents[agent.name] = agent
        return agent

    def __iter__(self) -> Iterator[DextaAgent]:
        return iter(self._agents.values())

    def run_all(
        self,
        ctx: AgentContext,
        *,
        on_skip: Callable[[str, list[str]], None] | None = None,
    ) -> list[Finding]:
        """Run every registered agent that meets its data requirement.

        Skipped agents are reported through ``on_skip`` with their unmet
        reasons — cold start is explicit, never silent. A raising agent is
        logged and isolated; the run continues.
        """
        findings: list[Finding] = []
        for agent in self._agents.values():
            reasons = agent.requires.unmet_reasons(ctx.gates)
            if reasons:
                logger.info("skipping agent %s: %s", agent.name, "; ".join(reasons))
                if on_skip is not None:
                    on_skip(agent.name, reasons)
                continue
            try:
                findings.extend(agent.run(ctx))
            except Exception:
                logger.exception("agent %s failed; continuing run", agent.name)
        return findings
