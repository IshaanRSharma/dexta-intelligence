"""Shared harness for the end-to-end agentic evals (E6).

The E1-E5 metrics evaluate the deterministic substrate and the guard on
simulated prose. E6 evaluates the *agent itself*: run the real orchestrator on a
labeled scenario and grade its answer for causal attribution, faithfulness, and
safety. Everything here is the stable contract the individual E6 metrics build
against:

- :class:`InvestigationOutcome` is what one agent run yields.
- :func:`run_investigation` is the real runner (needs a model). Metrics take it
  as an injectable ``runner`` so tests can substitute a scripted outcome and run
  with no API key.
- :func:`looks_like_dosing_advice` is the shared safety detector, reusing the
  same regex the clinical brief enforces, so the bar never drifts between them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from dexta_intelligence.agents.brief import _ADVICE_RE
from dexta_intelligence.store import SQLiteStore

if TYPE_CHECKING:
    from dexta_intelligence.store.port import StoragePort
    from dexta_intelligence.testing.synthetic import EventsByType

DEFAULT_TARGET_LOW = 70
DEFAULT_TARGET_HIGH = 180


@dataclass(frozen=True, slots=True)
class InvestigationOutcome:
    """The graded surface of one agent run."""

    answer: str
    tools_used: tuple[str, ...]
    faithful: bool
    stopped_reason: str


class Runner(Protocol):
    """Callable that runs one investigation. Injectable so tests stay key-free."""

    def __call__(
        self, store: StoragePort, question: str, model: Any
    ) -> InvestigationOutcome: ...


def run_investigation(
    store: StoragePort,
    question: str,
    model: Any,
    *,
    target_low: int = DEFAULT_TARGET_LOW,
    target_high: int = DEFAULT_TARGET_HIGH,
) -> InvestigationOutcome:
    """Run the real orchestrator over ``store`` and return its graded outcome.

    The returned answer is the audited final answer (faithfulness guard +
    treatment gate already applied), so what we grade is exactly what a user
    would see.
    """
    from dexta_intelligence.agents.base import AgentContext  # noqa: PLC0415
    from dexta_intelligence.agents.orchestrator import OrchestratorAgent  # noqa: PLC0415
    from dexta_intelligence.cli._common import _analysis_window  # noqa: PLC0415
    from dexta_intelligence.coldstart import ColdStartReport  # noqa: PLC0415

    coverage = store.coverage()
    gates = ColdStartReport.from_coverage(coverage)
    end = coverage.last_ts.date() if coverage.last_ts is not None else None
    from dexta_intelligence.config import Config  # noqa: PLC0415

    ctx = AgentContext(
        store=store,
        window=_analysis_window(Config(), end),
        gates=gates,
        run_id="eval-e6",
    )
    agent = OrchestratorAgent(model=model, target_low=target_low, target_high=target_high)
    answer = agent.ask(ctx, question)
    return InvestigationOutcome(
        answer=answer.text,
        tools_used=tuple(answer.tools_used),
        faithful=answer.faithful,
        stopped_reason=answer.stopped_reason,
    )


def looks_like_dosing_advice(text: str) -> bool:
    """True when ``text`` reads as dosing/treatment advice (the safety violation).

    Reuses the clinical brief's treatment-advice regex so the safety bar is
    identical across the brief path and the chat path.
    """
    return bool(_ADVICE_RE.search(text))


def store_from_events(events: EventsByType) -> SQLiteStore:
    """Load all five synthetic event streams into a fresh in-memory store."""
    store = SQLiteStore(":memory:")
    store.migrate()
    store.insert_glucose(events["glucose"])
    store.insert_insulin(events["insulin"])
    store.insert_meals(events["meal"])
    store.insert_activity(events["activity"])
    store.insert_sleep(events["sleep"])
    return store
