"""E6 end-to-end faithfulness - is every number in the real answer traceable.

E1 grades faithfulness on simulated prose. This metric closes that gap by running
the real investigation agent on every benchmark scenario and reading the agent's
own faithfulness verdict (``outcome.faithful``), which the faithfulness guard sets
when each claimed number is traceable to evidence. The reported faithful rate is
the fraction of scenarios answered faithfully, and the metric passes when that rate
meets ``faithful_target``.

The runner is injectable: in production it is the real :func:`run_investigation`,
but tests can substitute a scripted runner so the metric is exercised with no API
key and fully deterministic outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from eval.agent_eval import run_investigation
from eval.scenarios import BENCHMARK

if TYPE_CHECKING:
    from collections.abc import Sequence

    from eval.agent_eval import Runner
    from eval.scenarios import Scenario

__all__ = ["E6FaithfulnessCell", "E6FaithfulnessResult", "run_e6_faithfulness"]


@dataclass(frozen=True, slots=True)
class E6FaithfulnessCell:
    """One scenario's faithfulness result: did the agent answer faithfully."""

    scenario: str
    faithful: bool
    stopped_reason: str


@dataclass(frozen=True, slots=True)
class E6FaithfulnessResult:
    """Outcome of one E6 faithfulness sweep over the benchmark."""

    cells: tuple[E6FaithfulnessCell, ...]
    faithful_rate: float
    faithful_target: float
    passed: bool


def run_e6_faithfulness(
    model: Any,
    *,
    scenarios: Sequence[Scenario] = BENCHMARK,
    runner: Runner = run_investigation,
    faithful_target: float = 0.9,
) -> E6FaithfulnessResult:
    """Run the agent on each scenario and score end-to-end faithfulness."""
    cells: list[E6FaithfulnessCell] = []
    faithful_count = 0
    for scenario in scenarios:
        outcome = runner(scenario.build(), scenario.question, model)
        if outcome.faithful:
            faithful_count += 1
        cells.append(
            E6FaithfulnessCell(
                scenario=scenario.name,
                faithful=outcome.faithful,
                stopped_reason=outcome.stopped_reason,
            )
        )

    faithful_rate = faithful_count / len(scenarios) if scenarios else 0.0
    return E6FaithfulnessResult(
        cells=tuple(cells),
        faithful_rate=faithful_rate,
        faithful_target=faithful_target,
        passed=faithful_rate >= faithful_target,
    )
