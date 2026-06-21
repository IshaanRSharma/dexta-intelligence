"""E6 causal-attribution accuracy - does the real agent name the planted cause.

Each benchmark scenario builds a deterministic synthetic store with a known
planted cause and the keywords a correct attribution must contain. This metric
runs the real investigation agent on every scenario and scores a hit when all of
the scenario's expected keywords appear, case-insensitively, in the agent's
answer. The reported accuracy is the fraction of scenarios attributed correctly,
and the metric passes when that accuracy meets ``accuracy_target``.

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

__all__ = ["E6AttributionCell", "E6AttributionResult", "run_e6_attribution"]


@dataclass(frozen=True, slots=True)
class E6AttributionCell:
    """One scenario's attribution result: did the answer name the planted cause."""

    scenario: str
    hit: bool


@dataclass(frozen=True, slots=True)
class E6AttributionResult:
    """Outcome of one E6 attribution sweep over the benchmark."""

    cells: tuple[E6AttributionCell, ...]
    accuracy: float
    accuracy_target: float
    passed: bool


def run_e6_attribution(
    model: Any,
    *,
    scenarios: Sequence[Scenario] = BENCHMARK,
    runner: Runner = run_investigation,
    accuracy_target: float = 0.66,
) -> E6AttributionResult:
    """Run the agent on each scenario and score causal-attribution accuracy."""
    cells: list[E6AttributionCell] = []
    hits = 0
    for scenario in scenarios:
        outcome = runner(scenario.build(), scenario.question, model)
        answer = outcome.answer.lower()
        hit = all(keyword.lower() in answer for keyword in scenario.expected_keywords)
        if hit:
            hits += 1
        cells.append(E6AttributionCell(scenario=scenario.name, hit=hit))

    accuracy = hits / len(scenarios) if scenarios else 0.0
    return E6AttributionResult(
        cells=tuple(cells),
        accuracy=accuracy,
        accuracy_target=accuracy_target,
        passed=accuracy >= accuracy_target,
    )
