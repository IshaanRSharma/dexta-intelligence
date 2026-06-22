"""E7 reasoning-process quality - grade the investigation path, not just the answer.

E6 scores the final answer (attribution, faithfulness, safety). E7 scores *how*
the agent got there, from the same run's tool trace:

- **cross-modal coverage**: did it consult the evidence classes the planted cause
  needs (``Scenario.required_evidence``)?
- **probe efficiency**: how few tool calls relative to a sane budget.
- **gap handling**: on a scenario whose cause is unlogged, did it flag the gap
  instead of inventing a value?
- **path soundness**: faithful, correctly attributed, and built on the required
  evidence (for gap scenarios: faithful and the gap was flagged).

These are the baselines every later intelligence-flow phase is measured against.
The runner is injectable so the metric runs key-free against scripted outcomes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from eval.agent_eval import attribution_hit, run_investigation
from eval.scenarios import REASONING_BENCHMARK

if TYPE_CHECKING:
    from collections.abc import Sequence

    from eval.agent_eval import InvestigationOutcome, Runner
    from eval.scenarios import Scenario

__all__ = [
    "E7ReasoningCell",
    "E7ReasoningResult",
    "gap_flagged",
    "modality_coverage",
    "path_sound",
    "probe_efficiency",
    "run_e7_reasoning",
]

#: Tool budget a focused investigation should attribute a cause within. Efficiency
#: is full at or under this and decays inversely above it.
IDEAL_PROBES = 6

#: Phrases that surface a missing-context gap rather than fabricating the cause.
#: Kept to clear missing-context framings so ordinary attribution prose ("no
#: carb-ratio issue") does not read as a gap flag.
_GAP_RE = re.compile(
    r"\b(no (?:logged|recorded) \w+|without (?:your|a) (?:meal|carb|note|log)|"
    r"missing (?:meal|carb|context|log|data)|cannot (?:separate|attribute|tell)|"
    r"no (?:meal|carb|note)s? (?:were |was )?(?:logged|recorded|on record)|"
    r"unlogged|please log)\b",
    re.IGNORECASE,
)


def modality_coverage(outcome: InvestigationOutcome, scenario: Scenario) -> float:
    """Fraction of required evidence classes the run probed (1.0 if none required)."""
    required = scenario.required_evidence
    if not required:
        return 1.0
    probed = set(outcome.tools_used)
    hit = sum(1 for group in required if probed & group)
    return hit / len(required)


def probe_efficiency(outcome: InvestigationOutcome, *, ideal: int = IDEAL_PROBES) -> float:
    """1.0 at or under ``ideal`` tool calls, decaying as ``ideal / count`` above it."""
    count = len(outcome.tools_used)
    if count <= ideal:
        return 1.0
    return ideal / count


def gap_flagged(outcome: InvestigationOutcome) -> bool:
    """True when the answer surfaces a missing-context gap."""
    return bool(_GAP_RE.search(outcome.answer))


def path_sound(outcome: InvestigationOutcome, scenario: Scenario) -> bool:
    """Whether the run reached its conclusion soundly.

    Gap scenarios are sound when the answer is faithful, flags the gap, and does
    *not* name a cause it cannot support (no fabricated attribution). Other
    scenarios require faithfulness, full evidence coverage, and a correct
    attribution.
    """
    if not outcome.faithful:
        return False
    hit = attribution_hit(outcome.answer, scenario.expected_keywords)
    if scenario.is_gap:
        return gap_flagged(outcome) and not hit
    return modality_coverage(outcome, scenario) == 1.0 and hit


@dataclass(frozen=True, slots=True)
class E7ReasoningCell:
    """One scenario's reasoning-process scores."""

    scenario: str
    coverage: float
    probes: int
    efficiency: float
    attribution_hit: bool
    gap_flagged: bool
    is_gap: bool
    sound: bool


@dataclass(frozen=True, slots=True)
class E7ReasoningResult:
    """Outcome of one E7 reasoning-process sweep over the benchmark."""

    cells: tuple[E7ReasoningCell, ...]
    mean_coverage: float
    mean_efficiency: float
    soundness_rate: float
    #: Reported, not gated: gap-handling already feeds ``soundness_rate`` (a gap
    #: scenario is only sound when its gap is flagged). Surfaced for visibility.
    gap_handling_rate: float
    coverage_target: float
    soundness_target: float
    passed: bool


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_e7_reasoning(
    model: Any,
    *,
    scenarios: Sequence[Scenario] = REASONING_BENCHMARK,
    runner: Runner = run_investigation,
    coverage_target: float = 0.75,
    soundness_target: float = 0.5,
) -> E7ReasoningResult:
    """Run the agent on each scenario and score its reasoning process."""
    cells: list[E7ReasoningCell] = []
    for scenario in scenarios:
        outcome = runner(scenario.build(), scenario.question, model)
        cells.append(
            E7ReasoningCell(
                scenario=scenario.name,
                coverage=modality_coverage(outcome, scenario),
                probes=len(outcome.tools_used),
                efficiency=probe_efficiency(outcome),
                attribution_hit=attribution_hit(outcome.answer, scenario.expected_keywords),
                gap_flagged=gap_flagged(outcome),
                is_gap=scenario.is_gap,
                sound=path_sound(outcome, scenario),
            )
        )

    gap_cells = [c for c in cells if c.is_gap]
    mean_coverage = _mean([c.coverage for c in cells])
    soundness_rate = _mean([float(c.sound) for c in cells])
    return E7ReasoningResult(
        cells=tuple(cells),
        mean_coverage=mean_coverage,
        mean_efficiency=_mean([c.efficiency for c in cells]),
        soundness_rate=soundness_rate,
        gap_handling_rate=_mean([float(c.gap_flagged) for c in gap_cells]),
        coverage_target=coverage_target,
        soundness_target=soundness_target,
        passed=mean_coverage >= coverage_target and soundness_rate >= soundness_target,
    )
