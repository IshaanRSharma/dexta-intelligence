"""Cold-start gating — every capability declares its minimum data.

The whole platform's promise is longitudinal intelligence, but most users
arrive with days of data, not months. The contract here: capabilities
degrade **explicitly** (skipped with a reason and a progress message),
never silently and never by fabricating confidence from thin data.

A single :class:`ColdStartReport` is computed per run from
:class:`~dexta_intelligence.models.CoverageStats` and injected into every
agent's context. The registry — not the agents — enforces the gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dexta_intelligence.agents.base import DataRequirement
    from dexta_intelligence.models import CoverageStats

__all__ = ["CAPABILITY_GATES", "CapabilitySet", "ColdStartReport", "Gate"]


@dataclass(frozen=True, slots=True)
class CapabilitySet:
    """Which data streams exist in the analysis window — the single mechanism
    deciding which tools a reasoning loop (or the MCP server) exposes.

    Distinct from :class:`ColdStartReport`: gates ask "is there *enough* data
    for an honest claim", capabilities ask "does this stream exist at all".
    A tool whose stream is absent is hidden, not error-prone."""

    has_insulin: bool
    has_meals: bool
    has_sleep: bool
    has_activity: bool
    has_predictions: bool = False

    def allows(self, need: str | None) -> bool:
        """Whether the named stream requirement is met (``None`` = no requirement)."""
        if need is None:
            return True
        return {
            "insulin": self.has_insulin,
            "meals": self.has_meals,
            "sleep": self.has_sleep,
            "activity": self.has_activity,
            "predictions": self.has_predictions,
        }.get(need, True)

    def missing_notes(self) -> list[str]:
        """Human/model-readable notes for absent streams, with the unlock path."""
        notes: list[str] = []
        if not self.has_insulin:
            notes.append(
                "insulin/pump data absent — treatment tools disabled "
                "(connect Tandem or Nightscout, then Sync now)"
            )
        if not self.has_meals:
            notes.append("carb entries absent — meal tools disabled (log carbs in Nightscout)")
        if not self.has_sleep:
            notes.append("sleep data absent (connect a wearable)")
        if not self.has_activity:
            notes.append("activity data absent (connect a wearable)")
        if not self.has_predictions:
            notes.append("no algorithm prediction curves (looping uploads unlock reconciliation)")
        return notes


@dataclass(frozen=True, slots=True)
class Gate:
    """Minimum data for one named capability, plus its unlock message."""

    capability: str
    min_span_days: float
    min_glucose_coverage_pct: float = 0.0
    needs_insulin: bool = False
    description: str = ""


#: The product's capability ladder. Surfaces render this as a progress
#: mechanic ("17 more days unlocks weekday patterns"), not as failure.
CAPABILITY_GATES: tuple[Gate, ...] = (
    Gate("metrics_snapshot", 7, 70, description="TIR / GMI / CV snapshot"),
    Gate("agp", 14, 70, description="Ambulatory glucose profile (consensus minimum)"),
    Gate("patterns_weekday_tod", 21, description="Weekday / time-of-day patterns"),
    Gate("what_changed", 60, description="30d-vs-prior comparison"),
    Gate("discovery", 60, description="Longitudinal discovery (early-signals mode below 90)"),
    Gate("trajectory", 90, description="Multi-month trajectory and long-term context"),
    Gate(
        "insulin_grounded",
        7,
        needs_insulin=True,
        description="Basal/meal/correction agents grounded in real dosing data",
    ),
)

#: Below this span the CLI prints the coverage report and refuses to analyze.
HARD_FLOOR_DAYS = 3.0

#: Fraction of days that must carry insulin events for insulin-grounded mode.
INSULIN_DAYS_PCT_MIN = 50.0


@dataclass(frozen=True, slots=True)
class ColdStartReport:
    """What this dataset can honestly support, with per-capability reasons."""

    coverage: CoverageStats
    unlocked: frozenset[str]
    pending: dict[str, str]
    """capability → human-readable unlock message."""

    @classmethod
    def from_coverage(cls, coverage: CoverageStats) -> ColdStartReport:
        unlocked: set[str] = set()
        pending: dict[str, str] = {}
        for gate in CAPABILITY_GATES:
            reasons: list[str] = []
            if coverage.span_days < gate.min_span_days:
                missing = gate.min_span_days - coverage.span_days
                reasons.append(f"{missing:.0f} more days of data")
            if coverage.glucose_coverage_pct < gate.min_glucose_coverage_pct:
                reasons.append(
                    f"sensor coverage {coverage.glucose_coverage_pct:.0f}% "
                    f"(needs {gate.min_glucose_coverage_pct:.0f}%)"
                )
            if gate.needs_insulin and coverage.days_with_insulin_pct < INSULIN_DAYS_PCT_MIN:
                reasons.append(
                    "insulin/pump data on "
                    f"{coverage.days_with_insulin_pct:.0f}% of days "
                    f"(needs {INSULIN_DAYS_PCT_MIN:.0f}%; connect Nightscout treatments)"
                )
            if reasons:
                pending[gate.capability] = f"{gate.description}: needs " + " and ".join(reasons)
            else:
                unlocked.add(gate.capability)
        return cls(coverage=coverage, unlocked=frozenset(unlocked), pending=pending)

    @property
    def below_hard_floor(self) -> bool:
        return self.coverage.span_days < HARD_FLOOR_DAYS

    def allows(self, capability: str) -> bool:
        return capability in self.unlocked

    def unmet(self, requirement: DataRequirement) -> list[str]:
        """Reasons an agent's :class:`DataRequirement` is not satisfied."""
        cov = self.coverage
        reasons: list[str] = []
        if cov.span_days < requirement.min_span_days:
            reasons.append(
                f"needs {requirement.min_span_days:.0f} days of data "
                f"(have {cov.span_days:.0f})"
            )
        if cov.glucose_coverage_pct < requirement.min_glucose_coverage_pct:
            reasons.append(
                f"needs {requirement.min_glucose_coverage_pct:.0f}% sensor coverage "
                f"(have {cov.glucose_coverage_pct:.0f}%)"
            )
        if requirement.needs_insulin and cov.days_with_insulin_pct < INSULIN_DAYS_PCT_MIN:
            reasons.append("needs insulin/pump data (connect Nightscout treatments)")
        if requirement.needs_sleep and cov.n_sleep == 0:
            reasons.append("needs sleep data (connect a wearable)")
        if requirement.needs_activity and cov.n_activity == 0:
            reasons.append("needs activity data (connect a wearable)")
        return reasons
