"""Investigations - composable, agent-callable lines of inquiry.

Distinct from ``workflows/`` (deterministic orchestration pipelines: sync,
deep_analysis, goals). An investigation is what the orchestrator and goal-seeker
compose toward a conclusion; ``spike`` is the one certified shortcut (kept
deterministic for eval, the offline path, and the treatment gate's guarantee).
New investigations live here, one module each.
"""

from dexta_intelligence.investigations.spike import (
    NO_TREATMENT_DISCLAIMER,
    OUTPUT_KEYS,
    SAFETY_LINE,
    SpikeEvidence,
    explain_spike,
    gather_spike_evidence,
)

__all__ = [
    "NO_TREATMENT_DISCLAIMER",
    "OUTPUT_KEYS",
    "SAFETY_LINE",
    "SpikeEvidence",
    "explain_spike",
    "gather_spike_evidence",
]
