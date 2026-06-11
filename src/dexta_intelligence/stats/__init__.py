"""Deterministic statistics: descriptive core + the rigor gate.

``core`` holds pure, stdlib-only primitives (correlation, tests, effect
sizes, bootstrap). ``rigor`` is the claim gate every discovery must pass:
permutation p-value -> BH-FDR -> split-half replication -> power check.
Import from the submodules directly; this package re-exports the
high-traffic entry points only.
"""

from dexta_intelligence.stats.core import (
    cliffs_delta,
    cohen_d,
    hedges_g,
    mann_whitney_u,
    pearson_r,
    spearman_rho,
    summarize,
    welch_t_test,
)
from dexta_intelligence.stats.rigor import (
    RigorVerdict,
    assess,
    benjamini_hochberg,
    permutation_pvalue,
    power_gate,
    split_half_replication,
)

__all__ = [
    "RigorVerdict",
    "assess",
    "benjamini_hochberg",
    "cliffs_delta",
    "cohen_d",
    "hedges_g",
    "mann_whitney_u",
    "pearson_r",
    "permutation_pvalue",
    "power_gate",
    "spearman_rho",
    "split_half_replication",
    "summarize",
    "welch_t_test",
]
