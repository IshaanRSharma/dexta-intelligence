"""The research command: pre-registered single-subject (n-of-1) tests."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, TextIO

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.cli._common import (
    MEDICAL_DISCLAIMER,
    StoreOpener,
    _analysis_window,
    _maybe_close_store,
    open_sqlite_store,
)
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.workflows.nof1 import (
    COMPARISONS,
    METRICS,
    Hypothesis,
    parse_hypothesis,
    result_to_finding,
    run_nof1,
)

if TYPE_CHECKING:
    from pathlib import Path

    from dexta_intelligence.config import Config


def cmd_nof1(
    *,
    config: Config,
    db_path: Path | None,
    out: TextIO,
    statement: str | None = None,
    compare: str | None = None,
    metric: str = "mean_glucose",
    save: bool = False,
    seed: int = 1729,
    opener: StoreOpener = open_sqlite_store,
) -> int:
    """Pre-register one hypothesis, run the rigor battery on the subject's data.

    Accepts either a free-text ``statement`` (``dexta research "weekends run
    higher"``) or structured flags (``--compare weekend --metric mean_glucose``).
    Prints the pre-registered hypothesis, the rigor results (n, effect, p_perm,
    replication, power), and the plain-English verdict. Deterministic; no model.
    Persists a ``kind="nof1"`` finding only with ``--save``.
    """
    hypothesis = _resolve_hypothesis(statement, compare, metric, out)
    if hypothesis is None:
        return 2

    store = opener(config, db_path)
    try:
        coverage = store.coverage()
        gates = ColdStartReport.from_coverage(coverage)
        if gates.below_hard_floor:
            out.write(
                f"Only {coverage.span_days:.1f} days of data — too little to test.\n"
            )
            return 1
        end_date = coverage.last_ts.date() if coverage.last_ts is not None else None
        window = _analysis_window(config, end_date)
        ctx = AgentContext(
            store=store, window=window, gates=gates, run_id=str(uuid.uuid4()),
            timezone=config.analysis.timezone,
        )
        result = run_nof1(ctx, hypothesis, seed=seed)
        persisted_id = store.insert_finding(result_to_finding(result, ctx)) if save else None
    finally:
        _maybe_close_store(store, opener)

    out.write("Pre-registered hypothesis\n")
    out.write(f"  {result.hypothesis.registered_statement()}\n")
    out.write(f"  comparison: {result.hypothesis.comparison} · ")
    out.write(f"outcome: {result.hypothesis.outcome_label()}\n\n")

    out.write("Rigor (single-subject, observational)\n")
    out.write(f"  groups: {result.label_a} (n={result.n_a}) vs {result.label_b} (n={result.n_b})\n")
    if result.mean_a is not None and result.mean_b is not None:
        out.write(f"  means: {result.mean_a:.1f} vs {result.mean_b:.1f}\n")
    if result.effect_size is not None:
        d_bit = f" (Cohen's d={result.cohen_d:.2f})" if result.cohen_d is not None else ""
        out.write(f"  effect: {result.effect_size:+.1f}{d_bit}\n")
    if result.p_perm is not None:
        out.write(f"  permutation p: {result.p_perm:.4g} ({result.n_permutations} permutations)\n")
    if result.replicated is not None:
        out.write(f"  split-half replication: {'yes' if result.replicated else 'no'}\n")
    out.write(f"  powered: {'yes' if result.powered else 'no'}\n\n")

    verdict_label = {
        "supported": "SUPPORTED",
        "not_supported": "NOT SUPPORTED",
        "underpowered": "UNDERPOWERED — collect more data",
    }[result.verdict]
    out.write(f"Verdict: {verdict_label}\n")
    out.write(f"  {result.reason}\n")
    if persisted_id is not None:
        out.write(f"  persisted as finding #{persisted_id}\n")

    out.write(f"\n{result.disclaimer()}\n")
    out.write(f"\n{MEDICAL_DISCLAIMER}\n")
    return 0


def _resolve_hypothesis(
    statement: str | None, compare: str | None, metric: str, out: TextIO
) -> Hypothesis | None:
    """Build a Hypothesis from structured flags or free text; explain on failure."""
    if compare is not None:
        if compare not in COMPARISONS:
            out.write(
                f"Unknown comparison {compare!r}. Choose one of: "
                f"{', '.join(sorted(COMPARISONS))}\n"
            )
            return None
        if metric not in METRICS:
            out.write(f"Unknown metric {metric!r}. Choose one of: {', '.join(sorted(METRICS))}\n")
            return None
        return Hypothesis(comparison=compare, metric=metric, statement=statement or "")

    if statement:
        parsed = parse_hypothesis(statement)
        if parsed is not None:
            return parsed
        out.write(
            "Could not recognize a comparison in that statement. Use --compare "
            f"({', '.join(sorted(COMPARISONS))}) with --metric "
            f"({', '.join(sorted(METRICS))}), or include a keyword like "
            "'weekend', 'sleep', 'workout', or 'meal'.\n"
        )
        return None

    out.write(
        'Provide a hypothesis, e.g. dexta research "weekends run higher than weekdays" '
        "or dexta research --compare weekend --metric mean_glucose\n"
    )
    return None
