"""View-model for the Evaluation / model-card page (PRD section 16).

The credibility artifact: how dexta is evaluated (E1-E6 + consensus), the live
safety invariant (zero dosing-advice across persisted outputs), the user's own
consensus glycemic metrics, the model in use, and the reproducible commands.

Page-load stays cheap: it describes the eval methodology and runs only a
substring safety scan and direct glycemic formulas — never a model call or a
permutation eval. The heavy suite is reproducible from the listed commands.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dexta_intelligence.config import Config
    from dexta_intelligence.store.port import StoragePort

__all__ = ["evals_page_view"]

_WIDE_START = datetime(1970, 1, 1)

#: The evaluation suite, described for the model card. Numbers are produced by
#: the reproducible commands, not on page load.
_METHODOLOGY: tuple[dict[str, str], ...] = (
    {
        "id": "E1",
        "name": "Numeric faithfulness",
        "measures": "Every number in generated prose traces to a tool result; "
        "the guard's catch rate.",
        "command": "python -m eval.runner e1",
    },
    {
        "id": "E2",
        "name": "Statistical power",
        "measures": "Plant a known effect at a controlled size; confirm the "
        "rigor-gated agent finds it.",
        "command": "python -m eval.runner e2",
    },
    {
        "id": "E3",
        "name": "Clinical accuracy",
        "measures": "oref0 forecast vs realized glucose: Clarke / Parkes error grid and MARD.",
        "command": "python -m eval.runner e3",
    },
    {
        "id": "E4",
        "name": "Null false-discovery rate",
        "measures": "Plant nothing; any surfaced pattern is a false discovery. "
        "Empirical FDR at alpha 0.10.",
        "command": "python -m eval.runner e4-null",
    },
    {
        "id": "E5",
        "name": "Robustness",
        "measures": "Deterministic agents under data corruption; findings must stay stable.",
        "command": "python -m eval.runner e5",
    },
    {
        "id": "E6",
        "name": "Agentic attribution, faithfulness, safety",
        "measures": "End-to-end: does the real agent name the planted cause, stay "
        "traceable, and never give dosing advice across a red-team set (target zero)?",
        "command": "python -m eval.runner e6",
    },
    {
        "id": "Ec",
        "name": "Consensus-formula agreement",
        "measures": "Rollup glycemic metrics exactly match the 2019 "
        "international-consensus definitions.",
        "command": "python -m eval.runner consensus",
    },
)


def _safety_scan(store: StoragePort) -> dict[str, Any]:
    """Live dosing-advice invariant: scan persisted outputs with the same
    detector the safety rail uses. Target is zero violations, always."""
    from dexta_intelligence.agents.brief import _ADVICE_RE  # noqa: PLC0415

    outputs: list[str] = []
    try:
        for f in store.get_findings(limit=1000):
            outputs.append(f"{f.headline} {f.body_md or ''}")
    except Exception:
        pass
    try:
        for run in store.get_investigation_runs(limit=200):
            if run.answer:
                outputs.append(run.answer)
    except Exception:
        pass
    violations = sum(1 for text in outputs if _ADVICE_RE.search(text))
    return {"scanned": len(outputs), "violations": violations, "clean": violations == 0}


def _consensus_metrics(store: StoragePort, now: datetime) -> dict[str, Any] | None:
    """The user's own glycemic metrics vs 2019 consensus targets, computed with
    the formulas E_consensus validates. None when there is too little glucose."""
    end = now + timedelta(days=1)
    glucose = store.get_glucose(_WIDE_START, end)
    vals = [float(g.mg_dl) for g in glucose]
    n = len(vals)
    if n < 2:
        return None
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    sd = var**0.5
    cv = 100.0 * sd / mean if mean else 0.0
    gmi = 3.31 + 0.02392 * mean
    tir = 100.0 * sum(1 for v in vals if 70.0 <= v <= 180.0) / n
    tbr = 100.0 * sum(1 for v in vals if v < 70.0) / n
    tar = 100.0 * sum(1 for v in vals if v > 180.0) / n
    return {
        "n": n,
        "tir_pct": round(tir, 1),
        "tbr_pct": round(tbr, 1),
        "tar_pct": round(tar, 1),
        "cv_pct": round(cv, 1),
        "gmi_pct": round(gmi, 1),
        "mean": round(mean, 1),
        "tir_on_target": tir >= 70.0,
        "tbr_on_target": tbr < 4.0,
        "cv_on_target": cv <= 36.0,
    }


def evals_page_view(store: StoragePort, config: Config, *, now: datetime) -> dict[str, Any]:
    """Assemble the model-card page: methodology, safety, glycemic metrics, model."""
    return {
        "methodology": list(_METHODOLOGY),
        "safety": _safety_scan(store),
        "glucose": _consensus_metrics(store, now),
        "model": {"provider": config.llm.provider, "model": config.llm.model},
    }
