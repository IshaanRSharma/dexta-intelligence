"""View-model for the Prediction Reconciliation page.

Shapes the reconciliation agent's findings into expected-vs-actual cards: what
the looping algorithm forecast, where reality diverged, the max error, the
likely mismatch, and recurrence. For Tier A (logged predBG curves) it also
reconstructs the expected and actual traces for a sparkline. Read-only,
retrospective; never dosing advice.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from dexta_intelligence.server.render import sparkline_svg

if TYPE_CHECKING:
    from dexta_intelligence.models import Finding
    from dexta_intelligence.store.port import StoragePort

__all__ = ["reconciliation_page_view"]

_STEP_MIN = 5

_CONTRIBUTOR_LABELS = {
    "carb_underestimate": "Carbs underestimated",
    "sensitivity_shift": "Insulin sensitivity shift",
    "absorption_timing": "Absorption timing",
    "unclassified_mismatch": "Unclassified mismatch",
}
_CURVE_LABELS = {
    "iob": "insulin-only (IOB)",
    "cob": "carbs-as-announced (COB)",
    "uam": "unannounced meal (UAM)",
    "zt": "zero-temp (ZT)",
    "loop": "Loop forecast",
}


def _fmt_ts(value: str) -> str:
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return value
    return dt.strftime("%b %d, %H:%M")


def _expected_actual(
    store: StoragePort, cycle_ts_iso: str, curve: str, horizon_min: int
) -> tuple[list[float], list[float]]:
    """Reconstruct the forecast curve and the realized CGM over the episode."""
    try:
        cycle = datetime.fromisoformat(cycle_ts_iso)
    except ValueError:
        return [], []
    end = cycle + timedelta(minutes=horizon_min + _STEP_MIN)
    preds = [
        p
        for p in store.get_predictions(cycle - timedelta(minutes=1), end)
        if p.ts == cycle and p.curve_kind == curve
    ]
    if not preds:
        return [], []
    steps = horizon_min // _STEP_MIN + 1
    expected = [float(v) for v in preds[0].values_mg_dl[:steps]]
    gmap = {g.ts: g.mg_dl for g in store.get_glucose(cycle, end)}
    actual = [
        float(gmap[cycle + timedelta(minutes=_STEP_MIN * i)])
        for i in range(len(expected))
        if (cycle + timedelta(minutes=_STEP_MIN * i)) in gmap
    ]
    return expected, actual


def _card(store: StoragePort, finding: Finding) -> dict[str, Any]:
    ev = finding.evidence
    episodes = ev.get("episodes") or []
    representative: dict[str, Any] = max(
        episodes,
        key=lambda e: abs(float(e.get("signed_error_mg_dl", 0.0))),
        default={},
    )
    contributor = str(ev.get("contributor", "unclassified_mismatch"))
    curve = str(representative.get("best_curve", ""))
    horizon = int(representative.get("horizon_min", 0))
    signed = float(representative.get("signed_error_mg_dl", 0.0))
    tier = str(ev.get("tier", "A"))
    expected, actual = (
        _expected_actual(store, str(representative.get("cycle_ts", "")), curve, horizon)
        if tier == "A"
        else ([], [])
    )
    direction = "above" if signed >= 0 else "below"
    return {
        "headline": finding.headline,
        "contributor": _CONTRIBUTOR_LABELS.get(contributor, contributor),
        "curve": _CURVE_LABELS.get(curve, curve or "n/a"),
        "max_error": f"{signed:+.0f} mg/dL at {horizon} min",
        "divergence": _fmt_ts(str(representative.get("cycle_ts", ""))),
        "summary": (
            f"The {_CURVE_LABELS.get(curve, curve)} forecast expected a return toward range; "
            f"actual glucose ran {abs(signed):.0f} mg/dL {direction} the forecast by {horizon} min."
        ),
        "recurrence": int(ev.get("recurrence_count", 1)),
        "n_episodes": int(ev.get("n_episodes", len(episodes))),
        "tier": tier,
        "tier_label": "logged forecast curves" if tier == "A" else "computed expectations (weaker)",
        "p": finding.stats.p_perm,
        "expected_spark": sparkline_svg(expected) if len(expected) >= 2 else "",
        "actual_spark": sparkline_svg(actual) if len(actual) >= 2 else "",
        "limited": tier == "B",
    }


def reconciliation_page_view(
    store: StoragePort, findings: list[Finding], *, now: datetime
) -> dict[str, Any]:
    """Shape reconciliation ``findings`` (from the agent) into page cards."""
    cards = [_card(store, f) for f in findings]
    return {"cards": cards, "any": bool(cards)}
