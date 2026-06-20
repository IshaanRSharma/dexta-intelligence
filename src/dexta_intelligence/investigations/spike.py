"""Spike Explanation workflow - the canonical "why did I spike?" surface.

This module splits **investigation** from **intelligence**:

- **Investigation (deterministic):** the trace is fixed code - orient → locate →
  zoom → carbs → boluses → basal → similar events. Tool choice is never delegated
  to the model so the audit trail stays reproducible.
- **Intelligence (LLM):** after evidence is gathered, an optional model *synthesizes*
  the finding headline from the evidence bundle + computed attribution - guard-audited,
  with the deterministic headline as fallback. ``confidence`` stays computed, never
  model-assessed. ``limitations`` lists every step that could not run.

Output schema is frozen (the contract tests assert it):
``{headline, evidence, confidence, limitations, trace, safety}``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from dexta_intelligence.agents.reason import ToolCall
from dexta_intelligence.agents.tools.toolkit import DiscoveryToolkit
from dexta_intelligence.agents.trace import render_trace
from dexta_intelligence.guard.faithfulness import audit

if TYPE_CHECKING:
    from dexta_intelligence.agents.base import AgentContext

logger = logging.getLogger(__name__)

__all__ = [
    "INSUFFICIENT_SENTENCE",
    "NO_TREATMENT_DISCLAIMER",
    "OUTPUT_KEYS",
    "SAFETY_LINE",
    "SpikeEvidence",
    "explain_spike",
    "gather_spike_evidence",
]

#: The frozen output contract.
OUTPUT_KEYS = ("headline", "evidence", "confidence", "limitations", "trace", "safety")

SAFETY_LINE = "Discussion support only. No dosing recommendation."
NO_TREATMENT_DISCLAIMER = "Insulin/carb data unavailable. This is glucose-shape inference only."
INSUFFICIENT_SENTENCE = (
    "I can describe the glucose pattern, but I cannot make a strong cause "
    "hypothesis because treatment context was missing or not inspected."
)

#: Bolus this many minutes after the carb entry counts as the late-bolus signal.
LATE_BOLUS_MIN = 15
#: Hours considered "overnight" for the basal-drift hypothesis.
_OVERNIGHT_HOURS = range(0, 8)
#: Recurrence bar for upgrading confidence to "high" (with a consistent
#: late-bolus delay separation between spiking and non-spiking events).
_HIGH_N_SIMILAR = 15
_HIGH_SPIKE_RATIO = 0.7
_HIGH_DELAY_SEPARATION_MIN = 10.0


@dataclass(frozen=True, slots=True)
class SpikeEvidence:
    """Deterministically gathered evidence for one spike - the bundle the
    orchestrator (or :func:`explain_spike`) reasons over. No LLM in here.

    ``headline`` is the computed working hypothesis (allowed-vocabulary
    attribution); ``pool`` holds the guard-auditable numbers from every inner
    tool so a downstream synthesis is still traceable."""

    headline: str
    contributor_found: bool
    evidence: list[str]
    confidence: str
    limitations: list[str]
    steps: list[ToolCall]
    pool: dict[str, Any]


def gather_spike_evidence(
    ctx: AgentContext,
    when: str,
    *,
    threshold: float = 200.0,
    target_low: int = 70,
    target_high: int = 180,
) -> SpikeEvidence:
    """Run the fixed investigation (orient → locate → zoom → carbs → boluses →
    basal → similar) and compute the deterministic attribution + confidence.

    Pure: no model, never raises on bad input. This is the *instrument*; the
    LLM (orchestrator or :func:`explain_spike`) reasons over what it returns."""
    toolkit = DiscoveryToolkit(ctx, target_low=target_low, target_high=target_high)
    caps = toolkit.capabilities()
    steps: list[ToolCall] = []
    pool: dict[str, Any] = {}
    limitations: list[str] = []
    evidence: list[str] = []

    def step(name: str, args: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        ok = not result.get("error")
        steps.append(ToolCall(name=name, args=args, ok=ok, result=result))
        if ok:
            _collect(pool, name, len(steps), result)
        return result

    step("list_segments", {}, toolkit.list_segments())

    ts, locate_error = _resolve_when(toolkit, when, threshold, step)
    if ts is None:
        limitations.append(locate_error or "could not locate a spike")
        return SpikeEvidence(
            headline=locate_error or INSUFFICIENT_SENTENCE,
            contributor_found=False,
            evidence=evidence,
            confidence="low",
            limitations=limitations,
            steps=steps,
            pool=pool,
        )

    zoom = step("zoom_event", {"timestamp": ts.isoformat(), "pad_hours": 6},
                toolkit.zoom_event(ts.isoformat(), pad_hours=6))
    peak = zoom.get("peak")
    if peak is not None:
        evidence.append(f"Peak: {peak:g} mg/dL")

    carbs, boluses, basal, bolus_delay = _inspect_treatments(
        toolkit, caps, ts, step, evidence, limitations
    )

    anchor = _nearest_meal_ts(carbs) or ts
    similar = step(
        "find_similar_events",
        {"timestamp": anchor.isoformat(), "threshold": threshold},
        toolkit.find_similar_events(anchor.isoformat(), threshold),
    )
    n_similar = similar.get("n_similar") or 0
    n_spiking = similar.get("n_spiking") or 0
    if n_similar:
        evidence.append(f"Similar pattern: {n_spiking}/{n_similar} same-time events spiked")
    else:
        limitations.append("not enough history for a similar-event comparison")

    if not caps.has_predictions:
        limitations.append("no algorithm prediction curves available (reconciliation skipped)")

    headline, contributor_found = _attribute(
        caps_insulin=caps.has_insulin,
        carbs=carbs,
        boluses=boluses,
        basal=basal,
        bolus_delay=bolus_delay,
        spike_ts=ts.astimezone(toolkit.tzinfo),  # local time for overnight check + display
        peak=peak,
    )
    confidence = _confidence(
        has_insulin=caps.has_insulin,
        contributor_found=contributor_found,
        similar=similar,
    )
    return SpikeEvidence(
        headline=headline,
        contributor_found=contributor_found,
        evidence=evidence,
        confidence=confidence,
        limitations=limitations,
        steps=steps,
        pool=pool,
    )


def explain_spike(
    ctx: AgentContext,
    when: str,
    *,
    model: Any = None,
    threshold: float = 200.0,
    target_low: int = 70,
    target_high: int = 180,
) -> dict[str, Any]:
    """Explain the spike at/around ``when`` and return the frozen output dict.

    Standalone path (CLI / no-key / eval): deterministic gather + an optional
    guarded LLM synthesis of the headline. When the orchestrator drives instead,
    it calls :func:`gather_spike_evidence` as a tool and reasons over the bundle
    itself. Never raises on bad input."""
    ev = gather_spike_evidence(
        ctx, when, threshold=threshold, target_low=target_low, target_high=target_high
    )
    headline = ev.headline
    if model is not None and ev.contributor_found:
        headline = _synthesize_headline(
            model,
            deterministic=ev.headline,
            evidence=ev.evidence,
            limitations=ev.limitations,
            pool=ev.pool,
        )
    return _finish(
        headline=headline,
        evidence=ev.evidence,
        confidence=ev.confidence,
        limitations=ev.limitations,
        steps=ev.steps,
    )


def _inspect_treatments(
    toolkit: DiscoveryToolkit,
    caps: Any,
    ts: datetime,
    step: Any,
    evidence: list[str],
    limitations: list[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], float | None]:
    """Steps 4-6: carbs, boluses+IOB, basal - each skipped (and listed
    as a limitation) when its stream is absent."""
    carbs: dict[str, Any] = {}
    boluses: dict[str, Any] = {}
    basal: dict[str, Any] = {}
    bolus_delay: float | None = None
    if caps.has_meals:
        carbs = step("get_carb_entries", {}, toolkit.get_carb_entries())
    else:
        limitations.append("no carb entries logged anywhere in the record")
    if caps.has_insulin:
        boluses = step("get_boluses", {}, toolkit.get_boluses())
        gap = toolkit._treatment_gap_note()
        if gap:
            limitations.append(gap)
        bolus_delay = _delay_of_nearest_bolus(boluses, ts)
        if bolus_delay is not None:
            evidence.append(f"Bolus: {bolus_delay:g} min after meal entry")
        iob = step("get_iob", {"timestamp": ts.isoformat()}, toolkit.get_iob(ts.isoformat()))
        if iob.get("iob_units") is not None:
            evidence.append(f"IOB at event: {iob['iob_units']:g} U (computed, tier B)")
        basal = step("get_basal_timeline", {}, toolkit.get_basal_timeline())
        evidence.append(
            "Basal stable in the window"
            if basal.get("basal_stable")
            else "Temp-basal/suspend activity in the window"
        )
    else:
        limitations.append(NO_TREATMENT_DISCLAIMER)
        limitations.append("Run Sync now in Settings if Tandem/Nightscout is connected.")
    if caps.has_meals:
        cob = step("get_cob", {"timestamp": ts.isoformat()}, toolkit.get_cob(ts.isoformat()))
        if cob.get("cob_g"):
            evidence.append(f"COB at event: {cob['cob_g']:g} g (computed, tier B)")
    return carbs, boluses, basal, bolus_delay


# ── deterministic attribution ─────────────────────────────────────────────────


def _attribute(  # noqa: PLR0911 - one return per allowed contributor pattern
    *,
    caps_insulin: bool,
    carbs: dict[str, Any],
    boluses: dict[str, Any],
    basal: dict[str, Any],
    bolus_delay: float | None,
    spike_ts: datetime,
    peak: Any,
) -> tuple[str, bool]:
    """Pick the most consistent contributor from inspected treatment context.

    Returns ``(headline, contributor_found)``. The vocabulary is the
    allowed list - patterns and hypotheses, never directives."""
    if not caps_insulin:
        desc = f"Glucose rose to {peak:g} mg/dL" if peak is not None else "Glucose rose"
        return (
            f"{desc} around {spike_ts.strftime('%H:%M')}. {NO_TREATMENT_DISCLAIMER}",
            False,
        )
    basal_stable = bool(basal.get("basal_stable"))
    comparator = " than basal drift" if basal_stable else ""
    n_carbs = carbs.get("n_entries", 0)
    n_boluses = boluses.get("n_boluses", 0)
    if boluses.get("note", "").startswith("pump/insulin data in dexta ends"):
        return (
            "Glucose rose in this window, but pump/insulin data in dexta does not cover it - "
            "upload recent pump history to Tandem Source and Sync now before inferring "
            "bolus/meal causes.",
            False,
        )
    if not basal_stable:
        return (
            "The window includes temp-basal/suspend activity - the pattern is "
            "consistent with algorithm-intervention context rather than a "
            "single meal effect.",
            True,
        )
    if n_carbs and bolus_delay is not None and bolus_delay >= LATE_BOLUS_MIN:
        return (
            "The pattern is more consistent with late/insufficient meal "
            f"insulin context{comparator}.",
            True,
        )
    if n_carbs and not n_boluses:
        return (
            f"The pattern is more consistent with a meal without bolus context{comparator} "
            "(possible missed bolus).",
            True,
        )
    if not n_carbs and n_boluses:
        return (
            "A bolus was logged with no carb entry nearby - the pattern is "
            f"consistent with unlogged meal context{comparator}.",
            True,
        )
    if not n_carbs and not n_boluses and spike_ts.hour in _OVERNIGHT_HOURS:
        return (
            "Overnight rise with no meal or bolus context - the pattern is "
            "consistent with a basal-drift hypothesis.",
            True,
        )
    return (INSUFFICIENT_SENTENCE, False)


def _confidence(
    *, has_insulin: bool, contributor_found: bool, similar: dict[str, Any]
) -> str:
    """Computed, never model-assessed: data class + recurrence decide."""
    if not has_insulin or not contributor_found:
        return "low"
    n_similar = similar.get("n_similar") or 0
    n_spiking = similar.get("n_spiking") or 0
    delay_a = similar.get("mean_bolus_delay_spiking_min")
    delay_b = similar.get("mean_bolus_delay_other_min")
    separated = (
        isinstance(delay_a, (int, float))
        and isinstance(delay_b, (int, float))
        and delay_a - delay_b >= _HIGH_DELAY_SEPARATION_MIN
    )
    if (
        n_similar >= _HIGH_N_SIMILAR
        and n_spiking / n_similar >= _HIGH_SPIKE_RATIO
        and separated
    ):
        return "high"
    return "moderate"


# ── helpers ───────────────────────────────────────────────────────────────────


def _resolve_when(
    toolkit: DiscoveryToolkit,
    when: str,
    threshold: float,
    step: Any,
) -> tuple[datetime | None, str | None]:
    """ISO datetime → that moment; ISO date → the day's largest excursion."""
    raw = str(when).strip()
    try:
        if len(raw) > 10:
            ts = datetime.fromisoformat(raw)
            return (ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts, None)
        day = datetime.fromisoformat(raw).date()
    except ValueError:
        return None, f"could not parse {raw!r} as an ISO date or datetime"
    step("set_window", {"start": day.isoformat(), "end": day.isoformat()},
         toolkit.set_window(day.isoformat(), day.isoformat()))
    spikes = step(
        "find_spikes", {"threshold": threshold}, toolkit.find_spikes(threshold)
    )
    rows = spikes.get("spikes") or []
    if not rows:
        return None, (
            f"no excursion ≥ {threshold:g} mg/dL found on {day.isoformat()} - "
            "nothing to explain at that threshold"
        )
    return datetime.fromisoformat(str(rows[0]["ts"])), None


def _delay_of_nearest_bolus(boluses: dict[str, Any], spike_ts: datetime) -> float | None:
    """minutes_after_carb_entry of the bolus nearest the spike (±3h)."""
    best: tuple[timedelta, float] | None = None
    for row in boluses.get("boluses") or []:
        delay = row.get("minutes_after_carb_entry")
        if not isinstance(delay, (int, float)):
            continue
        try:
            ts = datetime.fromisoformat(str(row.get("ts")))
        except (TypeError, ValueError):
            continue
        gap = abs(spike_ts - ts)
        if gap <= timedelta(hours=3) and (best is None or gap < best[0]):
            best = (gap, float(delay))
    return best[1] if best else None


def _nearest_meal_ts(carbs: dict[str, Any]) -> datetime | None:
    entries = carbs.get("entries") or []
    if not entries:
        return None
    try:
        return datetime.fromisoformat(str(entries[0]["ts"]))
    except (KeyError, TypeError, ValueError):
        return None


def _collect(pool: dict[str, Any], name: str, idx: int, result: dict[str, Any]) -> None:
    """Flatten a tool result's numbers (one list level deep) into the guard pool."""
    numbers: dict[str, Any] = {
        k: v for k, v in result.items() if isinstance(v, (int, float))
    }
    for k, v in result.items():
        if isinstance(v, list):
            for i, row in enumerate(v[:50]):
                if isinstance(row, dict):
                    nums = {kk: vv for kk, vv in row.items() if isinstance(vv, (int, float))}
                    if nums:
                        numbers[f"{k}_{i}"] = nums
    if numbers:
        pool[f"{name}_{idx}"] = numbers


def _synthesize_headline(
    model: Any,
    *,
    deterministic: str,
    evidence: list[str],
    limitations: list[str],
    pool: dict[str, Any],
) -> str:
    """One guarded LLM call to synthesize the finding from gathered evidence.

    The deterministic attribution is a *working hypothesis* the model may refine
    but must not contradict without evidence. Falls back on guard failure or errors.
    """
    lim_text = "; ".join(limitations) if limitations else "none"
    pool_text = json.dumps(pool, indent=2, default=str)
    if len(pool_text) > 6000:
        pool_text = pool_text[:6000] + "\n…"
    prompt = (
        "You are explaining one glucose spike to a Type-1 patient.\n"
        "Write 1-2 observation-only sentences as the finding headline.\n\n"
        "WORKING HYPOTHESIS (from computed rules - refine using evidence, "
        "do not invent contradictions):\n"
        f"{deterministic}\n\n"
        "EVIDENCE LINES (every number you cite MUST appear here or in COMPUTED DATA):\n"
        + ("\n".join(f"- {line}" for line in evidence) if evidence else "- (none)")
        + f"\n\nLIMITATIONS: {lim_text}\n\n"
        "COMPUTED DATA (JSON - numbers here are fair game for the guard):\n"
        f"{pool_text}\n\n"
        "RULES:\n"
        "- Discussion support only - no dosing amounts, ratios, ISF changes, or "
        '"bolus X minutes earlier" as a directive.\n'
        "- Weave peak, meal/bolus timing, basal stability, and recurrence when present.\n"
        "- If evidence is thin, say so plainly.\n"
        "- Prefer meal-insulin vs basal-drift framing when the working hypothesis does."
    )
    try:
        response = model.invoke([{"role": "user", "content": prompt}])
        text = _text_of(response)
    except Exception:
        logger.warning("explain_spike: synthesis failed; keeping deterministic")
        return deterministic
    if not text:
        return deterministic
    report = audit(text, pool)
    if not report.ok:
        logger.warning(
            "explain_spike: synthesized headline failed the guard (%d violations); "
            "keeping deterministic",
            len(report.violations),
        )
        return deterministic
    return text


def _text_of(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
    return str(content).strip()


def _finish(
    *,
    headline: str,
    evidence: list[str],
    confidence: str,
    limitations: list[str],
    steps: list[ToolCall],
) -> dict[str, Any]:
    return {
        "headline": headline,
        "evidence": evidence,
        "confidence": confidence,
        "limitations": limitations,
        "trace": [line.text for line in render_trace(steps)],
        "safety": SAFETY_LINE,
    }
