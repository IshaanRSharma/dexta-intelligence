"""Active Context Acquisition - dexta asks for what it cannot observe.

Determinism detects a gap: a glucose spike with no meal and no note logged
nearby. dexta does not guess what caused it. It asks the user to log what was
happening so the deterministic engine can later check for a recurring pattern.
The question is observation-only and never fabricates the missing value; the
same dosing-advice gate that guards the clinical brief refuses any request that
reads as treatment guidance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from dexta_intelligence.agents.brief import _ADVICE_RE
from dexta_intelligence.agents.reason import ToolSpec
from dexta_intelligence.agents.tools.toolkit import DiscoveryToolkit
from dexta_intelligence.models import ContextRequest

if TYPE_CHECKING:
    from dexta_intelligence.agents.base import AgentContext

__all__ = ["ContextAcquisitionAgent", "context_request_at", "request_context_tool"]

#: Proximity for the mid-loop check: a meal/note within this of the moment counts
#: as context the agent should probe rather than ask about.
_MID_LOOP_PROXIMITY_MIN = 90

_REFINE_SYSTEM = (
    "You reword a short, factual observation about a glucose spike into more "
    "natural language. Keep it observation-only. Do not add any numbers. Do not "
    "give any dosing, insulin, or treatment advice. Return only the reworded "
    "sentence."
)


@dataclass
class ContextAcquisitionAgent:
    """Turn unexplained spikes into logging requests, never fabricated causes."""

    model: Any = None
    threshold: float = 200.0
    max_requests: int = 5
    proximity_min: int = 90

    def build(self, ctx: AgentContext) -> list[ContextRequest]:
        """Deterministically find unexplained spikes and ask the user to log them."""
        tk = DiscoveryToolkit(ctx)
        spikes = tk.find_spikes(self.threshold, top_n=self.max_requests * 3).get("spikes", [])

        win_start = datetime.combine(ctx.window[0], time.min, tzinfo=UTC)
        win_end = datetime.combine(ctx.window[1], time.max, tzinfo=UTC)
        meals = ctx.store.get_meals(win_start, win_end)
        try:
            manual = ctx.store.get_manual_events(win_start, win_end)
        except (AttributeError, NotImplementedError):  # minimal/partial stores
            manual = []

        prox = timedelta(minutes=self.proximity_min)
        requests: list[ContextRequest] = []
        for spike in spikes:
            spike_ts = datetime.fromisoformat(spike["ts"])
            meal_near = any(abs(m.ts - spike_ts) <= prox for m in meals)
            note_near = any(abs(e.event_ts - spike_ts) <= prox for e in manual)
            if meal_near or note_near:
                continue
            requests.append(self._request(spike, spike_ts, tk))
            if len(requests) >= self.max_requests:
                break

        gated = [r for r in requests if not _ADVICE_RE.search(r.question)]
        if self.model is None:
            return gated
        return [self._refine(r) for r in gated]

    def _request(
        self, spike: dict[str, Any], spike_ts: datetime, tk: DiscoveryToolkit
    ) -> ContextRequest:
        peak = float(spike["peak_mg_dl"])
        local = spike_ts.astimezone(tk.tzinfo).strftime("%b %d %H:%M")
        question = (
            f"Around {local}, glucose rose to {peak:.0f} mg/dL with no meal or note "
            "logged nearby. If you log what was happening then (a meal, illness, "
            "stress, or activity), dexta can check whether it is part of a recurring "
            "pattern."
        )
        return ContextRequest(
            kind="unexplained_spike",
            event_ts=spike_ts,
            question=question,
            suggested_event_type="meal",
            evidence={"peak_mg_dl": round(peak), "ts": spike["ts"]},
        )

    def _refine(self, req: ContextRequest) -> ContextRequest:
        """Reword the question via the model; keep the template on any failure.

        The model may not add numbers or treatment advice; if it does (or errors,
        or returns nothing), the deterministic template stands.
        """
        try:
            response = self.model.invoke(
                [
                    {"role": "system", "content": _REFINE_SYSTEM},
                    {"role": "user", "content": req.question},
                ]
            )
            text = getattr(response, "content", response)
        except Exception:
            return req
        if not isinstance(text, str):
            return req
        text = text.strip()
        if not text or _ADVICE_RE.search(text):
            return req
        return req.model_copy(update={"question": text})


def _parse_moment(when: str) -> datetime | None:
    when = when.strip()
    if not when:
        return None
    try:
        dt = datetime.fromisoformat(when)
    except ValueError:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def context_request_at(
    ctx: AgentContext, when: str, *, proximity_min: int = _MID_LOOP_PROXIMITY_MIN
) -> ContextRequest | None:
    """A gated logging request for a moment the agent is blind at.

    ``None`` when ``when`` is unparseable, or when a meal/note is already logged
    nearby (the agent should probe that instead of asking). Reuses the batch
    detector's proximity rule and dosing gate, so 'unexplained' means the same
    thing whether dexta finds a gap in the background or mid-investigation.
    """
    ts = _parse_moment(when)
    if ts is None:
        return None
    prox = timedelta(minutes=proximity_min)
    lo, hi = ts - prox, ts + prox
    meals = ctx.store.get_meals(lo, hi)
    try:
        manual = ctx.store.get_manual_events(lo, hi)
    except (AttributeError, NotImplementedError):  # minimal/partial stores
        manual = []
    if meals or manual:
        return None
    try:
        local = ts.astimezone(ZoneInfo(ctx.timezone)).strftime("%b %d %H:%M")
    except Exception:  # pragma: no cover - defensive over a bad zone string
        local = ts.strftime("%b %d %H:%M")
    question = (
        f"Around {local} there is no meal or note logged. If you log what was "
        "happening then (a meal, illness, stress, or activity), dexta can factor "
        "it into this pattern instead of guessing."
    )
    if _ADVICE_RE.search(question):  # the template is safe; guard against edits
        return None
    return ContextRequest(
        kind="blocked_discrimination",
        event_ts=ts,
        question=question,
        suggested_event_type="meal",
        evidence={"ts": ts.isoformat()},
    )


def request_context_tool(
    ctx: AgentContext, *, proximity_min: int = _MID_LOOP_PROXIMITY_MIN
) -> ToolSpec:
    """A tool the model calls when a gap blocks it: fetch the logged context near a
    moment, or a precise logging request to surface instead of guessing."""

    def _fn(args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        when = str(args.get("when", "")).strip()
        if _parse_moment(when) is None:
            return {
                "error": "request_context needs `when` as an ISO datetime, e.g. 2026-03-14T08:30"
            }, {}
        req = context_request_at(ctx, when, proximity_min=proximity_min)
        if req is None:
            return {
                "context_present": True,
                "note": "A meal or note is already logged near that time. Probe it, do not ask.",
            }, {}
        return {"context_missing": True, "ask_user": req.question}, {}

    return ToolSpec(
        name="request_context",
        description=(
            "When a gap blocks you (you cannot separate two hypotheses without data "
            "the user never logged), call this with the moment you are blind at. It "
            "returns either the logged context near that time, or a precise request "
            "to include in your answer asking the user to log what happened. It never "
            "fabricates the missing value."
        ),
        parameters={
            "type": "object",
            "properties": {
                "when": {
                    "type": "string",
                    "description": "ISO datetime of the moment you cannot explain",
                }
            },
            "required": ["when"],
        },
        fn=_fn,
    )
