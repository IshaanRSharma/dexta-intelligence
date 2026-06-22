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

from dexta_intelligence.agents.brief import _ADVICE_RE
from dexta_intelligence.agents.tools.toolkit import DiscoveryToolkit
from dexta_intelligence.models import ContextRequest

if TYPE_CHECKING:
    from dexta_intelligence.agents.base import AgentContext

__all__ = ["ContextAcquisitionAgent"]

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
