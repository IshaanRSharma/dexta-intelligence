"""Working belief state - the agent's evolving understanding across one investigation.

A real investigation carries a working memory across steps: competing
hypotheses, the evidence gathered, the gaps still open, and a running
confidence. Today that lives only implicitly in the chat transcript. This is the
explicit, first-class version: a plain mutable record the model reads and revises
through the ``update_belief`` tool, not a hard-coded controller.

The loop threads it through every step (see ``run_reasoning_loop``); later phases
read it to steer the next probe, to ask when blind, and to stop. The model still
does the thinking; this only gives that thinking a place to live.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from dexta_intelligence.agents.reason import ToolSpec
from dexta_intelligence.models import HypothesisStatus as StoredHypothesisStatus

if TYPE_CHECKING:
    from dexta_intelligence.agents.base import AgentContext

logger = logging.getLogger(__name__)

__all__ = [
    "BeliefState",
    "Hypothesis",
    "HypothesisStatus",
    "seed_belief_from_store",
]

#: Cap on prior hypotheses seeded into a live investigation. A handful keeps the
#: model discriminating between real competitors, not wading through a backlog.
_SEED_LIMIT = 5


class HypothesisStatus(StrEnum):
    """Where a hypothesis stands as evidence accrues."""

    OPEN = "open"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    UNDETERMINED = "undetermined"


@dataclass(slots=True)
class Hypothesis:
    """One competing explanation and its standing."""

    id: str
    statement: str
    status: HypothesisStatus = HypothesisStatus.OPEN
    note: str = ""


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


@dataclass(slots=True)
class BeliefState:
    """The agent's evolving understanding within one investigation.

    Mutable by design: the model revises it through :meth:`tool`. ``hypotheses``
    is keyed by id so an update can change a hypothesis's status in place;
    ``evidence`` accumulates (append-only), ``gaps`` is replaced when supplied
    (gaps close as evidence arrives), ``confidence`` is the running 0..1 belief
    in the leading explanation, and ``summary`` is the one-line "understanding so
    far".
    """

    hypotheses: dict[str, Hypothesis] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    confidence: float = 0.0
    summary: str = ""
    probed: list[str] = field(default_factory=list)
    _auto_id: int = 1

    def apply(self, update: dict[str, Any]) -> None:
        """Merge a model-supplied partial update. Unknown keys are ignored."""
        for entry in update.get("hypotheses") or ():
            self._apply_hypothesis(entry)
        for line in update.get("evidence") or ():
            text = str(line).strip()
            if text and text not in self.evidence:
                self.evidence.append(text)
        if "gaps" in update:
            self.gaps = [str(g).strip() for g in (update.get("gaps") or ()) if str(g).strip()]
        if "confidence" in update:
            self.confidence = _clamp01(update.get("confidence"))
        if "summary" in update:
            self.summary = str(update.get("summary") or "").strip()

    def _apply_hypothesis(self, entry: Any) -> None:
        if not isinstance(entry, dict):
            return
        hid = str(entry.get("id") or "").strip()
        existing = self.hypotheses.get(hid) if hid else None
        if existing is None:
            statement = str(entry.get("statement") or "").strip()
            if not statement:
                return
            hid = hid or self._fresh_id()
            raw_status = entry.get("status")
            self.hypotheses[hid] = Hypothesis(
                id=hid,
                statement=statement,
                status=_coerce_status(raw_status)
                if raw_status is not None
                else HypothesisStatus.OPEN,
                note=str(entry.get("note") or "").strip(),
            )
            return
        if entry.get("statement"):
            existing.statement = str(entry["statement"]).strip()
        if entry.get("status"):
            existing.status = _coerce_status(entry.get("status"))
        if entry.get("note") is not None:
            existing.note = str(entry.get("note") or "").strip()

    def _fresh_id(self) -> str:
        hid = f"h{self._auto_id}"
        self._auto_id += 1
        return hid

    def note_probe(self, name: str) -> None:
        """Record that a real (non-belief) tool was called, for probe selection."""
        if name and name != "update_belief":
            self.probed.append(name)

    def suggested_probe(self) -> str:
        """The most discriminating modality not yet examined for an open hypothesis.

        An information-gain proxy: the most useful next probe gathers evidence a
        live hypothesis depends on but the run has not collected. Advisory only -
        the model may probe otherwise. Empty when nothing open needs a new
        modality.
        """
        text = " ".join(
            h.statement.lower()
            for h in self.hypotheses.values()
            if h.status in (HypothesisStatus.OPEN, HypothesisStatus.UNDETERMINED)
        )
        if not text:
            return ""
        relevant = [m for m, kws in _MODALITY_KEYWORDS.items() if any(k in text for k in kws)]
        probed_modalities = {
            m for name in self.probed for m, tools in _MODALITY_TOOLS.items() if name in tools
        }
        missing = [m for m in relevant if m not in probed_modalities]
        if not missing:
            return ""
        nxt = missing[0]
        examples = ", ".join(sorted(_MODALITY_TOOLS[nxt])[:3])
        return f"{nxt} (e.g. {examples}): no {nxt} evidence yet for the open hypotheses"

    def snapshot(self) -> dict[str, Any]:
        """JSON-serializable view of the current state."""
        return {
            "hypotheses": [
                {"id": h.id, "statement": h.statement, "status": h.status.value, "note": h.note}
                for h in self.hypotheses.values()
            ],
            "evidence": list(self.evidence),
            "gaps": list(self.gaps),
            "confidence": round(self.confidence, 3),
            "summary": self.summary,
            "suggested_probe": self.suggested_probe(),
        }

    def as_text(self) -> str:
        """Compact human-readable rendering for a prompt or trace line."""
        lines = [f"confidence {self.confidence:.2f}"]
        if self.summary:
            lines.append(f"summary: {self.summary}")
        lines.extend(f"[{h.status.value}] {h.statement}" for h in self.hypotheses.values())
        lines.extend(f"gap: {g}" for g in self.gaps)
        return "\n".join(lines)

    def tool(self) -> ToolSpec:
        """The ``update_belief`` tool the model calls to revise this state.

        Returns the merged snapshot so the model always sees the current state.
        Emits no evidence numbers: the belief is meta-reasoning, not data the
        faithfulness guard should audit the answer against.
        """

        def _fn(args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
            self.apply(args or {})
            return self.snapshot(), {}

        return ToolSpec(
            name="update_belief",
            description=(
                "Record your evolving understanding before the next probe: the "
                "competing hypotheses and their status (open/supported/refuted/"
                "undetermined), evidence gathered, gaps still blocking you, your "
                "confidence (0..1) in the leading explanation, and a one-line "
                "summary. Call it after each probe; it returns the merged state."
            ),
            parameters=_BELIEF_SCHEMA,
            fn=_fn,
        )


def _coerce_status(value: Any) -> HypothesisStatus:
    try:
        return HypothesisStatus(str(value))
    except ValueError:
        return HypothesisStatus.UNDETERMINED


#: Evidence modalities for next-probe selection. ``_MODALITY_TOOLS`` are the tools
#: that read each modality directly (so a call marks it examined);
#: ``_MODALITY_KEYWORDS`` are the hypothesis terms that point to it. Insertion
#: order is the tie-break when several modalities are unexamined. The buckets are a
#: deliberate simplification for a heuristic: manual-event tools sit under carbs,
#: and "activity" is the event-proximity modality, not an exercise-only feed.
_MODALITY_TOOLS: dict[str, frozenset[str]] = {
    "carbs": frozenset(
        {
            "get_carb_entries",
            "get_cob",
            "meal_response",
            "get_manual_events",
            "search_manual_events",
        }
    ),
    "insulin": frozenset(
        {
            "get_boluses",
            "get_iob",
            "get_basal_timeline",
            "basal_overnight",
            "correction_outcome",
            "get_insulin_profile",
        }
    ),
    "activity": frozenset({"event_proximity", "find_similar_events"}),
    "temporal": frozenset({"get_weekday", "tod_compare"}),
}

_MODALITY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "carbs": ("carb", "meal", "breakfast", "lunch", "dinner", "snack", "food", "cob", "eat"),
    "insulin": ("bolus", "insulin", "iob", "basal", "correction", "dose", "dosing", "units"),
    "activity": ("exercise", "workout", "activity", "walk", "run", "gym", "steps"),
    "temporal": ("weekday", "weekend", "morning", "afternoon", "evening", "time of day", "dawn"),
}


def seed_belief_from_store(ctx: AgentContext, *, limit: int = _SEED_LIMIT) -> BeliefState:
    """Seed a belief state from the open hypotheses banked by prior runs.

    Prior wonders re-enter as live competing hypotheses for the model to
    discriminate, so a new investigation builds on what came before instead of a
    blank slate. Returns an empty state when the store holds none.
    """
    state = BeliefState()
    try:
        stored = ctx.store.get_hypotheses(status=StoredHypothesisStatus.OPEN.value)
    except Exception:  # pragma: no cover - defensive over an optional backend
        logger.debug("seed_belief_from_store: hypothesis store unavailable", exc_info=True)
        return state
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for h in stored:
        if len(entries) >= limit:
            break
        statement = h.statement.strip()
        if not statement or statement in seen:
            continue
        seen.add(statement)
        entry: dict[str, Any] = {
            "statement": statement,
            "status": StoredHypothesisStatus.OPEN.value,
        }
        if h.id is not None:
            entry["id"] = f"stored-{h.id}"
        entries.append(entry)
    if entries:
        state.apply({"hypotheses": entries})
    return state


_BELIEF_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "hypotheses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Reuse to update an existing one"},
                    "statement": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": [s.value for s in HypothesisStatus],
                    },
                    "note": {"type": "string"},
                },
            },
        },
        "evidence": {"type": "array", "items": {"type": "string"}},
        "gaps": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "summary": {"type": "string"},
    },
}
