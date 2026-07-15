"""Curated context for the ask surfaces - the librarian wired into chat.

Builds the typed context block (FACTS / BELIEFS / CONVENTIONS / HISTORY) that
:func:`~dexta_intelligence.memory.curator.select_context` chose for one
question, plus one trace line per drop receipt so every pruning decision is
visible in the answer trace. Deterministic: no model call, no wall-clock read
(``now`` is a parameter), and the same store state + question always produce
the same block.

The block is context, never evidence: it tells the model what dexta already
believes and instructs it to recompute any number with tools before citing it,
so the faithfulness guard's evidence pool stays tool-results-only.
"""

from __future__ import annotations

from datetime import UTC, datetime, time
from typing import TYPE_CHECKING

from dexta_intelligence.agents.trace import TraceLine
from dexta_intelligence.analytics.episodes import TARGET_HIGH, TARGET_LOW, detect_episodes
from dexta_intelligence.memory.curator import ContextType, select_context
from dexta_intelligence.models import FindingStatus

if TYPE_CHECKING:
    from dexta_intelligence.agents.base import AgentContext

__all__ = ["curated_context"]

#: Default prompt-space budget for the curated block (chars/4 heuristic).
DEFAULT_CONTEXT_BUDGET_TOKENS = 600
_FINDINGS_LIMIT = 50

_HEADER = (
    "CURATED CONTEXT (deterministically selected from stored episodes and "
    "findings; the label is the memory type). This is orientation, not "
    "evidence: recompute any number with tools before citing it."
)

_SECTION_ORDER = (
    ContextType.FACTS,
    ContextType.BELIEFS,
    ContextType.CONVENTIONS,
    ContextType.HISTORY,
)


def curated_context(
    ctx: AgentContext,
    question: str,
    *,
    now: datetime,
    budget_tokens: int = DEFAULT_CONTEXT_BUDGET_TOKENS,
    target_low: int = TARGET_LOW,
    target_high: int = TARGET_HIGH,
) -> tuple[str, tuple[TraceLine, ...]]:
    """The (system-prompt block, drop-receipt trace lines) for one question.

    Returns ``("", ())`` when the curator selects nothing, so callers can
    treat the integration as a no-op on an empty or irrelevant memory.
    """
    start = datetime.combine(ctx.window[0], time.min, tzinfo=UTC)
    end = datetime.combine(ctx.window[1], time.max, tzinfo=UTC)
    episodes = detect_episodes(
        ctx.store, start, end, target_low=target_low, target_high=target_high
    )
    findings = ctx.store.get_findings(status=FindingStatus.ACTIVE, limit=_FINDINGS_LIMIT)
    edges = ctx.store.get_finding_edges()
    selection = select_context(
        question,
        budget_tokens=budget_tokens,
        now=now,
        window=(start, end),
        episodes=episodes,
        findings=findings,
        edges=edges,
    )
    if not selection.selected:
        return "", ()
    lines = [_HEADER]
    for type_ in _SECTION_ORDER:
        items = [i for i in selection.selected if i.type is type_]
        if not items:
            continue
        lines.append(f"{type_.value.upper()}:")
        lines.extend(f"- {item.text}" for item in items)
    receipts = tuple(TraceLine("scope", text) for text in selection.trace_lines())
    return "\n".join(lines), receipts
