"""Agentic-wiki synthesis - the LLM authors the connective narrative across findings.

The deterministic templater (``wiki.py``) lists findings separately; this layer
reasons *across* them - "your weekend TIR drop and your late-Saturday boluses are
the same story" - in one LLM call, then renders into the wiki.

Two honesty rules keep it safe and make it the place the model earns its keep:
every paragraph and connection passes :func:`guard.faithfulness.audit` against the
involved findings' evidence + stats (a fabricated number drops the line, never the
run), and the synthesis is regenerated from the store, never hand-authored. The
prose is observation only - no dosing, no numbers absent from evidence.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dexta_intelligence.guard.faithfulness import audit
from dexta_intelligence.models import Finding, FindingStatus

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date

    from dexta_intelligence.store.port import StoragePort

logger = logging.getLogger(__name__)

__all__ = [
    "SynthesisResult",
    "load_latest",
    "save",
    "synthesize",
]

#: Findings authored by this layer carry these markers - never re-synthesized,
#: never recalled as patterns, queried back as the persisted synthesis.
_SYNTHESIS_AGENT = "synthesis"
_SYNTHESIS_KIND = "wiki_synthesis"
_SYNTHESIS_SCOPE = "memory"

#: Connection lines longer than this are dropped - the prompt caps them, the
#: guard can't see word count, so we enforce it deterministically.
_MAX_CONNECTION_WORDS = 30

_SYSTEM = (
    "You are the synthesis layer of a Type-1 diabetes findings wiki. You connect "
    "already-established findings into a short narrative. Observation only - never "
    "dosing, basal, or treatment advice. Cite ONLY numbers that appear in the "
    "findings given to you; invent no figures. Each connection line is at most "
    f"{_MAX_CONNECTION_WORDS} words. Respond with ONE JSON object only, no prose."
)

_USER_TEMPLATE = """Active findings (each: kind, headline, evidence, stats):

{findings}

Write connective narrative. Output STRICT JSON, no prose:
{{"topic_paragraphs": {{"<finding.kind>": "<one short paragraph about that \
finding's topic>"}},
  "connections": ["<cross-finding observation linking two or more findings, \
<= {max_words} words>"]}}"""


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    """LLM-authored, guard-checked narrative over the active findings.

    ``topic_paragraphs`` maps a ``finding.kind`` to a synthesis paragraph for
    that topic page; ``connections`` are cross-finding lines for the index.
    Empty when the model is absent, fails, or every line is unfaithful.
    """

    topic_paragraphs: dict[str, str] = field(default_factory=dict)
    connections: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        """Whether there is nothing to render."""
        return not self.topic_paragraphs and not self.connections


def synthesize(findings: Sequence[Finding], model: Any) -> SynthesisResult:
    """Author guard-checked connective narrative over active findings.

    One JSON LLM call given the active findings' headlines, evidence, and stats.
    Every paragraph is audited against its finding's evidence pool; every
    connection against the union of all active findings' evidence. Unfaithful or
    over-long lines are dropped and logged - never rendered. A missing model,
    a malformed response, or no active findings yields an empty result.
    """
    active = [
        f
        for f in findings
        if f.status == FindingStatus.ACTIVE and f.agent != _SYNTHESIS_AGENT
    ]
    if model is None or not active:
        return SynthesisResult()

    data = _invoke(active, model)
    if data is None:
        return SynthesisResult()

    by_kind: dict[str, list[Finding]] = {}
    for finding in active:
        by_kind.setdefault(finding.kind, []).append(finding)
    pool_all = _evidence_pool(active)

    paragraphs = _audited_paragraphs(data.get("topic_paragraphs"), by_kind)
    connections = _audited_connections(data.get("connections"), pool_all)
    return SynthesisResult(topic_paragraphs=paragraphs, connections=connections)


# ── persistence (supersede-on-save, read-back the newest) ────────────────────────


def save(store: StoragePort, result: SynthesisResult, *, today: date) -> None:
    """Persist the synthesis, superseding any prior one so only the latest is ACTIVE.

    Stored Hypothesis-free: as a single ``Finding`` (agent ``synthesis``, kind
    ``wiki_synthesis``) whose ``evidence`` carries the topic paragraphs,
    connections, and date. Every currently-ACTIVE synthesis finding is flipped to
    SUPERSEDED first, so :func:`load_latest` always reads exactly one. An empty
    result still supersedes - the prior narrative should not linger once stale.
    """
    for prior in store.get_findings(
        agent=_SYNTHESIS_AGENT, status=FindingStatus.ACTIVE
    ):
        if prior.id is not None:
            store.set_finding_status(prior.id, FindingStatus.SUPERSEDED)

    finding = Finding(
        agent=_SYNTHESIS_AGENT,
        kind=_SYNTHESIS_KIND,
        scope=_SYNTHESIS_SCOPE,
        headline=f"Wiki synthesis · {today.isoformat()}",
        evidence={
            "topic_paragraphs": dict(result.topic_paragraphs),
            "connections": list(result.connections),
            "date": today.isoformat(),
        },
        confidence=0.5,
        status=FindingStatus.ACTIVE,
    )
    store.insert_finding(finding)


def load_latest(store: StoragePort) -> SynthesisResult | None:
    """The newest persisted synthesis, or ``None`` if none was ever saved.

    Reads the most recent ACTIVE ``synthesis`` finding and rehydrates its
    paragraphs and connections from ``evidence``. A malformed payload degrades to
    empty fields rather than raising.
    """
    rows = store.get_findings(
        agent=_SYNTHESIS_AGENT, status=FindingStatus.ACTIVE, limit=1
    )
    if not rows:
        return None
    evidence = rows[0].evidence
    raw_paras = evidence.get("topic_paragraphs")
    raw_conns = evidence.get("connections")
    paragraphs = {
        str(k): str(v)
        for k, v in raw_paras.items()
        if isinstance(v, str)
    } if isinstance(raw_paras, dict) else {}
    connections = (
        [str(c) for c in raw_conns if isinstance(c, str)]
        if isinstance(raw_conns, list)
        else []
    )
    return SynthesisResult(topic_paragraphs=paragraphs, connections=connections)


# ── LLM call ─────────────────────────────────────────────────────────────────


def _invoke(active: list[Finding], model: Any) -> dict[str, Any] | None:
    prompt = _USER_TEMPLATE.format(
        findings=_render_findings(active), max_words=_MAX_CONNECTION_WORDS
    )
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": prompt},
    ]
    try:
        response = model.invoke(messages)
        data = json.loads(_text_of(response))
    except Exception:
        logger.warning("wiki synthesis LLM failed; rendering without synthesis", exc_info=True)
        return None
    return data if isinstance(data, dict) else None


def _render_findings(active: list[Finding]) -> str:
    blocks: list[str] = []
    for finding in active:
        stats = {k: v for k, v in finding.stats.model_dump().items() if v is not None}
        blocks.append(
            json.dumps(
                {
                    "kind": finding.kind,
                    "headline": finding.headline,
                    "evidence": finding.evidence,
                    "stats": stats,
                },
                sort_keys=True,
                default=str,
            )
        )
    return "\n".join(blocks)


# ── guard gate ───────────────────────────────────────────────────────────────


def _audited_paragraphs(
    raw: Any, by_kind: dict[str, list[Finding]]
) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for kind, group in by_kind.items():
        text = raw.get(kind)
        if not isinstance(text, str) or not text.strip():
            continue
        report = audit(text, _evidence_pool(group))
        if report.ok:
            out[kind] = text.strip()
        else:
            logger.info("synthesis paragraph dropped for kind %r: %s", kind, report.violations)
    return out


def _audited_connections(raw: Any, pool: list[Any]) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        line = item.strip()
        if not line:
            continue
        if len(line.split()) > _MAX_CONNECTION_WORDS:
            logger.info("synthesis connection dropped (too long): %s", line)
            continue
        report = audit(line, pool)
        if report.ok:
            out.append(line)
        else:
            logger.info("synthesis connection dropped: %s", report.violations)
    return out


def _evidence_pool(findings: list[Finding]) -> list[Any]:
    """Union of the involved findings' evidence + stats - the guard's pool."""
    pool: list[Any] = []
    for finding in findings:
        pool.append(finding.evidence)
        pool.append(finding.stats.model_dump())
    return pool


def _text_of(response: Any) -> str:
    content = getattr(response, "content", response)
    if not isinstance(content, str):
        return str(content)
    stripped = content.strip().removeprefix("```json").removeprefix("```")
    return stripped.removesuffix("```").strip()
