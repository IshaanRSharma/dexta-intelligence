"""Clinical Brief Agent — the physician-visit brief.

A port of the donor codebase's clinical brief (which ran against real
endocrinologist review): the model ranks the active findings and explains
them in prose, every number is audited against the deterministic evidence
pool, and any unfaithful or unsafe section falls back to a deterministic
render. The brief is observation only — it summarizes what the data shows,
never what to do about it (spec §A4, donor row "Clinical Brief Agent").

Two safety layers, both enforced in pure code (no model is trusted to
self-police):

- **Faithfulness** — :func:`guard.faithfulness.audit` rejects any summary or
  section body citing a number absent from the involved findings' evidence +
  stats; the section falls back to its deterministic render.
- **No treatment advice** — a hard regex refuses any section body that reads
  as dosing/titration guidance ("increase basal", "take 2 units"); the
  section falls back to its deterministic render. The model prompt forbids it
  too, but the prompt is advisory and the regex is the gate.

This module makes no LLM call when ``model is None`` and never calls
``datetime.now`` — the caller passes ``today`` so the output is pure and
testable. A brief is never empty when findings exist; with no findings it
renders a graceful "insufficient data" brief.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dexta_intelligence.guard.faithfulness import audit
from dexta_intelligence.memory.findings import count_recurrence
from dexta_intelligence.models import FindingStatus

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date

    from dexta_intelligence.models import CoverageStats, Finding

logger = logging.getLogger(__name__)

__all__ = [
    "BriefSection",
    "ClinicalBrief",
    "build_brief",
    "render_markdown",
]

#: Top findings carried into the brief — an endo visit is minutes long.
_MAX_SECTIONS = 5

#: Treatment-advice gate. Matches an action verb followed (within a short
#: window) by a dosing noun — "increase basal", "take 2 units", "adjust the
#: bolus". The brief is observation only, so any match is refused and the
#: deterministic section is rendered instead. The model prompt forbids this
#: too; this regex is the enforcement, not the request.
_ADVICE_RE = re.compile(
    r"(?i)\b(increase|decrease|adjust|take)\b.{0,40}\b(insulin|units|basal|bolus|dose)"
)


@dataclass(frozen=True, slots=True)
class BriefSection:
    """One titled section of the brief — a finding rendered for a clinician.

    ``evidence`` is the number pool the body was audited against (the union of
    the involved findings' evidence + stats), kept so a reviewer can trace any
    figure in ``body`` back to its source.
    """

    title: str
    body: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ClinicalBrief:
    """A physician-visit summary of the active findings.

    ``provenance`` records how the brief was produced — the model id or
    ``"deterministic"``, the count of findings considered, and the generation
    date supplied by the caller — so the brief is self-describing for review.
    Never empty when findings exist; an "insufficient data" brief otherwise.
    """

    headline_summary: str
    sections: list[BriefSection] = field(default_factory=list)
    data_sources_line: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)


def build_brief(
    findings: Sequence[Finding],
    coverage: CoverageStats,
    *,
    model: Any = None,
    today: date,
) -> ClinicalBrief:
    """Build the physician-visit brief from the active findings.

    Selects ACTIVE findings and ranks them by confidence (desc) then
    recurrence (desc), keeping the top five. With a model, one JSON LLM call
    ranks and explains them; each section body and the summary are audited
    against the involved findings' evidence pool and refused if they read as
    treatment advice — failures fall back to the deterministic render of that
    finding. Without a model (or on total model failure), the whole brief is
    deterministic. The brief is never empty when findings exist; with none, a
    graceful "insufficient data" brief is returned.
    """
    active = _rank(findings)
    sources_line = _data_sources_line(coverage)

    if not active:
        return ClinicalBrief(
            headline_summary="Insufficient data for a clinical brief.",
            sections=[],
            data_sources_line=sources_line,
            provenance={
                "model": "deterministic",
                "findings_considered": 0,
                "generated": today.isoformat(),
            },
        )

    top = active[:_MAX_SECTIONS]
    model_id = _model_id(model)

    if model is not None:
        composed = _compose_with_model(top, findings, model)
        if composed is not None:
            summary, sections = composed
            return ClinicalBrief(
                headline_summary=summary,
                sections=sections,
                data_sources_line=sources_line,
                provenance={
                    "model": model_id,
                    "findings_considered": len(active),
                    "generated": today.isoformat(),
                },
            )

    return ClinicalBrief(
        headline_summary=_deterministic_summary(top, len(active)),
        sections=[_deterministic_section(f) for f in top],
        data_sources_line=sources_line,
        provenance={
            "model": "deterministic",
            "findings_considered": len(active),
            "generated": today.isoformat(),
        },
    )


# ── ranking & deterministic render ───────────────────────────────────────────


def _rank(findings: Sequence[Finding]) -> list[Finding]:
    """Active findings, highest confidence first, recurrence breaking ties."""
    active = [f for f in findings if f.status == FindingStatus.ACTIVE]
    return sorted(
        active,
        key=lambda f: (f.confidence, count_recurrence(f, findings)),
        reverse=True,
    )


def _deterministic_summary(top: Sequence[Finding], total: int) -> str:
    """A counts line — the safe headline when no model authored one."""
    shown = len(top)
    return (
        f"{total} active finding(s); top {shown} summarized below. "
        "Observation only — no treatment recommendations."
    )


def _deterministic_section(finding: Finding) -> BriefSection:
    """Render one finding from its own evidence, citing only its numbers."""
    pool = _evidence_pool([finding])
    stats = _stats_line(finding)
    numbers = _evidence_numbers_line(finding)

    body_parts = [finding.headline.strip()]
    if stats:
        body_parts.append(stats)
    if numbers:
        body_parts.append(numbers)
    body = " ".join(part for part in body_parts if part)

    return BriefSection(title=_title_of(finding), body=body, evidence=pool)


def _title_of(finding: Finding) -> str:
    kind = finding.kind.replace("_", " ").strip().title()
    return kind or finding.agent.title() or "Finding"


def _stats_line(finding: Finding) -> str:
    stats = finding.stats
    parts: list[str] = []
    if stats.effect_size is not None:
        parts.append(f"effect {stats.effect_size:g}")
    if stats.n is not None:
        parts.append(f"n={stats.n}")
    if stats.p_perm is not None:
        parts.append(f"p={stats.p_perm:g}")
    if stats.q_fdr is not None:
        parts.append(f"q={stats.q_fdr:g}")
    if stats.replicated is not None:
        parts.append("replicated" if stats.replicated else "not replicated")
    return f"Stats: {', '.join(parts)}." if parts else ""


def _evidence_numbers_line(finding: Finding) -> str:
    """A compact key=number line from the finding's numeric evidence."""
    parts: list[str] = []
    for key, value in finding.evidence.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        parts.append(f"{key}={value:g}")
    return f"Evidence: {', '.join(parts)}." if parts else ""


# ── model composition (house LLM-call pattern) ───────────────────────────────


_SYSTEM = (
    "You are the clinical brief layer for a Type-1 diabetes review. You rank "
    "already-established findings and explain each for an endocrinologist. "
    "Observation only — NEVER dosing, basal, bolus, titration, or any treatment "
    "advice. Cite ONLY numbers that appear in the findings given to you; invent "
    "no figures. Respond with ONE JSON object only, no prose."
)

_USER_TEMPLATE = """Active findings, ranked, each with an index (each: index, kind, \
headline, evidence, stats):

{findings}

Write the brief. Output STRICT JSON, no prose:
{{"summary": "<one or two sentence headline over the findings>",
  "sections": [{{"title": "<short section title>",
                "body": "<2-4 sentence explanation citing only this finding's numbers>",
                "finding_idx": <index of the finding this section explains>}}]}}"""


def _compose_with_model(
    top: list[Finding], all_findings: Sequence[Finding], model: Any
) -> tuple[str, list[BriefSection]] | None:
    """One LLM call; audited per-section with deterministic per-section fallback.

    Returns ``None`` only on total model failure (so the caller falls fully
    deterministic). Otherwise returns the audited summary and one section per
    top finding — each section is the model's prose if it is faithful and free
    of treatment advice, else that finding's deterministic render.
    """
    data = _invoke(top, model)
    if data is None:
        return None

    pool_all = _evidence_pool(top)
    summary = _audited_summary(data.get("summary"), pool_all, top, len(_rank(all_findings)))
    sections = _audited_sections(data.get("sections"), top)
    return summary, sections


def _invoke(top: list[Finding], model: Any) -> dict[str, Any] | None:
    prompt = _USER_TEMPLATE.format(findings=_render_findings(top))
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": prompt},
    ]
    try:
        response = model.invoke(messages)
        data = json.loads(_text_of(response))
    except Exception:
        logger.warning("clinical brief LLM failed; rendering deterministically", exc_info=True)
        return None
    return data if isinstance(data, dict) else None


def _render_findings(top: list[Finding]) -> str:
    blocks: list[str] = []
    for idx, finding in enumerate(top):
        stats = {k: v for k, v in finding.stats.model_dump().items() if v is not None}
        blocks.append(
            json.dumps(
                {
                    "index": idx,
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


def _audited_summary(
    raw: Any, pool: dict[str, Any], top: Sequence[Finding], total: int
) -> str:
    """The model summary if faithful and advice-free, else the counts line."""
    if isinstance(raw, str) and raw.strip():
        text = raw.strip()
        if _ADVICE_RE.search(text):
            logger.info("brief summary rejected (treatment advice); using deterministic")
        elif not audit(text, pool).ok:
            logger.info("brief summary dropped (unfaithful); using deterministic")
        else:
            return text
    return _deterministic_summary(top, total)


def _audited_sections(raw: Any, top: list[Finding]) -> list[BriefSection]:
    """One section per top finding, model prose where faithful and safe.

    Each finding's section is keyed by its index in ``top``; a section is used
    only when its body is faithful to that finding's evidence and free of
    treatment advice. Any miss falls back to the deterministic render so the
    brief always has one section per top finding, in rank order.
    """
    by_idx = _sections_by_index(raw)
    out: list[BriefSection] = []
    for idx, finding in enumerate(top):
        section = _section_for(by_idx.get(idx), finding)
        out.append(section)
    return out


def _sections_by_index(raw: Any) -> dict[int, dict[str, Any]]:
    by_idx: dict[int, dict[str, Any]] = {}
    if not isinstance(raw, list):
        return by_idx
    for item in raw:
        if not isinstance(item, dict):
            continue
        idx = item.get("finding_idx")
        if isinstance(idx, int) and idx not in by_idx:
            by_idx[idx] = item
    return by_idx


def _section_for(item: dict[str, Any] | None, finding: Finding) -> BriefSection:
    pool = _evidence_pool([finding])
    if item is None:
        return _deterministic_section(finding)

    body = item.get("body")
    if not isinstance(body, str) or not body.strip():
        return _deterministic_section(finding)
    body = body.strip()

    if _ADVICE_RE.search(body):
        logger.info("brief section rejected (treatment advice) for %r", finding.kind)
        return _deterministic_section(finding)
    if not audit(body, pool).ok:
        logger.info("brief section dropped (unfaithful) for %r", finding.kind)
        return _deterministic_section(finding)

    title = item.get("title")
    title = title.strip() if isinstance(title, str) and title.strip() else _title_of(finding)
    return BriefSection(title=title, body=body, evidence=pool)


# ── shared helpers ───────────────────────────────────────────────────────────


def _evidence_pool(findings: Sequence[Finding]) -> dict[str, Any]:
    """Union of the involved findings' evidence + stats — the guard's pool.

    A dict (vs. synthesis's list) so it doubles as the section's traceable
    ``evidence`` payload; the guard walks either shape identically.
    """
    pool: dict[str, Any] = {}
    for i, finding in enumerate(findings):
        pool[f"evidence_{i}"] = finding.evidence
        pool[f"stats_{i}"] = finding.stats.model_dump()
    return pool


def _data_sources_line(coverage: CoverageStats) -> str:
    """e.g. ``"glucose + insulin, 90 days, 94% coverage"``."""
    streams: list[str] = []
    if coverage.n_glucose > 0:
        streams.append("glucose")
    if coverage.n_insulin > 0:
        streams.append("insulin")
    if coverage.n_meals > 0:
        streams.append("meals")
    if coverage.n_sleep > 0:
        streams.append("sleep")
    if coverage.n_activity > 0:
        streams.append("activity")
    streams_str = " + ".join(streams) if streams else "no data streams"

    days = round(coverage.span_days)
    coverage_pct = round(coverage.glucose_coverage_pct)
    return f"{streams_str}, {days} days, {coverage_pct}% coverage"


def _model_id(model: Any) -> str:
    for attr in ("model", "model_name", "model_id", "name"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value
    return type(model).__name__


def _text_of(response: Any) -> str:
    content = getattr(response, "content", response)
    if not isinstance(content, str):
        return str(content)
    stripped = content.strip().removeprefix("```json").removeprefix("```")
    return stripped.removesuffix("```").strip()


# ── markdown render ──────────────────────────────────────────────────────────


def render_markdown(brief: ClinicalBrief) -> str:
    """Render the brief as clean printable markdown for an endo appointment."""
    lines: list[str] = ["# Clinical Brief", ""]
    if brief.data_sources_line:
        lines.append(f"*Data sources: {brief.data_sources_line}*")
        lines.append("")
    lines.append(brief.headline_summary)
    lines.append("")

    if brief.sections:
        for section in brief.sections:
            lines.append(f"## {section.title}")
            lines.append("")
            lines.append(section.body)
            lines.append("")
    else:
        lines.append("_No findings to report for this window._")
        lines.append("")

    provenance = brief.provenance
    if provenance:
        model = provenance.get("model", "deterministic")
        considered = provenance.get("findings_considered", 0)
        generated = provenance.get("generated", "")
        lines.append("---")
        lines.append("")
        lines.append(
            f"*Generated {generated} · source: {model} · "
            f"{considered} active finding(s) considered · observation only, "
            "not medical advice.*"
        )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
