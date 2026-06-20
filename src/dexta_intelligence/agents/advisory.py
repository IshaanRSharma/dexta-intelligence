"""Clinical Advisory - an AMIE-style discussion brief for the clinician.

Adopts Google AMIE's disease-management architecture (analyze -> set goals ->
structured plan, every item grounded, schema-constrained generation) but NOT its
output class. dexta is patient-facing, so the "plan" is DISCUSSION options for
the clinician -- what to review, monitor, and ask -- never patient dosing. Two
groundings make each item defensible: the patient's own findings AND, when a
literature backend is available, published evidence (PubMed PMIDs). The treatment
gate is the hard backstop: any item that reads as dosing advice is dropped.

With ``model=None`` the brief is fully deterministic (built from the ranked
findings), so it runs with no API key like the rest of dexta; a model only
refines the analysis/goals/phrasing under the same grounding + gate.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime  # noqa: TC003 - pydantic resolves this field type at runtime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from dexta_intelligence.agents.brief import _ADVICE_RE, _rank

if TYPE_CHECKING:
    from dexta_intelligence.evidence.base import EvidenceBackend
    from dexta_intelligence.models import CoverageStats, Finding

logger = logging.getLogger(__name__)

__all__ = [
    "SAFETY_LINE",
    "ClinicalAdvisoryAgent",
    "DiscussionBrief",
    "DiscussionItem",
    "render_markdown",
]

SAFETY_LINE = "Pattern analysis only. Not a dosing recommendation."
_MAX_ITEMS = 5
_CITE_TOP = 3  # only the strongest findings get a (bounded) literature lookup


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DiscussionItem(_FrozenModel):
    """One thing to raise with the clinician, grounded and never imperative.

    ``evidence_refs`` are the patient's own findings/numbers behind it;
    ``citations`` are PMIDs from the literature backend (may be empty).
    """

    item: str
    rationale: str
    evidence_refs: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)


class DiscussionBrief(_FrozenModel):
    """AMIE-shaped output: reasoning (analysis + goals) plus a discussion plan
    (discuss now / monitoring / questions). Discussion support, not dosing."""

    question: str | None = None
    analysis: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    discuss_now: list[DiscussionItem] = Field(default_factory=list)
    monitoring: list[DiscussionItem] = Field(default_factory=list)
    questions_for_clinician: list[DiscussionItem] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    generated_at: datetime


def _is_safe(text: str) -> bool:
    """True unless the text reads as treatment/dosing advice (the hard gate)."""
    return not _ADVICE_RE.search(text)


def _topic(finding: Finding) -> str:
    """A literature query for a finding: its scope + kind, e.g. 'overnight basal'."""
    return f"{finding.scope} {finding.kind} type 1 diabetes".replace("_", " ").strip()


def _number_refs(finding: Finding) -> list[str]:
    refs: list[str] = []
    if finding.stats.n is not None:
        refs.append(f"n={finding.stats.n}")
    if finding.stats.effect_size is not None:
        refs.append(f"effect={finding.stats.effect_size:g}")
    return refs


@dataclass
class ClinicalAdvisoryAgent:
    """Builds a :class:`DiscussionBrief` from the active findings.

    ``evidence`` (any :class:`EvidenceBackend`) is optional and best-effort: a
    slow or failed literature lookup yields no citations, never an error.
    """

    model: Any = None
    evidence: EvidenceBackend | None = None
    max_items: int = _MAX_ITEMS

    def build(
        self,
        findings: list[Finding],
        coverage: CoverageStats,
        *,
        question: str | None = None,
        now: datetime,
    ) -> DiscussionBrief:
        active = _rank(findings)[: self.max_items]
        if not active:
            return DiscussionBrief(
                question=question,
                analysis=["No active findings yet - not enough to prepare a discussion."],
                limitations=[SAFETY_LINE],
                generated_at=now,
            )

        items = [it for f in active if _is_safe((it := self._item(f)).item + " " + it.rationale)]
        analysis = [f"dexta has {len(active)} active finding(s) over the analysed window."]
        goals = self._goals(active)
        monitoring = self._monitoring(active)
        questions = self._questions(active)
        limitations = [
            f"Coverage {coverage.glucose_coverage_pct:.0f}% over {coverage.span_days:.0f} days.",
            "These are discussion points for your clinician, not dosing recommendations.",
            SAFETY_LINE,
        ]
        brief = DiscussionBrief(
            question=question,
            analysis=analysis,
            goals=goals,
            discuss_now=items,
            monitoring=monitoring,
            questions_for_clinician=questions,
            limitations=limitations,
            generated_at=now,
        )
        return self._refine_with_model(brief, active) if self.model is not None else brief

    # ── deterministic construction ───────────────────────────────────────────

    def _item(self, finding: Finding) -> DiscussionItem:
        rationale = (finding.body_md or finding.headline).split("\n", 1)[0][:240]
        return DiscussionItem(
            item=f"Review the pattern: {finding.headline}",
            rationale=rationale,
            evidence_refs=[finding.headline, *_number_refs(finding)],
            citations=self._cite(finding),
        )

    def _cite(self, finding: Finding) -> list[str]:
        if self.evidence is None:
            return []
        try:
            hits = self.evidence.search(_topic(finding), limit=2)
        except Exception:
            logger.debug("advisory: literature lookup failed", exc_info=True)
            return []
        return [h.id for h in hits if h.source == "pubmed"]

    def _goals(self, active: list[Finding]) -> list[str]:
        kinds = sorted({f.kind.replace("_", " ") for f in active})
        return [f"Understand and address the recurring {k} pattern." for k in kinds[:3]]

    def _monitoring(self, active: list[Finding]) -> list[DiscussionItem]:
        return [
            DiscussionItem(
                item=f"Keep watching {f.scope.replace('_', ' ')} going forward.",
                rationale="Confirm whether the pattern persists or resolves.",
                evidence_refs=[f.headline],
            )
            for f in active[:2]
        ]

    def _questions(self, active: list[Finding]) -> list[DiscussionItem]:
        return [
            DiscussionItem(
                item=f"Is the {f.kind.replace('_', ' ')} pattern something to act on?",
                rationale="Bring the evidence card so your clinician can weigh it.",
                evidence_refs=[f.headline],
                citations=self._cite(f),
            )
            for f in active[:_CITE_TOP]
        ]

    # ── optional model refinement (schema-constrained, grounded, gated) ────────

    def _refine_with_model(self, brief: DiscussionBrief, active: list[Finding]) -> DiscussionBrief:
        """Let the model rewrite the analysis + goals prose. Phrasing only - the
        grounded, gated discussion items are never replaced by free text."""
        prompt = _REFINE_PROMPT.format(
            question=brief.question or "general review",
            findings="\n".join(f"- {f.headline}" for f in active),
        )
        try:
            response = self.model.invoke(
                [
                    {"role": "system", "content": "Respond with ONE JSON object only."},
                    {"role": "user", "content": prompt},
                ]
            )
            data = _parse_json(response.content)
        except Exception:
            logger.debug("advisory: model refine failed; keeping deterministic", exc_info=True)
            return brief
        if not isinstance(data, dict):
            return brief
        analysis = [s for s in _str_list(data.get("analysis")) if _is_safe(s)]
        goals = [s for s in _str_list(data.get("goals")) if _is_safe(s)]
        return brief.model_copy(
            update={
                "analysis": analysis or brief.analysis,
                "goals": goals or brief.goals,
            }
        )


_REFINE_PROMPT = """Summarize these diabetes findings for a clinician visit about: {question}

FINDINGS:
{findings}

Write a short analysis and 2-3 management GOALS. Do NOT give dosing, insulin,
basal, or carb-ratio instructions - goals are directions to discuss, not actions.

Output STRICT JSON: {{"analysis": ["..."], "goals": ["..."]}}"""


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [s.strip() for s in value if isinstance(s, str) and s.strip()]


def _parse_json(content: Any) -> dict[str, Any] | None:
    text = content if isinstance(content, str) else ""
    if isinstance(content, list):
        text = "".join(
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        )
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


# ── rendering ──────────────────────────────────────────────────────────────────


def _items_md(items: list[DiscussionItem]) -> list[str]:
    lines: list[str] = []
    for it in items:
        cite = f"  [{', '.join(f'PMID {p}' for p in it.citations)}]" if it.citations else ""
        lines.append(f"- **{it.item}**{cite}")
        if it.rationale:
            lines.append(f"  - {it.rationale}")
        if it.evidence_refs:
            lines.append(f"  - Evidence: {'; '.join(it.evidence_refs)}")
    return lines


def render_markdown(brief: DiscussionBrief) -> str:
    """The brief as Markdown for the doctor export (PMIDs linkify when shown)."""
    out: list[str] = ["# dexta discussion brief", ""]
    out.append(
        "_Decision support for a clinician visit. Generated from this person's own "
        "data; evidence and uncertainty are shown. Your clinician decides. "
        f"{SAFETY_LINE}_"
    )
    if brief.question:
        out += ["", f"**Focus:** {brief.question}"]
    if brief.analysis:
        out += ["", "## Analysis", *[f"- {a}" for a in brief.analysis]]
    if brief.goals:
        out += ["", "## Goals to discuss", *[f"- {g}" for g in brief.goals]]
    if brief.discuss_now:
        out += ["", "## Discuss now", *_items_md(brief.discuss_now)]
    if brief.monitoring:
        out += ["", "## Suggested monitoring", *_items_md(brief.monitoring)]
    if brief.questions_for_clinician:
        out += ["", "## Questions for your clinician", *_items_md(brief.questions_for_clinician)]
    if brief.limitations:
        out += ["", "## Limitations", *[f"- {limit}" for limit in brief.limitations]]
    return "\n".join(out) + "\n"
