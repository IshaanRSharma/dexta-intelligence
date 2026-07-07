"""Endo-visit brief - the three things worth appointment time.

A deterministic composer (no model call anywhere) that selects the top
discussion items from the active findings and the window's episodes, ranked by
severity, recurrence, and recency. Each item carries the pattern, its receipts
(the verified numbers behind every figure in the pattern text), and a neutral
question for the care team.

Two structural rails, enforced in code:

- **Treatment gate.** Questions are template-generated and observation-only;
  any candidate whose text reads as dosing/titration guidance (the same
  ``_ADVICE_RE`` the clinical brief enforces) is excluded outright. The brief
  describes patterns and asks questions; it never instructs.
- **Faithfulness.** Every candidate's pattern text is audited against its own
  receipts before selection; a candidate citing a number its receipts lack is
  dropped and the next candidate promoted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dexta_intelligence.agents.brief import _ADVICE_RE
from dexta_intelligence.guard.faithfulness import audit
from dexta_intelligence.memory.findings import recurrence_line
from dexta_intelligence.models import FindingStatus

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date

    from dexta_intelligence.analytics.episodes import Episode
    from dexta_intelligence.models import Finding

__all__ = ["DiscussionItem", "EndoBrief", "build_endo_brief", "render_markdown"]

_MAX_ITEMS = 3
_RECENCY_HORIZON_DAYS = 90.0
#: Ranking weights: clinical severity leads, recurrence next, recency breaks ties.
_W_SEVERITY = 0.5
_W_RECURRENCE = 0.3
_W_RECENCY = 0.2
#: seen_count at which recurrence saturates.
_RECURRENCE_CAP = 5

_FOOTER = (
    "Observation only: patterns and questions to discuss, never treatment "
    "instructions."
)


@dataclass(frozen=True, slots=True)
class DiscussionItem:
    """One thing worth the appointment time.

    ``receipts`` holds every number ``pattern`` is allowed to cite (the item
    was audited against it at composition time). ``question`` is a neutral,
    template-generated prompt for the care team, never an instruction.
    """

    title: str
    pattern: str
    receipts: dict[str, Any]
    question: str
    source: str  # "finding" | "episodes"
    score: float


@dataclass(frozen=True, slots=True)
class EndoBrief:
    """The visit brief: at most three ranked discussion items plus provenance."""

    items: tuple[DiscussionItem, ...]
    generated: str
    findings_considered: int
    episodes_considered: int


def build_endo_brief(
    findings: Sequence[Finding],
    episodes: Sequence[Episode],
    *,
    today: date,
    max_items: int = _MAX_ITEMS,
) -> EndoBrief:
    """Compose the brief from findings + episodes, deterministically.

    Candidates are scored on severity/recurrence/recency, then admitted in
    rank order only if they clear both rails (no treatment advice, every
    number faithful to the candidate's own receipts).
    """
    candidates = _finding_candidates(findings, today) + _episode_candidates(episodes, today)
    candidates.sort(key=lambda item: (-item.score, item.title, item.pattern))
    items: list[DiscussionItem] = []
    for candidate in candidates:
        if len(items) >= max_items:
            break
        if _ADVICE_RE.search(f"{candidate.pattern} {candidate.question}"):
            continue
        if not audit(candidate.pattern, candidate.receipts).ok:
            continue
        items.append(candidate)
    return EndoBrief(
        items=tuple(items),
        generated=today.isoformat(),
        findings_considered=len(findings),
        episodes_considered=len(episodes),
    )


# ── candidates from findings ─────────────────────────────────────────────────


def _finding_candidates(findings: Sequence[Finding], today: date) -> list[DiscussionItem]:
    out: list[DiscussionItem] = []
    for finding in findings:
        if finding.status is not FindingStatus.ACTIVE:
            continue
        if _ADVICE_RE.search(f"{finding.headline} {finding.body_md or ''}"):
            continue
        pattern = finding.headline.strip()
        recurrence = recurrence_line(finding)
        if recurrence:
            pattern = f"{pattern} ({recurrence})."
        receipts = _finding_receipts(finding)
        score = _score(
            severity=finding.confidence,
            recurrence=min(finding.seen_count, _RECURRENCE_CAP) / _RECURRENCE_CAP,
            recency=_recency(finding.window_end.date() if finding.window_end else None, today),
        )
        out.append(
            DiscussionItem(
                title=_finding_title(finding),
                pattern=pattern,
                receipts=receipts,
                question=(
                    "Does this pattern suggest anything you would want to look "
                    "into with us?"
                ),
                source="finding",
                score=score,
            )
        )
    return out


def _finding_receipts(finding: Finding) -> dict[str, Any]:
    receipts: dict[str, Any] = {
        k: v
        for k, v in finding.evidence.items()
        if isinstance(v, (int, float, str)) and not isinstance(v, bool)
    }
    receipts.update(
        {
            k: v
            for k, v in finding.stats.model_dump().items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
    )
    receipts["seen_count"] = finding.seen_count
    if finding.window_start is not None:
        receipts["window_start"] = finding.window_start.isoformat()
    if finding.window_end is not None:
        receipts["window_end"] = finding.window_end.isoformat()
    return receipts


def _finding_title(finding: Finding) -> str:
    kind = finding.kind.replace("_", " ").strip().title()
    return kind or finding.agent.title() or "Finding"


# ── candidates from episodes ─────────────────────────────────────────────────


def _episode_candidates(episodes: Sequence[Episode], today: date) -> list[DiscussionItem]:
    out: list[DiscussionItem] = []
    for kind, title, noun, question in (
        (
            "hypo",
            "Low Episodes",
            "lowest",
            "What might be contributing to these lows, and is this frequency "
            "something to review?",
        ),
        (
            "hyper",
            "High Episodes",
            "highest",
            "What might be contributing to these sustained highs, and is this "
            "worth reviewing together?",
        ),
    ):
        cluster = [e for e in episodes if e.kind == kind and e.clinically_significant]
        if not cluster:
            continue
        n = len(cluster)
        longest = max(e.duration_min for e in cluster)
        extremes = [e.extreme_mg_dl for e in cluster if e.extreme_mg_dl is not None]
        extreme = (min(extremes) if kind == "hypo" else max(extremes)) if extremes else None
        n_severe = sum(1 for e in cluster if e.severe)
        parts = [f"{n} clinically significant {kind} episode(s) in this window"]
        parts.append(f"longest {longest:g} min")
        if extreme is not None:
            parts.append(f"{noun} {extreme:g} mg/dL")
        if n_severe:
            parts.append(f"{n_severe} severe")
        pattern = "; ".join(parts) + "."
        receipts: dict[str, Any] = {
            "n_episodes": n,
            "longest_min": longest,
            "n_severe": n_severe,
        }
        if extreme is not None:
            receipts[f"{noun}_mg_dl"] = extreme
        last_end = max(e.end for e in cluster)
        score = _score(
            severity=(1.0 if n_severe else 0.75) if kind == "hypo" else (0.9 if n_severe else 0.6),
            recurrence=min(n, _RECURRENCE_CAP) / _RECURRENCE_CAP,
            recency=_recency(last_end.date(), today),
        )
        out.append(
            DiscussionItem(
                title=title,
                pattern=pattern,
                receipts=receipts,
                question=question,
                source="episodes",
                score=score,
            )
        )
    return out


# ── scoring ──────────────────────────────────────────────────────────────────


def _score(*, severity: float, recurrence: float, recency: float) -> float:
    return _W_SEVERITY * severity + _W_RECURRENCE * recurrence + _W_RECENCY * recency


def _recency(anchor: date | None, today: date) -> float:
    if anchor is None:
        return 0.5
    age_days = (today - anchor).days
    if age_days <= 0:
        return 1.0
    return max(0.0, 1.0 - age_days / _RECENCY_HORIZON_DAYS)


# ── markdown render ──────────────────────────────────────────────────────────


def render_markdown(brief: EndoBrief) -> str:
    """The brief as printable markdown for the appointment."""
    lines: list[str] = ["# Endo Visit Brief", ""]
    lines.append(f"*Generated {brief.generated}. {_FOOTER}*")
    lines.append("")
    if not brief.items:
        lines.append("_Not enough verified patterns to suggest discussion items yet._")
        lines.append("")
    for idx, item in enumerate(brief.items, start=1):
        lines.append(f"## {idx}. {item.title}")
        lines.append("")
        lines.append(item.pattern)
        lines.append("")
        lines.append(f"**Ask the care team:** {item.question}")
        lines.append("")
        receipts = ", ".join(
            f"{k}={v:g}"
            if isinstance(v, (int, float)) and not isinstance(v, bool)
            else f"{k}={v}"
            for k, v in item.receipts.items()
        )
        if receipts:
            lines.append(f"*Receipts: {receipts}*")
            lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        f"*{brief.findings_considered} finding(s) and {brief.episodes_considered} "
        f"episode(s) considered. {_FOOTER}*"
    )
    return "\n".join(lines).rstrip() + "\n"
