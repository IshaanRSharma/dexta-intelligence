"""Wiki - the human-readable markdown projection of the findings store.

The findings table is the only memory; the wiki is a generated view of it,
rebuildable byte-identically (``dexta wiki``). Nothing is deleted: retracted
findings go to the graveyard with skeptic notes, stale ones decay by score
(age vs confidence and recurrence) without leaving history, and each
generation is a git commit so ``git log`` is the record of belief. Rendering
is deterministic templating, so every number traces to the store.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dexta_intelligence.memory.findings import count_recurrence
from dexta_intelligence.models import Finding, FindingStatus, HypothesisStatus

if TYPE_CHECKING:
    from datetime import date
    from pathlib import Path
    from typing import Any

    from dexta_intelligence.memory.synthesis import SynthesisResult
    from dexta_intelligence.models import FindingStats, Hypothesis
    from dexta_intelligence.store.port import StoragePort

__all__ = [
    "STALE_THRESHOLD",
    "WikiReport",
    "generate_wiki",
    "staleness",
    "topic_slug",
]

STALE_THRESHOLD = 20.0
"""Findings scoring above this are demoted to the index's stale section."""

#: Staleness accrues at half a point per day past the finding's window.
_AGE_WEIGHT = 0.5
#: Each point of confidence buys 10 staleness points of grace.
_CONFIDENCE_RELIEF = 10.0
#: Each recurrence buys 5 points of grace (recurrence is our reinforcement).
_RECURRENCE_RELIEF = 5.0
_MAX_RECURRENCE_BONUS = 4
#: ``get_findings`` paginates; the wiki always projects the whole store.
_FETCH_ALL = 1_000_000

_GRAVEYARD_STATUSES = (
    FindingStatus.REJECTED,
    FindingStatus.SUPERSEDED,
    FindingStatus.DISMISSED,
    FindingStatus.CONTRADICTED,
)
_STATUS_MARKS = {
    FindingStatus.ACTIVE: "✓",
    FindingStatus.SUPERSEDED: "⊘",
    FindingStatus.REJECTED: "✗",
    FindingStatus.DISMISSED: "-",
    FindingStatus.CONTRADICTED: "≠",
}
_HYPOTHESIS_ORDER = (
    HypothesisStatus.OPEN,
    HypothesisStatus.SUPPORTED,
    HypothesisStatus.REFUTED,
    HypothesisStatus.STALE,
)


@dataclass(frozen=True, slots=True)
class WikiReport:
    """Outcome of one wiki generation."""

    root: Path
    pages: tuple[Path, ...]
    committed: bool


def topic_slug(kind: str) -> str:
    """Kebab-case page slug for a finding kind; stable so links never break."""
    slug = re.sub(r"[^a-z0-9]+", "-", kind.lower()).strip("-")
    return slug or "general"


def staleness(finding: Finding, *, today: date, recurrence: int = 0) -> float:
    """Decay score: age worked against confidence and recurrence.

    Higher is staler; > ``STALE_THRESHOLD`` demotes the finding from the
    index's active list. Findings without a window never decay.
    """
    if finding.window_end is None:
        return 0.0
    age_days = max(0.0, float((today - finding.window_end.date()).days))
    relief = finding.confidence * _CONFIDENCE_RELIEF
    relief += _RECURRENCE_RELIEF * min(recurrence, _MAX_RECURRENCE_BONUS)
    return age_days * _AGE_WEIGHT - relief


def generate_wiki(
    store: StoragePort,
    *,
    root: Path,
    today: date,
    new_findings: tuple[Finding, ...] = (),
    git: bool = True,
    synthesis: SynthesisResult | None = None,
) -> WikiReport:
    """Project the findings store into markdown pages under ``root``.

    Same store + same ``today`` ⇒ identical bytes. ``new_findings`` (this
    run's output, if generation follows an analyze) adds a dated changelog
    page under ``runs/``. With ``git=True`` the wiki directory becomes its
    own repository and every generation that changes a page is committed.

    ``synthesis`` (optional) is the LLM-authored, guard-checked narrative
    layer: when given, topic pages gain a ``## Synthesis`` section
    and the index a ``## Connections`` section. With ``synthesis=None`` the
    output is byte-identical to the deterministic templating.
    """
    findings = store.get_findings(status=None, limit=_FETCH_ALL)
    hypotheses = store.get_hypotheses()
    active = [f for f in findings if f.status == FindingStatus.ACTIVE]
    recurrence = {id(f): count_recurrence(f, active) for f in active}

    pages: list[Path] = []

    def write(rel: str, content: str) -> None:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        pages.append(path)

    write(
        "index.md",
        _render_index(
            store, findings, hypotheses, recurrence, today=today, synthesis=synthesis
        ),
    )
    by_kind: dict[str, list[Finding]] = {}
    for finding in findings:
        by_kind.setdefault(finding.kind, []).append(finding)
    for kind, group in sorted(by_kind.items()):
        write(
            f"topics/{topic_slug(kind)}.md",
            _render_topic(kind, group, recurrence, today=today, synthesis=synthesis),
        )
    write("hypotheses.md", _render_hypotheses(hypotheses))
    write("graveyard.md", _render_graveyard(findings))
    goals = store.get_goals()
    if goals:
        write("goals.md", _render_goals(store, goals))
    if new_findings:
        write(f"runs/{today.isoformat()}.md", _render_run(new_findings, today=today))

    committed = git and _git_commit(root, message=f"dexta wiki: {today.isoformat()}")
    return WikiReport(root=root, pages=tuple(pages), committed=committed)


# ── rendering ────────────────────────────────────────────────────────────────


def _render_index(
    store: StoragePort,
    findings: list[Finding],
    hypotheses: list[Hypothesis],
    recurrence: dict[int, int],
    *,
    today: date,
    synthesis: SynthesisResult | None = None,
) -> str:
    coverage = store.coverage()
    active = [f for f in findings if f.status == FindingStatus.ACTIVE]
    fresh_ids = {
        id(f)
        for f in active
        if staleness(f, today=today, recurrence=recurrence[id(f)]) <= STALE_THRESHOLD
    }
    fresh = [f for f in active if id(f) in fresh_ids]
    stale = [f for f in active if id(f) not in fresh_ids]
    open_count = sum(1 for h in hypotheses if h.status == HypothesisStatus.OPEN)
    buried = sum(1 for f in findings if f.status in _GRAVEYARD_STATUSES)

    lines = [
        "# dexta wiki",
        "",
        f"_Generated {today.isoformat()} from the findings store - do not edit; run"
        " `dexta wiki` to regenerate. Belief history: `git log` in this directory._",
        "",
        f"Coverage: {coverage.span_days:.1f} days · {coverage.glucose_coverage_pct:.0f}%"
        f" glucose slots · {len(active)} active finding(s)",
        "",
        "## Current beliefs",
        "",
    ]
    if fresh:
        lines += ["| finding | topic | confidence | seen | stats |", "|---|---|---|---|---|"]
        lines += [_index_row(f, recurrence[id(f)]) for f in _ranked(fresh, recurrence)]
    elif stale:
        lines.append("_All active findings have gone stale - `dexta sync && dexta analyze`._")
    else:
        lines.append("_No active findings yet - run `dexta analyze`._")
    if stale:
        lines += [
            "",
            "## Stale - awaiting fresh data",
            "",
            "_Not retracted; the data has just stopped reinforcing them._",
            "",
        ]
        lines += [
            f"- {f.headline} ([{topic_slug(f.kind)}](topics/{topic_slug(f.kind)}.md))"
            for f in _ranked(stale, recurrence)
        ]
    if synthesis is not None and synthesis.connections:
        lines += ["", "## Connections", ""]
        lines += [f"- {line}" for line in synthesis.connections]
    lines += [
        "",
        "## Boards",
        "",
        f"- [Hypotheses](hypotheses.md) - {open_count} open",
        f"- [Graveyard](graveyard.md) - {buried} retracted or superseded",
    ]
    goals = store.get_goals()
    if goals:
        active_goals = sum(1 for g in goals if g.status.value == "active")
        lines.append(f"- [Goals](goals.md) - {active_goals} active")
    lines.append("")
    return "\n".join(lines)


def _render_topic(
    kind: str,
    group: list[Finding],
    recurrence: dict[int, int],
    *,
    today: date,
    synthesis: SynthesisResult | None = None,
) -> str:
    active = [f for f in group if f.status == FindingStatus.ACTIVE]
    lines = [f"# {kind}", ""]
    lines += ["## Current belief", ""]
    if active:
        for finding in _ranked(active, recurrence):
            lines += _belief_section(finding, recurrence[id(finding)], today=today)
    else:
        lines += ["_Nothing currently believed - see history below._", ""]
    if synthesis is not None and (paragraph := synthesis.topic_paragraphs.get(kind)):
        lines += ["## Synthesis", "", paragraph, ""]
    lines += ["## History", ""]
    lines += [_history_line(f) for f in _by_recency(group)]
    lines.append("")
    return "\n".join(lines)


def _render_hypotheses(hypotheses: list[Hypothesis]) -> str:
    lines = [
        "# Hypotheses",
        "",
        "_Candidate patterns that have not cleared the rigor bar (yet)._",
        "",
    ]
    for status in _HYPOTHESIS_ORDER:
        matching = [h for h in hypotheses if h.status == status]
        if not matching:
            continue
        lines += [f"## {status.value.capitalize()}", ""]
        lines += [f"- {h.statement}" for h in matching]
        lines.append("")
    if len(lines) == 4:
        lines += ["_None yet._", ""]
    return "\n".join(lines)


def _render_graveyard(findings: list[Finding]) -> str:
    lines = [
        "# Graveyard",
        "",
        "_What the system stopped believing, and why. Nothing is deleted -"
        " retraction with reasons is the trust artifact._",
        "",
    ]
    buried = [f for f in findings if f.status in _GRAVEYARD_STATUSES]
    if not buried:
        lines += ["_Empty - no finding has been retracted yet._", ""]
        return "\n".join(lines)
    for finding in _by_recency(buried):
        mark = _STATUS_MARKS[finding.status]
        lines.append(f"### {mark} {finding.headline}")
        lines.append(
            f"- {finding.status.value} · agent `{finding.agent}` ·"
            f" topic [{topic_slug(finding.kind)}](topics/{topic_slug(finding.kind)}.md)"
            f"{_window_clause(finding)}"
        )
        if finding.superseded_by is not None:
            lines.append(f"- superseded by finding #{finding.superseded_by}")
        if finding.skeptic_notes:
            lines.append(f"- skeptic: {finding.skeptic_notes}")
        lines.append("")
    return "\n".join(lines)


def _render_goals(store: StoragePort, goals: list[Any]) -> str:
    lines = [
        "# Goals",
        "",
        "_What the background agents are working toward. Progress is a measured arc,"
        " never a model's opinion._",
        "",
    ]
    for goal in goals:
        lines.append(f"## #{goal.id} - {goal.statement}")
        lines.append(
            f"- {goal.status.value} · metric `{goal.metric.value}` ({goal.direction})"
            f" · every {goal.cadence_days}d"
        )
        checkpoints = store.get_goal_checkpoints(goal.id) if goal.id is not None else []
        if checkpoints:
            lines.append("- arc:")
            lines += [
                f"  - {cp.ts.date().isoformat()} · {cp.metric_value} · {cp.note}"
                for cp in checkpoints
            ]
        else:
            lines.append("- no checkpoints yet")
        lines.append("")
    return "\n".join(lines)


def _render_run(new_findings: tuple[Finding, ...], *, today: date) -> str:
    accepted = [f for f in new_findings if f.status == FindingStatus.ACTIVE]
    rejected = [f for f in new_findings if f.status != FindingStatus.ACTIVE]
    lines = [
        f"# Run - {today.isoformat()}",
        "",
        f"{len(accepted)} finding(s) survived the skeptic; {len(rejected)} did not.",
        "",
    ]
    if accepted:
        lines += ["## New beliefs", ""]
        lines += [f"- {f.headline} ({f.agent})" for f in accepted]
        lines.append("")
    if rejected:
        lines += ["## Rejected this run", ""]
        for finding in rejected:
            note = f" - skeptic: {finding.skeptic_notes}" if finding.skeptic_notes else ""
            lines.append(f"- {finding.headline} ({finding.agent}){note}")
        lines.append("")
    return "\n".join(lines)


# ── fragments ────────────────────────────────────────────────────────────────


def _belief_section(finding: Finding, recurrence: int, *, today: date) -> list[str]:
    lines = [f"### {finding.headline}", ""]
    seen = f" · seen x{recurrence + 1}" if recurrence else ""
    stale_score = staleness(finding, today=today, recurrence=recurrence)
    stale_badge = " · **stale**" if stale_score > STALE_THRESHOLD else ""
    lines.append(
        f"- confidence {finding.confidence:.2f}{seen}{_window_clause(finding)}{stale_badge}"
    )
    if stats_line := _stats_line(finding.stats):
        lines.append(f"- stats: {stats_line}")
    if finding.evidence:
        lines.append("- evidence:")
        lines += [
            f"  - {key}: {_fmt_value(value)}" for key, value in sorted(finding.evidence.items())
        ]
    if finding.skeptic_notes:
        lines.append(f"- skeptic: {finding.skeptic_notes}")
    if finding.body_md:
        lines += ["", finding.body_md]
    lines.append("")
    return lines


def _index_row(finding: Finding, recurrence: int) -> str:
    slug = topic_slug(finding.kind)
    headline = finding.headline.replace("|", "\\|")
    stats = _stats_line(finding.stats) or "-"
    return (
        f"| {headline} | [{slug}](topics/{slug}.md) | {finding.confidence:.2f}"
        f" | x{recurrence + 1} | {stats} |"
    )


def _history_line(finding: Finding) -> str:
    mark = _STATUS_MARKS[finding.status]
    ref = f" (#{finding.id})" if finding.id is not None else ""
    superseded = (
        f" → superseded by #{finding.superseded_by}" if finding.superseded_by is not None else ""
    )
    when = finding.window_end.date().isoformat() if finding.window_end else "undated"
    return f"- {mark} {finding.status.value} · {when} · {finding.headline}{ref}{superseded}"


def _stats_line(stats: FindingStats) -> str | None:
    bits: list[str] = []
    if stats.effect_size is not None:
        bits.append(f"effect={stats.effect_size:g}")
    if stats.n is not None:
        bits.append(f"n={stats.n}")
    if stats.p_perm is not None:
        bits.append(f"p_perm={stats.p_perm:g}")
    if stats.q_fdr is not None:
        bits.append(f"q_fdr={stats.q_fdr:g}")
    if stats.replicated is not None:
        bits.append("replicated ✓" if stats.replicated else "replicated ✗")
    return " · ".join(bits) if bits else None


def _window_clause(finding: Finding) -> str:
    if finding.window_start is None or finding.window_end is None:
        return ""
    return (
        f" · window {finding.window_start.date().isoformat()}"
        f" → {finding.window_end.date().isoformat()}"
    )


#: Evidence lists longer than this render abbreviated; the store keeps every value.
_MAX_LIST_VALUES = 8


def _fmt_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (str, int, bool)) or value is None:
        return str(value)
    if isinstance(value, list) and len(value) > _MAX_LIST_VALUES:
        head = ", ".join(_fmt_value(v) for v in value[:3])
        return f"[{head}, … {len(value)} values]"
    return json.dumps(value, sort_keys=True, default=str)


def _ranked(findings: list[Finding], recurrence: dict[int, int]) -> list[Finding]:
    """Confidence, then recurrence, then recency - deterministic tie-break on headline."""

    def key(f: Finding) -> tuple[float, int, str, str]:
        when = f.window_end.isoformat() if f.window_end else ""
        return (-f.confidence, -recurrence.get(id(f), 0), when, f.headline)

    return sorted(findings, key=key)


def _by_recency(findings: list[Finding]) -> list[Finding]:
    def key(f: Finding) -> tuple[str, int]:
        when = f.window_end.isoformat() if f.window_end else ""
        return (when, f.id or 0)

    return sorted(findings, key=key, reverse=True)


# ── git (forensic belief history) ────────────────────────────────────────────


def _git_commit(root: Path, *, message: str) -> bool:
    """Commit the wiki directory as its own repository; False when unchanged.

    Best-effort by design: a missing git binary or a failed command degrades
    to an uncommitted (but fully written) wiki, never an error.
    """
    git = shutil.which("git")
    if git is None:
        return False
    try:
        if not (root / ".git").is_dir():
            subprocess.run(
                [git, "init", "--quiet", str(root)],
                check=True,
                capture_output=True,
            )
        subprocess.run([git, "-C", str(root), "add", "-A"], check=True, capture_output=True)
        result = subprocess.run(
            [
                git,
                "-C",
                str(root),
                "-c",
                "user.name=dexta",
                "-c",
                "user.email=wiki@dexta.local",
                "commit",
                "--quiet",
                "-m",
                message,
            ],
            check=False,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return result.returncode == 0
