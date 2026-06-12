"""Goal workflows — user objectives pursued by self-pacing background agents.

A user states a want ("reduce my overnight lows"); the model composes it into a
:class:`Goal` with a *deterministic* success metric and a plan of read-only
tool calls. Each background tick measures the metric, runs the plan to refresh
evidence, and records a checkpoint on the goal's progress arc. The metric is
deterministic by design so an ambient agent cannot drift into declaring success
the data does not show. Goals investigate and report; they never prescribe.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from typing import TYPE_CHECKING, Any, Literal

from dexta_intelligence.agents.discovery_tools import DiscoveryToolkit
from dexta_intelligence.analytics.rollups import daily_rollup
from dexta_intelligence.models import Goal, GoalCheckpoint, GoalMetric, GoalStatus, Hypothesis
from dexta_intelligence.stats.core import mean

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.language_models.chat_models import BaseChatModel

    from dexta_intelligence.agents.base import AgentContext
    from dexta_intelligence.models import GlucoseEvent

logger = logging.getLogger(__name__)

Direction = Literal["increase", "decrease"]

__all__ = [
    "METRIC_LABELS",
    "GoalTick",
    "compose_goal",
    "goal_due",
    "measure_metric",
    "tick_goal",
]

_NIGHT_HOURS = (0, 6)
_LOW_MG_DL = 70

METRIC_LABELS: dict[GoalMetric, str] = {
    GoalMetric.TIR: "time in range (%)",
    GoalMetric.NOCTURNAL_TBR: "overnight time below range, 00-06h (%)",
    GoalMetric.TBR: "time below range (%)",
    GoalMetric.MEAN_GLUCOSE: "mean glucose (mg/dL)",
    GoalMetric.CV: "glucose variability, CV (%)",
}

_DEFAULT_DIRECTION: dict[GoalMetric, Direction] = {
    GoalMetric.TIR: "increase",
    GoalMetric.NOCTURNAL_TBR: "decrease",
    GoalMetric.TBR: "decrease",
    GoalMetric.MEAN_GLUCOSE: "decrease",
    GoalMetric.CV: "decrease",
}

_KEYWORD_METRIC: tuple[tuple[frozenset[str], GoalMetric], ...] = (
    (frozenset({"low", "lows", "hypo", "below", "overnight low"}), GoalMetric.NOCTURNAL_TBR),
    (frozenset({"range", "tir", "in-range", "target"}), GoalMetric.TIR),
    (frozenset({"variabilit", "stable", "swing", "spik", "cv", "smooth"}), GoalMetric.CV),
    (
        frozenset({"high", "highs", "hyper", "average", "mean", "a1c", "lower"}),
        GoalMetric.MEAN_GLUCOSE,
    ),
)

_COMPOSE_PROMPT = """A Type-1 patient stated this goal:
  "{statement}"

Pick the single best DETERMINISTIC success metric and a short plan of read-only
tools to investigate it. Metrics: tir | nocturnal_tbr | tbr | mean_glucose | cv.

{tool_schema}

Output STRICT JSON, no prose:
{{"metric": "<one metric>",
  "direction": "increase"|"decrease",
  "cadence_days": <int 1-14>,
  "target": <number or null>,
  "tools": [{{"tool": "<name>", "args": {{...}}}}]}}"""


@dataclass(frozen=True, slots=True)
class GoalTick:
    """Result of advancing one goal by one background cycle."""

    checkpoint: GoalCheckpoint
    achieved: bool


def measure_metric(metric: GoalMetric, ctx: AgentContext) -> float | None:
    """Compute a goal metric over the context window. ``None`` when unmeasurable."""
    start = datetime.combine(ctx.window[0], time.min, tzinfo=UTC)
    end = datetime.combine(ctx.window[1], time.max, tzinfo=UTC)
    glucose = ctx.store.get_glucose(start, end)
    if not glucose:
        return None

    if metric is GoalMetric.NOCTURNAL_TBR:
        night = [g.mg_dl for g in glucose if _NIGHT_HOURS[0] <= g.ts.hour < _NIGHT_HOURS[1]]
        if not night:
            return None
        return round(100.0 * sum(1 for v in night if v < _LOW_MG_DL) / len(night), 2)

    if metric is GoalMetric.MEAN_GLUCOSE:
        return round(mean([float(g.mg_dl) for g in glucose]), 1)
    return _rollup_metric(metric, glucose)


def _rollup_metric(metric: GoalMetric, glucose: Sequence[GlucoseEvent]) -> float | None:
    rollups = _daily_rollups(glucose)
    if not rollups:
        return None
    if metric is GoalMetric.TIR:
        return round(mean([r.tir for r in rollups]), 1)
    if metric is GoalMetric.TBR:
        return round(mean([r.tbr for r in rollups]), 1)
    cvs = [r.cv for r in rollups if r.cv is not None]
    return round(mean(cvs), 1) if cvs else None


def compose_goal(
    statement: str,
    *,
    model: BaseChatModel | None = None,
    now: datetime,
    target: float | None = None,
) -> Goal:
    """Turn a free-text objective into a measurable, tool-backed goal.

    An explicit ``target`` wins; otherwise the LLM compose path may supply one.
    Without a target the goal tracks progress but never auto-flips to ACHIEVED.
    """
    plan = _llm_compose(statement, model) if model is not None else None
    if plan is None:
        plan = _keyword_compose(statement)
    resolved_target = target if target is not None else plan.target
    return Goal(
        statement=statement,
        metric=plan.metric,
        direction=plan.direction,
        target=resolved_target,
        tools=plan.tools,
        cadence_days=plan.cadence_days,
        status=GoalStatus.ACTIVE,
        created_at=now,
    )


def tick_goal(
    goal: Goal,
    ctx: AgentContext,
    *,
    now: datetime,
    model: BaseChatModel | None = None,
) -> GoalTick:
    """Advance a goal: measure the metric, investigate, record a checkpoint.

    With a model, the investigation is a reasoning loop scoped to the goal —
    the model picks which tools to run this cycle. Without one, the goal's
    stored plan is replayed deterministically. Either way the metric value is
    computed deterministically and the note is faithfulness-audited.
    """
    if goal.id is None:
        msg = "tick_goal requires a persisted goal (id is None)"
        raise ValueError(msg)

    value = measure_metric(goal.metric, ctx)
    prior = ctx.store.get_goal_checkpoints(goal.id)
    salient = _investigate(goal, ctx, model)
    note = _progress_note(goal, value, prior_value=_last_value(prior), salient=salient)
    achieved = _is_achieved(goal, value)

    checkpoint = GoalCheckpoint(goal_id=goal.id, ts=now, metric_value=value, note=note)
    return GoalTick(checkpoint=checkpoint, achieved=achieved)


def goal_due(goal: Goal, checkpoints: Sequence[GoalCheckpoint], *, now: datetime) -> bool:
    """Whether the goal's cadence calls for a tick now."""
    if not checkpoints:
        return True
    return now - checkpoints[-1].ts >= timedelta(days=goal.cadence_days)


# ── composition internals ──────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _Plan:
    metric: GoalMetric
    direction: Direction
    cadence_days: int
    tools: list[dict[str, Any]]
    target: float | None = None


def _keyword_compose(statement: str) -> _Plan:
    text = statement.lower()
    metric = GoalMetric.TIR
    for keywords, candidate in _KEYWORD_METRIC:
        if any(k in text for k in keywords):
            metric = candidate
            break
    return _Plan(
        metric=metric,
        direction=_DEFAULT_DIRECTION[metric],
        cadence_days=7,
        tools=_default_tools(metric),
    )


def _llm_compose(statement: str, model: BaseChatModel) -> _Plan | None:
    prompt = _COMPOSE_PROMPT.format(statement=statement, tool_schema=_tool_schema())
    messages = [
        {"role": "system", "content": "Respond with ONE JSON object only, no prose."},
        {"role": "user", "content": prompt},
    ]
    try:
        response = model.invoke(messages)
        data = json.loads(_text_of(response))
    except Exception:
        logger.warning("goal composition LLM failed; using keyword fallback", exc_info=True)
        return None
    try:
        metric = GoalMetric(str(data["metric"]))
    except (KeyError, ValueError):
        return None
    direction: Direction = (
        data["direction"]
        if data.get("direction") in ("increase", "decrease")
        else _DEFAULT_DIRECTION[metric]
    )
    tools = [t for t in data.get("tools", []) if isinstance(t, dict) and "tool" in t]
    cadence = data.get("cadence_days", 7)
    raw_target = data.get("target")
    target = float(raw_target) if isinstance(raw_target, (int, float)) else None
    return _Plan(
        metric=metric,
        direction=direction,
        cadence_days=int(cadence) if isinstance(cadence, int) else 7,
        tools=tools or _default_tools(metric),
        target=target,
    )


def _default_tools(metric: GoalMetric) -> list[dict[str, Any]]:
    if metric is GoalMetric.NOCTURNAL_TBR:
        return [{"tool": "event_proximity", "args": {"event_type": "bolus", "window_min": 180}}]
    if metric is GoalMetric.CV:
        return [{"tool": "tod_compare", "args": {"hours_a": [0, 6], "hours_b": [11, 15]}}]
    return [{"tool": "groupby_compare", "args": {"group_by": "weekend", "target": "tir_pct"}}]


# ── tick internals ─────────────────────────────────────────────────────────────

_INVESTIGATE_SYSTEM = """You are a background health agent working on one goal for a \
Type-1/2 diabetic patient. Investigate ONLY this goal using the tools; then report the single \
most relevant observation in one sentence. Every number must come from a tool result. \
Observation only — never dosing or treatment advice. If nothing notable, say so."""


def _investigate(goal: Goal, ctx: AgentContext, model: BaseChatModel | None) -> str | None:
    if model is None:
        return _run_plan(goal, ctx)

    from dexta_intelligence.agents.discovery_tools import tool_specs  # noqa: PLC0415
    from dexta_intelligence.agents.reason import run_reasoning_loop  # noqa: PLC0415
    from dexta_intelligence.guard.faithfulness import audit  # noqa: PLC0415

    toolkit = DiscoveryToolkit(ctx)
    result = run_reasoning_loop(
        model,
        tool_specs(ctx, toolkit),
        system=_INVESTIGATE_SYSTEM,
        user=f'Goal: "{goal.statement}" — what is this cycle\'s most relevant observation?',
        max_steps=4,
    )
    if result.answer and audit(result.answer, result.evidence).ok:
        return result.answer
    return _run_plan(goal, ctx)


def _run_plan(goal: Goal, ctx: AgentContext) -> str | None:
    """Run the plan's tools; return the most salient effect and bank strong ones."""
    toolkit = DiscoveryToolkit(ctx)
    best: tuple[float, str, str] | None = None
    for call in goal.tools:
        tool = str(call.get("tool", ""))
        args = call.get("args") or {}
        result = toolkit.run(tool, args)
        if not result.ok:
            continue
        delta = result.summary.get("delta")
        if not isinstance(delta, (int, float)):
            continue
        interpretation = str(result.summary.get("interpretation", "n/a"))
        line = (
            f"{tool}: {result.summary.get('label_a', 'A')} vs "
            f"{result.summary.get('label_b', 'B')} differs by {abs(delta)} "
            f"({interpretation} effect)"
        )
        if best is None or abs(delta) > best[0]:
            best = (abs(delta), line, interpretation)
    if best is None:
        return None
    if best[2] in ("moderate", "large"):
        _bank_observation(goal, best[1], ctx)
    return best[1]


def _bank_observation(goal: Goal, line: str, ctx: AgentContext) -> None:
    """Persist a strong goal observation as an open hypothesis, once."""
    statement = f"[goal #{goal.id}] {line}"
    existing = {h.statement for h in ctx.store.get_hypotheses(status="open")}
    if statement not in existing:
        ctx.store.insert_hypothesis(Hypothesis(statement=statement))


def _progress_note(
    goal: Goal,
    value: float | None,
    *,
    prior_value: float | None,
    salient: str | None,
) -> str:
    label = METRIC_LABELS[goal.metric]
    if value is None:
        head = f"Not enough data to measure {label} yet."
    elif prior_value is None:
        head = f"Baseline {label}: {value}."
    else:
        moved = value - prior_value
        toward = (goal.direction == "decrease" and moved < 0) or (
            goal.direction == "increase" and moved > 0
        )
        arrow = "→ toward goal" if toward else "→ away from goal" if moved else "→ no change"
        head = f"{label}: {prior_value} → {value} ({arrow})."
    return f"{head} {salient}" if salient else head


def _is_achieved(goal: Goal, value: float | None) -> bool:
    if value is None or goal.target is None:
        return False
    return value <= goal.target if goal.direction == "decrease" else value >= goal.target


def _last_value(checkpoints: Sequence[GoalCheckpoint]) -> float | None:
    for cp in reversed(checkpoints):
        if cp.metric_value is not None:
            return cp.metric_value
    return None


def _daily_rollups(glucose: Sequence[GlucoseEvent]) -> list[Any]:
    by_day: dict[Any, list[GlucoseEvent]] = {}
    for g in glucose:
        by_day.setdefault(g.ts.date(), []).append(g)
    out = [daily_rollup(day, events) for day, events in by_day.items()]
    return [r for r in out if r is not None]


def _tool_schema() -> str:
    from dexta_intelligence.agents.discovery_tools import TOOL_SCHEMA_FOR_LLM  # noqa: PLC0415

    return TOOL_SCHEMA_FOR_LLM


def _text_of(response: Any) -> str:
    content = getattr(response, "content", response)
    if not isinstance(content, str):
        return str(content)
    stripped = content.strip().removeprefix("```json").removeprefix("```")
    return stripped.removesuffix("```").strip()
