"""Native tool-calling reasoning loop — the model decides which tool to call and when to stop.

Dependency-light: a ``model`` is anything with ``bind_tools(schemas)`` and
``invoke(messages)`` whose response exposes ``.content`` and ``.tool_calls``
(LangChain's ``AIMessage`` shape); messages are plain dicts, so the hot path
never imports ``langchain_core``. Tools are read-only; the union of their
results becomes the evidence pool the faithfulness guard audits the final
answer against.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "ReasoningEvent",
    "ReasoningResult",
    "ToolCall",
    "ToolSpec",
    "run_reasoning_loop",
]

#: Default ceiling on reasoning turns — insurance against a model that loops.
_DEFAULT_MAX_STEPS = 6


@dataclass(frozen=True, slots=True)
class ReasoningEvent:
    """One streamed step of the loop, for live surfaces (SSE, CLI trace).

    ``kind`` is ``tool_call`` (the agent is about to run a tool), ``tool_result``
    (it came back), or ``answer`` (final prose). ``payload`` is JSON-serializable
    so a surface can forward it verbatim; it mirrors the run → tool-call →
    tool-result → text shape of agent-UI streaming protocols.
    """

    kind: str
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """One instrument the model may choose to call.

    ``parameters`` is a JSON Schema object (OpenAI/Anthropic function-calling
    shape). ``fn`` takes the validated argument dict and returns a
    JSON-serializable result plus the numbers it produced — the tuple
    ``(public_result, evidence_numbers)`` so the loop can both show the model
    the result and accumulate the guard's evidence pool.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[[dict[str, Any]], tuple[Any, dict[str, Any]]]

    def schema(self) -> dict[str, Any]:
        """OpenAI-style function schema accepted by ``bind_tools``."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A single executed tool call, for the transparency trace."""

    name: str
    args: dict[str, Any]
    ok: bool
    result: Any


@dataclass
class ReasoningResult:
    """Outcome of one reasoning loop."""

    answer: str
    steps: list[ToolCall] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    stopped_reason: str = "answered"


def run_reasoning_loop(
    model: Any,  # any LangChain chat model — duck-typed on .bind_tools/.invoke
    tools: Sequence[ToolSpec],
    *,
    system: str,
    user: str,
    max_steps: int = _DEFAULT_MAX_STEPS,
    on_event: Callable[[ReasoningEvent], None] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> ReasoningResult:
    """Run the model in a tool-calling loop until it answers or hits the cap.

    Each turn: invoke the model; if it requested tool calls, execute every one
    (read-only), feed the results back, and let it decide again; otherwise its
    text is the answer. The evidence pool grows with every tool result so the
    caller can audit the final prose.

    ``on_event`` (optional) receives a :class:`ReasoningEvent` as each tool runs
    and when the answer lands, so a live surface can stream the agent's path. It
    is best-effort: a failing sink is logged and never breaks the loop.

    ``history`` (optional) is prior conversation turns as ``{role, content}``
    dicts, seeded before the current ``user`` turn so follow-up questions resolve
    against the conversation. The faithfulness guard still runs per-turn against
    this turn's evidence, so history adds context without weakening the rail.
    """
    bound = model.bind_tools([t.schema() for t in tools])
    by_name = {t.name: t for t in tools}
    messages: list[Any] = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user})
    steps: list[ToolCall] = []
    evidence: dict[str, Any] = {}

    for _ in range(max_steps):
        try:
            response = bound.invoke(messages)
        except Exception:
            logger.warning("reasoning: model.invoke failed", exc_info=True)
            return ReasoningResult(
                answer="", steps=steps, evidence=evidence, stopped_reason="model_error"
            )

        tool_calls = list(getattr(response, "tool_calls", None) or [])
        if not tool_calls:
            answer = _text_of(response)
            _emit(on_event, ReasoningEvent("answer", {"text": answer}))
            return ReasoningResult(
                answer=answer,
                steps=steps,
                evidence=evidence,
                stopped_reason="answered",
            )

        messages.append(response)
        for call in tool_calls:
            name = call.get("name", "")
            args = call.get("args") or {}
            _emit(on_event, ReasoningEvent("tool_call", {"name": name, "args": args}))
            spec = by_name.get(name)
            if spec is None:
                result: Any = {"error": f"unknown tool {name!r}"}
                ok = False
            else:
                try:
                    result, numbers = spec.fn(args)
                    ok = not (isinstance(result, dict) and result.get("error"))
                    _merge_evidence(evidence, name, len(steps), numbers)
                except Exception as exc:  # a tool fault must not kill the loop
                    logger.debug("reasoning: tool %s raised: %s", name, exc)
                    result, ok = {"error": f"{type(exc).__name__}: {exc}"}, False
            _emit(on_event, ReasoningEvent("tool_result", {"name": name, "ok": ok}))
            steps.append(ToolCall(name=name, args=args, ok=ok, result=result))
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", name),
                    "content": json.dumps(result, default=str)[:4000],
                }
            )

    return ReasoningResult(
        answer="",
        steps=steps,
        evidence=evidence,
        stopped_reason="max_steps",
    )


def _emit(on_event: Callable[[ReasoningEvent], None] | None, event: ReasoningEvent) -> None:
    """Fire a stream event, best-effort — a failing sink never breaks the loop."""
    if on_event is None:
        return
    try:
        on_event(event)
    except Exception:
        logger.debug("reasoning: on_event sink raised", exc_info=True)


def _merge_evidence(pool: dict[str, Any], name: str, idx: int, numbers: dict[str, Any]) -> None:
    if numbers:
        pool[f"{name}_{idx}"] = numbers


def _text_of(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
    return str(content).strip()
