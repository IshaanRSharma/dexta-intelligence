"""Native tool-calling reasoning loop - the model decides which tool to call and when to stop.

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

    from dexta_intelligence.agents.investigation import BeliefState

logger = logging.getLogger(__name__)

__all__ = [
    "ReasoningEvent",
    "ReasoningResult",
    "ToolCall",
    "ToolSpec",
    "run_reasoning_loop",
]

#: Default ceiling on reasoning turns - insurance against a model that loops.
_DEFAULT_MAX_STEPS = 6


@dataclass(frozen=True, slots=True)
class ReasoningEvent:
    """One streamed step of the loop, for live surfaces (SSE, CLI trace).

    ``kind`` is ``tool_call`` / ``tool_result`` (tool work), ``answer_start`` /
    ``answer_delta`` (token/chunk streaming of the final prose), or ``answer``
    (legacy full-text; prefer deltas for live surfaces). ``payload`` is
    JSON-serializable so a surface can forward it verbatim.
    """

    kind: str
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """One instrument the model may choose to call.

    ``parameters`` is a JSON Schema object (OpenAI/Anthropic function-calling
    shape). ``fn`` takes the validated argument dict and returns a
    JSON-serializable result plus the numbers it produced - the tuple
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
    error_detail: str = ""
    #: The working belief state, when one was threaded through this run.
    belief: BeliefState | None = None


def run_reasoning_loop(
    model: Any,  # any LangChain chat model - duck-typed on .bind_tools/.invoke
    tools: Sequence[ToolSpec],
    *,
    system: str,
    user: str,
    max_steps: int = _DEFAULT_MAX_STEPS,
    on_event: Callable[[ReasoningEvent], None] | None = None,
    history: list[dict[str, Any]] | None = None,
    belief: BeliefState | None = None,
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

    ``belief`` (optional) is a working belief state the model maintains across
    steps: its ``update_belief`` tool is offered alongside ``tools``, the merged
    state streams as a ``belief`` event after each tool round, and the final
    state rides home on the result. It scaffolds the reasoning; it never decides.
    """
    active_tools = [belief.tool(), *tools] if belief is not None else list(tools)
    bound = model.bind_tools([t.schema() for t in active_tools])
    by_name = {t.name: t for t in active_tools}
    messages: list[Any] = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user})
    steps: list[ToolCall] = []
    evidence: dict[str, Any] = {}

    for _ in range(max_steps):
        try:
            response, answer_started = _invoke_model(bound, messages, on_event)
        except Exception as exc:
            logger.warning("reasoning: model call failed", exc_info=True)
            return ReasoningResult(
                answer="",
                steps=steps,
                evidence=evidence,
                stopped_reason="model_error",
                error_detail=_model_error_message(exc),
                belief=belief,
            )

        tool_calls = list(getattr(response, "tool_calls", None) or [])
        if not tool_calls:
            answer = _content_text(response)
            if not answer_started and answer:
                _emit(on_event, ReasoningEvent("answer_start", {}))
                _emit(on_event, ReasoningEvent("answer_delta", {"delta": answer}))
            return ReasoningResult(
                answer=answer,
                steps=steps,
                evidence=evidence,
                stopped_reason="answered",
                belief=belief,
            )

        messages.append(response)
        _run_tool_calls(
            tool_calls,
            by_name,
            steps=steps,
            evidence=evidence,
            messages=messages,
            on_event=on_event,
        )
        if belief is not None:
            _emit(on_event, ReasoningEvent("belief", belief.snapshot()))

    return ReasoningResult(
        answer="",
        steps=steps,
        evidence=evidence,
        stopped_reason="max_steps",
        belief=belief,
    )


def _invoke_model(
    bound: Any,
    messages: list[Any],
    on_event: Callable[[ReasoningEvent], None] | None,
) -> tuple[Any, bool]:
    """One model turn, streaming answer deltas when supported. Returns the
    response and whether answer streaming already began."""
    stream_fn = getattr(bound, "stream", None)
    if stream_fn is None:
        return bound.invoke(messages), False
    chunks: list[Any] = []
    answer_started = False
    for chunk in stream_fn(messages):
        chunks.append(chunk)
        delta = _chunk_text(chunk)
        if delta:
            if not answer_started:
                _emit(on_event, ReasoningEvent("answer_start", {}))
                answer_started = True
            _emit(on_event, ReasoningEvent("answer_delta", {"delta": delta}))
    response = _merge_chunks(chunks) if chunks else bound.invoke(messages)
    return response, answer_started


def _run_tool_calls(
    tool_calls: list[Any],
    by_name: dict[str, ToolSpec],
    *,
    steps: list[ToolCall],
    evidence: dict[str, Any],
    messages: list[Any],
    on_event: Callable[[ReasoningEvent], None] | None,
) -> None:
    """Execute every requested tool call, appending steps and tool messages."""
    for call in tool_calls:
        name = call.get("name", "")
        args = call.get("args") or {}
        _emit(on_event, ReasoningEvent("tool_call", {"name": name, "args": args}))
        result, ok = _execute_tool(by_name.get(name), name, args, len(steps), evidence)
        _emit(on_event, ReasoningEvent("tool_result", {"name": name, "ok": ok}))
        steps.append(ToolCall(name=name, args=args, ok=ok, result=result))
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.get("id", name),
                "content": json.dumps(result, default=str)[:4000],
            }
        )


def _execute_tool(
    spec: ToolSpec | None,
    name: str,
    args: dict[str, Any],
    idx: int,
    evidence: dict[str, Any],
) -> tuple[Any, bool]:
    """Run one tool; a fault becomes an error result, never an exception."""
    if spec is None:
        return {"error": f"unknown tool {name!r}"}, False
    try:
        result, numbers = spec.fn(args)
        ok = not (isinstance(result, dict) and result.get("error"))
        _merge_evidence(evidence, name, idx, numbers)
    except Exception as exc:  # a tool fault must not kill the loop
        logger.debug("reasoning: tool %s raised: %s", name, exc)
        return {"error": f"{type(exc).__name__}: {exc}"}, False
    return result, ok


def _model_error_message(exc: Exception) -> str:
    """Turn provider failures into something actionable in chat."""
    body = str(exc).strip()
    lowered = body.lower()
    if "credit balance is too low" in lowered or (
        "insufficient" in lowered and "credit" in lowered
    ):
        return (
            "Your Anthropic account has no API credits. Add billing at "
            "https://console.anthropic.com/settings/billing, or switch Settings → "
            "LLM provider to OpenRouter/Ollama."
        )
    if "invalid x-api-key" in lowered or "authentication" in lowered or "401" in body:
        return (
            "The API key was rejected. Check Settings → LLM provider and update "
            "your key in ~/.dexta/secrets.env."
        )
    if "model" in lowered and ("not found" in lowered or "does not exist" in lowered):
        return (
            "The configured model is no longer available from Anthropic. "
            "Update Settings → LLM provider → Model to claude-sonnet-4-6."
        )
    if body:
        return f"The language model call failed: {body[:240]}"
    return "The language model is unavailable right now."


def _emit(on_event: Callable[[ReasoningEvent], None] | None, event: ReasoningEvent) -> None:
    """Fire a stream event, best-effort - a failing sink never breaks the loop."""
    if on_event is None:
        return
    try:
        on_event(event)
    except Exception:
        logger.debug("reasoning: on_event sink raised", exc_info=True)


def _merge_evidence(pool: dict[str, Any], name: str, idx: int, numbers: dict[str, Any]) -> None:
    if numbers:
        pool[f"{name}_{idx}"] = numbers


def _chunk_text(chunk: Any) -> str:
    """Incremental text from one streamed model chunk."""
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _merge_chunks(chunks: list[Any]) -> Any:
    """Fold streamed chunks into one response-shaped object."""
    if not chunks:
        return type("_Empty", (), {"content": "", "tool_calls": []})()
    merged = chunks[0]
    for chunk in chunks[1:]:
        add = getattr(merged, "__add__", None)
        if callable(add):
            try:
                merged = add(chunk)
                continue
            except TypeError:
                pass
        break
    else:
        return merged
    content = "".join(_chunk_text(c) for c in chunks)
    tool_calls: list[Any] = []
    for chunk in chunks:
        tool_calls.extend(getattr(chunk, "tool_calls", None) or [])
    return type("_Merged", (), {"content": content, "tool_calls": tool_calls})()


def _content_text(response: Any) -> str:
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
