"""BYOM model factory - the only place a chat model is ever constructed.

Every LLM call site in the system asks for a model **by role**, never by
provider. Vanilla mode runs everything on the single configured model;
power users override per role in ``dexta.toml``::

    [llm]
    provider = "anthropic"
    model = "claude-sonnet-4-6"

    [llm.roles.skeptic]
    provider = "ollama"
    model = "llama3"

Provider support comes from LangChain's ``init_chat_model`` (Anthropic,
OpenAI, Ollama, Groq, Mistral, …) - we do not write provider clients.
``provider = "openrouter"`` is special-cased onto the OpenAI-compatible
endpoint, so one ``OPENROUTER_API_KEY`` unlocks every model on OpenRouter
(``model = "anthropic/claude-sonnet-4"``, ``"openai/gpt-4o"``,
``"meta-llama/llama-3.3-70b-instruct"``, …) - the lowest-friction BYOM path.
The ``llm`` extra is optional: deterministic agents import nothing from
this module, so ``pip install dexta-intelligence`` works with no LLM at all.

CI enforces the boundary: ``ChatAnthropic|ChatOpenAI|init_chat_model`` may
not appear outside ``dexta_intelligence/llm/``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.language_models.chat_models import BaseChatModel

__all__ = ["ROLE_DEFAULTS", "ModelSpec", "RoleDefaults", "get_model"]


@dataclass(frozen=True, slots=True)
class RoleDefaults:
    """Sampling defaults for one role. ``None`` means provider default."""

    temperature: float | None
    max_tokens: int | None


#: Known LLM roles and their sampling defaults. Deterministic agents
#: (observation, pattern, basal, meal, correction, rollups, analytics)
#: have NO role here on purpose - they are not allowed to ask for a model.
ROLE_DEFAULTS: dict[str, RoleDefaults] = {
    "plan": RoleDefaults(temperature=0.0, max_tokens=1024),
    "discovery": RoleDefaults(temperature=0.2, max_tokens=1800),
    "skeptic": RoleDefaults(temperature=0.0, max_tokens=1200),
    "research": RoleDefaults(temperature=0.2, max_tokens=1200),
    "brief": RoleDefaults(temperature=0.2, max_tokens=2200),
    "explain": RoleDefaults(temperature=0.2, max_tokens=1500),
    "polish": RoleDefaults(temperature=0.2, max_tokens=600),
}


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Resolved (provider, model, sampling) for one role."""

    provider: str
    model: str
    temperature: float | None
    max_tokens: int | None


def resolve_spec(
    role: str,
    *,
    provider: str,
    model: str,
    role_overrides: dict[str, dict[str, Any]] | None = None,
) -> ModelSpec:
    """Merge global config, role defaults, and per-role overrides."""
    if role not in ROLE_DEFAULTS:
        known = ", ".join(sorted(ROLE_DEFAULTS))
        msg = f"unknown LLM role {role!r} (known roles: {known})"
        raise KeyError(msg)
    defaults = ROLE_DEFAULTS[role]
    override = (role_overrides or {}).get(role, {})
    return ModelSpec(
        provider=str(override.get("provider", provider)),
        model=str(override.get("model", model)),
        temperature=override.get("temperature", defaults.temperature),
        max_tokens=override.get("max_tokens", defaults.max_tokens),
    )


def get_model(spec: ModelSpec) -> BaseChatModel:
    """Construct the chat model for a resolved spec.

    Raises a clear, actionable error when the optional ``llm`` extra is not
    installed instead of an opaque ``ModuleNotFoundError`` deep in a run.
    """
    try:
        # Deliberately lazy: LLM support is an optional extra and deterministic
        # agents must be importable without it.
        from langchain.chat_models import init_chat_model as _init  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - import-path guard
        msg = (
            "LLM support is not installed. "
            "Install it with: pip install 'dexta-intelligence[llm]'"
        )
        raise RuntimeError(msg) from exc

    # init_chat_model is typed -> Any; pin the return so callers get BaseChatModel.
    init_chat_model = cast("Callable[..., BaseChatModel]", _init)

    kwargs: dict[str, Any] = {}
    if spec.temperature is not None:
        kwargs["temperature"] = spec.temperature
    if spec.max_tokens is not None:
        kwargs["max_tokens"] = spec.max_tokens

    if spec.provider == "openrouter":
        # OpenRouter speaks the OpenAI API; route through the openai provider
        # with its endpoint + key. One key, every model - the BYOM default.
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            msg = (
                "provider 'openrouter' requires the OPENROUTER_API_KEY "
                "environment variable (get one at https://openrouter.ai/keys)"
            )
            raise RuntimeError(msg)
        return init_chat_model(
            spec.model,
            model_provider="openai",
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
            **kwargs,
        )

    if spec.provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            msg = (
                "provider 'anthropic' requires the ANTHROPIC_API_KEY "
                "environment variable (https://console.anthropic.com/settings/keys)"
            )
            raise RuntimeError(msg)
        return init_chat_model(
            spec.model,
            model_provider=spec.provider,
            api_key=key,
            **kwargs,
        )

    if spec.provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            msg = (
                "provider 'openai' requires the OPENAI_API_KEY "
                "environment variable (https://platform.openai.com/api-keys)"
            )
            raise RuntimeError(msg)
        return init_chat_model(
            spec.model,
            model_provider=spec.provider,
            api_key=key,
            **kwargs,
        )

    return init_chat_model(spec.model, model_provider=spec.provider, **kwargs)
