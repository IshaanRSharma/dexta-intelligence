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
OpenAI, Google DeepMind Gemini, Ollama, Groq, Mistral, …) - we do not write
provider clients. ``provider = "gemini"`` / ``"google"`` is normalized to
LangChain's ``google_genai`` (Gemini via the AI Studio API, ``GOOGLE_API_KEY``).

Local-first paths need no API key and run offline:

    [llm]
    provider = "ollama"            # a local Ollama daemon; model = "llama3"
    # or
    provider = "llamacpp"          # a local weights file; model = the .gguf path
    model = "~/models/llama-3.1-8b-instruct.Q4_K_M.gguf"

``ollama`` honors ``OLLAMA_HOST`` for a non-default/remote daemon; ``llamacpp``
(aliases ``gguf`` / ``local_path``) loads ``model`` as a filesystem path through
llama.cpp (the optional ``local`` extra).
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


#: provider id -> (API-key env var, help url) for single-key cloud providers.
_CLOUD_KEYS: dict[str, tuple[str, str]] = {
    "anthropic": ("ANTHROPIC_API_KEY", "https://console.anthropic.com/settings/keys"),
    "openai": ("OPENAI_API_KEY", "https://platform.openai.com/api-keys"),
}
_GEMINI_ALIASES = ("google_genai", "gemini", "google")
_OLLAMA_ALIASES = ("ollama", "local")
_LLAMACPP_ALIASES = ("llamacpp", "llama_cpp", "gguf", "local_path")


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
        # OpenRouter speaks the OpenAI API, routed through that provider.
        key = _require_key("openrouter", "OPENROUTER_API_KEY", "https://openrouter.ai/keys")
        return init_chat_model(
            spec.model,
            model_provider="openai",
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
            **kwargs,
        )
    if spec.provider in _OLLAMA_ALIASES:
        return _ollama_model(init_chat_model, spec, kwargs)
    if spec.provider in _LLAMACPP_ALIASES:
        return _local_model_file(spec)
    cloud = _cloud_keyed_model(init_chat_model, spec, kwargs)
    if cloud is not None:
        return cloud
    return init_chat_model(spec.model, model_provider=spec.provider, **kwargs)


def _require_key(provider: str, env_var: str, url: str) -> str:
    """Return the env-var value or raise a clear, actionable RuntimeError."""
    key = os.environ.get(env_var)
    if not key:
        msg = f"provider {provider!r} requires the {env_var} environment variable ({url})"
        raise RuntimeError(msg)
    return key


def _cloud_keyed_model(
    init_chat_model: Callable[..., BaseChatModel],
    spec: ModelSpec,
    kwargs: dict[str, Any],
) -> BaseChatModel | None:
    """A cloud provider gated on an API key, or ``None`` if not one of them."""
    if spec.provider in _CLOUD_KEYS:
        env_var, url = _CLOUD_KEYS[spec.provider]
        key = _require_key(spec.provider, env_var, url)
        return init_chat_model(spec.model, model_provider=spec.provider, api_key=key, **kwargs)
    if spec.provider in _GEMINI_ALIASES:
        _require_key("google_genai", "GOOGLE_API_KEY", "https://aistudio.google.com/apikey")
        return init_chat_model(spec.model, model_provider="google_genai", **kwargs)
    return None


def _ollama_model(
    init_chat_model: Callable[..., BaseChatModel],
    spec: ModelSpec,
    kwargs: dict[str, Any],
) -> BaseChatModel:
    """Local Ollama daemon (no key). ``OLLAMA_HOST`` overrides the endpoint."""
    host = os.environ.get("OLLAMA_HOST")
    if host:
        if not host.startswith(("http://", "https://")):
            host = f"http://{host}"
        kwargs.setdefault("base_url", host)
    return init_chat_model(spec.model, model_provider="ollama", **kwargs)


def _local_model_file(spec: ModelSpec) -> BaseChatModel:
    """Load a local weights file (``spec.model`` is a GGUF path) via llama.cpp."""
    from pathlib import Path  # noqa: PLC0415

    path = Path(spec.model).expanduser()
    if not path.is_file():
        msg = (
            f"provider 'llamacpp' expects a path to a local model file; "
            f"{path} is not a file (set [llm].model to a .gguf path)"
        )
        raise RuntimeError(msg)
    try:
        from langchain_community.chat_models import ChatLlamaCpp  # noqa: PLC0415

        cpp_kwargs: dict[str, Any] = {"model_path": str(path)}
        if spec.temperature is not None:
            cpp_kwargs["temperature"] = spec.temperature
        if spec.max_tokens is not None:
            cpp_kwargs["max_tokens"] = spec.max_tokens
        model = ChatLlamaCpp(**cpp_kwargs)
    except ImportError as exc:
        msg = (
            "local model files need the optional 'local' extra: "
            "pip install 'dexta-intelligence[local]'"
        )
        raise RuntimeError(msg) from exc
    return cast("BaseChatModel", model)
