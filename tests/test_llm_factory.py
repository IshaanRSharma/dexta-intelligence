"""Provider construction in the BYOM model factory.

The role-resolution tests (test_cli_roles.py) mock ``get_model`` wholesale;
these exercise ``get_model`` itself by monkeypatching LangChain's
``init_chat_model`` so no real provider client is built. Focus: the Google
DeepMind Gemini branch normalizes its provider id and gates on the key.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from dexta_intelligence.llm.factory import ModelSpec, get_model

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

# get_model imports init_chat_model from langchain; skip cleanly without the extra.
pytest.importorskip("langchain.chat_models")


@pytest.fixture
def captured_init(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    """Patch the lazily-imported init_chat_model; capture model + kwargs."""
    import langchain.chat_models as lcm  # noqa: PLC0415

    captured: dict[str, Any] = {}

    def _fake_init(model: str, **kwargs: Any) -> object:
        captured["model"] = model
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(lcm, "init_chat_model", _fake_init)
    yield captured


def _spec(provider: str) -> ModelSpec:
    return ModelSpec(provider=provider, model="gemini-2.0-flash", temperature=0.2, max_tokens=512)


@pytest.mark.parametrize("provider", ["google_genai", "gemini", "google"])
def test_gemini_aliases_normalize_to_google_genai(
    provider: str, captured_init: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "k-123")
    get_model(_spec(provider))
    assert captured_init["model"] == "gemini-2.0-flash"
    assert captured_init["model_provider"] == "google_genai"
    assert captured_init["temperature"] == 0.2
    assert captured_init["max_tokens"] == 512


def test_gemini_requires_google_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        get_model(_spec("google_genai"))


@pytest.mark.parametrize("provider", ["ollama", "local"])
def test_ollama_needs_no_key_and_uses_ollama_provider(
    provider: str, captured_init: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    get_model(ModelSpec(provider=provider, model="llama3", temperature=0.0, max_tokens=256))
    assert captured_init["model"] == "llama3"
    assert captured_init["model_provider"] == "ollama"
    assert "base_url" not in captured_init  # default daemon, no key required


def test_ollama_host_env_sets_base_url(
    captured_init: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "10.0.0.5:11434")  # bare host:port
    get_model(ModelSpec(provider="ollama", model="llama3", temperature=None, max_tokens=None))
    assert captured_init["base_url"] == "http://10.0.0.5:11434"  # normalized to a URL


def test_local_model_file_requires_existing_path(tmp_path: Path) -> None:
    missing = tmp_path / "nope.gguf"
    with pytest.raises(RuntimeError, match="not a file"):
        get_model(_spec_path("llamacpp", str(missing)))


def test_local_model_file_passes_model_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lcm = pytest.importorskip("langchain_community.chat_models")
    weights = tmp_path / "model.gguf"
    weights.write_bytes(b"\x00")
    captured: dict[str, Any] = {}

    class _FakeChatLlamaCpp:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(lcm, "ChatLlamaCpp", _FakeChatLlamaCpp, raising=False)
    get_model(_spec_path("gguf", str(weights)))
    assert captured["model_path"] == str(weights)
    assert captured["temperature"] == 0.2


def _spec_path(provider: str, path: str) -> ModelSpec:
    return ModelSpec(provider=provider, model=path, temperature=0.2, max_tokens=512)


def test_gemini_listed_as_a_settings_provider() -> None:
    from dexta_intelligence.server.settings_schema import SETTINGS_PANELS  # noqa: PLC0415

    providers: set[str] = set()
    for panel in SETTINGS_PANELS:
        for field in panel.fields:
            if field.name == "provider" and field.options:
                providers.update(value for value, _label in field.options)
    assert "google_genai" in providers
