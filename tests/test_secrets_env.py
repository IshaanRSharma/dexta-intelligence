"""~/.dexta/secrets.env loading and Settings UI persistence."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from dexta_intelligence.config import (
    load_config,
    load_secrets_env,
    save_secret,
    secrets_path_for,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_load_secrets_env_does_not_override_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secrets = tmp_path / "secrets.env"
    secrets.write_text('ANTHROPIC_API_KEY="from-file"\n', encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-shell")
    load_secrets_env(secrets)
    assert os.environ["ANTHROPIC_API_KEY"] == "from-shell"


def test_load_secrets_env_fills_missing_vars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    secrets = tmp_path / "secrets.env"
    secrets.write_text('OPENROUTER_API_KEY="sk-or-test"\n', encoding="utf-8")
    load_secrets_env(secrets)
    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-test"


def test_save_secret_writes_0600_and_updates_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    toml = tmp_path / "dexta.toml"
    toml.write_text("[llm]\nprovider = \"anthropic\"\n", encoding="utf-8")
    secrets = secrets_path_for(toml)
    save_secret("ANTHROPIC_API_KEY", "sk-ant-test-key-abcd", path=secrets)
    assert secrets.stat().st_mode & 0o777 == 0o600
    assert "sk-ant-test-key-abcd" in secrets.read_text(encoding="utf-8")
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-test-key-abcd"
    load_config(toml)
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-test-key-abcd"


def test_save_secret_rejects_unknown_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown secret"):
        save_secret("MADE_UP_KEY", "x", path=tmp_path / "secrets.env")
