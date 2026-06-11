"""Configuration — one TOML file, environment overrides, working defaults.

Vanilla contract: ``dexta init`` writes ``~/.dexta/dexta.toml`` with exactly
two values the user must supply (Nightscout URL + token); every other key
has a default good enough to run. Secrets (API keys) come from the
environment, never from the TOML file, so configs are safe to share when
asking for help.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "AnalysisConfig",
    "Config",
    "DataConfig",
    "DexcomConfig",
    "LLMConfig",
    "WhoopConfig",
    "load_config",
]

DEFAULT_CONFIG_PATH = Path("~/.dexta/dexta.toml")


class _Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DataConfig(_Section):
    backend: str = "sqlite"
    """``sqlite`` (zero-setup quick start) or ``postgres`` (reference deployment)."""
    sqlite_path: Path = Path("~/.dexta/dexta.db")
    database_url: str | None = None
    """Postgres DSN; also settable via the ``DATABASE_URL`` environment variable."""


class NightscoutConfig(_Section):
    url: str = ""
    token: str = ""


class WhoopConfig(_Section):
    access_token: str = ""
    """OAuth access token; settable via the ``WHOOP_ACCESS_TOKEN`` environment variable."""
    refresh_token: str = ""
    """OAuth refresh token - with client id/secret, enables automatic refresh on 401."""
    client_id: str = ""
    client_secret: str = ""


class DexcomConfig(_Section):
    """Dexcom Share credentials (the *user's* login, not a follower account).

    Prefer the ``DEXCOM_USERNAME`` / ``DEXCOM_PASSWORD`` / ``DEXCOM_OUS``
    environment variables over the TOML file - these are real account
    secrets, and configs should stay safe to share.
    """

    username: str = ""
    password: str = ""
    ous: bool = False
    """True for accounts registered outside the US (Dexcom's OUS region)."""


class LLMConfig(_Section):
    provider: str = "anthropic"
    """Any LangChain provider (``anthropic``, ``openai``, ``ollama``, …) or
    ``openrouter`` — one OPENROUTER_API_KEY unlocks every hosted model, the
    lowest-friction BYOM path (model names like ``anthropic/claude-sonnet-4``)."""
    model: str = "claude-sonnet-4-20250514"
    roles: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Per-role overrides, e.g. ``roles.skeptic = {provider="ollama", model="llama3"}``."""


class AnalysisConfig(_Section):
    target_low: int = 70
    target_high: int = 180
    deep_analysis_window_days: int = 90


class Config(_Section):
    data: DataConfig = Field(default_factory=DataConfig)
    nightscout: NightscoutConfig = Field(default_factory=NightscoutConfig)
    whoop: WhoopConfig = Field(default_factory=WhoopConfig)
    dexcom: DexcomConfig = Field(default_factory=DexcomConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)


def load_config(path: Path | None = None) -> Config:
    """Load config from TOML with environment overrides applied.

    Precedence (highest wins): environment variables → TOML file → defaults.
    A missing file is not an error — defaults make the library usable
    programmatically without any setup.
    """
    resolved = (path or DEFAULT_CONFIG_PATH).expanduser()
    raw: dict[str, Any] = {}
    if resolved.is_file():
        with resolved.open("rb") as fh:
            raw = tomllib.load(fh)

    if database_url := os.environ.get("DATABASE_URL"):
        raw.setdefault("data", {})["database_url"] = database_url
        raw["data"].setdefault("backend", "postgres")
    if ns_url := os.environ.get("NIGHTSCOUT_URL"):
        raw.setdefault("nightscout", {})["url"] = ns_url
    if ns_token := os.environ.get("NIGHTSCOUT_TOKEN"):
        raw.setdefault("nightscout", {})["token"] = ns_token
    if whoop_access_token := os.environ.get("WHOOP_ACCESS_TOKEN"):
        raw.setdefault("whoop", {})["access_token"] = whoop_access_token
    if whoop_refresh_token := os.environ.get("WHOOP_REFRESH_TOKEN"):
        raw.setdefault("whoop", {})["refresh_token"] = whoop_refresh_token
    if whoop_client_id := os.environ.get("WHOOP_CLIENT_ID"):
        raw.setdefault("whoop", {})["client_id"] = whoop_client_id
    if whoop_client_secret := os.environ.get("WHOOP_CLIENT_SECRET"):
        raw.setdefault("whoop", {})["client_secret"] = whoop_client_secret
    if dx_username := os.environ.get("DEXCOM_USERNAME"):
        raw.setdefault("dexcom", {})["username"] = dx_username
    if dx_password := os.environ.get("DEXCOM_PASSWORD"):
        raw.setdefault("dexcom", {})["password"] = dx_password
    if dx_ous := os.environ.get("DEXCOM_OUS"):
        # pydantic coerces "true"/"false"/"1"/"0"/"yes"/"no" to bool
        raw.setdefault("dexcom", {})["ous"] = dx_ous

    return Config.model_validate(raw)
