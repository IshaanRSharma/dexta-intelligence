"""Configuration — one TOML file, environment overrides, working defaults.

Vanilla contract: ``dexta init`` writes ``~/.dexta/dexta.toml`` with exactly
two values the user must supply (Nightscout URL + token); every other key
has a default good enough to run. Secrets (API keys) live in
``~/.dexta/secrets.env`` (or shell env vars), never in the TOML file, so
configs are safe to share when asking for help.
"""

from __future__ import annotations

import contextlib
import enum
import os
import tempfile
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "AnalysisConfig",
    "CareLinkConfig",
    "Config",
    "DataConfig",
    "DexcomApiConfig",
    "DexcomConfig",
    "EvidenceConfig",
    "LLMConfig",
    "LensConfig",
    "LibreConfig",
    "LibreRegion",
    "OuraConfig",
    "ServerConfig",
    "TandemConfig",
    "TidepoolConfig",
    "WhoopConfig",
    "WikiConfig",
    "env_override_for",
    "load_config",
    "load_secrets_env",
    "save_config_values",
    "save_secret",
    "secret_is_set",
    "secrets_path_for",
]

DEFAULT_CONFIG_PATH = Path("~/.dexta/dexta.toml")
SECRETS_FILENAME = "secrets.env"

#: Keys the Settings UI may persist via :func:`save_secret`.
PANEL_SECRET_VARS: frozenset[str] = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENEVIDENCE_API_KEY",
    }
)


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


class OuraConfig(_Section):
    access_token: str = ""
    """Personal access token; settable via the ``OURA_ACCESS_TOKEN`` environment variable."""


class TidepoolConfig(_Section):
    export_path: Path = Path("")
    """Path to a Tidepool JSON export (tidepool.org → Upload → Export)."""


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


class LibreRegion(enum.StrEnum):
    """LibreLinkUp regional API hosts - mirrors pylibrelinkup's ``APIUrl`` identifiers."""

    US = "us"
    EU = "eu"
    EU2 = "eu2"
    AE = "ae"
    AP = "ap"
    AU = "au"
    CA = "ca"
    DE = "de"
    FR = "fr"
    JP = "jp"
    LA = "la"
    RU = "ru"


class LibreConfig(_Section):
    """LibreLinkUp follower credentials (a follower account that accepted a
    sharing invitation, not the wearer's own LibreLink login).

    Prefer the ``LIBRE_EMAIL`` / ``LIBRE_PASSWORD`` / ``LIBRE_REGION``
    environment variables over the TOML file - these are real account
    secrets, and configs should stay safe to share.
    """

    email: str = ""
    password: str = ""
    region: LibreRegion = LibreRegion.US
    """Regional API host the account was registered against (``us``, ``eu``, …)."""
    patient_id: str = ""
    """LibreLinkUp patient UUID to follow; empty selects the account's first patient."""


class TandemConfig(_Section):
    """Tandem t:slim X2 via the t:connect cloud (tconnectsync) — **direct pump
    access, no Nightscout required**.

    Unofficial/reverse-engineered API: opt-in, may break without notice. Prefer
    the ``TANDEM_EMAIL`` / ``TANDEM_PASSWORD`` environment variables over the
    TOML file — these are real account secrets.
    """

    email: str = ""
    password: str = ""
    region: str = "us"
    """``us`` (Tandem Source US) or ``eu`` (Tandem Source EU)."""
    pump_serial: str = ""
    """Optional pump serial; empty selects the pump with the most recent data."""


class CareLinkConfig(_Section):
    """Medtronic pump + CGM via CareLink — **direct pump access, no Nightscout
    required**.

    Unofficial/reverse-engineered API: opt-in, may break without notice; auth is
    region-split and fragile. Prefer the ``CARELINK_USERNAME`` /
    ``CARELINK_PASSWORD`` environment variables over the TOML file.
    """

    username: str = ""
    password: str = ""
    country: str = "us"
    """ISO country code the CareLink account is registered in (``us``, ``gb``, …)."""
    patient: str = ""
    """For care-partner accounts: the patient username to pull; empty = own account."""


class DexcomApiConfig(_Section):
    """Dexcom **official** API (OAuth2 ``/egvs``) — ToS-clean, ~1-3h delayed.

    Complements the reverse-engineered Share path (:class:`DexcomConfig`) for
    users who want the sanctioned integration. Token-based in OSS (no embedded
    web OAuth flow); prefer ``DEXCOM_API_*`` environment variables.
    """

    access_token: str = ""
    refresh_token: str = ""
    client_id: str = ""
    client_secret: str = ""
    sandbox: bool = False
    """True targets Dexcom's sandbox host instead of production."""


class LLMConfig(_Section):
    provider: str = "anthropic"
    """Any LangChain provider (``anthropic``, ``openai``, ``ollama``, …) or
    ``openrouter`` — one OPENROUTER_API_KEY unlocks every hosted model, the
    lowest-friction BYOM path (model names like ``anthropic/claude-sonnet-4``)."""
    model: str = "claude-sonnet-4-6"
    roles: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Per-role overrides, e.g. ``roles.skeptic = {provider="ollama", model="llama3"}``."""


class AnalysisConfig(_Section):
    target_low: int = 70
    target_high: int = 180
    max_reasoning_steps: int = Field(default=20, ge=4, le=64)
    """Max tool calls per chat / orchestrator question before the loop stops."""
    deep_analysis_window_days: int = 90
    timezone: str = "UTC"
    """IANA zone for date/time-of-day bucketing (e.g. ``America/New_York``).
    Storage stays UTC; this only sets how days and clock-hours are grouped, so
    "dinner"/"overnight"/per-day analysis lands at the patient's local time. Set
    it to your zone in ``dexta.toml``; unknown values fall back to UTC."""


class LensConfig(_Section):
    """A named agent route: a subset of producers plus an optional window.

    Built-ins live in :mod:`dexta_intelligence.workflows.lenses`; user
    ``[lens.<name>]`` entries override or extend them. The skeptic post-pass is
    never listed here — it is non-routable and always appended at build time.
    """

    agents: list[str]
    window_days: int | None = None
    """Override ``[analysis].deep_analysis_window_days`` for this lens (or inherit)."""


class EvidenceConfig(_Section):
    """Clinical-literature grounding for confirmed personal patterns.

    ``backend`` selects the provider (``pubmed`` is zero-auth and the default;
    ``openevidence`` is gated behind ``OPENEVIDENCE_API_KEY``). ``email`` is the
    NCBI-etiquette contact sent with PubMed requests — recommended, never a
    secret, so it lives in the shareable TOML rather than the environment.
    """

    backend: str = "pubmed"
    email: str = ""
    enabled: bool = True


class WikiConfig(_Section):
    path: Path = Path("~/.dexta/wiki")
    """Where ``dexta wiki`` writes the generated knowledge base."""
    git: bool = True
    """Commit each generation so ``git log`` is the forensic belief history."""


class ServerConfig(_Section):
    auto_sync_minutes: int = 0
    """When > 0, ``dexta serve`` runs a background sync every N minutes so the GUI
    alone keeps the local DB fresh (0 = off). It reuses the configured connectors
    and storage backend, so any source the user sets up — Nightscout, a pump, or a
    self-hosted Postgres — is covered without extra wiring. For the full cadence
    (sync + monitor + goal ticks) use ``dexta daemon`` instead."""


class Config(_Section):
    data: DataConfig = Field(default_factory=DataConfig)
    nightscout: NightscoutConfig = Field(default_factory=NightscoutConfig)
    whoop: WhoopConfig = Field(default_factory=WhoopConfig)
    oura: OuraConfig = Field(default_factory=OuraConfig)
    dexcom: DexcomConfig = Field(default_factory=DexcomConfig)
    dexcom_api: DexcomApiConfig = Field(default_factory=DexcomApiConfig)
    libre: LibreConfig = Field(default_factory=LibreConfig)
    tidepool: TidepoolConfig = Field(default_factory=TidepoolConfig)
    tandem: TandemConfig = Field(default_factory=TandemConfig)
    carelink: CareLinkConfig = Field(default_factory=CareLinkConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    evidence: EvidenceConfig = Field(default_factory=EvidenceConfig)
    wiki: WikiConfig = Field(default_factory=WikiConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    lens: dict[str, LensConfig] = Field(default_factory=dict)


def secrets_path_for(config_path: Path | None = None) -> Path:
    """``secrets.env`` beside the config file (default ``~/.dexta/secrets.env``)."""
    return (config_path or DEFAULT_CONFIG_PATH).expanduser().parent / SECRETS_FILENAME


def _parse_secrets_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[7:].strip()
    if "=" not in stripped:
        return None
    key, _, value = stripped.partition("=")
    key = key.strip()
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    if not key:
        return None
    return key, value


def load_secrets_env(path: Path | None = None) -> None:
    """Load ``secrets.env`` into ``os.environ`` without overriding existing vars."""
    resolved = (path or secrets_path_for()).expanduser()
    if not resolved.is_file():
        return
    for line in resolved.read_text(encoding="utf-8").splitlines():
        parsed = _parse_secrets_line(line)
        if parsed is None:
            continue
        key, value = parsed
        if key not in os.environ:
            os.environ[key] = value


def secret_is_set(name: str) -> bool:
    return bool(os.environ.get(name))


def save_secret(name: str, value: str, *, path: Path | None = None) -> None:
    """Persist one secret to ``secrets.env`` (mode 0600) and refresh ``os.environ``."""
    if name not in PANEL_SECRET_VARS:
        msg = f"unknown secret {name!r}"
        raise ValueError(msg)
    resolved = (path or secrets_path_for()).expanduser()
    secrets: dict[str, str] = {}
    if resolved.is_file():
        for line in resolved.read_text(encoding="utf-8").splitlines():
            parsed = _parse_secrets_line(line)
            if parsed is None:
                continue
            key, existing = parsed
            secrets[key] = existing
    if value:
        secrets[name] = value
        os.environ[name] = value
    else:
        secrets.pop(name, None)
        os.environ.pop(name, None)
    lines = [
        "# Dexta Intelligence secrets — mode 0600, never commit or share",
        "# Edit here or paste keys in Settings → LLM / Literature",
        "",
    ]
    for key in sorted(secrets):
        lines.append(f"{key}={_toml_value(secrets[key])}")
    text = "\n".join(lines) + "\n"
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=resolved.parent, prefix=".secrets.env.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.chmod(tmp, 0o600)
        os.replace(tmp, resolved)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def load_config(path: Path | None = None) -> Config:
    """Load config from TOML with environment overrides applied.

    Precedence (highest wins): shell environment → ``secrets.env`` → TOML → defaults.
    A missing file is not an error — defaults make the library usable
    programmatically without any setup.
    """
    resolved = (path or DEFAULT_CONFIG_PATH).expanduser()
    load_secrets_env(secrets_path_for(resolved))
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
    _apply_oura_env(raw)
    if dx_username := os.environ.get("DEXCOM_USERNAME"):
        raw.setdefault("dexcom", {})["username"] = dx_username
    if dx_password := os.environ.get("DEXCOM_PASSWORD"):
        raw.setdefault("dexcom", {})["password"] = dx_password
    if dx_ous := os.environ.get("DEXCOM_OUS"):
        # pydantic coerces "true"/"false"/"1"/"0"/"yes"/"no" to bool
        raw.setdefault("dexcom", {})["ous"] = dx_ous
    _apply_libre_env(raw)
    _apply_pump_env(raw)
    if max_steps := os.environ.get("DEXTA_MAX_REASONING_STEPS"):
        raw.setdefault("analysis", {})["max_reasoning_steps"] = int(max_steps)

    return Config.model_validate(raw)


def _apply_pump_env(raw: dict[str, Any]) -> None:
    """Direct-pump + official-Dexcom environment overrides (no Nightscout path)."""
    if tandem_email := os.environ.get("TANDEM_EMAIL"):
        raw.setdefault("tandem", {})["email"] = tandem_email
    if tandem_password := os.environ.get("TANDEM_PASSWORD"):
        raw.setdefault("tandem", {})["password"] = tandem_password
    if tandem_region := os.environ.get("TANDEM_REGION"):
        raw.setdefault("tandem", {})["region"] = tandem_region.lower()
    if tandem_serial := os.environ.get("TANDEM_PUMP_SERIAL"):
        raw.setdefault("tandem", {})["pump_serial"] = tandem_serial.strip()
    if carelink_username := os.environ.get("CARELINK_USERNAME"):
        raw.setdefault("carelink", {})["username"] = carelink_username
    if carelink_password := os.environ.get("CARELINK_PASSWORD"):
        raw.setdefault("carelink", {})["password"] = carelink_password
    if carelink_country := os.environ.get("CARELINK_COUNTRY"):
        raw.setdefault("carelink", {})["country"] = carelink_country.lower()
    if dx_api_token := os.environ.get("DEXCOM_API_ACCESS_TOKEN"):
        raw.setdefault("dexcom_api", {})["access_token"] = dx_api_token
    if dx_api_refresh := os.environ.get("DEXCOM_API_REFRESH_TOKEN"):
        raw.setdefault("dexcom_api", {})["refresh_token"] = dx_api_refresh


def _apply_oura_env(raw: dict[str, Any]) -> None:
    """``OURA_*`` environment overrides for the ``[oura]`` section."""
    if oura_access_token := os.environ.get("OURA_ACCESS_TOKEN"):
        raw.setdefault("oura", {})["access_token"] = oura_access_token


def _apply_libre_env(raw: dict[str, Any]) -> None:
    """``LIBRE_*`` environment overrides for the ``[libre]`` section."""
    if libre_email := os.environ.get("LIBRE_EMAIL"):
        raw.setdefault("libre", {})["email"] = libre_email
    if libre_password := os.environ.get("LIBRE_PASSWORD"):
        raw.setdefault("libre", {})["password"] = libre_password
    if libre_region := os.environ.get("LIBRE_REGION"):
        # region identifiers are case-insensitive in the wild ("EU2" == "eu2")
        raw.setdefault("libre", {})["region"] = libre_region.lower()


#: Environment overrides recognized by :func:`load_config`, keyed by
#: ``(section, field)``. Kept in lockstep with the override block above so the
#: GUI can label env-managed fields without re-deriving the mapping.
ENV_OVERRIDES: dict[tuple[str, str], str] = {
    ("data", "database_url"): "DATABASE_URL",
    ("nightscout", "url"): "NIGHTSCOUT_URL",
    ("nightscout", "token"): "NIGHTSCOUT_TOKEN",
    ("whoop", "access_token"): "WHOOP_ACCESS_TOKEN",
    ("whoop", "refresh_token"): "WHOOP_REFRESH_TOKEN",
    ("whoop", "client_id"): "WHOOP_CLIENT_ID",
    ("whoop", "client_secret"): "WHOOP_CLIENT_SECRET",
    ("oura", "access_token"): "OURA_ACCESS_TOKEN",
    ("dexcom", "username"): "DEXCOM_USERNAME",
    ("dexcom", "password"): "DEXCOM_PASSWORD",
    ("dexcom", "ous"): "DEXCOM_OUS",
    ("libre", "email"): "LIBRE_EMAIL",
    ("libre", "password"): "LIBRE_PASSWORD",
    ("libre", "region"): "LIBRE_REGION",
    ("tandem", "email"): "TANDEM_EMAIL",
    ("tandem", "password"): "TANDEM_PASSWORD",
    ("tandem", "region"): "TANDEM_REGION",
    ("tandem", "pump_serial"): "TANDEM_PUMP_SERIAL",
    ("carelink", "username"): "CARELINK_USERNAME",
    ("carelink", "password"): "CARELINK_PASSWORD",
    ("carelink", "country"): "CARELINK_COUNTRY",
    ("dexcom_api", "access_token"): "DEXCOM_API_ACCESS_TOKEN",
    ("dexcom_api", "refresh_token"): "DEXCOM_API_REFRESH_TOKEN",
    ("analysis", "max_reasoning_steps"): "DEXTA_MAX_REASONING_STEPS",
}


def env_override_for(section: str, field: str) -> str | None:
    """The env var currently overriding ``[section].field``, or ``None``."""
    var = ENV_OVERRIDES.get((section, field))
    return var if var is not None and os.environ.get(var) else None


def save_config_values(updates: dict[str, dict[str, Any]], path: Path | None = None) -> None:
    """Merge per-section field updates into the TOML file atomically.

    Reads the existing file verbatim (no env overrides), merges ``updates``,
    and re-validates the exact serialized bytes through the loader before
    committing via tempfile + ``os.replace`` (mode 0600) — a failed
    validation never touches the existing file.
    """
    resolved = (path or DEFAULT_CONFIG_PATH).expanduser()
    raw: dict[str, Any] = {}
    if resolved.is_file():
        with resolved.open("rb") as fh:
            raw = tomllib.load(fh)
    for section, fields in updates.items():
        existing = raw.get(section)
        if not isinstance(existing, dict):
            existing = {}
            raw[section] = existing
        existing.update(fields)

    text = _dump_toml(raw)
    Config.model_validate(tomllib.loads(text))

    resolved.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=resolved.parent, prefix=".dexta.toml.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.chmod(tmp, 0o600)
        os.replace(tmp, resolved)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, str):
        escaped = (
            value.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
        )
        return f'"{escaped}"'
    msg = f"unsupported TOML value type: {type(value).__name__}"
    raise TypeError(msg)


def _dump_toml(doc: dict[str, Any]) -> str:
    """Serialize a parsed-TOML-shaped dict back to TOML (flat known schema)."""
    lines = ["# dexta-intelligence configuration", ""]

    def emit(name: str, table: dict[str, Any]) -> None:
        scalars = {k: v for k, v in table.items() if not isinstance(v, dict)}
        subtables = {k: v for k, v in table.items() if isinstance(v, dict)}
        if scalars or not subtables:
            lines.append(f"[{name}]")
            lines.extend(f"{key} = {_toml_value(v)}" for key, v in scalars.items())
            lines.append("")
        for key, sub in subtables.items():
            emit(f"{name}.{key}", sub)

    for key, table in doc.items():
        if isinstance(table, dict):
            emit(key, table)
    return "\n".join(lines)
