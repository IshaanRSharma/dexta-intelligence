"""Settings form schema — the contract between ``Config``, TOML, and the GUI.

Each :class:`PanelSchema` maps 1:1 to a ``[section]`` in ``dexta.toml``.
Field ``name`` values must match the corresponding Pydantic model on
:class:`~dexta_intelligence.config.Config`. Environment overrides come from
:const:`~dexta_intelligence.config.ENV_OVERRIDES` — never duplicated here.

The GUI, CLI ``dexta init`` hints, and connector docs should all derive from
this module so labels, types, and persistence stay aligned.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "ANALYSIS_PANEL",
    "DATA_FIELDS",
    "SETTINGS_OVERVIEW",
    "SETTINGS_PANELS",
    "FieldKind",
    "FieldSchema",
    "PanelCategory",
    "PanelSchema",
    "PanelTier",
    "SetupLink",
    "panel_by_key",
    "source_nav",
]

PanelCategory = Literal["connection", "intelligence"]
PanelTier = Literal["official", "unofficial"]


class FieldKind(enum.StrEnum):
    """HTML control + coercion contract for one config key."""

    TEXT = "text"
    PASSWORD = "password"
    EMAIL = "email"
    URL = "url"
    CHECKBOX = "checkbox"
    NUMBER = "number"
    SELECT = "select"


@dataclass(frozen=True, slots=True)
class FieldSchema:
    """One persisted field in a config section."""

    name: str
    label: str
    kind: FieldKind = FieldKind.TEXT
    secret: bool = False
    optional: bool = False
    placeholder: str = ""
    hint: str = ""
    autocomplete: str = "off"
    min: int | None = None
    max: int | None = None
    options: tuple[tuple[str, str], ...] = ()
    """``(value, label)`` pairs when ``kind`` is :attr:`FieldKind.SELECT`."""

    @property
    def input_type(self) -> str:
        if self.kind == FieldKind.CHECKBOX:
            return "checkbox"
        if self.kind == FieldKind.PASSWORD or self.secret:
            return "password"
        if self.kind == FieldKind.EMAIL:
            return "email"
        if self.kind == FieldKind.URL:
            return "url"
        if self.kind == FieldKind.NUMBER:
            return "number"
        return "text"


@dataclass(frozen=True, slots=True)
class SetupLink:
    """External doc or product page shown on a connection panel."""

    label: str
    url: str


@dataclass(frozen=True, slots=True)
class PanelSchema:
    """One settings panel (sidebar entry + form)."""

    key: str
    title: str
    section: str
    fields: tuple[FieldSchema, ...]
    category: PanelCategory = "connection"
    connector: str | None = None
    tier: PanelTier = "official"
    subtitle: str = ""
    note: str = ""
    env_keys: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    """Extra env-only secrets not tied to a single field (``(VAR, label)``)."""
    setup_flows: tuple[tuple[str, ...], ...] = ()
    """Human-readable pipelines, e.g. ``("Tandem", "t:connect", "Nightscout", "Dexta")``."""
    setup_links: tuple[SetupLink, ...] = ()
    """Curated setup guides — rendered as outbound links, never persisted."""


_UNOFFICIAL_BANNER = "unofficial API — may break without notice"

_NS = "https://nightscout.github.io"
_TP = "https://www.tidepool.org"

SETTINGS_PANELS: tuple[PanelSchema, ...] = (
    PanelSchema(
        key="nightscout",
        title="Nightscout",
        section="nightscout",
        connector="nightscout",
        note="Hub for most pump/CGM loops — point Dexta at your public site URL.",
        setup_flows=(
            ("Tandem", "t:connect / tconnectsync", "Nightscout", "Dexta"),
            ("Libre", "Juggluco / xDrip+", "Nightscout", "Dexta"),
            ("OpenAPS / AAPS / Loop", "Nightscout devicestatus", "Dexta"),
        ),
        setup_links=(
            SetupLink("Nightscout — new site setup", f"{_NS}/nightscout/new_user/"),
            SetupLink("Tandem t:connect", "https://www.tandemdiabetes.com/products/tconnect"),
            SetupLink("Tandem uploader guide", f"{_NS}/uploader/setup/editors/tandem/"),
            SetupLink("Juggluco (Libre → Nightscout)", "https://github.com/jkal77/Juggluco"),
            SetupLink("xDrip+ (CGM → Nightscout)", "https://github.com/NightscoutFoundation/xDrip"),
            SetupLink("Android APS (AAPS)", "https://androidaps.readthedocs.io/"),
            SetupLink("Loop (iOS)", "https://loopkit.github.io/loopdocs/"),
            SetupLink("OpenAPS", "https://openaps.org/"),
        ),
        fields=(
            FieldSchema(
                "url",
                "Base URL",
                kind=FieldKind.URL,
                placeholder="https://your-site.herokuapp.com",
                hint="Public Nightscout URL — treatments and glucose arrive through here.",
                autocomplete="url",
            ),
            FieldSchema(
                "token",
                "API token",
                kind=FieldKind.PASSWORD,
                secret=True,
                placeholder="API secret",
                hint="Nightscout API secret (``AUTH_DEFAULT``). Leave blank to keep the stored "
                "value.",
            ),
        ),
    ),
    PanelSchema(
        key="dexcom",
        title="Dexcom Share",
        subtitle="pydexcom",
        section="dexcom",
        connector="dexcom",
        tier="unofficial",
        note="Direct Dexcom Share login — bypasses Nightscout.",
        setup_flows=(("Dexcom G6/G7", "Dexcom Share", "Dexta"),),
        setup_links=(
            SetupLink("Dexcom Share", "https://share2.dexcom.com/"),
            SetupLink("pydexcom (library we use)", "https://github.com/themotleyfool/pydexcom"),
        ),
        fields=(
            FieldSchema(
                "username",
                "Username",
                placeholder="Dexcom Share username",
                hint="The account that owns the CGM — not a follower login.",
                autocomplete="username",
            ),
            FieldSchema(
                "password",
                "Password",
                kind=FieldKind.PASSWORD,
                secret=True,
                autocomplete="current-password",
            ),
            FieldSchema(
                "ous",
                "Account registered outside the US",
                kind=FieldKind.CHECKBOX,
                hint="Enable for Dexcom OUS (non-US) accounts.",
            ),
        ),
    ),
    PanelSchema(
        key="libre",
        title="LibreLinkUp",
        subtitle="Libre",
        section="libre",
        connector="libre",
        tier="unofficial",
        note="LibreLinkUp follower account — or route Libre via Nightscout instead.",
        setup_flows=(
            ("Libre", "LibreLinkUp follower", "Dexta"),
            ("Libre", "Juggluco / xDrip+", "Nightscout", "Dexta"),
        ),
        setup_links=(
            SetupLink("LibreLinkUp", "https://www.libreview.com/"),
            SetupLink("Juggluco → Nightscout", "https://github.com/jkal77/Juggluco"),
            SetupLink("Configure Nightscout instead", f"{_NS}/nightscout/new_user/"),
        ),
        fields=(
            FieldSchema(
                "email",
                "Follower email",
                kind=FieldKind.EMAIL,
                placeholder="follower@example.com",
                autocomplete="email",
            ),
            FieldSchema(
                "password",
                "Password",
                kind=FieldKind.PASSWORD,
                secret=True,
                autocomplete="current-password",
            ),
            FieldSchema(
                "region",
                "Region",
                kind=FieldKind.SELECT,
                options=(
                    ("us", "United States"),
                    ("eu", "Europe"),
                    ("eu2", "Europe 2"),
                    ("ae", "UAE"),
                    ("ap", "Asia Pacific"),
                    ("au", "Australia"),
                    ("ca", "Canada"),
                    ("de", "Germany"),
                    ("fr", "France"),
                    ("jp", "Japan"),
                    ("la", "Latin America"),
                    ("ru", "Russia"),
                ),
                hint="LibreLinkUp API region for this follower account.",
            ),
            FieldSchema(
                "patient_id",
                "Patient ID",
                optional=True,
                placeholder="Optional — first shared patient if empty",
                hint="LibreLinkUp patient UUID when following multiple people.",
            ),
        ),
    ),
    PanelSchema(
        key="whoop",
        title="Whoop",
        section="whoop",
        connector="whoop",
        setup_flows=(("Whoop", "OAuth / API token", "Dexta"),),
        setup_links=(
            SetupLink("Whoop Developer", "https://developer.whoop.com/"),
        ),
        fields=(
            FieldSchema("access_token", "Access token", kind=FieldKind.PASSWORD, secret=True),
            FieldSchema("refresh_token", "Refresh token", kind=FieldKind.PASSWORD, secret=True),
            FieldSchema("client_id", "Client ID"),
            FieldSchema("client_secret", "Client secret", kind=FieldKind.PASSWORD, secret=True),
        ),
    ),
    PanelSchema(
        key="oura",
        title="Oura",
        section="oura",
        connector="oura",
        setup_flows=(("Oura Ring", "Personal access token", "Dexta"),),
        setup_links=(
            SetupLink("Oura Cloud — access tokens", "https://cloud.ouraring.com/"),
        ),
        fields=(
            FieldSchema(
                "access_token",
                "Personal access token",
                kind=FieldKind.PASSWORD,
                secret=True,
                hint="Generate at cloud.ouraring.com → Personal access tokens.",
            ),
        ),
    ),
    PanelSchema(
        key="tidepool",
        title="Tidepool",
        subtitle="JSON export",
        section="tidepool",
        connector="tidepool",
        tier="official",
        note="Offline import — export from tidepool.org, then sync.",
        setup_flows=(
            ("Tidepool", "JSON export", "Dexta"),
            ("Tidepool", "XLSX export → convert to JSON", "Dexta"),
        ),
        setup_links=(
            SetupLink("Tidepool", _TP),
            SetupLink("Export your data", "https://support.tidepool.org/hc/en-us/articles/115001953852"),
            SetupLink("Tidepool data model", "https://developer.tidepool.org/"),
        ),
        fields=(
            FieldSchema(
                "export_path",
                "Export file",
                placeholder="~/Downloads/tidepool-export.json",
                hint="Tidepool → Upload → Export device data (JSON). Sync re-reads this file.",
            ),
        ),
    ),
    PanelSchema(
        key="tandem",
        title="Tandem",
        subtitle="t:connect",
        section="tandem",
        connector="tandem",
        tier="unofficial",
        note="Direct t:slim X2 / Mobi access via the reverse-engineered t:connect cloud — "
        "no Nightscout required.",
        setup_flows=(("Tandem t:slim X2 / Mobi", "t:connect cloud", "tconnectsync", "Dexta"),),
        setup_links=(
            SetupLink("tconnectsync (library we use)", "https://github.com/jwoglom/tconnectsync"),
            SetupLink("pumpx2 (Tandem BT protocol)", "https://github.com/jwoglom/pumpx2"),
            SetupLink(
                "Tandem Source",
                "https://www.tandemdiabetes.com/products/software-apps/tandem-source",
            ),
        ),
        fields=(
            FieldSchema(
                "email",
                "Email",
                kind=FieldKind.EMAIL,
                placeholder="you@example.com",
                autocomplete="email",
            ),
            FieldSchema(
                "password",
                "Password",
                kind=FieldKind.PASSWORD,
                secret=True,
                autocomplete="current-password",
            ),
            FieldSchema(
                "region",
                "Region",
                kind=FieldKind.SELECT,
                options=(
                    ("us", "Tandem Source (US)"),
                    ("eu", "Tandem Source (EU)"),
                ),
                hint="Which Tandem Source region the account is registered against.",
            ),
            FieldSchema(
                "pump_serial",
                "Pump serial (optional)",
                placeholder="12345678",
                hint="Numeric serial on the pump label; leave blank to use the most "
                "recently active pump.",
            ),
        ),
    ),
    PanelSchema(
        key="carelink",
        title="CareLink",
        subtitle="Medtronic",
        section="carelink",
        connector="carelink",
        tier="unofficial",
        note="Medtronic pump + CGM via the CareLink cloud — direct, no Nightscout. Region-split "
        "auth is fragile. The MiniMed 780G exposes no live unofficial API (CareLink export only).",
        setup_flows=(("Medtronic pump + CGM", "CareLink cloud", "carelink connector", "Dexta"),),
        setup_links=(
            SetupLink("CareLink", "https://carelink.minimed.com/"),
            SetupLink(
                "carelink-python-client (community)",
                "https://github.com/ondrej1024/carelink-python-client",
            ),
        ),
        fields=(
            FieldSchema(
                "username",
                "Username",
                placeholder="CareLink username",
                autocomplete="username",
            ),
            FieldSchema(
                "password",
                "Password",
                kind=FieldKind.PASSWORD,
                secret=True,
                autocomplete="current-password",
            ),
            FieldSchema(
                "country",
                "Country",
                placeholder="us",
                hint="ISO country code the CareLink account is registered in.",
            ),
            FieldSchema(
                "patient",
                "Patient",
                optional=True,
                hint="Care-partner accounts: the patient username to pull; empty = your own "
                "account.",
            ),
        ),
    ),
    PanelSchema(
        key="dexcom_api",
        title="Dexcom",
        subtitle="official API",
        section="dexcom_api",
        connector="dexcom_api",
        tier="official",
        note="Dexcom's sanctioned OAuth /egvs API — ToS-clean, ~1-3h delayed. Complements Dexcom "
        "Share for users who want the official integration.",
        setup_links=(
            SetupLink("Dexcom Developer", "https://developer.dexcom.com/"),
            SetupLink(
                "Dexcom v3 endpoints",
                "https://developer.dexcom.com/docs/dexcomv3/endpoint-overview/",
            ),
        ),
        fields=(
            FieldSchema("access_token", "Access token", kind=FieldKind.PASSWORD, secret=True),
            FieldSchema("refresh_token", "Refresh token", kind=FieldKind.PASSWORD, secret=True),
            FieldSchema("client_id", "Client ID"),
            FieldSchema("client_secret", "Client secret", kind=FieldKind.PASSWORD, secret=True),
            FieldSchema(
                "sandbox",
                "Use sandbox host",
                kind=FieldKind.CHECKBOX,
                hint="Target Dexcom's sandbox host instead of production.",
            ),
        ),
    ),
    PanelSchema(
        key="llm",
        title="LLM provider",
        section="llm",
        category="intelligence",
        note="API keys save to ~/.dexta/secrets.env (never in this file).",
        env_keys=(
            ("ANTHROPIC_API_KEY", "Anthropic models"),
            ("OPENROUTER_API_KEY", "OpenRouter (any hosted model)"),
        ),
        fields=(
            FieldSchema(
                "provider",
                "Provider",
                kind=FieldKind.SELECT,
                options=(
                    ("anthropic", "Anthropic"),
                    ("openai", "OpenAI"),
                    ("openrouter", "OpenRouter (BYOM)"),
                    ("ollama", "Ollama (local)"),
                ),
                hint="LangChain provider id. API keys stay in the environment below.",
            ),
            FieldSchema(
                "model",
                "Model",
                placeholder="claude-sonnet-4-6",
                hint="Model slug for the provider (e.g. ``anthropic/claude-sonnet-4-6`` on "
                "OpenRouter).",
            ),
        ),
    ),
    PanelSchema(
        key="evidence",
        title="Literature",
        subtitle="optional",
        section="evidence",
        category="intelligence",
        env_keys=(("OPENEVIDENCE_API_KEY", "OpenEvidence literature"),),
        fields=(
            FieldSchema(
                "backend",
                "Backend",
                kind=FieldKind.SELECT,
                options=(
                    ("pubmed", "PubMed (free, default)"),
                    ("openevidence", "OpenEvidence (API key required)"),
                ),
                hint="Literature source for grounding confirmed patterns.",
            ),
            FieldSchema(
                "email",
                "PubMed contact email",
                kind=FieldKind.EMAIL,
                optional=True,
                placeholder="you@example.com",
                hint="Sent with PubMed requests (NCBI etiquette). Not a secret.",
            ),
            FieldSchema(
                "enabled",
                "Enable literature grounding",
                kind=FieldKind.CHECKBOX,
            ),
        ),
    ),
)

SETTINGS_OVERVIEW: tuple[dict[str, Any], ...] = (
    {
        "title": "Via Nightscout",
        "examples": "Tandem (t:connect), Libre (Juggluco/xDrip+), Loop / AAPS / OpenAPS",
        "panel": "nightscout",
    },
    {
        "title": "Direct",
        "examples": "Dexcom Share, LibreLinkUp, Tidepool JSON export",
        "panel": None,
    },
    {
        "title": "Omnipod (DIY only)",
        "examples": "Omnipod DASH via DIY Loop / AAPS → Nightscout; Omnipod 5 is a closed "
        "ecosystem (no integration)",
        "panel": "nightscout",
    },
)

DATA_FIELDS: tuple[FieldSchema, ...] = (
    FieldSchema(
        "backend",
        "Storage backend",
        kind=FieldKind.SELECT,
        options=(
            ("sqlite", "SQLite (local file, zero setup)"),
            ("postgres", "PostgreSQL (reference deployment)"),
        ),
        hint="Where findings, glucose, and memory live. Switching backends does not migrate data.",
    ),
    FieldSchema(
        "sqlite_path",
        "SQLite path",
        placeholder="~/.dexta/dexta.db",
        hint="Used when backend is SQLite.",
    ),
    FieldSchema(
        "database_url",
        "Postgres DSN",
        optional=True,
        placeholder="postgresql://user:pass@localhost/dexta",
        hint="Used when backend is postgres. ``DATABASE_URL`` env overrides this field.",
    ),
)

ANALYSIS_PANEL: PanelSchema = PanelSchema(
    key="analysis",
    title="Analysis & storage",
    section="analysis",
    category="connection",  # nav group overridden in source_nav()
    fields=(
        FieldSchema("target_low", "Target low (mg/dL)", kind=FieldKind.NUMBER, min=40, max=120),
        FieldSchema("target_high", "Target high (mg/dL)", kind=FieldKind.NUMBER, min=120, max=300),
        FieldSchema(
            "max_reasoning_steps",
            "Max tool calls per question",
            kind=FieldKind.NUMBER,
            min=4,
            max=64,
            hint="How many tools chat can call before stopping (orchestrator loop). Raise for complex multi-step questions.",
        ),
        FieldSchema(
            "deep_analysis_window_days",
            "Analysis window (days)",
            kind=FieldKind.NUMBER,
            min=7,
            max=365,
        ),
    ),
    note=(
        "Memory recall uses lightweight lexical matching at query time — no vector DB or "
        "embedding API keys. Findings persist in the storage backend above."
    ),
)

WIKI_FIELDS: tuple[FieldSchema, ...] = (
    FieldSchema(
        "path",
        "Wiki path",
        placeholder="~/.dexta/wiki",
        hint="Directory where ``dexta wiki`` writes markdown pages.",
    ),
    FieldSchema(
        "git",
        "Commit each generation to git",
        kind=FieldKind.CHECKBOX,
        hint="Keeps a forensic history of belief updates.",
    ),
)

PANELS_BY_KEY: dict[str, PanelSchema] = {p.key: p for p in SETTINGS_PANELS}


def panel_by_key(key: str) -> PanelSchema | None:
    return PANELS_BY_KEY.get(key)


def source_nav() -> tuple[dict[str, Any], ...]:
    """Sidebar catalog derived from the schema (stable even before runtime card views)."""
    items: list[dict[str, Any]] = []
    for spec in SETTINGS_PANELS:
        group = "Intelligence" if spec.category == "intelligence" else "Connections"
        label = spec.title if not spec.subtitle else f"{spec.title} ({spec.subtitle})"
        items.append({"key": spec.key, "label": label, "group": group})
    items.append({"key": "analysis", "label": ANALYSIS_PANEL.title, "group": "System"})
    return tuple(items)


def panel_display_title(spec: PanelSchema) -> str:
    if spec.subtitle:
        return f"{spec.title} ({spec.subtitle})"
    return spec.title


def panel_banner(spec: PanelSchema) -> str:
    return _UNOFFICIAL_BANNER if spec.tier == "unofficial" else ""
