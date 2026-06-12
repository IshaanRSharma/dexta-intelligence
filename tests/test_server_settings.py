"""Settings & credentials page — per-source cards, TOML writes, secret hygiene.

Drive the FastAPI app through TestClient against a tmp_path-pointed config
file. No network: the test-connection endpoint is exercised through a fake
connector patched in via the ``build_connectors`` factory seam.
"""

from __future__ import annotations

import tomllib
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from dexta_intelligence.config import Config, load_config, save_config_values
from dexta_intelligence.connectors.base import HealthReport
from dexta_intelligence.models import RawEvent
from dexta_intelligence.server import create_app
from dexta_intelligence.store import SQLiteStore

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dexta_intelligence.store.port import StoragePort

_ENV_VARS = (
    "NIGHTSCOUT_URL",
    "NIGHTSCOUT_TOKEN",
    "DEXCOM_USERNAME",
    "DEXCOM_PASSWORD",
    "DEXCOM_OUS",
    "LIBRE_EMAIL",
    "LIBRE_PASSWORD",
    "LIBRE_REGION",
    "WHOOP_ACCESS_TOKEN",
    "WHOOP_REFRESH_TOKEN",
    "WHOOP_CLIENT_ID",
    "WHOOP_CLIENT_SECRET",
    "OURA_ACCESS_TOKEN",
    "DATABASE_URL",
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENEVIDENCE_API_KEY",
    "DEXTA_ALLOW_REMOTE_SETTINGS",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _opener(db_path: Path) -> Callable[[Config, Path | None], StoragePort]:
    def _open(_config: Config, _db: Path | None = None) -> StoragePort:
        store = SQLiteStore(db_path)
        store.migrate()
        return store

    return _open


def _client(tmp_path: Path, *, host: str = "127.0.0.1") -> tuple[TestClient, Path]:
    db = tmp_path / "settings.db"
    store = SQLiteStore(db)
    store.migrate()
    store.close()
    toml_path = tmp_path / "dexta.toml"
    app = create_app(Config(), store_opener=_opener(db), config_path=toml_path, host=host)
    return TestClient(app), toml_path


def _card(body: str, key: str) -> str:
    start = body.index(f'id="card-{key}"')
    end = body.find('id="card-', start + 1)
    return body[start : end if end != -1 else len(body)]


# ── GET: per-source cards ─────────────────────────────────────────────────────


def test_settings_shows_cards_with_configured_state(tmp_path: Path) -> None:
    client, toml_path = _client(tmp_path)
    save_config_values(
        {"nightscout": {"url": "https://ns.example.com", "token": "tok-aaaa-bbbb"}},
        path=toml_path,
    )
    body = client.get("/settings").text
    for key in ("nightscout", "dexcom", "libre", "whoop", "oura", "llm", "evidence"):
        assert f'id="card-{key}"' in body, key
    assert "Not configured" in _card(body, "oura")
    ns = _card(body, "nightscout")
    assert "Not configured" not in ns
    assert "Connected" in ns
    # Unofficial-tier banner on Dexcom Share and Libre, at the point of decision.
    assert "unofficial API" in _card(body, "dexcom")
    assert "unofficial API" in _card(body, "libre")
    assert "unofficial API" not in _card(body, "nightscout")


def test_settings_shows_setup_paths_and_links(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    body = client.get("/settings").text
    assert "How data reaches Dexta" in body
    assert "Via Nightscout" in body
    assert "Tandem" in body
    ns = _card(body, "nightscout")
    assert "Nightscout" in ns and "Dexta" in ns
    assert "nightscout.github.io" in ns
    assert 'target="_blank"' in ns
    tp = _card(body, "tidepool")
    assert "JSON export" in tp or "Tidepool" in tp
    assert "tidepool.org" in tp


def test_settings_shows_data_freshness(tmp_path: Path) -> None:
    client, _toml_path = _client(tmp_path)
    store = SQLiteStore(tmp_path / "settings.db")
    store.upsert_raw_events(
        [
            RawEvent(
                source="nightscout",
                source_id="sgv-1",
                source_ts=datetime(2025, 6, 10, 12, 0, tzinfo=UTC),
                payload={},
            )
        ]
    )
    store.close()
    body = client.get("/settings").text
    assert "Latest data" in _card(body, "nightscout")
    assert "2025-06-10" in _card(body, "nightscout")
    assert "Latest data" not in _card(body, "oura")


# ── POST: atomic 0600 TOML writes ─────────────────────────────────────────────


def test_post_source_writes_toml_with_0600(tmp_path: Path) -> None:
    client, toml_path = _client(tmp_path)
    resp = client.post(
        "/settings/nightscout",
        data={"url": "https://ns.example.com", "token": "supersecrettoken"},
    )
    assert resp.status_code == 200
    assert "Saved." in resp.text
    assert toml_path.stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob(".dexta.toml.*"))  # no temp files left behind
    reloaded = load_config(toml_path)
    assert reloaded.nightscout.url == "https://ns.example.com"
    assert reloaded.nightscout.token == "supersecrettoken"


def test_empty_secret_field_keeps_stored_value(tmp_path: Path) -> None:
    client, toml_path = _client(tmp_path)
    save_config_values(
        {"nightscout": {"url": "https://old.example.com", "token": "keep-me-secret"}},
        path=toml_path,
    )
    resp = client.post(
        "/settings/nightscout",
        data={"url": "https://new.example.com", "token": ""},
    )
    assert resp.status_code == 200
    reloaded = load_config(toml_path)
    assert reloaded.nightscout.url == "https://new.example.com"
    assert reloaded.nightscout.token == "keep-me-secret"


def test_failed_validation_never_clobbers_file(tmp_path: Path) -> None:
    client, toml_path = _client(tmp_path)
    save_config_values(
        {"libre": {"email": "f@example.com", "password": "pw-libre-1234"}},
        path=toml_path,
    )
    before = toml_path.read_text(encoding="utf-8")
    resp = client.post(
        "/settings/libre",
        data={"email": "f@example.com", "password": "", "region": "mars", "patient_id": ""},
    )
    assert resp.status_code == 400
    assert toml_path.read_text(encoding="utf-8") == before


def test_general_save_preserves_card_credentials(tmp_path: Path) -> None:
    client, toml_path = _client(tmp_path)
    client.post(
        "/settings/oura",
        data={"access_token": "oura-token-9876"},
    )
    resp = client.post(
        "/settings",
        data={
            "target_low": "65",
            "target_high": "180",
            "deep_analysis_window_days": "90",
            "path": str(tmp_path / "wiki"),
            "git": "off",
            "backend": "sqlite",
            "sqlite_path": str(tmp_path / "dexta.db"),
            "database_url": "",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    reloaded = load_config(toml_path)
    assert reloaded.analysis.target_low == 65
    assert reloaded.oura.access_token == "oura-token-9876"


# ── env precedence: read-only status, never overridable ──────────────────────


def test_env_field_disabled_and_post_cannot_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NIGHTSCOUT_TOKEN", "env-secret-token1")
    client, toml_path = _client(tmp_path)

    ns = _card(client.get("/settings").text, "nightscout")
    assert "NIGHTSCOUT_TOKEN" in ns
    assert "managed in your shell" in ns
    assert "env-secret-token1" not in ns

    resp = client.post(
        "/settings/nightscout",
        data={"url": "https://ns.example.com", "token": "attacker-token"},
    )
    assert resp.status_code == 200
    doc = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    assert doc["nightscout"]["url"] == "https://ns.example.com"
    assert "token" not in doc["nightscout"]  # env-managed field never persisted


# ── secret hygiene: bullets + last 4, never the value ─────────────────────────


def test_stored_secret_renders_last4_only(tmp_path: Path) -> None:
    client, toml_path = _client(tmp_path)
    save_config_values(
        {"nightscout": {"url": "https://ns.example.com", "token": "abcd1234efgh"}},
        path=toml_path,
    )
    body = client.get("/settings").text
    assert "abcd1234efgh" not in body
    assert "••••efgh" in body
    assert "leave blank to keep" in body


def test_env_secret_shows_no_last4(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OURA_ACCESS_TOKEN", "oura-env-secret-value")
    client, _toml_path = _client(tmp_path)
    body = client.get("/settings").text
    assert "oura-env-secret-value" not in body
    assert "••••alue" not in body  # not even the masked last 4 chars


# ── remote bind: status-only mode ─────────────────────────────────────────────


def test_non_loopback_host_shows_warning_but_allows_editing(tmp_path: Path) -> None:
    client, toml_path = _client(tmp_path, host="0.0.0.0")
    body = client.get("/settings").text
    assert "Network exposure" in body
    assert "credential editing is disabled" not in body
    assert "Nightscout" in body
    resp = client.post(
        "/settings/nightscout",
        data={"url": "https://ns.example.com", "token": "x" * 12},
    )
    assert resp.status_code == 200
    assert load_config(toml_path).nightscout.token == "x" * 12


def test_remote_editing_works_without_env_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DEXTA_ALLOW_REMOTE_SETTINGS", raising=False)
    client, toml_path = _client(tmp_path, host="0.0.0.0")
    resp = client.post(
        "/settings/nightscout",
        data={"url": "https://ns.example.com", "token": "remote-ok-token"},
    )
    assert resp.status_code == 200
    assert load_config(toml_path).nightscout.token == "remote-ok-token"


def test_loopback_host_no_network_warning(tmp_path: Path) -> None:
    client, _toml_path = _client(tmp_path, host="localhost")
    body = client.get("/settings").text
    assert "Network exposure" not in body
    assert "Nightscout" in body


# ── test-connection endpoint ──────────────────────────────────────────────────


class _FakeConnector:
    source = "nightscout"

    def __init__(self, report: HealthReport | None = None, exc: Exception | None = None) -> None:
        self._report = report
        self._exc = exc

    def check(self) -> HealthReport:
        if self._exc is not None:
            raise self._exc
        assert self._report is not None
        return self._report

    def pull(self, since: object) -> object:
        raise NotImplementedError


def test_test_connection_renders_health_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = HealthReport(
        ok=True,
        source="nightscout",
        detail="reachable",
        latest_data_ts=datetime(2025, 6, 10, 12, 0, tzinfo=UTC),
    )
    monkeypatch.setattr(
        "dexta_intelligence.server.app.build_connectors",
        lambda _cfg: [_FakeConnector(report=report)],
    )
    client, _toml_path = _client(tmp_path)
    resp = client.post("/settings/nightscout/test")
    assert resp.status_code == 200
    assert "Connection OK" in resp.text
    assert "reachable" in resp.text
    assert "Jun 10" in resp.text


def test_test_connection_failure_rendered_inline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "dexta_intelligence.server.app.build_connectors",
        lambda _cfg: [_FakeConnector(exc=RuntimeError("auth failed (401)"))],
    )
    client, _toml_path = _client(tmp_path)
    resp = client.post("/settings/nightscout/test")
    assert resp.status_code == 200
    assert "Connection failed" in resp.text
    assert "auth failed (401)" in resp.text


def test_test_connection_unconfigured_source(tmp_path: Path) -> None:
    client, _toml_path = _client(tmp_path)
    resp = client.post("/settings/oura/test")
    assert resp.status_code == 200
    assert "Not configured" in resp.text


def test_unknown_source_is_404(tmp_path: Path) -> None:
    client, _toml_path = _client(tmp_path)
    assert client.post("/settings/ghost", data={}).status_code == 404
    assert client.post("/settings/ghost/test").status_code == 404
    assert client.post("/settings/llm/test").status_code == 404  # no connector to test
