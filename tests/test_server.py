"""GUI tests - gated on the optional [gui] extra.

Drive the FastAPI app through TestClient against a seeded SQLiteStore injected
via the ``store_opener`` seam. Skipped wholesale when fastapi is absent.
"""

from __future__ import annotations

import io
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from dexta_intelligence.cli.serve import cmd_serve
from dexta_intelligence.config import Config
from dexta_intelligence.models import (
    Finding,
    FindingStats,
    FindingStatus,
    GlucoseEvent,
    Goal,
    GoalCheckpoint,
    GoalMetric,
    InsulinEvent,
    InsulinKind,
    InvestigationRun,
    MealEvent,
    OpenInvestigation,
    RunFinding,
)
from dexta_intelligence.server import create_app
from dexta_intelligence.server.render import emit_toml, markdown_to_html, sparkline_svg
from dexta_intelligence.store import SQLiteStore

if TYPE_CHECKING:
    from collections.abc import Callable

    from dexta_intelligence.store.port import StoragePort

FIXED_NOW = datetime(2025, 6, 10, 12, 0, tzinfo=UTC)


def _seed_glucose(store: SQLiteStore, days: float = 10.0) -> None:
    ts = FIXED_NOW - timedelta(days=days)
    while ts <= FIXED_NOW:
        store.insert_glucose([GlucoseEvent(ts=ts, mg_dl=120)])
        ts += timedelta(minutes=5)


def _db_path(tmp_path: Path) -> Path:
    return tmp_path / "gui.db"


def _store(tmp_path: Path) -> SQLiteStore:
    store = SQLiteStore(_db_path(tmp_path))
    store.migrate()
    return store


def _opener(db_path: Path) -> Callable[[Config, Path | None], StoragePort]:
    """Open a fresh connection per call against a fixed path.

    The TestClient runs sync handlers in a threadpool and sqlite connections
    are thread-bound, so we re-open (the production behaviour) rather than
    sharing one connection across threads.
    """

    def _open(_config: Config, _db: Path | None = None) -> StoragePort:
        store = SQLiteStore(db_path)
        store.migrate()
        return store

    return _open


def _client(store: SQLiteStore, config: Config | None = None) -> TestClient:
    app = create_app(config or Config(), store_opener=_opener(Path(store._path)))
    return TestClient(app)


# ── dashboard ─────────────────────────────────────────────────────────────────


def test_dashboard_lists_active_finding(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_glucose(store)
    store.insert_finding(
        Finding(
            agent="pattern",
            kind="overnight-lows",
            scope="global",
            headline="Overnight lows cluster after evening exercise",
            confidence=0.82,
            stats=FindingStats(n=24, effect_size=0.6),
            status=FindingStatus.ACTIVE,
        )
    )
    resp = _client(store).get("/")
    assert resp.status_code == 200
    body = resp.text
    assert "Overnight lows cluster after evening exercise" in body
    assert "strong" in body
    assert "supported" in body
    store.close()


def test_dashboard_graveyard_holds_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_glucose(store)
    store.insert_finding(
        Finding(
            agent="pattern",
            kind="noise",
            scope="global",
            headline="Spurious weekday effect",
            status=FindingStatus.REJECTED,
            skeptic_notes="reject: failed permutation test",
        )
    )
    body = _client(store).get("/").text
    assert "Graveyard" in body
    assert "Spurious weekday effect" in body
    store.close()


def test_dashboard_empty_state(tmp_path: Path) -> None:
    store = _store(tmp_path)
    body = _client(store).get("/").text
    assert "No active findings yet" in body
    assert "Run analyze" in body
    assert "Sync in Connectors" in body
    assert 'action="/actions/sync"' not in body
    store.close()


# ── wiki ──────────────────────────────────────────────────────────────────────


def test_wiki_page_renders_markdown(tmp_path: Path) -> None:
    store = _store(tmp_path)
    wiki_root = tmp_path / "wiki"
    wiki_root.mkdir()
    (wiki_root / "index.md").write_text(
        "# dexta wiki\n\n"
        "Coverage: **10 days**\n\n"
        "| finding | confidence |\n|---|---|\n| Lows | 0.82 |\n\n"
        "- [topic](topics/lows.md)\n",
        encoding="utf-8",
    )
    config = Config.model_validate({"wiki": {"path": str(wiki_root)}})
    body = _client(store, config).get("/wiki").text
    assert "wiki-shell" in body
    assert "wiki-nav" in body
    assert "<h1>dexta wiki</h1>" in body
    assert "<strong>10 days</strong>" in body
    assert "wiki-table-wrap" in body
    assert 'href="/wiki/topics/lows"' in body
    store.close()


def test_wiki_missing_page_empty_state(tmp_path: Path) -> None:
    store = _store(tmp_path)
    config = Config.model_validate({"wiki": {"path": str(tmp_path / "nope")}})
    body = _client(store, config).get("/wiki/topics/ghost").text
    assert "dexta wiki" in body
    store.close()


# ── goals ─────────────────────────────────────────────────────────────────────


def test_goals_page_shows_svg_arc(tmp_path: Path) -> None:
    store = _store(tmp_path)
    goal_id = store.insert_goal(
        Goal(
            statement="Reduce overnight lows",
            metric=GoalMetric.NOCTURNAL_TBR,
            direction="decrease",
        )
    )
    for i, val in enumerate((5.0, 4.2, 3.1)):
        store.insert_goal_checkpoint(
            GoalCheckpoint(
                goal_id=goal_id,
                ts=FIXED_NOW + timedelta(days=i),
                metric_value=val,
                note=f"tick {i}",
            )
        )
    body = _client(store).get("/goals").text
    assert "Reduce overnight lows" in body
    assert "<svg" in body and "<polyline" in body
    assert "tick 2" in body  # only the latest note is surfaced
    store.close()


def test_goals_empty_state(tmp_path: Path) -> None:
    store = _store(tmp_path)
    body = _client(store).get("/goals").text
    assert "No goals yet" in body
    assert "Add goal" in body
    assert 'name="statement"' in body
    assert "Tick goals now" in body
    store.close()


def test_goals_post_creates_goal(tmp_path: Path) -> None:
    store = _store(tmp_path)
    client = _client(store)
    resp = client.post(
        "/goals",
        data={"statement": "Reduce overnight lows", "target": "5"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/goals?saved=1"
    body = client.get("/goals?saved=1").text
    assert "Goal added." in body
    assert "Reduce overnight lows" in body
    assert store.get_goals()[0].target == 5.0
    store.close()


def test_goals_post_rejects_empty_statement(tmp_path: Path) -> None:
    store = _store(tmp_path)
    resp = _client(store).post("/goals", data={"statement": "   "})
    assert resp.status_code == 200
    assert "Describe what you want to improve" in resp.text
    assert not store.get_goals()
    store.close()


def test_goals_post_rejects_duplicate_active_statement(tmp_path: Path) -> None:
    store = _store(tmp_path)
    client = _client(store)
    assert (
        client.post(
            "/goals",
            data={"statement": "Increase time in range"},
            follow_redirects=False,
        ).status_code
        == 303
    )
    resp = client.post("/goals", data={"statement": "increase time in range"})
    assert resp.status_code == 200
    assert "already have an active goal" in resp.text
    assert len(store.get_goals()) == 1
    store.close()


def test_goals_abandon_hides_from_page(tmp_path: Path) -> None:
    store = _store(tmp_path)
    goal_id = store.insert_goal(
        Goal(
            statement="Reduce overnight lows",
            metric=GoalMetric.NOCTURNAL_TBR,
            direction="decrease",
        )
    )
    client = _client(store)
    resp = client.post(f"/goals/{goal_id}/abandon", follow_redirects=False)
    assert resp.status_code == 303
    body = client.get("/goals").text
    assert "Reduce overnight lows" not in body
    store.close()


# ── chat ──────────────────────────────────────────────────────────────────────


class _FakeAnswer:
    text = "Your time-in-range was 68% over the last 10 days (n=2880)."
    tools_used = ("tir_snapshot",)
    faithful = True
    violations: tuple[str, ...] = ()
    stopped_reason = "answer"


class _FakeAgent:
    def __init__(self, *_a: object, **_kw: object) -> None:
        pass

    def ask(self, _ctx: object, _question: str) -> _FakeAnswer:
        return _FakeAnswer()


def test_api_ask_with_fake_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store(tmp_path)
    _seed_glucose(store)
    monkeypatch.setattr(
        "dexta_intelligence.server.app.discovery_model", lambda _cfg: object()
    )
    monkeypatch.setattr(
        "dexta_intelligence.agents.orchestrator.OrchestratorAgent", _FakeAgent
    )
    resp = _client(store).post("/api/ask", data={"question": "how is my TIR?"})
    assert resp.status_code == 200
    assert "time-in-range was 68%" in resp.text
    assert "tir_snapshot" in resp.text
    store.close()


def test_chat_empty_state_without_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store(tmp_path)
    monkeypatch.setattr(
        "dexta_intelligence.server.app.discovery_model", lambda _cfg: None
    )
    body = _client(store).get("/chat").text
    assert "Chat needs a language model" in body
    assert "Open Settings" in body
    store.close()


def test_chat_with_model_shows_history_rail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    monkeypatch.setattr(
        "dexta_intelligence.server.app.discovery_model", lambda _cfg: object()
    )
    body = _client(store).get("/chat").text
    assert 'id="new-chat-btn"' in body
    assert 'id="session-list"' in body
    store.close()


# ── settings ──────────────────────────────────────────────────────────────────


def test_settings_shows_env_status_without_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-super-secret-value")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    body = _client(store).get("/settings").text
    assert "ANTHROPIC_API_KEY" in body
    assert "OPENROUTER_API_KEY" in body
    assert "sk-super-secret-value" not in body  # never leak the value
    store.close()


def _settings_form(**overrides: str) -> dict[str, str]:
    data = {
        "target_low": "70",
        "target_high": "180",
        "max_reasoning_steps": "20",
        "deep_analysis_window_days": "90",
        "path": "/tmp/wiki",
        "git": "off",
        "backend": "sqlite",
        "sqlite_path": "~/.dexta/dexta.db",
        "database_url": "",
    }
    data.update(overrides)
    return data


def test_settings_post_roundtrips_target_low(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    toml_path = tmp_path / "dexta.toml"
    monkeypatch.setattr(
        "dexta_intelligence.cli._common.resolve_config_path", lambda _explicit: toml_path
    )
    config = Config()
    wiki = str(tmp_path / "wiki")
    db = str(tmp_path / "custom.db")
    resp = _client(store, config).post(
        "/settings",
        data=_settings_form(
            target_low="65",
            path=wiki,
            git="on",
            sqlite_path=db,
        ),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    written = toml_path.read_text(encoding="utf-8")
    assert "target_low = 65" in written
    assert f'sqlite_path = "{db}"' in written
    from dexta_intelligence.config import load_config  # noqa: PLC0415

    reloaded = load_config(toml_path)
    assert reloaded.analysis.target_low == 65
    assert str(reloaded.data.sqlite_path.expanduser()) == db
    store.close()


# ── pure render helpers ───────────────────────────────────────────────────────


def test_markdown_escapes_and_renders() -> None:
    html = markdown_to_html("## Heading\n\n- **bold** and `code`\n\n<script>alert(1)</script>")
    assert "<h2>Heading</h2>" in html
    assert "<strong>bold</strong>" in html
    assert "<code>code</code>" in html
    assert "<script>" not in html


def test_markdown_table_without_divider_row() -> None:
    md = "| Data stream | Status |\n| Glucose readings | 8 days of data |"
    html = markdown_to_html(md)
    assert "<table>" in html
    assert "Glucose readings" in html
    assert "8 days" in html


def test_emit_toml_is_loadable(tmp_path: Path) -> None:
    config = Config.model_validate({"analysis": {"target_low": 72}})
    out = emit_toml(config)
    path = tmp_path / "c.toml"
    path.write_text(out, encoding="utf-8")
    from dexta_intelligence.config import load_config  # noqa: PLC0415

    assert load_config(path).analysis.target_low == 72


def test_sparkline_flat_for_sparse_data() -> None:
    assert "spark-flat" in sparkline_svg([])
    assert "spark-flat" in sparkline_svg([1.0])
    assert "polyline" in sparkline_svg([1.0, 2.0, 1.5])


# ── security: wiki path-traversal containment ─────────────────────────────────


def _wiki_config(wiki_root: Path) -> Config:
    return Config.model_validate({"wiki": {"path": str(wiki_root)}})


def test_wiki_blocks_sibling_prefix_traversal(tmp_path: Path) -> None:
    store = _store(tmp_path)
    wiki_root = tmp_path / "wiki"
    wiki_root.mkdir()
    (wiki_root / "index.md").write_text("# ok\n", encoding="utf-8")
    # A sibling dir that shares the wiki root's string prefix (the old bug).
    evil = tmp_path / "wiki_evil"
    evil.mkdir()
    (evil / "pwn.md").write_text("SIBLING-PREFIX-LEAK secret\n", encoding="utf-8")

    client = _client(store, _wiki_config(wiki_root))
    resp = client.get("/wiki/../wiki_evil/pwn", follow_redirects=False)
    # Either rejected outright (3xx/404) or rendered as the empty state - never leaked.
    assert "SIBLING-PREFIX-LEAK" not in resp.text
    store.close()


def test_wiki_blocks_dotdot_and_absolute_and_encoded(tmp_path: Path) -> None:
    store = _store(tmp_path)
    wiki_root = tmp_path / "wiki"
    wiki_root.mkdir()
    (wiki_root / "index.md").write_text("# ok\n", encoding="utf-8")
    secret = tmp_path / "secret.md"
    secret.write_text("OUTSIDE-ROOT-LEAK\n", encoding="utf-8")

    client = _client(store, _wiki_config(wiki_root))
    for path in (
        "/wiki/../secret",
        "/wiki/..%2fsecret",
        "/wiki/%2e%2e/secret",
        f"/wiki{secret.with_suffix('')}",  # absolute-path variant
    ):
        resp = client.get(path, follow_redirects=False)
        assert "OUTSIDE-ROOT-LEAK" not in resp.text, path
    store.close()


# ── security: settings writes to the launched config path ─────────────────────


def test_settings_writes_to_launched_config_not_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    launched = tmp_path / "launched.toml"
    cwd_default = tmp_path / "dexta.toml"
    # If the handler wrongly re-resolved per request, it would hit this path.
    monkeypatch.setattr(
        "dexta_intelligence.cli._common.resolve_config_path", lambda _explicit: cwd_default
    )
    app = create_app(Config(), store_opener=_opener(_db_path(tmp_path)), config_path=launched)
    client = TestClient(app)
    resp = client.post(
        "/settings",
        data=_settings_form(path=str(tmp_path / "wiki")),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert launched.is_file()
    assert "target_low = 70" in launched.read_text(encoding="utf-8")
    assert not cwd_default.exists()  # never touched the cwd default
    store.close()


# ── security: settings validation ─────────────────────────────────────────────


def _post_settings(client: TestClient, **overrides: str) -> Any:
    return client.post(
        "/settings",
        data=_settings_form(**overrides),
        follow_redirects=False,
    )


def test_settings_rejects_invalid_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    launched = tmp_path / "launched.toml"
    app = create_app(Config(), store_opener=_opener(_db_path(tmp_path)), config_path=launched)
    client = TestClient(app)

    # Non-numeric, negative, and low>=high are all rejected with a re-rendered form.
    for overrides in (
        {"target_low": "abc"},
        {"target_low": "-50"},
        {"target_low": "200", "target_high": "180"},
        {"max_reasoning_steps": "3"},
        {"max_reasoning_steps": "99"},
    ):
        resp = _post_settings(client, **overrides)
        assert resp.status_code == 400, overrides
        assert not launched.exists(), overrides  # garbage never persisted
    store.close()


# ── security: javascript: link sanitised in rendered markdown ─────────────────


def test_markdown_sanitizes_javascript_link() -> None:
    html = markdown_to_html("[click me](javascript:alert(1))")
    assert "javascript:alert" not in html
    assert 'href="#"' in html
    assert "click me" in html  # link text preserved, only the scheme is stripped


def test_markdown_keeps_safe_link_schemes() -> None:
    html = markdown_to_html(
        "[a](https://example.com) [b](mailto:x@y.z) [c](topics/lows.md)"
    )
    assert 'href="https://example.com"' in html
    assert 'href="mailto:x@y.z"' in html
    assert 'href="topics/lows.md"' in html


# ── security: 0.0.0.0 LAN-exposure warning ────────────────────────────────────


def test_serve_warns_on_lan_bind(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("uvicorn.run", lambda *a, **kw: None)
    out = io.StringIO()
    cmd_serve(config=Config(), db_path=None, out=out, host="0.0.0.0", port=8787)
    text = out.getvalue()
    assert "WARNING" in text
    assert "0.0.0.0" in text


def test_serve_no_warning_on_localhost(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("uvicorn.run", lambda *a, **kw: None)
    out = io.StringIO()
    cmd_serve(config=Config(), db_path=None, out=out, host="127.0.0.1", port=8787)
    assert "WARNING" not in out.getvalue()


# ── treatment logger ──────────────────────────────────────────────────────────


def _logged_treatments(store: SQLiteStore) -> tuple[list[Any], list[Any]]:
    lo = FIXED_NOW - timedelta(days=2)
    hi = FIXED_NOW + timedelta(days=2)
    return store.get_meals(lo, hi), store.get_insulin(lo, hi)


def test_log_treatment_writes_paired_meal_and_bolus(tmp_path: Path) -> None:
    store = _store(tmp_path)
    client = _client(store)
    resp = client.post(
        "/actions/log-treatment",
        data={
            "treatment_ts": "2025-06-10T12:00",
            "carbs_g": "58",
            "units": "5.2",
            "note": "dinner",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/log?flash=treatment_ok"
    meals, insulin = _logged_treatments(store)
    assert len(meals) == 1 and meals[0].carbs_g == 58.0 and meals[0].note == "dinner"
    assert len(insulin) == 1 and insulin[0].units == 5.2
    assert insulin[0].automatic is False
    assert meals[0].ts == insulin[0].ts  # dt = 0, so the episode graph pairs them
    store.close()


def test_log_treatment_carbs_only_writes_meal_only(tmp_path: Path) -> None:
    store = _store(tmp_path)
    client = _client(store)
    client.post(
        "/actions/log-treatment",
        data={"treatment_ts": "2025-06-10T12:00", "carbs_g": "15", "units": ""},
        follow_redirects=False,
    )
    meals, insulin = _logged_treatments(store)
    assert len(meals) == 1 and insulin == []
    store.close()


def test_log_treatment_rejects_empty_and_bad_numbers(tmp_path: Path) -> None:
    store = _store(tmp_path)
    client = _client(store)
    empty = client.post(
        "/actions/log-treatment",
        data={"treatment_ts": "2025-06-10T12:00", "carbs_g": "", "units": ""},
        follow_redirects=False,
    )
    assert empty.headers["location"] == "/log?flash=treatment_empty"
    for carbs, units in (("abc", ""), ("-5", ""), ("", "900"), ("", "0")):
        bad = client.post(
            "/actions/log-treatment",
            data={"treatment_ts": "2025-06-10T12:00", "carbs_g": carbs, "units": units},
            follow_redirects=False,
        )
        assert bad.headers["location"] == "/log?flash=treatment_badnum", (carbs, units)
    meals, insulin = _logged_treatments(store)
    assert meals == [] and insulin == []
    store.close()


def test_log_page_shows_treatment_form(tmp_path: Path) -> None:
    store = _store(tmp_path)
    body = _client(store).get("/log").text
    assert "Log a treatment" in body
    assert 'action="/actions/log-treatment"' in body
    assert "never suggests a dose" in body
    store.close()


# ── sync targets the served store ─────────────────────────────────────────────


def test_action_sync_uses_served_store_opener(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = _store(tmp_path)
    opener = _opener(_db_path(tmp_path))
    app = create_app(Config(), store_opener=opener)
    captured: dict[str, Any] = {}

    def fake_cmd_sync(**kwargs: Any) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr("dexta_intelligence.cli.data.cmd_sync", fake_cmd_sync)
    resp = TestClient(app).post("/actions/sync", follow_redirects=False)
    assert resp.status_code == 303
    assert captured["opener"] is opener
    store.close()


# ── demo isolation: throwaway db, no real-data ingress ────────────────────────


def _demo_client(tmp_path: Path) -> TestClient:
    store = _store(tmp_path)
    app = create_app(Config(), store_opener=_opener(Path(store._path)), demo=True)
    return TestClient(app)


def test_demo_blocks_all_sync_actions(tmp_path: Path) -> None:
    client = _demo_client(tmp_path)
    for path, data in (
        ("/actions/sync", {}),
        ("/actions/connectors/sync", {"scope": "all"}),
        ("/actions/connectors/autosync", {"interval": "15"}),
    ):
        resp = client.post(path, data=data, follow_redirects=False)
        assert resp.status_code == 303, path
        assert resp.headers["location"] == "/connectors?flash=demo_sync", path


def test_demo_autosync_stays_disabled_after_post(tmp_path: Path) -> None:
    client = _demo_client(tmp_path)
    client.post("/actions/connectors/autosync", data={"interval": "15"}, follow_redirects=False)
    assert client.app.state.autosync.status().enabled is False


def test_demo_connectors_page_shows_notice_and_flash(tmp_path: Path) -> None:
    client = _demo_client(tmp_path)
    html = client.get("/connectors?flash=demo_sync").text
    assert "Demo mode: connector sync is disabled" in html


def test_serve_demo_uses_throwaway_db(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("uvicorn.run", lambda *a, **kw: None)
    iso = tmp_path / "demo-iso"
    iso.mkdir()
    monkeypatch.setattr("tempfile.mkdtemp", lambda prefix="": str(iso))
    out = io.StringIO()
    cmd_serve(config=Config(), db_path=None, out=out, demo=True, sync_every=5)
    text = out.getvalue()
    assert str(iso / "demo.db") in text
    assert "seeded the synthetic demo patient" in text
    assert "connector sync is disabled" in text
    assert "auto-sync every" not in text
    assert (iso / "demo.db").exists()


# ── CSV upload ────────────────────────────────────────────────────────────────


def test_upload_csv_ingests_glucose(tmp_path: Path) -> None:
    store = _store(tmp_path)
    client = _client(store)
    fixture = Path(__file__).parent / "fixtures" / "clarity_sample.csv"
    with fixture.open("rb") as fh:
        resp = client.post(
            "/actions/upload",
            files={"file": ("clarity_sample.csv", fh, "text/csv")},
            follow_redirects=False,
        )
    assert resp.status_code == 303
    assert "upload_ok" in resp.headers["location"]
    assert store.coverage().n_glucose > 0
    store.close()


def test_upload_empty_file_flashes(tmp_path: Path) -> None:
    store = _store(tmp_path)
    resp = _client(store).post(
        "/actions/upload",
        files={"file": ("empty.csv", b"", "text/csv")},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert "upload_empty" in resp.headers["location"]


# ── investigate redirect + lens picker ───────────────────────────────────────


def test_investigate_post_redirects_to_investigations(tmp_path: Path) -> None:
    store = _store(tmp_path)
    resp = _client(store).post(
        "/actions/investigate",
        data={"question": "what drives my overnight lows?"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/investigations?q=what%20drives%20my%20overnight%20lows%3F"
    store.close()


def test_investigate_empty_question_redirects(tmp_path: Path) -> None:
    store = _store(tmp_path)
    resp = _client(store).post(
        "/actions/investigate",
        data={"question": "   "},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/investigations"
    store.close()


def test_dashboard_has_investigation_cta_not_form(tmp_path: Path) -> None:
    store = _store(tmp_path)
    body = _client(store).get("/").text
    assert "Start an investigation" in body
    assert "/investigations" in body
    assert 'action="/actions/investigate"' not in body
    assert "Ask a question →" not in body
    store.close()


def _seed_run(store: SQLiteStore, *, question: str = "what drives overnight lows?") -> None:
    store.insert_investigation_run(
        InvestigationRun(
            run_id="r1",
            kind="question",
            status="completed",
            question=question,
            window_start=date(2025, 6, 1),
            window_end=date(2025, 6, 10),
            plan=["observation", "pattern"],
            trace=[
                "Planned: observation, pattern",
                "Round 1: ran observation, pattern -> 1 finding(s)",
            ],
            findings=[
                RunFinding(
                    headline="Overnight lows cluster after evening exercise",
                    kind="overnight-lows",
                    confidence=0.8,
                    status="active",
                )
            ],
            n_findings=1,
            started_at=FIXED_NOW,
            finished_at=FIXED_NOW,
        )
    )


def test_investigations_page_lists_runs(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_run(store)
    body = _client(store).get("/investigations").text
    assert "what drives overnight lows?" in body
    assert "Overnight lows cluster after evening exercise" in body
    assert "observation" in body
    assert "Round 1: ran observation, pattern" in body
    assert "trace-timeline" in body
    store.close()


def test_investigations_empty_state(tmp_path: Path) -> None:
    store = _store(tmp_path)
    body = _client(store).get("/investigations").text
    assert "No investigations yet" in body
    assert "Use the form above" in body
    store.close()


def test_investigations_page_shows_open_queue(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.insert_open_investigation(
        OpenInvestigation(
            question="Why does severe high keep happening?",
            condition_type="event_count",
            subject="severe_high",
            target=3.0,
            current=1.0,
            status="collecting",
            created_at=FIXED_NOW,
        )
    )
    body = _client(store).get("/investigations").text
    assert "Open investigations" in body
    assert "Why does severe high keep happening?" in body
    assert "1/3 seen" in body
    store.close()


def test_findings_page_renders_tabs_and_cards(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_glucose(store)
    store.insert_finding(
        Finding(
            agent="pattern",
            kind="overnight-lows",
            scope="global",
            headline="Overnight lows after evening exercise",
            confidence=0.82,
            stats=FindingStats(n=24, effect_size=0.6),
            status=FindingStatus.ACTIVE,
        )
    )
    _seed_run(store)
    body = _client(store).get("/findings").text
    assert "Active findings" in body
    assert "Overnight lows after evening exercise" in body
    assert "Open hypotheses" in body
    assert "Investigation log" in body
    assert "evidence" in body
    store.close()


def test_goals_page_shows_progress_and_checkpoints(tmp_path: Path) -> None:
    store = _store(tmp_path)
    goal_id = store.insert_goal(
        Goal(
            statement="increase time in range",
            metric=GoalMetric.TIR,
            direction="increase",
            target=70.0,
        )
    )
    for i, val in enumerate((31.6, 45.0, 58.0)):
        store.insert_goal_checkpoint(
            GoalCheckpoint(
                goal_id=goal_id,
                ts=FIXED_NOW + timedelta(days=i),
                metric_value=val,
                note=f"tick {i}",
            )
        )
    body = _client(store).get("/goals").text
    assert "increase time in range" in body
    assert "baseline" in body
    assert "target" in body
    assert "58.0" in body  # the current (latest) value, not the baseline
    assert "checkpoint(s)" in body
    store.close()


def test_goal_cadence_is_configurable(tmp_path: Path) -> None:
    store = _store(tmp_path)
    resp = _client(store).post(
        "/goals",
        data={"statement": "reduce overnight lows", "cadence": "14"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert store.get_goals()[0].cadence_days == 14
    store.close()


def test_goal_cadence_rejects_non_positive(tmp_path: Path) -> None:
    store = _store(tmp_path)
    resp = _client(store).post("/goals", data={"statement": "x", "cadence": "0"})
    assert resp.status_code == 200
    assert "at least 1 day" in resp.text
    assert not store.get_goals()
    store.close()


def test_goals_tick_action_invokes_cmd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    called: dict[str, Any] = {}

    def _fake_cmd_goals(*, action: str, **_kw: Any) -> int:
        called["action"] = action
        return 0

    monkeypatch.setattr(
        "dexta_intelligence.cli.intelligence.cmd_goals", _fake_cmd_goals
    )
    resp = _client(store).post("/actions/goals/tick", follow_redirects=False)
    assert resp.status_code == 303
    assert "flash=ticked_ok" in resp.headers["location"]
    assert called["action"] == "tick"
    store.close()


def test_wiki_rebuild_action_invokes_cmd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)

    def _fake_cmd_wiki(**_kw: Any) -> int:
        return 0

    monkeypatch.setattr(
        "dexta_intelligence.cli.intelligence.cmd_wiki", _fake_cmd_wiki
    )
    resp = _client(store).post("/actions/wiki", follow_redirects=False)
    assert resp.status_code == 303
    assert "flash=wiki_ok" in resp.headers["location"]
    store.close()


# ── connectors page ─────────────────────────────────────────────────────────


def test_connectors_page_lists_all_sources(tmp_path: Path) -> None:
    store = _store(tmp_path)
    body = _client(store).get("/connectors").text
    assert "Connectors" in body
    assert "Continuous sync" in body
    assert "Nightscout" in body
    assert "read-only" in body
    assert "not configured" in body
    store.close()


def test_connectors_sync_requires_a_selection(tmp_path: Path) -> None:
    store = _store(tmp_path)
    resp = _client(store).post(
        "/actions/connectors/sync",
        data={"scope": "selected"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert "flash=sync_none" in resp.headers["location"]
    store.close()


def test_connectors_autosync_sets_interval_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    toml_path = tmp_path / "dexta.toml"
    monkeypatch.setattr(
        "dexta_intelligence.cli._common.resolve_config_path", lambda _explicit: toml_path
    )
    client = _client(store)
    resp = client.post(
        "/actions/connectors/autosync",
        data={"interval": "15"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert "flash=autosync_ok" in resp.headers["location"]
    # The live controller was retuned without a restart.
    assert client.app.state.autosync.status().interval_min == 15
    client.app.state.autosync.stop()
    store.close()


def test_connectors_autosync_htmx_returns_panel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    toml_path = tmp_path / "dexta.toml"
    monkeypatch.setattr(
        "dexta_intelligence.cli._common.resolve_config_path", lambda _explicit: toml_path
    )
    client = _client(store)
    resp = client.post(
        "/actions/connectors/autosync",
        data={"interval": "30"},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200
    assert 'id="autosync-panel"' in resp.text
    assert "Continuous sync updated." in resp.text
    assert client.app.state.autosync.status().interval_min == 30
    client.app.state.autosync.stop()
    store.close()


def test_dashboard_shows_lens_picker(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_glucose(store)
    body = _client(store).get("/").text
    assert 'name="lens"' in body
    assert ">analyze</option>" in body
    assert ">watch</option>" in body
    store.close()


def test_analyze_passes_selected_lens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    _seed_glucose(store)
    captured: dict[str, str] = {}

    def _fake_cmd_analyze(
        *, config: Config, db_path: Path | None, out: Any, lens: str = "analyze"
    ) -> int:
        captured["lens"] = lens
        return 0

    monkeypatch.setattr(
        "dexta_intelligence.cli.analysis.cmd_analyze", _fake_cmd_analyze
    )
    resp = _client(store).post(
        "/actions/analyze",
        data={"lens": "watch"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert captured["lens"] == "watch"
    store.close()
    store.close()


def test_mask_dsn_hides_password() -> None:
    from dexta_intelligence.server.app import _mask_dsn  # noqa: PLC0415

    masked = _mask_dsn("postgresql://user:secret@db.example.com:5432/dexta")
    assert "secret" not in masked
    assert "***" in masked
    assert "user" in masked
    assert "db.example.com:5432/dexta" in masked
    # no password -> unchanged; empty -> empty
    assert _mask_dsn("postgresql://db.example.com/dexta") == "postgresql://db.example.com/dexta"
    assert _mask_dsn("") == ""


# ── timeline (temporal episode graph) ───────────────────────────────────────────


def _seed_excursions(store: SQLiteStore) -> None:
    """Seed a flat trace with one clinically significant high and one low run,
    so the episode graph has hyper/hypo nodes to render."""
    ts = FIXED_NOW - timedelta(days=2)
    rows: list[GlucoseEvent] = []
    while ts <= FIXED_NOW:
        offset = (ts - (FIXED_NOW - timedelta(days=1))).total_seconds() / 60.0
        if 0 <= offset < 40:
            mg = 210  # hyper run (>180), ~40 min
        elif 120 <= offset < 160:
            mg = 60  # hypo run (<70), ~40 min
        else:
            mg = 120
        rows.append(GlucoseEvent(ts=ts, mg_dl=mg))
        ts += timedelta(minutes=5)
    store.insert_glucose(rows)


def test_timeline_page_renders(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_excursions(store)
    resp = _client(store).get("/timeline")
    assert resp.status_code == 200
    body = resp.text
    assert "Temporal episode graph" in body
    assert "tl-shell" in body
    assert "timeline.js" in body
    assert "High episodes" in body
    # Navigator strip and focus relation view are both present.
    assert "tl-navigator" in body
    assert "tl-focus" in body
    # Both lenses of the focus view: the Curve / Graph toggle drives off the same
    # selected episode.
    assert 'data-view="curve"' in body
    assert 'data-view="graph"' in body
    # Default selection is populated server-side, never blank: the shell carries a
    # default episode id and the facts card is pre-rendered.
    assert 'data-default-episode="' in body
    assert "data-default-episode=\"\"" not in body
    assert "episode-card" in body
    store.close()


def test_episodes_json_shape(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_excursions(store)
    resp = _client(store).get("/episodes.json")
    assert resp.status_code == 200
    data = resp.json()
    assert set(data) >= {"summary", "nodes", "window"}
    assert isinstance(data["nodes"], list)
    assert data["summary"]["num_hyper"] >= 1
    assert data["summary"]["num_hypo"] >= 1
    assert data["window"]["start"] < data["window"]["end"]
    node = next(n for n in data["nodes"] if n["kind"] in ("hypo", "hyper"))
    assert {"id", "kind", "start", "end", "duration_min", "links"} <= set(node)
    store.close()


def test_timeline_graph_lens_toggle(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_excursions(store)
    body = _client(store).get("/timeline").text
    # The Curve / Graph switcher renders and defaults to the curve lens; both
    # lenses drive off the same selected episode client-side.
    assert 'class="tl-viewtoggle"' in body
    assert 'data-view="curve"' in body
    assert 'data-view="graph"' in body
    assert 'id="tl-view-curve"' in body and 'aria-pressed="true"' in body
    store.close()


def test_episode_json_relation_view(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_excursions(store)
    client = _client(store)
    nodes = client.get("/episodes.json").json()["nodes"]
    hyper = next(n for n in nodes if n["kind"] == "hyper")
    resp = client.get("/episode.json", params={"id": hyper["id"]})
    assert resp.status_code == 200
    data = resp.json()
    assert set(data) >= {"episode", "series", "window", "target"}
    assert data["episode"]["id"] == hyper["id"]
    # The local glucose curve is a non-empty series sliced from the store, and it
    # spans the episode plus padding on each side.
    assert isinstance(data["series"], list) and len(data["series"]) > 1
    assert all({"t", "v"} <= set(pt) for pt in data["series"])
    assert data["window"]["start"] < data["episode"]["start"]
    assert data["window"]["end"] > data["episode"]["end"]
    store.close()


def test_episode_json_carries_labelled_edges(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_excursions(store)
    # A meal shortly before the seeded high gives the episode a typed edge whose
    # signed offset is what the focus view turns into "N min before".
    high_start = FIXED_NOW - timedelta(days=1)
    store.insert_meals([MealEvent(ts=high_start - timedelta(minutes=20), carbs_g=45)])
    client = _client(store)
    nodes = client.get("/episodes.json").json()["nodes"]
    hyper = next(n for n in nodes if n["kind"] == "hyper")
    data = client.get("/episode.json", params={"id": hyper["id"]}).json()
    meals = [link for link in data["episode"]["links"] if link["kind"] == "meal"]
    assert meals, "expected a meal edge on the episode"
    assert meals[0]["detail"]["carbs_g"] == 45
    assert meals[0]["offset_min"] < 0
    store.close()


def _chained_episode_dict() -> dict[str, Any]:
    """An explain_episode-shaped dict: a rebound high with an incoming chain from
    a low, bridged by rescue carbs."""
    return {
        "id": "hyper:2026-03-08T17:10:00+00:00",
        "kind": "hyper",
        "start": "2026-03-08T17:10:00+00:00",
        "end": "2026-03-08T17:50:00+00:00",
        "duration_min": 40.0,
        "extreme_mg_dl": 206.0,
        "links": [],
        "chain": {
            "in": [
                {
                    "src_id": "hypo:2026-03-08T15:50:00+00:00",
                    "dst_id": "hyper:2026-03-08T17:10:00+00:00",
                    "relation": "rebound_after_low",
                    "gap_min": 55.0,
                    "bridge": {
                        "kind": "meal",
                        "ts": "2026-03-08T16:15:00+00:00",
                        "offset_min": 25.0,
                        "detail": {"carbs_g": 16.0, "note": "rescue carbs"},
                    },
                }
            ],
            "out": [],
        },
    }


def test_episode_chain_view_builds_the_sequence() -> None:
    from zoneinfo import ZoneInfo  # noqa: PLC0415

    from dexta_intelligence.server.views_episode import episode_chain_view  # noqa: PLC0415

    view = episode_chain_view(_chained_episode_dict(), ZoneInfo("UTC"))
    assert view is not None
    assert view["this"]["short"] == "High"
    assert len(view["incoming"]) == 1 and view["outgoing"] == []
    step = view["incoming"][0]
    assert step["relation"] == "rebound after low"
    assert step["relation_key"] == "rebound_after_low"
    assert step["bridge"] == "16 g carbs, rescue carbs"
    assert step["gap"] == "55 min"
    assert step["node"]["short"] == "Low"


def test_episode_chain_view_none_without_chain() -> None:
    from zoneinfo import ZoneInfo  # noqa: PLC0415

    from dexta_intelligence.server.views_episode import episode_chain_view  # noqa: PLC0415

    assert episode_chain_view({"id": "hyper:x", "kind": "hyper"}, ZoneInfo("UTC")) is None
    assert episode_chain_view(
        {"id": "hyper:x", "kind": "hyper", "chain": {"in": [], "out": []}}, ZoneInfo("UTC")
    ) is None


def test_chain_partial_renders_chain_strip() -> None:
    from importlib import resources  # noqa: PLC0415
    from zoneinfo import ZoneInfo  # noqa: PLC0415

    from jinja2 import Environment, FileSystemLoader, select_autoescape  # noqa: PLC0415

    from dexta_intelligence.server.views_episode import episode_chain_view  # noqa: PLC0415

    templates_dir = str(resources.files("dexta_intelligence.server") / "templates")
    env = Environment(
        loader=FileSystemLoader(templates_dir), autoescape=select_autoescape()
    )
    chain = episode_chain_view(_chained_episode_dict(), ZoneInfo("UTC"))
    html = env.get_template("_episode_chain.html").render(chain=chain)
    assert "chain-strip" in html
    assert "rebound after low" in html
    assert "via 16 g carbs, rescue carbs" in html
    assert "this episode" in html


def test_episodes_json_includes_chain_edges(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_excursions(store)
    data = _client(store).get("/episodes.json").json()
    assert "edges" in data
    # The seeded high and low sit ~85 min apart with nothing in the gap: a
    # weak "follows", never a confident causal name.
    assert any(e["relation"] == "follows" for e in data["edges"])
    store.close()


def test_episode_json_chain_names_bridge_and_neighbour(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_excursions(store)
    high_start = FIXED_NOW - timedelta(days=1)
    # A correction bolus in the gap between the seeded high and low makes the
    # chain a confident low_after_high with the bolus as its bridge.
    store.insert_insulin([
        InsulinEvent(
            ts=high_start + timedelta(minutes=70), kind=InsulinKind.BOLUS, units=2.0
        )
    ])
    client = _client(store)
    nodes = client.get("/episodes.json").json()["nodes"]
    hypo = next(n for n in nodes if n["kind"] == "hypo")
    data = client.get("/episode.json", params={"id": hypo["id"]}).json()
    incoming = data["chain"]["in"]
    assert len(incoming) == 1
    edge = incoming[0]
    assert edge["relation"] == "low_after_high"
    assert edge["bridge"]["kind"] == "bolus"
    assert edge["bridge"]["detail"]["units"] == 2.0
    assert edge["other"]["kind"] == "hyper"
    assert edge["other"]["id"] == edge["src_id"]
    store.close()


def test_episode_json_missing_and_unknown(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_excursions(store)
    client = _client(store)
    assert client.get("/episode.json").status_code == 400
    assert client.get("/episode.json", params={"id": "hyper:nope"}).status_code == 404
    store.close()


def test_timeline_empty_store_degrades(tmp_path: Path) -> None:
    store = _store(tmp_path)
    resp = _client(store).get("/timeline")
    assert resp.status_code == 200
    body = resp.text
    assert "No episodes in this window" in body
    assert "tl-shell" not in body
    store.close()


def test_episodes_json_empty_store(tmp_path: Path) -> None:
    store = _store(tmp_path)
    data = _client(store).get("/episodes.json").json()
    assert data["nodes"] == []
    assert data["summary"]["num_hyper"] == 0
    assert "window" in data
    store.close()
