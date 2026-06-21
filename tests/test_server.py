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
    InvestigationRun,
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
    # confidence 0.82 + n=24 + survived -> strong evidence; seen once -> supported
    assert "strong evidence" in body
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
    assert "Sync now" in body
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
    assert "dexta wiki" in body  # install hint mentions the command
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
    assert "tick 2" in body  # latest note
    store.close()


def test_goals_empty_state(tmp_path: Path) -> None:
    store = _store(tmp_path)
    body = _client(store).get("/goals").text
    assert "No goals yet" in body
    assert "Add goal" in body
    assert 'name="statement"' in body
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
    stopped_reason = "answer"


class _FakeAgent:
    def __init__(self, *_a: object, **_kw: object) -> None:
        pass

    def ask(self, _ctx: object, _question: str) -> _FakeAnswer:
        return _FakeAnswer()


def test_api_ask_with_fake_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store(tmp_path)
    _seed_glucose(store)
    # Inject a fake model + fake ChatAgent so no real LLM is needed.
    monkeypatch.setattr(
        "dexta_intelligence.server.app.discovery_model", lambda _cfg: object()
    )
    monkeypatch.setattr("dexta_intelligence.agents.chat.ChatAgent", _FakeAgent)
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
    assert "<script>" not in html  # escaped


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
    assert "click me" in html  # text preserved


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


# ── investigate + lens picker ──────────────────────────────────────────────────


class _FakeCoordinator:
    """Coordinator stand-in - echoes the goal as one finding, no model needed."""

    def __init__(self, *_a: object, **_kw: object) -> None:
        pass

    def investigate(self, _ctx: object, goal: str | None = None) -> list[Finding]:
        return [
            Finding(
                agent="coordinator",
                kind="overnight-lows",
                scope="global",
                headline=f"Investigated: {goal}",
                confidence=0.7,
                status=FindingStatus.ACTIVE,
            )
        ]


def test_investigate_persists_findings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    _seed_glucose(store)  # 10 days → above the hard floor
    monkeypatch.setattr(
        "dexta_intelligence.agents.coordinator.CoordinatorAgent", _FakeCoordinator
    )
    monkeypatch.setattr(
        "dexta_intelligence.server.app.discovery_model", lambda _cfg: object()
    )
    resp = _client(store).post(
        "/actions/investigate",
        data={"question": "what drives my overnight lows?"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert "investigate_ok:1" in resp.headers["location"]
    headlines = [f.headline for f in store.get_findings(status=None, limit=100)]
    assert "Investigated: what drives my overnight lows?" in headlines
    store.close()


def test_investigate_below_floor_skips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    _seed_glucose(store, days=1.0)  # below the 3-day hard floor
    monkeypatch.setattr(
        "dexta_intelligence.server.app.discovery_model", lambda _cfg: object()
    )
    resp = _client(store).post(
        "/actions/investigate",
        data={"question": "anything"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert "investigate_skip" in resp.headers["location"]
    assert store.get_findings(status=None, limit=100) == []
    store.close()


def test_investigate_empty_question_flashes(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_glucose(store)
    resp = _client(store).post(
        "/actions/investigate",
        data={"question": "   "},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert "investigate_empty" in resp.headers["location"]
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
    assert "Round 1: ran observation, pattern" in body  # the trace is shown
    store.close()


def test_investigations_empty_state(tmp_path: Path) -> None:
    store = _store(tmp_path)
    body = _client(store).get("/investigations").text
    assert "No investigations yet" in body
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
    _seed_run(store)  # populates the investigation log tab
    body = _client(store).get("/findings").text
    assert "Active findings" in body
    assert "Overnight lows after evening exercise" in body
    assert "Open hypotheses" in body
    assert "Investigation log" in body
    assert "evidence" in body  # evidence-strength chip
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
    assert "baseline" in body  # progress labels
    assert "target" in body
    assert "58.0" in body  # current value surfaced
    assert "checkpoint(s)" in body  # history affordance
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
    assert "Nightscout" in body  # a connector-backed source row
    assert "read-only" in body  # the read-only badge
    assert "not configured" in body  # nothing configured in a fresh store
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
