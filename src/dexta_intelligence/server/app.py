"""FastAPI app factory for ``dexta serve``.

``create_app(config, store_opener)`` wires the routes against the same store
seam the CLI uses (the ``store_opener`` callable), so tests can inject a seeded
SQLiteStore and the real server can pass ``open_sqlite_store``. All third-party
imports are lazy with a clear install hint — the base package never depends on
the GUI stack.
"""

from __future__ import annotations

import io
import json
import os
import queue
import threading
import uuid
from datetime import UTC, datetime, time
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dexta_intelligence.cli._common import (
    _analysis_window,
    build_connectors,
    discovery_model,
    is_carelink_configured,
    is_dexcom_api_configured,
    is_dexcom_configured,
    is_libre_configured,
    is_nightscout_configured,
    is_oura_configured,
    is_tandem_configured,
    is_tidepool_configured,
    is_whoop_configured,
    model_for_role,
    open_sqlite_store,
)
from dexta_intelligence.coldstart import CAPABILITY_GATES, HARD_FLOOR_DAYS, ColdStartReport
from dexta_intelligence.config import (
    env_override_for,
    load_config,
    save_config_values,
    save_secret,
    secrets_path_for,
)
from dexta_intelligence.connectors.base import HealthReport
from dexta_intelligence.models import ChatTurn, FindingStatus
from dexta_intelligence.server.render import markdown_to_html, sparkline_svg
from dexta_intelligence.server.settings_render import field_to_view, panel_to_view
from dexta_intelligence.server.settings_schema import (
    ANALYSIS_PANEL,
    DATA_FIELDS,
    PANELS_BY_KEY,
    SETTINGS_OVERVIEW,
    SETTINGS_PANELS,
    WIKI_FIELDS,
    FieldKind,
    source_nav,
)

# FastAPI resolves route annotations against this module's globals, so the GUI
# types must live here at import time. They are an optional extra, so a missing
# install degrades to ``None`` and ``create_app`` raises a friendly RuntimeError.
try:
    from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
    from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
except ModuleNotFoundError:  # pragma: no cover - exercised only without the extra
    FastAPI = Form = HTTPException = Request = HTMLResponse = RedirectResponse = None  # type: ignore[assignment,misc]
    StreamingResponse = File = UploadFile = None  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from collections.abc import Callable

    from dexta_intelligence.cli._common import StoreOpener
    from dexta_intelligence.config import Config
    from dexta_intelligence.models import Finding, Goal, InvestigationRun
    from dexta_intelligence.store.port import StoragePort

_INSTALL_HINT = (
    "The web GUI needs the optional GUI stack. Install it with:\n"
    "  pip install 'dexta-intelligence[gui]'"
)

_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")

#: Cap on retained chat turns per session (messages, not turns) — bounds context.
_MAX_SESSION_MESSAGES = 12

_SOURCE_CONFIGURED: dict[str, Callable[[Config], bool]] = {
    "nightscout": is_nightscout_configured,
    "dexcom": is_dexcom_configured,
    "libre": is_libre_configured,
    "whoop": is_whoop_configured,
    "oura": is_oura_configured,
    "tidepool": is_tidepool_configured,
    "tandem": is_tandem_configured,
    "carelink": is_carelink_configured,
    "dexcom_api": is_dexcom_api_configured,
}


def _require_gui() -> None:
    if FastAPI is None:  # the [gui] extra is not installed
        raise RuntimeError(_INSTALL_HINT)
    try:
        import jinja2  # noqa: F401, PLC0415
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via install
        raise RuntimeError(_INSTALL_HINT) from exc


def create_app(  # noqa: PLR0915 - a route table; each handler is small
    config: Config,
    store_opener: StoreOpener = open_sqlite_store,
    config_path: Path | None = None,
    host: str = "127.0.0.1",
) -> FastAPI:
    """Build the GUI app bound to a config and a store-opener seam.

    ``config_path`` is the config file the running server was launched with;
    the settings panel reads back from and writes to *that* file (captured once
    here, never re-resolved per request). Falls back to the boot-time default.
    ``host`` is the bind address: a non-loopback bind disables credential
    editing (status-only) unless ``DEXTA_ALLOW_REMOTE_SETTINGS=1``.
    """
    _require_gui()

    from dexta_intelligence.cli._common import resolve_config_path  # noqa: PLC0415

    settings_path = config_path if config_path is not None else resolve_config_path(None)

    from fastapi.staticfiles import StaticFiles  # noqa: PLC0415
    from fastapi.templating import Jinja2Templates  # noqa: PLC0415

    pkg = resources.files("dexta_intelligence.server")
    templates_dir = Path(str(pkg / "templates"))
    static_dir = Path(str(pkg / "static"))

    app = FastAPI(title="dexta", docs_url=None, redoc_url=None)
    app.state.config_path = settings_path
    app.state.bind_host = host
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    templates = Jinja2Templates(directory=str(templates_dir))
    _static_stamps = [f.stat().st_mtime for f in static_dir.iterdir() if f.is_file()]
    templates.env.globals["static_version"] = str(int(max(_static_stamps, default=0)))
    templates.env.globals["nav_items"] = (
        ("/", "Dashboard"),
        ("/investigations", "Investigations"),
        ("/wiki", "Wiki"),
        ("/goals", "Goals"),
        ("/chat", "Chat"),
        ("/settings", "Settings"),
    )
    templates.env.globals["source_nav"] = source_nav()

    def _render(
        name: str,
        request: Request,
        active: str = "",
        status_code: int = 200,
        status_pill: str | None = None,
        **kw: Any,
    ) -> Any:
        if status_pill is None:
            store = store_opener(config, None)
            try:
                status_pill = _status_pill_text(store.coverage())
            finally:
                _close(store, store_opener)
        return templates.TemplateResponse(
            request,
            name,
            {"active": active, "status_pill": status_pill, **kw},
            status_code=status_code,
        )

    # ── dashboard ─────────────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request) -> Any:
        flash = request.query_params.get("flash")
        store = store_opener(config, None)
        try:
            coverage = store.coverage()
            gates = ColdStartReport.from_coverage(coverage)
            findings = store.get_findings(status=None, limit=1_000_000)
            has_connectors = bool(build_connectors(config))
            hero = _hero_metrics(store, config, coverage, findings)
            sidebar = _status_sidebar(config, coverage, gates, store)
            status_pill = _status_pill_text(coverage)
        finally:
            _close(store, store_opener)

        active_findings = [
            f
            for f in findings
            if f.status == FindingStatus.ACTIVE and f.kind not in _INTERNAL_FINDING_KINDS
        ]
        graveyard = [
            f
            for f in findings
            if f.status
            in (FindingStatus.REJECTED, FindingStatus.SUPERSEDED, FindingStatus.DISMISSED)
        ]
        cards = [_finding_card(f, active_findings) for f in _ranked(active_findings)]
        banners = _dashboard_banners(
            coverage,
            gates,
            flash=flash,
            has_connectors=has_connectors,
        )
        return _render(
            "dashboard.html",
            request,
            "/",
            status_pill=status_pill,
            hero=hero,
            sidebar=sidebar,
            banners=banners,
            has_connectors=has_connectors,
            below_floor=gates.below_hard_floor,
            cards=cards,
            graveyard=[_graveyard_row(f) for f in graveyard],
            lenses=_lens_names(config),
        )

    @app.post("/actions/sync")
    def action_sync() -> Any:
        from dexta_intelligence.cli.data import cmd_sync  # noqa: PLC0415

        buf = io.StringIO()
        code = cmd_sync(config=config, db_path=None, out=buf)
        flash = "sync_ok" if code == 0 else "sync_fail"
        return RedirectResponse(f"/?flash={flash}", status_code=303)

    @app.post("/actions/analyze")
    def action_analyze(lens: str = Form("analyze")) -> Any:
        from dexta_intelligence.cli.analysis import cmd_analyze  # noqa: PLC0415

        buf = io.StringIO()
        try:
            code = cmd_analyze(config=config, db_path=None, out=buf, lens=lens)
        except ValueError:  # unknown lens — should not happen from the picker
            return RedirectResponse("/?flash=analyze_fail", status_code=303)
        if code == 0:
            flash = "analyze_ok"
        elif "Need at least" in buf.getvalue():
            flash = "analyze_skip"
        else:
            flash = "analyze_fail"
        return RedirectResponse(f"/?flash={flash}", status_code=303)

    @app.post("/actions/investigate")
    def action_investigate(question: str = Form(...)) -> Any:
        from dexta_intelligence.agents.base import AgentContext  # noqa: PLC0415
        from dexta_intelligence.agents.coordinator import CoordinatorAgent  # noqa: PLC0415
        from dexta_intelligence.workflows.deep_analysis import persist_findings  # noqa: PLC0415

        goal = question.strip()
        if not goal:
            return RedirectResponse("/?flash=investigate_empty", status_code=303)
        store = store_opener(config, None)
        try:
            coverage = store.coverage()
            gates = ColdStartReport.from_coverage(coverage)
            if gates.below_hard_floor:
                return RedirectResponse("/?flash=investigate_skip", status_code=303)
            end = coverage.last_ts.date() if coverage.last_ts is not None else None
            ctx = AgentContext(
                store=store,
                window=_analysis_window(config, end),
                gates=gates,
                run_id=f"gui-investigate-{uuid.uuid4()}",
                timezone=config.analysis.timezone,
            )
            coordinator = CoordinatorAgent(model=discovery_model(config), config=config)
            findings = coordinator.investigate(ctx, goal=goal)
            persisted = persist_findings(store, findings)
            flash = f"investigate_ok:{len(persisted)}"
        except Exception:
            flash = "investigate_fail"
        finally:
            _close(store, store_opener)
        return RedirectResponse(f"/?flash={flash}", status_code=303)

    @app.get("/investigations", response_class=HTMLResponse)
    def investigations(request: Request) -> Any:
        store = store_opener(config, None)
        try:
            runs = store.get_investigation_runs(limit=50)
        finally:
            _close(store, store_opener)
        now = datetime.now(tz=UTC)
        return _render(
            "investigations.html",
            request,
            "/investigations",
            runs=[_run_view(r, now) for r in runs],
        )

    @app.post("/actions/upload")
    async def action_upload(file: UploadFile = File(...)) -> Any:  # noqa: B008 - FastAPI idiom
        """Ingest a Dexcom Clarity / LibreView CSV export through the same sync
        path as a live connector (format auto-detected; re-uploading is safe)."""
        import tempfile  # noqa: PLC0415

        from dexta_intelligence.connectors.csv_upload import CSVUploadConnector  # noqa: PLC0415
        from dexta_intelligence.workflows.sync import sync  # noqa: PLC0415

        data = await file.read()
        if not data:
            return RedirectResponse("/?flash=upload_empty", status_code=303)

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        store = store_opener(config, None)
        try:
            connector = CSVUploadConnector(
                tmp_path, format="auto", tz=config.analysis.timezone
            )
            report = sync(connector, store)
            flash = f"upload_ok:{report.inserted.get('glucose', 0)}"
        except Exception:
            flash = "upload_fail"
        finally:
            _close(store, store_opener)
            tmp_path.unlink(missing_ok=True)
        return RedirectResponse(f"/?flash={flash}", status_code=303)

    # ── wiki ──────────────────────────────────────────────────────────────────

    @app.get("/wiki", response_class=HTMLResponse)
    def wiki_index(request: Request) -> Any:
        return _wiki_page(request, "index")

    @app.get("/wiki/{page:path}", response_class=HTMLResponse)
    def wiki_page(request: Request, page: str) -> Any:
        return _wiki_page(request, page)

    def _wiki_page(request: Request, page: str) -> Any:
        root = config.wiki.path.expanduser().resolve()
        slug = page[:-3] if page.endswith(".md") else page
        md_path = (root / f"{slug}.md").resolve()
        # Containment check — proper path containment, not a prefix match, so a
        # sibling dir sharing the root's string prefix (or ``..``) can't escape.
        contained = md_path == root or md_path.is_relative_to(root)
        if not contained or not md_path.is_file():
            return _render(
                "wiki.html",
                request,
                "/wiki",
                body=None,
                slug=slug,
                root=str(root),
                nav=_wiki_nav(root, slug),
            )
        raw_md = md_path.read_text(encoding="utf-8")
        body = _rewrite_wiki_links(_wrap_wiki_tables(markdown_to_html(raw_md)))
        return _render(
            "wiki.html",
            request,
            "/wiki",
            body=body,
            slug=slug,
            root=str(root),
            nav=_wiki_nav(root, slug),
            title=_wiki_title(raw_md, slug),
        )

    # ── goals ─────────────────────────────────────────────────────────────────

    def _goals_page(
        request: Request,
        *,
        saved: bool = False,
        error: str | None = None,
    ) -> Any:
        from dexta_intelligence.models import GoalStatus  # noqa: PLC0415

        store = store_opener(config, None)
        try:
            cards = [
                _goal_card(store, g)
                for g in store.get_goals()
                if g.status in (GoalStatus.ACTIVE, GoalStatus.ACHIEVED)
            ]
        finally:
            _close(store, store_opener)
        return _render(
            "goals.html",
            request,
            "/goals",
            cards=cards,
            saved=saved,
            error=error,
        )

    @app.get("/goals", response_class=HTMLResponse)
    def goals(request: Request) -> Any:
        qp = request.query_params
        return _goals_page(
            request,
            saved=qp.get("saved") == "1",
            error=qp.get("error"),
        )

    @app.post("/goals", response_class=HTMLResponse)
    async def goals_add(request: Request) -> Any:
        from dexta_intelligence.models import GoalStatus  # noqa: PLC0415
        from dexta_intelligence.workflows.goals import compose_goal  # noqa: PLC0415

        form = await request.form()
        raw_statement = form.get("statement")
        statement = raw_statement.strip() if isinstance(raw_statement, str) else ""
        if not statement:
            return _goals_page(request, error="Describe what you want to improve.")

        target: float | None = None
        raw_target = form.get("target")
        if isinstance(raw_target, str) and raw_target.strip():
            try:
                target = float(raw_target.strip())
            except ValueError:
                return _goals_page(request, error="Target must be a number.")

        store = store_opener(config, None)
        try:
            normalized = statement.casefold()
            if any(
                g.statement.strip().casefold() == normalized
                for g in store.get_goals(status=GoalStatus.ACTIVE)
            ):
                return _goals_page(
                    request,
                    error="You already have an active goal with that statement.",
                )
            goal = compose_goal(
                statement,
                model=model_for_role(config, "plan"),
                now=datetime.now(tz=UTC),
                target=target,
            )
            store.insert_goal(goal)
        finally:
            _close(store, store_opener)
        return RedirectResponse("/goals?saved=1", status_code=303)

    @app.post("/goals/{goal_id}/abandon", response_class=HTMLResponse)
    def goals_abandon(request: Request, goal_id: int) -> Any:
        from dexta_intelligence.models import GoalStatus  # noqa: PLC0415

        store = store_opener(config, None)
        try:
            match = next((g for g in store.get_goals() if g.id == goal_id), None)
            if match is None:
                raise HTTPException(status_code=404)
            if match.status is GoalStatus.ACTIVE:
                store.set_goal_status(goal_id, GoalStatus.ABANDONED)
        finally:
            _close(store, store_opener)
        return RedirectResponse("/goals", status_code=303)

    # ── chat ──────────────────────────────────────────────────────────────────

    @app.get("/chat", response_class=HTMLResponse)
    def chat(request: Request) -> Any:
        has_model = discovery_model(config) is not None
        return _render("chat.html", request, "/chat", has_model=has_model)

    @app.post("/api/ask", response_class=HTMLResponse)
    def api_ask(request: Request, question: str = Form(...)) -> Any:
        from dexta_intelligence.agents.base import AgentContext  # noqa: PLC0415
        from dexta_intelligence.agents.chat import ChatAgent  # noqa: PLC0415

        model = getattr(request.app.state, "chat_model", None) or discovery_model(config)
        if model is None:
            return _render("_answer.html", request, answer=None, question=question)
        store = store_opener(config, None)
        try:
            coverage = store.coverage()
            gates = ColdStartReport.from_coverage(coverage)
            end = coverage.last_ts.date() if coverage.last_ts is not None else None
            from dexta_intelligence.cli._common import _analysis_window  # noqa: PLC0415

            ctx = AgentContext(
                store=store,
                window=_analysis_window(config, end),
                gates=gates,
                run_id="gui",
                timezone=config.analysis.timezone,
            )
            agent = ChatAgent(
                model=model,
                max_steps=config.analysis.max_reasoning_steps,
                target_low=config.analysis.target_low,
                target_high=config.analysis.target_high,
            )
            answer = agent.ask(ctx, question)
        finally:
            _close(store, store_opener)
        return _render(
            "_answer.html",
            request,
            question=question,
            answer=answer.text,
            answer_html=markdown_to_html(answer.text),
            tools=list(answer.tools_used),
            faithful=answer.faithful,
        )

    @app.get("/api/ask/stream")
    async def api_ask_stream(  # noqa: PLR0915 - queue + worker + drain in one handler
        request: Request, q: str, sid: str | None = None
    ) -> Any:
        """Stream an orchestrator run to the browser as Server-Sent Events.

        The orchestrator is synchronous and calls the model synchronously, so it
        runs in a worker thread; its ``on_event`` sink pushes each
        ``ReasoningEvent`` onto a thread-safe queue that this async generator
        drains into ``text/event-stream`` frames. The stream ends after the
        ``answer`` event (or a terminal ``error``).

        ``sid`` (optional) keys an in-memory conversation: prior turns are seeded
        so follow-ups carry context, and this turn is appended after it answers.
        """
        question = q.strip()
        model = getattr(request.app.state, "chat_model", None) or discovery_model(config)
        if model is None:

            async def _no_model() -> Any:
                yield _sse(
                    {
                        "kind": "error",
                        "payload": {
                            "text": (
                                "Chat needs a language model. Set a provider in Settings "
                                "and add an API key to your environment."
                            )
                        },
                    }
                )

            return StreamingResponse(_no_model(), media_type="text/event-stream")

        events: queue.Queue[Any] = queue.Queue()
        done = object()

        def _run() -> None:
            from dexta_intelligence.agents.base import AgentContext  # noqa: PLC0415
            from dexta_intelligence.agents.orchestrator import OrchestratorAgent  # noqa: PLC0415

            store = store_opener(config, None)
            try:
                coverage = store.coverage()
                gates = ColdStartReport.from_coverage(coverage)
                end = coverage.last_ts.date() if coverage.last_ts is not None else None
                ctx = AgentContext(
                    store=store,
                    window=_analysis_window(config, end),
                    gates=gates,
                    run_id="gui-stream",
                    timezone=config.analysis.timezone,
                )
                agent = OrchestratorAgent(
                    model=model,
                    max_steps=config.analysis.max_reasoning_steps,
                    target_low=config.analysis.target_low,
                    target_high=config.analysis.target_high,
                )
                history = None
                if sid:
                    prior = store.get_chat_turns(sid, limit=_MAX_SESSION_MESSAGES)
                    history = [{"role": t.role, "content": t.content} for t in prior]

                def _sink(event: Any) -> None:
                    # Drop the loop's legacy full-text answer — the endpoint emits
                    # audited prose after guard/treatment rails.
                    if event.kind == "answer":
                        return
                    events.put({"kind": event.kind, "payload": event.payload})

                answer = agent.ask(ctx, question, on_event=_sink, history=history)
                # Persist before signalling the answer so a session-list refresh
                # triggered by the answer event already sees this conversation.
                if sid:
                    now = datetime.now(UTC)
                    store.append_chat_turn(
                        ChatTurn(session_id=sid, role="user", content=question, ts=now)
                    )
                    store.append_chat_turn(
                        ChatTurn(session_id=sid, role="assistant", content=answer.text, ts=now)
                    )
                events.put(
                    {
                        "kind": "answer",
                        "payload": {
                            "text": answer.text,
                            "html": markdown_to_html(answer.text),
                            "tools": list(answer.tools_used),
                            "faithful": answer.faithful,
                        },
                    }
                )
            except Exception as exc:
                events.put(
                    {
                        "kind": "error",
                        "payload": {"text": f"{type(exc).__name__}: {exc}"},
                    }
                )
            finally:
                _close(store, store_opener)
                events.put(done)

        async def _drain() -> Any:
            import asyncio  # noqa: PLC0415

            worker = threading.Thread(target=_run, daemon=True)
            worker.start()
            while True:
                if await request.is_disconnected():
                    break
                try:
                    item = await asyncio.to_thread(events.get, True, 0.25)
                except queue.Empty:
                    continue
                if item is done:
                    break
                yield _sse(item)

        return StreamingResponse(
            _drain(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/history")
    def api_history(sid: str | None = None) -> Any:
        """Past turns for a session so returning to Chat restores the conversation.
        Durable: read from the storage backend, so it survives a server restart."""
        if not sid:
            return {"turns": []}
        store = store_opener(config, None)
        try:
            turns = store.get_chat_turns(sid, limit=_MAX_SESSION_MESSAGES)
        finally:
            _close(store, store_opener)
        return {
            "turns": [
                {
                    "role": t.role,
                    "content": t.content,
                    "html": markdown_to_html(t.content) if t.role == "assistant" else "",
                }
                for t in turns
            ]
        }

    @app.get("/api/sessions")
    def api_sessions() -> Any:
        """Past conversations (newest-active first) for the chat history rail."""
        store = store_opener(config, None)
        try:
            sessions = store.get_chat_sessions(limit=50)
        finally:
            _close(store, store_opener)
        now = datetime.now(tz=UTC)
        return {
            "sessions": [
                {
                    "session_id": s.session_id,
                    "preview": s.preview or "(no message)",
                    "turn_count": s.turn_count,
                    "relative": _relative_time(s.last_ts, now),
                }
                for s in sessions
            ]
        }

    @app.delete("/api/sessions/{session_id}")
    def api_delete_session(session_id: str) -> Any:
        """Remove one chat conversation and all its turns."""
        store = store_opener(config, None)
        try:
            deleted = store.delete_chat_session(session_id)
        finally:
            _close(store, store_opener)
        if deleted == 0:
            return JSONResponse({"ok": False, "deleted": 0}, status_code=404)
        return {"ok": True, "deleted": deleted}

    # ── settings ──────────────────────────────────────────────────────────────

    def _remote_bind() -> bool:
        return app.state.bind_host not in _LOOPBACK_HOSTS

    def _settings_cfg() -> Config:
        # Re-read the launched file so card saves show up without a restart;
        # env precedence is applied by the loader, same as boot.
        return load_config(app.state.config_path)

    def _freshness_map() -> dict[str, datetime]:
        try:
            store = store_opener(config, None)
        except Exception:
            return {}
        try:
            sources = [s.connector for s in SETTINGS_PANELS if s.connector is not None]
            marks = {src: store.get_watermark(src) for src in sources}
            return {src: ts for src, ts in marks.items() if ts is not None}
        except Exception:
            return {}
        finally:
            _close(store, store_opener)

    def _settings_context(request: Request, **extra: Any) -> dict[str, Any]:
        cfg = _settings_cfg()
        remote_bind = _remote_bind()
        fresh = _freshness_map()
        cards = [
            _card_view(
                spec,
                cfg,
                freshness=fresh.get(spec.connector) if spec.connector else None,
            )
            for spec in SETTINGS_PANELS
        ]
        connection_cards = [c for c in cards if c.get("category", "connection") == "connection"]
        intel_cards = [c for c in cards if c.get("category") == "intelligence"]
        settings_nav = [
            *[
                {"key": c["key"], "label": c["title"], "group": "Connections",
                 "ok": c["configured"]}
                for c in connection_cards
            ],
            *[
                {"key": c["key"], "label": c["title"], "group": "Intelligence",
                 "ok": c["configured"]}
                for c in intel_cards
            ],
            {"key": "analysis", "label": "Analysis & storage", "group": "System", "ok": True},
        ]
        default_panel = request.query_params.get("panel") or (
            connection_cards[0]["key"] if connection_cards else "analysis"
        )
        lenses = sorted(cfg.lens) or ["analyze", "watch", "why", "insulin"]
        analysis_fields = [
            field_to_view("analysis", f, cfg.analysis, editable=True) for f in ANALYSIS_PANEL.fields
        ]
        data_fields = [
            field_to_view("data", f, cfg.data, editable=True) for f in DATA_FIELDS
        ]
        wiki_fields = [
            field_to_view("wiki", f, cfg.wiki, editable=True) for f in WIKI_FIELDS
        ]
        return {
            "cfg": cfg,
            "config_path": str(request.app.state.config_path),
            "cards": cards,
            "cards_by_key": {c["key"]: c for c in cards},
            "connection_cards": connection_cards,
            "intel_cards": intel_cards,
            "settings_nav": settings_nav,
            "default_panel": default_panel,
            "analysis_fields": analysis_fields,
            "data_fields": data_fields,
            "analysis_note": ANALYSIS_PANEL.note,
            "wiki_fields": wiki_fields,
            "setup_overview": SETTINGS_OVERVIEW,
            "editable": True,
            "remote_bind": remote_bind,
            "lenses": lenses,
            **extra,
        }

    @app.get("/settings", response_class=HTMLResponse)
    def settings(request: Request) -> Any:
        return _render(
            "settings.html",
            request,
            "/settings",
            **_settings_context(request, saved=request.query_params.get("saved") == "1"),
        )

    @app.post("/settings")
    def save_settings(
        request: Request,
        target_low: str = Form(...),
        target_high: str = Form(...),
        max_reasoning_steps: str = Form(...),
        deep_analysis_window_days: str = Form(...),
        path: str = Form(...),
        git: str = Form("off"),
        backend: str = Form(...),
        sqlite_path: str = Form(...),
        database_url: str = Form(""),
    ) -> Any:
        try:
            low = _positive_int("target low", target_low)
            high = _positive_int("target high", target_high)
            steps = _positive_int("max tool calls per question", max_reasoning_steps)
            window = _positive_int("analysis window (days)", deep_analysis_window_days)
            if low >= high:
                raise _SettingsError("Target low must be below target high.")
            if steps < 4 or steps > 64:
                raise _SettingsError("Max tool calls per question must be between 4 and 64.")
            if backend not in ("sqlite", "postgres"):
                raise _SettingsError("Storage backend must be sqlite or postgres.")
            updates: dict[str, dict[str, Any]] = {
                "analysis": {
                    "target_low": low,
                    "target_high": high,
                    "max_reasoning_steps": steps,
                    "deep_analysis_window_days": window,
                },
                "wiki": {"path": path, "git": git in ("on", "true", "1")},
                "data": {
                    "backend": backend,
                    "sqlite_path": sqlite_path.strip() or "~/.dexta/dexta.db",
                },
            }
            dsn = database_url.strip()
            if dsn:
                updates["data"]["database_url"] = dsn
            save_config_values(updates, path=request.app.state.config_path)
        except (ValueError, TypeError) as exc:
            return _render(
                "settings.html",
                request,
                "/settings",
                status_code=400,
                **_settings_context(request, error=str(exc)),
            )
        return RedirectResponse("/settings?saved=1", status_code=303)

    @app.post("/settings/{source_key}", response_class=HTMLResponse)
    async def save_source(request: Request, source_key: str) -> Any:
        spec = PANELS_BY_KEY.get(source_key)
        if spec is None:
            raise HTTPException(status_code=404)

        form = await request.form()
        secrets_path = secrets_path_for(request.app.state.config_path)
        for var, _label in spec.env_keys:
            raw = form.get(f"env__{var}")
            if raw is None:
                continue
            value = raw.strip() if isinstance(raw, str) else ""
            if not value:
                continue
            try:
                save_secret(var, value, path=secrets_path)
            except ValueError as exc:
                card = _card_view(spec, _settings_cfg(), error=str(exc))
                return _render("_settings_card.html", request, status_code=400, card=card)

        updates: dict[str, Any] = {}
        for f in spec.fields:
            if env_override_for(spec.section, f.name) is not None:
                continue
            if f.kind == FieldKind.CHECKBOX:
                updates[f.name] = form.get(f.name) in ("on", "true", "1")
                continue
            raw = form.get(f.name)
            value = raw.strip() if isinstance(raw, str) else ""
            if f.secret and not value:
                continue
            updates[f.name] = value
        try:
            save_config_values({spec.section: updates}, path=request.app.state.config_path)
        except (ValueError, TypeError) as exc:
            card = _card_view(spec, _settings_cfg(), error=str(exc))
            return _render("_settings_card.html", request, status_code=400, card=card)
        card = _card_view(
            spec,
            _settings_cfg(),
            freshness=_freshness_map().get(spec.connector) if spec.connector else None,
            saved=True,
        )
        return _render("_settings_card.html", request, card=card)

    @app.post("/settings/{source_key}/test", response_class=HTMLResponse)
    def test_source(request: Request, source_key: str) -> Any:
        spec = PANELS_BY_KEY.get(source_key)
        if spec is None or spec.connector is None:
            raise HTTPException(status_code=404)
        cfg = _settings_cfg()
        connector = next(
            (c for c in build_connectors(cfg) if c.source == spec.connector), None
        )
        if connector is None:
            return _render("_settings_test.html", request, report=None)
        try:
            # Tandem's check() already defaults to a 25s timeout; the protocol's
            # check() takes no args, so call it plainly for every connector.
            report = connector.check()
        except Exception as exc:
            report = HealthReport(ok=False, source=spec.connector, detail=str(exc))
        return _render("_settings_test.html", request, report=report)

    @app.post("/settings/{source_key}/sync", response_class=HTMLResponse)
    def sync_source(request: Request, source_key: str) -> Any:
        from dexta_intelligence.workflows.sync import sync  # noqa: PLC0415

        spec = PANELS_BY_KEY.get(source_key)
        if spec is None or spec.connector is None:
            raise HTTPException(status_code=404)
        cfg = _settings_cfg()
        connector = next(
            (c for c in build_connectors(cfg) if c.source == spec.connector), None
        )
        if connector is None:
            card = _card_view(spec, cfg, error="Save credentials first, then sync.")
            return _render("_settings_card.html", request, status_code=400, card=card)
        store = store_opener(cfg, None)
        try:
            report = sync(connector, store)
            if report.ok:
                parts = [f"{k}={v}" for k, v in sorted(report.inserted.items()) if v]
                detail = ", ".join(parts) if parts else "no new rows"
                sync_msg = f"Sync OK · {detail}"
            else:
                sync_msg = report.errors[0] if report.errors else "Sync failed"
            card = _card_view(
                spec,
                cfg,
                freshness=_freshness_map().get(spec.connector),
                sync_msg=sync_msg,
                sync_ok=report.ok,
            )
        finally:
            _close(store, store_opener)
        return _render("_settings_card.html", request, card=card)

    return app


# ── view helpers (pure, testable) ─────────────────────────────────────────────

_SOURCE_LABELS: dict[str, str] = {
    "nightscout": "Nightscout",
    "dexcom": "Dexcom Share",
    "libre": "LibreLinkUp",
    "whoop": "Whoop",
    "oura": "Oura",
    "tidepool": "Tidepool",
    "tandem": "Tandem",
    "carelink": "CareLink",
    "dexcom_api": "Dexcom API",
}


def _lens_names(config: Config) -> list[str]:
    """Lens picker options: built-ins overlaid with user ``[lens.*]`` entries."""
    from dexta_intelligence.workflows.lenses import BUILTIN_LENSES  # noqa: PLC0415

    return sorted({**BUILTIN_LENSES, **config.lens})


def _sse(event: dict[str, Any]) -> str:
    """Serialize one ``{kind, payload}`` event as a Server-Sent Events frame."""
    return f"data: {json.dumps(event, default=str)}\n\n"


def _status_pill_text(coverage: Any) -> str:
    if coverage.n_glucose == 0:
        return "Local only · no data"
    days = int(coverage.span_days)
    suffix = "s" if days != 1 else ""
    return f"Local only · {days} day{suffix}"


#: Coordinator investigation receipts (kind="investigation") are planning memory
#: — recalled to avoid re-running the same belt — not user-facing findings.
_INTERNAL_FINDING_KINDS = frozenset({"investigation"})


def _hero_metrics(
    store: StoragePort, config: Config, coverage: Any, findings: list[Any]
) -> dict[str, Any]:
    active = sum(
        1
        for f in findings
        if f.status == FindingStatus.ACTIVE and f.kind not in _INTERNAL_FINDING_KINDS
    )
    span = coverage.span_days if coverage.n_glucose else None
    cov_pct = coverage.glucose_coverage_pct if coverage.n_glucose else None
    tir = _recent_tir(store, config, coverage) if coverage.n_glucose else None
    return {
        "span_days": span,
        "coverage_pct": cov_pct,
        "tir_pct": tir,
        "active_findings": active,
    }


def _recent_tir(store: StoragePort, config: Config, coverage: Any) -> float | None:
    from dexta_intelligence.models import RollupPeriod  # noqa: PLC0415

    if coverage.last_ts is None:
        return None
    window = _analysis_window(config, coverage.last_ts.date())
    start_date, end_date = window
    start = datetime.combine(start_date, time.min, tzinfo=UTC)
    end = datetime.combine(end_date, time.max, tzinfo=UTC)
    rollups = store.get_rollups(RollupPeriod.DAILY, start, end)
    vals = [r.tir for r in rollups if r.tir is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _dashboard_banners(
    coverage: Any,
    gates: ColdStartReport,
    *,
    flash: str | None,
    has_connectors: bool,
) -> list[dict[str, str]]:
    banners: list[dict[str, str]] = []
    flash_msgs: dict[str, tuple[str, str]] = {
        "sync_ok": ("ok", "Sync finished successfully."),
        "sync_fail": ("bad", "Sync failed — check Settings and try again."),
        "analyze_ok": ("ok", "Analysis complete — findings refreshed."),
        "analyze_skip": (
            "setup",
            f"Need at least {HARD_FLOOR_DAYS:.0f} days of data before analysis.",
        ),
        "analyze_fail": ("bad", "Analysis failed — see the CLI log for details."),
        "investigate_skip": (
            "setup",
            f"Need at least {HARD_FLOOR_DAYS:.0f} days of data before an investigation.",
        ),
        "investigate_empty": ("bad", "Type a question to investigate."),
        "investigate_fail": ("bad", "Investigation failed — see the CLI log for details."),
    }
    if flash in flash_msgs:
        kind, message = flash_msgs[flash]
        banners.append({"kind": kind, "message": message})
    elif flash and flash.startswith("investigate_ok"):
        n = flash.split(":", 1)[1] if ":" in flash else "0"
        plural = "s" if n != "1" else ""
        banners.append(
            {"kind": "ok", "message": f"Investigation complete — {n} finding{plural} banked."}
        )
    elif flash and flash.startswith("upload_ok"):
        n = flash.split(":", 1)[1] if ":" in flash else "0"
        banners.append({"kind": "ok", "message": f"Imported {n} glucose rows from CSV."})
    elif flash == "upload_fail":
        banners.append(
            {
                "kind": "bad",
                "message": "CSV import failed — expected a Dexcom Clarity or LibreView export.",
            }
        )
    elif flash == "upload_empty":
        banners.append({"kind": "bad", "message": "No file was selected to upload."})

    if coverage.n_glucose == 0:
        banners.append(
            {
                "kind": "setup",
                "message": "Connect a data source or upload a CSV to get started.",
                "href": "/settings",
                "label": "Open Settings",
            }
        )
    elif not has_connectors and coverage.n_glucose > 0:
        banners.append(
            {
                "kind": "info",
                "message": "Data loaded locally — connect a source in Settings to keep syncing.",
                "href": "/settings",
                "label": "Settings",
            }
        )
    elif gates.below_hard_floor and flash not in ("analyze_skip",):
        banners.append(
            {
                "kind": "setup",
                "message": (
                    f"Collect {HARD_FLOOR_DAYS:.0f}+ days of data to unlock analysis "
                    f"(have {coverage.span_days:.0f})."
                ),
            }
        )
    return banners


def _status_sidebar(
    config: Config,
    coverage: Any,
    gates: ColdStartReport,
    store: StoragePort,
) -> dict[str, Any]:
    now = datetime.now(tz=UTC)
    sources: list[dict[str, str]] = []
    for conn in build_connectors(config):
        label = _SOURCE_LABELS.get(conn.source, conn.source)
        ts = store.get_watermark(conn.source)
        fresh = _relative_time(ts, now) if ts is not None else ""
        sources.append({"label": label, "fresh": fresh})

    if not sources and coverage.n_glucose:
        for label in _detect_sources(coverage):
            sources.append({"label": label.title(), "fresh": ""})

    pending = list(gates.pending.values())[:3]
    unlocked_count = sum(1 for g in CAPABILITY_GATES if g.capability in gates.unlocked)

    last_ts: datetime | None = None
    for conn in build_connectors(config):
        ts = store.get_watermark(conn.source)
        if ts is not None and (last_ts is None or ts > last_ts):
            last_ts = ts
    last_sync = _relative_time(last_ts, now) if last_ts is not None else None

    return {
        "sources": sources,
        "pending": pending,
        "unlocked_count": unlocked_count,
        "last_sync": last_sync,
        "stream_counts": {
            "glucose": coverage.n_glucose,
            "insulin": coverage.n_insulin,
            "meals": coverage.n_meals,
        },
        "storage": _storage_view(config),
    }


def _storage_view(config: Config) -> dict[str, str]:
    """Where data actually lives — so 'is my DB local?' has a visible answer."""
    if config.data.backend == "sqlite":
        path = config.data.sqlite_path.expanduser()
        try:
            kb = path.stat().st_size / 1024
            size = f"{kb / 1024:.1f} MB" if kb >= 1024 else f"{kb:.0f} KB"
        except OSError:
            size = "new"
        return {
            "backend": "SQLite",
            "detail": "local file · no server",
            "path": str(path),
            "size": size,
        }
    return {
        "backend": "PostgreSQL",
        "detail": "external server",
        "path": config.data.database_url or "(DATABASE_URL)",
        "size": "",
    }


def _relative_time(ts: datetime, now: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    delta = now - ts.astimezone(UTC)
    secs = int(delta.total_seconds())
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{delta.days}d ago"


def _card_configured(spec: Any, cfg: Config) -> bool:
    if spec.key == "llm":
        return any(os.environ.get(var) for var, _ in spec.env_keys)
    if spec.key == "evidence":
        return cfg.evidence.enabled and (
            cfg.evidence.backend != "openevidence"
            or bool(os.environ.get("OPENEVIDENCE_API_KEY"))
        )
    return _SOURCE_CONFIGURED[spec.key](cfg)


def _card_view(
    spec: Any,
    cfg: Config,
    *,
    freshness: datetime | None = None,
    saved: bool = False,
    error: str | None = None,
    sync_msg: str | None = None,
    sync_ok: bool | None = None,
) -> dict[str, Any]:
    return panel_to_view(
        spec,
        cfg,
        configured=_card_configured(spec, cfg),
        editable=True,
        freshness=freshness,
        saved=saved,
        error=error,
        sync_msg=sync_msg,
        sync_ok=sync_ok,
    )


class _SettingsError(ValueError):
    """A user-facing settings validation failure (re-rendered, never persisted)."""


def _positive_int(label: str, raw: str) -> int:
    """Parse a non-negative integer or raise a user-facing error."""
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise _SettingsError(f"{label.capitalize()} must be a whole number.") from None
    if value < 0:
        raise _SettingsError(f"{label.capitalize()} must not be negative.")
    return value


def _close(store: StoragePort, opener: StoreOpener) -> None:
    if opener is open_sqlite_store and hasattr(store, "close"):
        store.close()


def _detect_sources(coverage: Any) -> list[str]:
    sources: list[str] = []
    if coverage.n_glucose:
        sources.append("glucose")
    if coverage.n_insulin:
        sources.append("insulin")
    if coverage.n_sleep:
        sources.append("sleep")
    if coverage.n_activity:
        sources.append("activity")
    return sources


def _ranked(findings: list[Finding]) -> list[Finding]:
    def key(f: Finding) -> tuple[float, str]:
        return (-f.confidence, f.headline)

    return sorted(findings, key=key)


def _count_recurrence(finding: Finding, active: list[Finding]) -> int:
    return sum(1 for f in active if f.kind == finding.kind and f.scope == finding.scope) - 1


def _stats_line(finding: Finding) -> str:
    s = finding.stats
    bits: list[str] = []
    if s.effect_size is not None:
        bits.append(f"effect {s.effect_size:g}")
    if s.n is not None:
        bits.append(f"n={s.n}")
    if s.p_perm is not None:
        bits.append(f"p={s.p_perm:g}")
    if s.q_fdr is not None:
        bits.append(f"q={s.q_fdr:g}")
    if s.replicated is not None:
        bits.append("replicated" if s.replicated else "not replicated")
    return " · ".join(bits)


def _finding_card(finding: Finding, active: list[Finding]) -> dict[str, Any]:
    recurrence = _count_recurrence(finding, active)
    survived = finding.skeptic_notes is None or "reject" not in finding.skeptic_notes.lower()
    return {
        "headline": finding.headline,
        "agent": finding.agent,
        "kind": finding.kind,
        "confidence": finding.confidence,
        "confidence_pct": round(finding.confidence * 100),
        "stats_line": _stats_line(finding),
        "skeptic_survived": survived,
        "skeptic_notes": finding.skeptic_notes,
        "recurrence": recurrence + 1,
        "body_html": markdown_to_html(finding.body_md) if finding.body_md else "",
    }


def _run_view(run: InvestigationRun, now: datetime) -> dict[str, Any]:
    """Shape one investigation run for the Investigations page."""
    return {
        "question": run.question or "Whole-record investigation",
        "kind": run.kind,
        "status": run.status,
        "when": _relative_time(run.finished_at, now),
        "window": f"{run.window_start.isoformat()} to {run.window_end.isoformat()}",
        "plan": run.plan,
        "trace": run.trace,
        "n_findings": run.n_findings,
        "findings": [
            {
                "headline": f.headline,
                "kind": f.kind,
                "confidence_pct": round(f.confidence * 100),
                "status": f.status,
            }
            for f in run.findings
        ],
    }


def _graveyard_row(finding: Finding) -> dict[str, Any]:
    return {
        "headline": finding.headline,
        "status": finding.status.value,
        "agent": finding.agent,
        "skeptic_notes": finding.skeptic_notes,
    }


def _goal_card(store: StoragePort, goal: Goal) -> dict[str, Any]:
    checkpoints = store.get_goal_checkpoints(goal.id) if goal.id is not None else []
    values = [cp.metric_value for cp in checkpoints if cp.metric_value is not None]
    latest = checkpoints[-1] if checkpoints else None
    return {
        "id": goal.id,
        "statement": goal.statement,
        "status": goal.status.value,
        "metric": goal.metric.value,
        "direction": goal.direction,
        "cadence_days": goal.cadence_days,
        "spark": sparkline_svg(values),
        "n_checkpoints": len(checkpoints),
        "latest_note": latest.note if latest else None,
        "latest_value": latest.metric_value if latest else None,
    }


def _rewrite_wiki_links(body: str) -> str:
    """Rewrite relative ``*.md`` links to ``/wiki/*`` GUI routes; leave URLs alone."""
    import re  # noqa: PLC0415

    def repl(m: re.Match[str]) -> str:
        target = m.group(1)
        if "://" in target or target.startswith(("/", "#", "mailto:")):
            return m.group(0)
        if target.endswith(".md"):
            target = target[:-3]
        return f'href="/wiki/{target}"'

    return re.sub(r'href="([^"]+)"', repl, body)


def _wrap_wiki_tables(html: str) -> str:
    """Scrollable card wrapper so wide belief tables don't blow the layout."""
    import re  # noqa: PLC0415

    return re.sub(
        r"<table>",
        '<div class="wiki-table-wrap"><table class="wiki-table">',
        html,
    ).replace("</table>", "</table></div>")


def _wiki_title(raw_md: str, slug: str) -> str:
    for line in raw_md.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return slug.replace("/", " · ").replace("-", " ").title()


def _wiki_nav(root: Path, slug: str) -> list[dict[str, Any]]:
    """Sidebar links for pages that exist under the wiki root."""
    pages: list[tuple[str, str, str]] = [
        ("index", "/wiki", "Overview"),
        ("hypotheses", "/wiki/hypotheses", "Hypotheses"),
        ("graveyard", "/wiki/graveyard", "Graveyard"),
        ("goals", "/wiki/goals", "Goals"),
    ]
    nav: list[dict[str, Any]] = []
    for key, href, label in pages:
        if not (root / f"{key}.md").is_file():
            continue
        nav.append({"href": href, "label": label, "active": slug == key})
    topics = root / "topics"
    if topics.is_dir():
        for path in sorted(topics.glob("*.md")):
            key = f"topics/{path.stem}"
            nav.append(
                {
                    "href": f"/wiki/{key}",
                    "label": path.stem.replace("-", " ").title(),
                    "active": slug == key,
                    "topic": True,
                }
            )
    return nav
