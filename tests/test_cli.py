"""Tests for the dexta CLI."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from dexta_intelligence.agents.base import AgentContext, AgentRegistry, DataRequirement
from dexta_intelligence.cli import (
    cmd_analyze,
    cmd_doctor,
    cmd_init,
    cmd_sync,
    init_config_path,
    is_dexcom_configured,
    is_libre_configured,
    is_nightscout_configured,
    is_whoop_configured,
    main,
    open_sqlite_store,
    resolve_config_path,
)
from dexta_intelligence.config import Config, load_config
from dexta_intelligence.connectors.base import HealthReport, NormalizedBatch
from dexta_intelligence.models import Finding, FindingStats, FindingStatus, GlucoseEvent, RawEvent
from dexta_intelligence.store import SQLiteStore

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest

    from dexta_intelligence.store.port import StoragePort
else:
    import pytest

FIXED_NOW = datetime(2025, 6, 10, 12, 0, tzinfo=UTC)


@dataclass
class FakeConnector:
    source: str
    health: HealthReport
    batch: NormalizedBatch = field(default_factory=NormalizedBatch)
    check_raises: BaseException | None = None

    def check(self) -> HealthReport:
        if self.check_raises is not None:
            raise self.check_raises
        return self.health

    def pull(self, since: datetime) -> NormalizedBatch:
        return self.batch


@dataclass
class StubAgent:
    name: str
    requires: DataRequirement = field(default_factory=DataRequirement)
    findings: list[Finding] = field(default_factory=list)
    fail: BaseException | None = None

    def run(self, ctx: AgentContext) -> list[Finding]:
        del ctx
        if self.fail is not None:
            raise self.fail
        return self.findings


def _capture(out: TextIO | None = None) -> io.StringIO:
    return out if isinstance(out, io.StringIO) else io.StringIO()


def _tmp_store(tmp_path: Path) -> SQLiteStore:
    store = SQLiteStore(tmp_path / "test.db")
    store.migrate()
    return store


def _opener_for(store: SQLiteStore) -> Callable[[Config, Path | None], StoragePort]:
    def _open(_config: Config, _db: Path | None = None) -> StoragePort:
        return store

    return _open


class TestConfigResolution:
    def test_resolve_prefers_local_dexta_toml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        local = tmp_path / "dexta.toml"
        local.write_text("[analysis]\ntarget_low = 80\n", encoding="utf-8")
        assert resolve_config_path(None).resolve() == local.resolve()

    def test_init_default_path_is_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        assert init_config_path(None) == Path("dexta.toml")

    def test_configured_detection(self) -> None:
        cfg = Config.model_validate(
            {
                "nightscout": {"url": "https://ns.example", "token": "abc"},
                "dexcom": {"username": "u", "password": "p"},
                "whoop": {"access_token": "tok"},
                "libre": {"email": "e@x.com", "password": "pw"},
            }
        )
        assert is_nightscout_configured(cfg)
        assert is_dexcom_configured(cfg)
        assert is_whoop_configured(cfg)
        assert is_libre_configured(cfg)

        empty = Config()
        assert not is_nightscout_configured(empty)
        assert not is_dexcom_configured(empty)
        assert not is_whoop_configured(empty)
        assert not is_libre_configured(empty)


class TestInit:
    def test_writes_valid_toml_and_db(self, tmp_path: Path) -> None:
        config_path = tmp_path / "dexta.toml"
        db_path = tmp_path / "dexta.db"
        out = _capture()

        code = cmd_init(
            config_path=config_path,
            db_path=db_path,
            force=False,
            out=out,
            opener=open_sqlite_store,
        )

        assert code == 0
        assert config_path.is_file()
        assert db_path.is_file()
        cfg = load_config(config_path)
        assert cfg.data.backend == "sqlite"
        assert cfg.analysis.target_low == 70
        assert "Next steps" in out.getvalue()

    def test_refuses_overwrite_without_force(self, tmp_path: Path) -> None:
        config_path = tmp_path / "dexta.toml"
        config_path.write_text("existing", encoding="utf-8")
        out = _capture()

        code = cmd_init(config_path=config_path, db_path=None, force=False, out=out)

        assert code == 1
        assert config_path.read_text(encoding="utf-8") == "existing"
        assert "Refusing to overwrite" in out.getvalue()

    def test_force_overwrites(self, tmp_path: Path) -> None:
        config_path = tmp_path / "dexta.toml"
        config_path.write_text("existing", encoding="utf-8")
        out = _capture()

        code = cmd_init(config_path=config_path, db_path=None, force=True, out=out)

        assert code == 0
        assert "[data]" in config_path.read_text(encoding="utf-8")


class TestDoctor:
    def test_healthy_and_failing_connectors(self, tmp_path: Path) -> None:
        store = _tmp_store(tmp_path)
        cfg = Config()
        healthy = FakeConnector(
            "nightscout",
            HealthReport(ok=True, source="nightscout", detail="ok"),
        )
        failing = FakeConnector(
            "dexcom",
            HealthReport(ok=False, source="dexcom", detail="auth failed"),
        )
        out = _capture()

        code = cmd_doctor(
            config=cfg,
            db_path=None,
            out=out,
            connector_factory=lambda _c: [healthy, failing],
            opener=_opener_for(store),
        )
        text = out.getvalue()

        assert code == 1
        assert "✓ nightscout" in text
        assert "✗ dexcom" in text
        assert "auth failed" in text
        store.close()

    def test_runtime_error_rendered_as_failed_check(self, tmp_path: Path) -> None:
        store = _tmp_store(tmp_path)
        cfg = Config()
        broken = FakeConnector(
            "libre",
            HealthReport(ok=True, source="libre", detail="unused"),
            check_raises=RuntimeError("pip install 'dexta-intelligence[libre]'"),
        )
        out = _capture()

        code = cmd_doctor(
            config=cfg,
            db_path=None,
            out=out,
            connector_factory=lambda _c: [broken],
            opener=_opener_for(store),
        )
        text = out.getvalue()

        assert code == 1
        assert "✗ libre" in text
        assert "pip install" in text
        store.close()

    def test_no_sources_configured_is_ok(self, tmp_path: Path) -> None:
        store = _tmp_store(tmp_path)
        out = _capture()

        code = cmd_doctor(
            config=Config(),
            db_path=None,
            out=out,
            connector_factory=lambda _c: [],
            opener=_opener_for(store),
        )

        assert code == 0
        assert "No data sources configured" in out.getvalue()
        store.close()


class TestSync:
    def test_happy_path_prints_counts(self, tmp_path: Path) -> None:
        store = _tmp_store(tmp_path)
        ts = FIXED_NOW - timedelta(hours=1)
        batch = NormalizedBatch(
            raw=[RawEvent(source="nightscout", source_id="1", source_ts=ts, payload={"v": 100})],
            glucose=[GlucoseEvent(ts=ts, mg_dl=100)],
        )
        connector = FakeConnector(
            "nightscout",
            HealthReport(ok=True, source="nightscout", detail="ok"),
            batch=batch,
        )
        out = _capture()

        code = cmd_sync(
            config=Config(),
            db_path=None,
            out=out,
            connector_factory=lambda _c: [connector],
            opener=_opener_for(store),
            now=FIXED_NOW,
        )
        text = out.getvalue()

        assert code == 0
        assert "raw new: 1" in text
        assert "glucose=1" in text
        assert store.coverage().n_glucose == 1
        store.close()

    def test_all_fail_exit_code(self, tmp_path: Path) -> None:
        store = _tmp_store(tmp_path)

        class BoomConnector:
            source = "nightscout"

            def check(self) -> HealthReport:
                return HealthReport(ok=True, source=self.source, detail="ok")

            def pull(self, since: datetime) -> NormalizedBatch:
                raise ConnectionError("offline")

        out = _capture()
        code = cmd_sync(
            config=Config(),
            db_path=None,
            out=out,
            connector_factory=lambda _c: [BoomConnector()],
            opener=_opener_for(store),
            now=FIXED_NOW,
        )

        assert code == 1
        assert "All sources failed" in out.getvalue()
        store.close()


class TestAnalyze:
    def _seed_glucose(self, store: SQLiteStore, days: float = 5.0) -> None:
        start = FIXED_NOW - timedelta(days=days)
        ts = start
        while ts <= FIXED_NOW:
            store.insert_glucose([GlucoseEvent(ts=ts, mg_dl=120)])
            ts += timedelta(minutes=5)

    def test_stub_agent_finding_printed_and_persisted(self, tmp_path: Path) -> None:
        store = _tmp_store(tmp_path)
        self._seed_glucose(store)
        finding = Finding(
            agent="stub",
            kind="demo",
            scope="global",
            headline="Demo finding",
            stats=FindingStats(n=10, effect_size=0.5),
            status=FindingStatus.ACTIVE,
        )
        registry = AgentRegistry()
        registry.register(StubAgent(name="stub", findings=[finding]))
        out = _capture()

        code = cmd_analyze(
            config=Config(),
            db_path=None,
            out=out,
            opener=_opener_for(store),
            registry=registry,
        )
        text = out.getvalue()

        assert code == 0
        assert "Demo finding" in text
        assert "evidence stats: n=10" in text
        persisted = store.get_findings(agent="stub")
        assert len(persisted) == 1
        assert persisted[0].headline == "Demo finding"
        store.close()

    def test_raising_agent_isolated(self, tmp_path: Path) -> None:
        store = _tmp_store(tmp_path)
        self._seed_glucose(store)
        good = Finding(
            agent="good",
            kind="demo",
            scope="global",
            headline="Still runs",
            status=FindingStatus.ACTIVE,
        )
        registry = AgentRegistry()
        registry.register(StubAgent(name="bad", fail=RuntimeError("boom")))
        registry.register(StubAgent(name="good", findings=[good]))
        out = _capture()

        code = cmd_analyze(
            config=Config(),
            db_path=None,
            out=out,
            opener=_opener_for(store),
            registry=registry,
        )
        text = out.getvalue()

        assert code == 0
        assert "✗ bad" in text
        assert "Still runs" in text
        assert len(store.get_findings()) == 1
        store.close()

    def test_no_agents_registered(self, tmp_path: Path) -> None:
        store = _tmp_store(tmp_path)
        out = _capture()

        code = cmd_analyze(
            config=Config(),
            db_path=None,
            out=out,
            opener=_opener_for(store),
            registry=AgentRegistry(),
        )

        assert code == 0
        assert "No agents registered" in out.getvalue()
        store.close()


class TestMain:
    def test_help(self) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        assert exc.value.code == 0

    def test_unknown_command(self) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["not-a-command"])
        assert exc.value.code == 2

    def test_no_command_prints_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main([]) == 0
        assert "init" in capsys.readouterr().out
