"""Per-role model resolution for the reasoning CLI commands.

Each command asks the BYOM factory for a model *by role* (plan/discovery/
skeptic/research/brief/explain/polish). These tests monkeypatch
``llm.factory.get_model`` to capture the resolved :class:`ModelSpec` without
constructing a real chat model, then assert the right role — with its sampling
defaults and any ``[llm.roles.*]`` override — flows through each call site.
"""

from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, ClassVar

from dexta_intelligence.cli._common import discovery_model, model_for_role
from dexta_intelligence.cli.intelligence import cmd_ask, cmd_brief, cmd_goals
from dexta_intelligence.config import Config
from dexta_intelligence.llm import factory
from dexta_intelligence.models import GlucoseEvent
from dexta_intelligence.store import SQLiteStore

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import pytest

    from dexta_intelligence.llm.factory import ModelSpec
    from dexta_intelligence.store.port import StoragePort

FIXED_NOW = datetime(2025, 6, 10, 12, 0, tzinfo=UTC)


class _FakeModel:
    """Stand-in chat model; never actually calls a provider."""

    def bind_tools(self, _schemas: object) -> _FakeModel:
        return self

    def invoke(self, _messages: object) -> object:
        class _Msg:
            content = "ok"
            tool_calls: ClassVar[list[object]] = []

        return _Msg()


def _capture_specs(monkeypatch: pytest.MonkeyPatch) -> list[ModelSpec]:
    """Patch ``get_model`` to record every spec it is handed; return that list."""
    specs: list[ModelSpec] = []

    def _fake_get_model(spec: ModelSpec) -> _FakeModel:
        specs.append(spec)
        return _FakeModel()

    monkeypatch.setattr(factory, "get_model", _fake_get_model)
    return specs


def _tmp_store(tmp_path: Path) -> SQLiteStore:
    store = SQLiteStore(tmp_path / "roles.db")
    store.migrate()
    return store


def _opener_for(store: SQLiteStore) -> Callable[[Config, Path | None], StoragePort]:
    def _open(_config: Config, _db: Path | None = None) -> StoragePort:
        return store

    return _open


def _seed_30d(store: SQLiteStore) -> None:
    start = datetime.now(tz=UTC) - timedelta(days=30)
    store.insert_glucose(
        [GlucoseEvent(ts=start + timedelta(hours=i), mg_dl=120) for i in range(24 * 30)]
    )


# ── direct resolution ──────────────────────────────────────────────────────────


def test_model_for_role_resolves_role_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    specs = _capture_specs(monkeypatch)
    model_for_role(Config(), "brief")
    assert len(specs) == 1
    assert specs[0].temperature == 0.2
    assert specs[0].max_tokens == 2200


def test_discovery_model_wrapper_resolves_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    specs = _capture_specs(monkeypatch)
    discovery_model(Config())
    assert specs[0].max_tokens == 1800  # the discovery role default


def test_model_for_role_returns_none_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(_spec: object) -> object:
        raise RuntimeError("no llm extra")

    monkeypatch.setattr(factory, "get_model", _boom)
    assert model_for_role(Config(), "explain") is None


# ── call sites ──────────────────────────────────────────────────────────────────


def test_cmd_brief_resolves_brief_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs = _capture_specs(monkeypatch)
    store = _tmp_store(tmp_path)
    code = cmd_brief(
        config=Config(),
        db_path=None,
        out=io.StringIO(),
        opener=_opener_for(store),
    )
    assert code == 0
    assert len(specs) == 1
    assert specs[0].temperature == 0.2
    assert specs[0].max_tokens == 2200
    store.close()


def test_cmd_ask_resolves_explain_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs = _capture_specs(monkeypatch)
    store = _tmp_store(tmp_path)
    _seed_30d(store)
    code = cmd_ask(
        question="how am I doing?",
        config=Config(),
        db_path=None,
        out=io.StringIO(),
        opener=_opener_for(store),
    )
    assert code == 0
    assert len(specs) == 1
    assert specs[0].temperature == 0.2
    assert specs[0].max_tokens == 1500  # explain role default
    store.close()


def test_cmd_goals_add_resolves_plan_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs = _capture_specs(monkeypatch)
    store = _tmp_store(tmp_path)
    code = cmd_goals(
        action="add",
        statement="reduce my overnight lows",
        config=Config(),
        db_path=None,
        out=io.StringIO(),
        opener=_opener_for(store),
        now=FIXED_NOW,
    )
    assert code == 0
    assert len(specs) == 1
    assert specs[0].temperature == 0.0
    assert specs[0].max_tokens == 1024  # plan role default
    store.close()


def test_brief_role_override_flows_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs = _capture_specs(monkeypatch)
    store = _tmp_store(tmp_path)
    config = Config.model_validate(
        {"llm": {"roles": {"brief": {"provider": "ollama", "model": "llama3", "max_tokens": 999}}}}
    )
    code = cmd_brief(
        config=config,
        db_path=None,
        out=io.StringIO(),
        opener=_opener_for(store),
    )
    assert code == 0
    assert specs[0].provider == "ollama"
    assert specs[0].model == "llama3"
    assert specs[0].max_tokens == 999
    store.close()
