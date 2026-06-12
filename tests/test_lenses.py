"""Tests for named agent routes (lenses) and the thinned analyze path."""

from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from dexta_intelligence.cli import cmd_analyze
from dexta_intelligence.config import Config
from dexta_intelligence.models import GlucoseEvent
from dexta_intelligence.store import SQLiteStore
from dexta_intelligence.workflows import lenses
from dexta_intelligence.workflows.lenses import SKEPTIC, build_registry

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dexta_intelligence.store.port import StoragePort

FIXED_NOW = datetime(2025, 6, 10, 12, 0, tzinfo=UTC)


def _names(registry: object) -> set[str]:
    return {agent.name for agent in registry}  # type: ignore[attr-defined]


def _tmp_store(tmp_path: Path) -> SQLiteStore:
    store = SQLiteStore(tmp_path / "test.db")
    store.migrate()
    return store


def _opener_for(store: SQLiteStore) -> Callable[[Config, Path | None], StoragePort]:
    def _open(_config: Config, _db: Path | None = None) -> StoragePort:
        return store

    return _open


class TestBuildRegistry:
    def test_analyze_lens_has_all_producers_plus_skeptic(self) -> None:
        registry, window_days = build_registry("analyze", Config())
        assert _names(registry) == {
            "observation",
            "pattern",
            "reconciliation",
            "discovery",
            "insulin",
            SKEPTIC,
        }
        assert window_days is None

    def test_watch_lens_is_observation_pattern_skeptic_7d(self) -> None:
        registry, window_days = build_registry("watch", Config())
        assert _names(registry) == {"observation", "pattern", SKEPTIC}
        assert window_days == 7

    def test_why_lens(self) -> None:
        registry, window_days = build_registry("why", Config())
        assert _names(registry) == {"reconciliation", "discovery", SKEPTIC}
        assert window_days is None

    def test_insulin_lens_30d(self) -> None:
        registry, window_days = build_registry("insulin", Config())
        assert _names(registry) == {"insulin", SKEPTIC}
        assert window_days == 30

    def test_skeptic_always_present(self) -> None:
        for name in ("analyze", "watch", "why", "insulin"):
            registry, _ = build_registry(name, Config())
            assert SKEPTIC in _names(registry), name

    def test_custom_lens_from_config(self) -> None:
        config = Config.model_validate(
            {"lens": {"morning": {"agents": ["observation", "pattern"], "window_days": 7}}}
        )
        registry, window_days = build_registry("morning", config)
        assert _names(registry) == {"observation", "pattern", SKEPTIC}
        assert window_days == 7

    def test_user_lens_overrides_builtin(self) -> None:
        config = Config.model_validate({"lens": {"watch": {"agents": ["insulin"]}}})
        registry, window_days = build_registry("watch", config)
        assert _names(registry) == {"insulin", SKEPTIC}
        assert window_days is None  # user entry dropped the builtin 7d override

    def test_unknown_lens_lists_known(self) -> None:
        with pytest.raises(ValueError, match="unknown lens") as exc:
            build_registry("nope", Config())
        msg = str(exc.value)
        assert "analyze" in msg
        assert "watch" in msg

    def test_unknown_agent_name_lists_known(self) -> None:
        config = Config.model_validate({"lens": {"bad": {"agents": ["observation", "ghost"]}}})
        with pytest.raises(ValueError, match="unknown agent 'ghost'") as exc:
            build_registry("bad", config)
        msg = str(exc.value)
        assert "observation" in msg
        assert "pattern" in msg

    def test_skeptic_cannot_be_excluded(self) -> None:
        # Even a lens with no producers still gets the skeptic.
        config = Config.model_validate({"lens": {"empty": {"agents": []}}})
        registry, _ = build_registry("empty", config)
        assert _names(registry) == {SKEPTIC}

    def test_listing_skeptic_does_not_duplicate(self) -> None:
        config = Config.model_validate(
            {"lens": {"dup": {"agents": ["observation", "skeptic"]}}}
        )
        registry, _ = build_registry("dup", config)
        assert _names(registry) == {"observation", SKEPTIC}


class TestCmdAnalyzeLens:
    def _seed_glucose(self, store: SQLiteStore, days: float = 10.0) -> None:
        ts = FIXED_NOW - timedelta(days=days)
        while ts <= FIXED_NOW:
            store.insert_glucose([GlucoseEvent(ts=ts, mg_dl=120)])
            ts += timedelta(minutes=5)

    def test_watch_lens_runs_only_lens_agents(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = _tmp_store(tmp_path)
        self._seed_glucose(store)

        seen: list[str] = []
        real_build = lenses.build_registry

        def _spy(name: str, config: Config, *, model: object = None) -> object:
            registry, window = real_build(name, config, model=model)
            seen.extend(agent.name for agent in registry)
            return registry, window

        monkeypatch.setattr(lenses, "build_registry", _spy)

        out = io.StringIO()
        code = cmd_analyze(
            config=Config(),
            db_path=None,
            out=out,
            opener=_opener_for(store),
            lens="watch",
        )

        assert code == 0
        assert set(seen) == {"observation", "pattern", SKEPTIC}
        store.close()

    def test_unknown_lens_raises_from_cmd(self, tmp_path: Path) -> None:
        store = _tmp_store(tmp_path)
        with pytest.raises(ValueError, match="unknown lens"):
            cmd_analyze(
                config=Config(),
                db_path=None,
                out=io.StringIO(),
                opener=_opener_for(store),
                lens="nope",
            )
        store.close()

    def test_custom_lens_window_days_applies(self, tmp_path: Path) -> None:
        store = _tmp_store(tmp_path)
        self._seed_glucose(store)
        # A 3-day lens window narrows analysis even though coverage spans 10 days.
        config = Config.model_validate(
            {"lens": {"narrow": {"agents": ["observation"], "window_days": 3}}}
        )
        out = io.StringIO()
        code = cmd_analyze(
            config=config,
            db_path=None,
            out=out,
            opener=_opener_for(store),
            lens="narrow",
        )
        assert code == 0
        store.close()
