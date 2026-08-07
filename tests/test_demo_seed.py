"""The one-command demo seeding: `dexta serve --demo` populates an empty store once."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dexta_intelligence.cli.main import build_parser
from dexta_intelligence.demo import build_demo_store, seed_demo_if_empty
from dexta_intelligence.models import RollupPeriod
from dexta_intelligence.store import SQLiteStore

if TYPE_CHECKING:
    from pathlib import Path


def _empty_store(tmp_path: Path) -> SQLiteStore:
    store = SQLiteStore(str(tmp_path / "demo.db"))
    store.migrate()
    return store


def test_seed_populates_an_empty_store(tmp_path: Path) -> None:
    store = _empty_store(tmp_path)
    assert store.coverage().first_ts is None
    assert seed_demo_if_empty(store) is True
    assert store.coverage().first_ts is not None
    store.close()


def test_seed_is_idempotent(tmp_path: Path) -> None:
    store = _empty_store(tmp_path)
    seed_demo_if_empty(store)
    first = store.coverage()
    assert seed_demo_if_empty(store) is False  # already has data, left untouched
    assert store.coverage().first_ts == first.first_ts
    store.close()


def test_build_demo_store_still_seeds() -> None:
    store = build_demo_store()
    assert store.coverage().first_ts is not None
    store.close()


def _rollup_count(store: SQLiteStore) -> int:
    coverage = store.coverage()
    assert coverage.first_ts is not None and coverage.last_ts is not None
    return len(store.get_rollups(RollupPeriod.DAILY, coverage.first_ts, coverage.last_ts))


def test_seed_computes_rollups(tmp_path: Path) -> None:
    # Rollups are normally a side effect of connector sync, which demo mode
    # disables. Without them the dashboard time-in-range tile renders empty on a
    # fully seeded database.
    store = _empty_store(tmp_path)
    seed_demo_if_empty(store)
    assert _rollup_count(store) > 0
    store.close()


def test_seed_backfills_rollups_on_a_store_seeded_without_them(tmp_path: Path) -> None:
    store = _empty_store(tmp_path)
    seed_demo_if_empty(store)
    seeded = _rollup_count(store)

    store._conn.execute("delete from rollups")
    store._conn.commit()
    assert _rollup_count(store) == 0

    assert seed_demo_if_empty(store) is False  # repaired in place, not re-seeded
    assert _rollup_count(store) == seeded
    store.close()


def test_serve_parses_the_demo_flag() -> None:
    args = build_parser().parse_args(["serve", "--demo"])
    assert args.demo is True


def test_serve_demo_defaults_off() -> None:
    args = build_parser().parse_args(["serve"])
    assert args.demo is False
