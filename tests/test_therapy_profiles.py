"""Versioned therapy profiles (PRD section 18).

Devices report only the current profile, so dexta records a new version when the
content changes. Covers the store versioning, the sync-time capture, and the
``get_active_profile`` tool that reads the version in effect at an event time.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.agents.tools.toolkit import DiscoveryToolkit, tool_specs
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.models import GlucoseEvent, RawEvent, TherapyProfile
from dexta_intelligence.store import SQLiteStore
from dexta_intelligence.workflows.sync import _capture_profile_versions

_MAR = datetime(2026, 3, 1, tzinfo=UTC)
_JUN = datetime(2026, 6, 1, tzinfo=UTC)


def _profile(name: str, when: datetime, isf: int) -> TherapyProfile:
    return TherapyProfile(
        source="tandem",
        name=name,
        content={"active_profile": name, "isf": isf},
        content_hash=f"{name}-{isf}",
        active_from=when,
        created_at=when,
    )


def _store() -> SQLiteStore:
    s = SQLiteStore(":memory:")
    s.migrate()
    return s


# ── store versioning ──────────────────────────────────────────────────────────


def test_unchanged_profile_does_not_create_a_new_version() -> None:
    store = _store()
    first = store.add_profile_version(_profile("Weekday", _MAR, 45))
    again = store.add_profile_version(_profile("Weekday", _MAR + timedelta(days=1), 45))
    assert first == again
    assert len(store.get_profile_versions()) == 1


def test_changed_profile_opens_a_new_version_and_closes_the_old() -> None:
    store = _store()
    store.add_profile_version(_profile("Weekday", _MAR, 45))
    store.add_profile_version(_profile("Summer", _JUN, 50))
    versions = store.get_profile_versions()
    assert [v.name for v in versions] == ["Weekday", "Summer"]
    assert versions[0].active_to == _JUN  # previous version closed
    assert versions[1].active_to is None  # newest stays open


def test_get_active_profile_reads_the_version_in_effect() -> None:
    store = _store()
    store.add_profile_version(_profile("Weekday", _MAR, 45))
    store.add_profile_version(_profile("Summer", _JUN, 50))
    assert store.get_active_profile(datetime(2026, 3, 15, tzinfo=UTC)).name == "Weekday"
    assert store.get_active_profile(datetime(2026, 7, 1, tzinfo=UTC)).name == "Summer"
    assert store.get_active_profile(datetime(2026, 1, 1, tzinfo=UTC)) is None


# ── sync-time capture ──────────────────────────────────────────────────────────


def test_capture_records_versions_from_profile_snapshots() -> None:
    store = _store()
    _capture_profile_versions(
        store,
        [RawEvent(
            source="tandem",
            source_id="tandem:profile:active",
            source_ts=_MAR,
            payload={"active_profile": "Weekday", "isf": 45},
        )],
    )
    _capture_profile_versions(
        store,
        [RawEvent(
            source="tandem",
            source_id="tandem:profile:active",
            source_ts=_JUN,
            payload={"active_profile": "Summer", "isf": 50},
        )],
    )
    versions = store.get_profile_versions()
    assert [v.name for v in versions] == ["Weekday", "Summer"]


# ── tool ────────────────────────────────────────────────────────────────────


def _toolkit(store: SQLiteStore) -> DiscoveryToolkit:
    base = datetime(2026, 3, 14, 12, tzinfo=UTC)
    store.insert_glucose(
        [GlucoseEvent(ts=base + timedelta(minutes=5 * i), mg_dl=120) for i in range(40)]
    )
    ctx = AgentContext(
        store=store,
        window=(date(2026, 3, 1), date(2026, 6, 30)),
        gates=ColdStartReport.from_coverage(store.coverage()),
        run_id="t",
    )
    return DiscoveryToolkit(ctx)


def test_get_active_profile_tool_picks_the_periods_profile() -> None:
    store = _store()
    store.add_profile_version(_profile("Weekday", _MAR, 45))
    store.add_profile_version(_profile("Summer", _JUN, 50))
    tk = _toolkit(store)
    march = tk.get_active_profile("2026-03-14T20:00:00+00:00")
    assert march["versioned"] is True
    assert march["version_name"] == "Weekday"
    june = tk.get_active_profile("2026-06-20T20:00:00+00:00")
    assert june["version_name"] == "Summer"


def test_get_active_profile_tool_on_belt_and_bad_timestamp() -> None:
    store = _store()
    tk = _toolkit(store)
    ctx = AgentContext(
        store=store,
        window=(date(2026, 3, 1), date(2026, 6, 30)),
        gates=ColdStartReport.from_coverage(store.coverage()),
        run_id="t",
    )
    assert "get_active_profile" in {t.name for t in tool_specs(ctx, tk)}
    assert "error" in tk.get_active_profile("not-a-date")
