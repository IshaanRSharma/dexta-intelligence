"""Treatment-inspection tools + capability filtering over golden datasets.

Pins the canonical numbers end to end: the March 14 dinner bolus is 22 minutes
late, the spike peaks at 246, and 14 of 18 similar dinners spike.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from tests.golden import make_store

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.agents.tools.time_tools import CALENDAR_TOOL_NAMES
from dexta_intelligence.agents.tools.toolkit import DiscoveryToolkit, tool_specs
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.connectors.tandem import PROFILE_SOURCE_ID, format_insulin_profile
from dexta_intelligence.models import RawEvent

_WINDOW = (date(2025, 12, 15), date(2026, 3, 15))
_SPIKE_TS = "2026-03-14T20:42:00+00:00"
_DINNER_TS = "2026-03-14T20:00:00+00:00"


def _ctx(name: str) -> AgentContext:
    store = make_store(name)
    return AgentContext(
        store=store,
        window=_WINDOW,
        gates=ColdStartReport.from_coverage(store.coverage()),
        run_id="test-run",
    )


@pytest.fixture(scope="module")
def late_bolus_toolkit() -> DiscoveryToolkit:
    return DiscoveryToolkit(_ctx("late_bolus"))


# ── the canonical March 14 numbers ────────────────────────────────────────────


def test_get_boluses_reports_22_minute_late_bolus(late_bolus_toolkit: DiscoveryToolkit) -> None:
    late_bolus_toolkit.set_window("2026-03-14", "2026-03-14")
    result = late_bolus_toolkit.get_boluses()
    assert result["n_boluses"] == 1
    assert result["boluses"][0]["minutes_after_carb_entry"] == 22


def test_get_carb_entries_lists_the_dinner(late_bolus_toolkit: DiscoveryToolkit) -> None:
    late_bolus_toolkit.set_window("2026-03-14", "2026-03-14")
    result = late_bolus_toolkit.get_carb_entries()
    assert result["n_entries"] == 1
    assert result["entries"][0]["carbs_g"] > 0


def test_basal_stable_in_window(late_bolus_toolkit: DiscoveryToolkit) -> None:
    late_bolus_toolkit.set_window("2026-03-14", "2026-03-14")
    result = late_bolus_toolkit.get_basal_timeline()
    assert result["basal_stable"] is True
    assert result["n_basal"] >= 1
    assert result["n_temp_basal"] == 0


def test_find_spikes_locates_the_246_peak(late_bolus_toolkit: DiscoveryToolkit) -> None:
    late_bolus_toolkit.set_window("2026-03-14", "2026-03-14")
    result = late_bolus_toolkit.find_spikes()
    assert result["n_spikes"] >= 1
    top = result["spikes"][0]
    assert top["peak_mg_dl"] == 246
    assert top["ts"] == _SPIKE_TS


def test_find_similar_events_14_of_18_dinners(late_bolus_toolkit: DiscoveryToolkit) -> None:
    result = late_bolus_toolkit.find_similar_events(_DINNER_TS)
    assert result["n_similar"] == 18
    assert result["n_spiking"] == 14
    # The late-bolus signal: spiking dinners have a larger mean bolus delay.
    assert result["mean_bolus_delay_spiking_min"] > result["mean_bolus_delay_other_min"]


def test_get_iob_positive_after_bolus(late_bolus_toolkit: DiscoveryToolkit) -> None:
    result = late_bolus_toolkit.get_iob(_SPIKE_TS)
    assert result["tier"] == "B"
    assert result["iob_units"] > 0
    assert result["n_recent_boluses"] >= 1


def test_get_cob_positive_after_dinner(late_bolus_toolkit: DiscoveryToolkit) -> None:
    result = late_bolus_toolkit.get_cob(_SPIKE_TS)
    assert result["tier"] == "B"
    assert result["cob_g"] > 0
    assert result["n_carb_entries"] >= 1


# ── graceful degradation ──────────────────────────────────────────────────────


def test_bad_timestamps_return_error_dicts(late_bolus_toolkit: DiscoveryToolkit) -> None:
    assert "error" in late_bolus_toolkit.get_iob("garbage")
    assert "error" in late_bolus_toolkit.get_cob("garbage")
    assert "error" in late_bolus_toolkit.find_similar_events("garbage")


def test_null_dataset_has_no_spikes() -> None:
    toolkit = DiscoveryToolkit(_ctx("null"))
    result = toolkit.find_spikes()
    assert result["n_spikes"] == 0
    assert "note" in result


def test_missing_carb_dataset_flags_empty_entries() -> None:
    toolkit = DiscoveryToolkit(_ctx("missing_carb"))
    toolkit.set_window("2026-03-10", "2026-03-10")
    result = toolkit.get_carb_entries()
    assert result["n_entries"] == 0
    assert "missing-carb" in result["note"]


# ── capability filtering ──────────────────────────────────────────────────────


def test_no_insulin_dataset_hides_treatment_tools() -> None:
    ctx = _ctx("no_insulin")
    toolkit = DiscoveryToolkit(ctx)
    names = {s.name for s in tool_specs(ctx, toolkit)}
    for hidden in (
        "get_boluses",
        "get_basal_timeline",
        "get_iob",
        "get_insulin_profile",
        "get_carb_entries",
        "get_cob",
        "meal_response",
        "correction_outcome",
    ):
        assert hidden not in names
    # Glucose-only and calendar tools remain.
    assert {"set_window", "zoom_event", "find_spikes", "find_similar_events"} <= names
    assert set(CALENDAR_TOOL_NAMES) <= names


def test_insulin_dataset_exposes_full_belt() -> None:
    ctx = _ctx("late_bolus")
    toolkit = DiscoveryToolkit(ctx)
    names = {s.name for s in tool_specs(ctx, toolkit)}
    assert {
        "get_boluses",
        "get_carb_entries",
        "get_basal_timeline",
        "get_iob",
        "get_insulin_profile",
        "get_cob",
        "find_spikes",
        "find_similar_events",
    } <= names


def test_get_insulin_profile_without_sync_returns_error(
    late_bolus_toolkit: DiscoveryToolkit,
) -> None:
    result = late_bolus_toolkit.get_insulin_profile()
    assert "error" in result
    assert "Sync now" in result["note"]


def test_get_insulin_profile_reads_synced_snapshot() -> None:
    store = make_store("late_bolus")
    ts = datetime(2026, 6, 5, tzinfo=UTC)
    store.replace_raw_events(
        [
            RawEvent(
                source="tandem",
                source_id=PROFILE_SOURCE_ID,
                source_ts=ts,
                payload=format_insulin_profile(
                    {
                        "profiles": {
                            "activeIdp": 1,
                            "profile": [
                                {
                                    "name": "Weekday",
                                    "idp": 1,
                                    "insulinDuration": 300,
                                    "maxBolus": 5000,
                                    "tDependentSegs": [
                                        {
                                            "startTime": 0,
                                            "basalRate": 800,
                                            "isf": 50,
                                            "carbRatio": 10000,
                                            "targetBg": 100,
                                        }
                                    ],
                                }
                            ],
                        },
                        "cgmSettings": {
                            "highGlucoseAlert": {"mgPerDl": 250, "enabled": 1},
                            "lowGlucoseAlert": {"mgPerDl": 70, "enabled": 1},
                        },
                    },
                    pump_serial="923983",
                    as_of=ts,
                ),
            )
        ]
    )
    ctx = AgentContext(
        store=store,
        window=_WINDOW,
        gates=ColdStartReport.from_coverage(store.coverage()),
        run_id="profile-run",
    )
    toolkit = DiscoveryToolkit(ctx)
    result = toolkit.get_insulin_profile()
    assert result["active_profile"] == "Weekday"
    assert result["tier"] == "B"
    assert result["active_segments"][0]["basal_u_hr"] == 0.8
    assert result["pump_serial"] == "923983"


def test_coverage_tool_reports_missing_streams() -> None:
    ctx = _ctx("no_insulin")
    toolkit = DiscoveryToolkit(ctx)
    coverage_spec = next(s for s in tool_specs(ctx, toolkit) if s.name == "coverage")
    result, numbers = coverage_spec.fn({})
    assert any("insulin" in note for note in result["unavailable"])
    assert numbers["n_insulin"] == 0


def test_capabilities_reflect_streams() -> None:
    caps = DiscoveryToolkit(_ctx("late_bolus")).capabilities()
    assert caps.has_insulin and caps.has_meals
    caps_empty = DiscoveryToolkit(_ctx("no_insulin")).capabilities()
    assert not caps_empty.has_insulin and not caps_empty.has_meals
