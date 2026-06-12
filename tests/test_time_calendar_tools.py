"""Calendar helpers and the always-on time ToolSpecs."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import pytest

from dexta_intelligence.agents._calendar import (
    is_weekend,
    parse_iso_date,
    parse_relative_date,
    weekday_name,
)
from dexta_intelligence.agents.time_tools import CALENDAR_TOOL_NAMES, time_tool_specs
from dexta_intelligence.guard.faithfulness import audit

# The donor repo's recorded LLM failure: the model called this date "Wednesday".
SATURDAY = "2026-05-02"

# Frozen "now": 2026-05-02 20:34 UTC (a Saturday).
FROZEN_NOW = datetime(2026, 5, 2, 20, 34, tzinfo=UTC)


def _frozen_now() -> datetime:
    return FROZEN_NOW


def _specs_by_name() -> dict[str, Any]:
    return {spec.name: spec for spec in time_tool_specs(now_fn=_frozen_now)}


# ── _calendar helpers ────────────────────────────────────────────────────────


class TestCalendarHelpers:
    def test_saturday_failure_case(self) -> None:
        assert weekday_name(SATURDAY) == "Saturday"
        assert weekday_name(SATURDAY, short=True) == "Sat"

    def test_weekday_case(self) -> None:
        assert weekday_name("2026-05-06") == "Wednesday"

    def test_is_weekend(self) -> None:
        assert is_weekend(SATURDAY) is True
        assert is_weekend("2026-05-03") is True  # Sunday
        assert is_weekend("2026-05-04") is False  # Monday
        assert is_weekend("not a date") is None

    def test_parse_iso_date_accepts_datetime_prefix(self) -> None:
        assert parse_iso_date("2026-05-02T20:34:00+00:00") == date(2026, 5, 2)
        assert parse_iso_date(None) is None
        assert parse_iso_date("") is None

    def test_parse_relative_date(self) -> None:
        today = date(2026, 5, 2)  # Saturday
        assert parse_relative_date("yesterday", today=today) == date(2026, 5, 1)
        assert parse_relative_date("3 days ago", today=today) == date(2026, 4, 29)
        assert parse_relative_date("last tuesday", today=today) == date(2026, 4, 28)
        assert parse_relative_date("last week", today=today) == date(2026, 4, 25)
        assert parse_relative_date("next saturday", today=today) == date(2026, 5, 9)
        assert parse_relative_date("the other day", today=today) is None
        assert parse_relative_date("", today=today) is None


# ── ToolSpec surface ─────────────────────────────────────────────────────────


class TestToolSpecs:
    def test_names_match_constant(self) -> None:
        specs = time_tool_specs()
        assert tuple(s.name for s in specs) == CALENDAR_TOOL_NAMES

    def test_parameters_are_json_schema_objects(self) -> None:
        for spec in time_tool_specs():
            assert spec.parameters["type"] == "object"
            assert isinstance(spec.parameters["properties"], dict)
            assert spec.description


class TestGetCurrentTime:
    def test_frozen_now_utc(self) -> None:
        result, evidence = _specs_by_name()["get_current_time"].fn({})
        assert result["date"] == SATURDAY
        assert result["weekday"] == "Saturday"
        assert result["is_weekend"] is True
        assert result["timezone"] == "UTC"
        assert result["utc_offset"] == "+0000"
        assert evidence == {"year": 2026, "month": 5, "day": 2, "hour": 20, "minute": 34}

    def test_timezone_math_new_york(self) -> None:
        result, evidence = _specs_by_name()["get_current_time"].fn(
            {"timezone": "America/New_York"}
        )
        assert result["utc_offset"] == "-0400"  # EDT in May
        assert result["date"] == SATURDAY
        assert evidence["hour"] == 16

    def test_local_date_crosses_midnight(self) -> None:
        specs = {
            s.name: s
            for s in time_tool_specs(
                now_fn=lambda: datetime(2026, 5, 3, 2, 0, tzinfo=UTC)
            )
        }
        result, _ = specs["get_current_time"].fn({"timezone": "America/New_York"})
        assert result["date"] == SATURDAY  # still Saturday evening locally
        assert result["weekday"] == "Saturday"

    def test_invalid_timezone_falls_back_to_utc(self) -> None:
        result, _ = _specs_by_name()["get_current_time"].fn({"timezone": "Mars/Olympus"})
        assert result["timezone"] == "UTC"
        assert result["utc_offset"] == "+0000"

    def test_default_timezone_used_when_arg_absent(self) -> None:
        specs = {
            s.name: s for s in time_tool_specs("America/New_York", now_fn=_frozen_now)
        }
        result, _ = specs["get_current_time"].fn({})
        assert result["timezone"] == "America/New_York"
        assert result["utc_offset"] == "-0400"

    def test_garbage_timezone_arg_never_raises(self) -> None:
        result, _ = _specs_by_name()["get_current_time"].fn({"timezone": 999})
        assert result["timezone"] == "UTC"


class TestGetWeekday:
    def test_saturday_failure_case(self) -> None:
        result, evidence = _specs_by_name()["get_weekday"].fn({"date": SATURDAY})
        assert result == {
            "ok": True,
            "date": SATURDAY,
            "weekday": "Saturday",
            "weekday_short": "Sat",
            "is_weekend": True,
        }
        assert evidence == {"year": 2026, "month": 5, "day": 2}

    def test_weekday_case(self) -> None:
        result, _ = _specs_by_name()["get_weekday"].fn({"date": "2026-05-06"})
        assert result["weekday"] == "Wednesday"
        assert result["is_weekend"] is False

    @pytest.mark.parametrize("args", [{}, {"date": "garbage"}, {"date": None}, {"date": 123}])
    def test_bad_args_return_error_dict(self, args: dict[str, Any]) -> None:
        result, evidence = _specs_by_name()["get_weekday"].fn(args)
        assert result["ok"] is False
        assert "error" in result
        assert evidence == {}


class TestParseRelativeDateTool:
    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("yesterday", "2026-05-01"),
            ("3 days ago", "2026-04-29"),
            ("last tuesday", "2026-04-28"),
            ("last week", "2026-04-25"),
        ],
    )
    def test_resolves_against_frozen_now(self, expression: str, expected: str) -> None:
        result, evidence = _specs_by_name()["parse_relative_date"].fn(
            {"expression": expression}
        )
        assert result["ok"] is True
        assert result["date"] == expected
        assert evidence["day"] == int(expected[-2:])

    def test_today_anchor_respects_timezone(self) -> None:
        specs = {
            s.name: s
            for s in time_tool_specs(
                now_fn=lambda: datetime(2026, 5, 3, 2, 0, tzinfo=UTC)
            )
        }
        utc, _ = specs["parse_relative_date"].fn({"expression": "today"})
        ny, _ = specs["parse_relative_date"].fn(
            {"expression": "today", "timezone": "America/New_York"}
        )
        assert utc["date"] == "2026-05-03"
        assert ny["date"] == SATURDAY

    @pytest.mark.parametrize("args", [{}, {"expression": "the other day"}, {"expression": None}])
    def test_unknown_phrase_returns_error_dict(self, args: dict[str, Any]) -> None:
        result, evidence = _specs_by_name()["parse_relative_date"].fn(args)
        assert result["ok"] is False
        assert "error" in result
        assert evidence == {}


class TestGuardEvidence:
    """Pins the choice: date components ARE evidence, because the guard's
    number extractor flags bare years/days cited in prose."""

    def test_year_untraceable_without_evidence(self) -> None:
        assert audit("the spike was on 2026-05-13", {}).ok is False

    def test_tool_evidence_makes_cited_date_traceable(self) -> None:
        _, evidence = _specs_by_name()["get_weekday"].fn({"date": "2026-05-13"})
        report = audit("that was Wednesday, 2026-05-13", evidence)
        assert report.ok
