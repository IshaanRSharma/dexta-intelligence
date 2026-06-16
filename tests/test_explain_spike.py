"""explain_spike workflow contract over golden datasets (metrics M3/M5/M10).

The canonical WAVE5 §1 question — "Why did I spike on March 14?" — must
produce the canonical trace and evidence numbers from the late_bolus plant.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pytest
from tests.golden import make_store

from dexta_intelligence.agents.base import AgentContext
from dexta_intelligence.coldstart import ColdStartReport
from dexta_intelligence.investigations.spike import (
    NO_TREATMENT_DISCLAIMER,
    OUTPUT_KEYS,
    SAFETY_LINE,
    explain_spike,
)

_WINDOW = (date(2025, 12, 15), date(2026, 3, 15))


def _ctx(name: str) -> AgentContext:
    store = make_store(name)
    return AgentContext(
        store=store,
        window=_WINDOW,
        gates=ColdStartReport.from_coverage(store.coverage()),
        run_id="test-run",
    )


@dataclass
class _FakeResponse:
    content: str


@dataclass
class _FakeModel:
    reply: str

    def invoke(self, _messages: list[dict[str, str]]) -> _FakeResponse:
        return _FakeResponse(self.reply)


# ── the canonical question (M3) ───────────────────────────────────────────────


@pytest.fixture(scope="module")
def canonical() -> dict[str, object]:
    return explain_spike(_ctx("late_bolus"), "2026-03-14")


def test_canonical_headline_names_late_meal_insulin(canonical: dict[str, object]) -> None:
    assert "late/insufficient meal insulin context" in str(canonical["headline"])
    assert "than basal drift" in str(canonical["headline"])


def test_canonical_evidence_numbers(canonical: dict[str, object]) -> None:
    evidence = "\n".join(str(e) for e in canonical["evidence"])  # type: ignore[union-attr]
    assert "Peak: 246 mg/dL" in evidence
    assert "Bolus: 22 min after meal entry" in evidence
    assert "Similar pattern: 14/18" in evidence
    assert "Basal stable in the window" in evidence


def test_canonical_trace_shows_treatment_path(canonical: dict[str, object]) -> None:
    trace = "\n".join(str(t) for t in canonical["trace"])  # type: ignore[union-attr]
    for fragment in (
        "scanned",          # list_segments
        "narrowed",         # set_window
        "zoomed",           # zoom_event
        "carb entries",     # get_carb_entries
        "bolus timing",     # get_boluses
        "basal/temp-basal", # get_basal_timeline
        "similar event",    # find_similar_events
    ):
        assert fragment in trace, f"missing {fragment!r} in trace:\n{trace}"


def test_canonical_confidence_and_safety(canonical: dict[str, object]) -> None:
    # 18 similar events, 14 spiking (78%), ~21 min delay separation → high.
    assert canonical["confidence"] == "high"
    assert canonical["safety"] == SAFETY_LINE
    assert canonical["limitations"], "limitations must list skipped steps (M10)"


def test_explicit_timestamp_path_matches() -> None:
    out = explain_spike(_ctx("late_bolus"), "2026-03-14T20:42:00+00:00")
    assert "late/insufficient meal insulin context" in str(out["headline"])


# ── frozen schema (M10) ───────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["late_bolus", "basal_drift", "missing_carb", "no_insulin"])
def test_output_schema_is_frozen(name: str) -> None:
    out = explain_spike(_ctx(name), "2026-03-14")
    assert tuple(out) == OUTPUT_KEYS
    assert out["confidence"] in ("low", "moderate", "high")
    assert isinstance(out["evidence"], list)
    assert isinstance(out["limitations"], list)
    assert isinstance(out["trace"], list) and out["trace"]


# ── degradation honesty (M5) ──────────────────────────────────────────────────


def test_no_insulin_discloses_and_makes_no_cause_claim() -> None:
    out = explain_spike(_ctx("no_insulin"), "2026-03-14")
    assert NO_TREATMENT_DISCLAIMER in str(out["headline"])
    assert out["confidence"] == "low"
    assert any(NO_TREATMENT_DISCLAIMER in str(x) for x in out["limitations"])  # type: ignore[union-attr]
    assert "consistent with late" not in str(out["headline"])


def test_null_day_finds_nothing_to_explain() -> None:
    out = explain_spike(_ctx("null"), "2026-03-10")
    assert "no excursion" in str(out["headline"])
    assert out["confidence"] == "low"


def test_garbage_input_degrades() -> None:
    out = explain_spike(_ctx("late_bolus"), "not-a-date")
    assert "could not parse" in str(out["headline"])
    assert out["confidence"] == "low"
    assert tuple(out) == OUTPUT_KEYS


# ── other planted contributors (M3) ───────────────────────────────────────────


def test_basal_drift_attributed_overnight() -> None:
    out = explain_spike(_ctx("basal_drift"), "2026-03-05T05:00:00+00:00")
    assert "basal-drift hypothesis" in str(out["headline"])
    assert "meal insulin" not in str(out["headline"])


def test_missing_carb_attributed_unlogged_meal() -> None:
    out = explain_spike(_ctx("missing_carb"), "2026-03-10")
    assert "unlogged meal context" in str(out["headline"])


# ── the single guarded LLM call ───────────────────────────────────────────────


def test_fabricating_model_falls_back_to_deterministic_headline() -> None:
    fake = _FakeModel("Your glucose hit 999 mg/dL because of a 73g meal.")
    out = explain_spike(_ctx("late_bolus"), "2026-03-14", model=fake)
    assert "999" not in str(out["headline"])
    assert "late/insufficient meal insulin context" in str(out["headline"])


def test_faithful_model_phrasing_is_kept() -> None:
    fake = _FakeModel(
        "This looks like late meal insulin rather than basal drift — a "
        "discussion point for your care team."
    )
    out = explain_spike(_ctx("late_bolus"), "2026-03-14", model=fake)
    assert str(out["headline"]).startswith("This looks like late meal insulin")


def test_no_treatment_data_skips_the_model_entirely() -> None:
    fake = _FakeModel("Anything")
    out = explain_spike(_ctx("no_insulin"), "2026-03-14", model=fake)
    assert NO_TREATMENT_DISCLAIMER in str(out["headline"])
