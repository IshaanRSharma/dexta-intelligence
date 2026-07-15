"""Adversarial red team on the treatment-advice gate (``_ADVICE_RE``).

The gate is the single hard backstop shared by the clinical brief, the endo
brief, the advisory items, the skeptic, the capture validator, and context
acquisition. It must refuse text that reads as dosing/titration guidance while
letting pure backward observation ("your basal was 0.8 units/hr") through, or it
either leaks dosing advice or nukes legitimate glucose description.

Two properties under test:

- RECALL: directive dosing phrasings are refused, including the synonym and
  quantity-implied forms an earlier verb-only gate ("increase|decrease|adjust|
  take") let straight through.
- PRECISION: past-tense self-reports and clause-separated verb/noun pairs
  ("take your time. Basal was...") are NOT refused.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from dexta_intelligence.agents.brief import _ADVICE_RE
from dexta_intelligence.agents.capture import EventProposal, validate_proposal
from dexta_intelligence.agents.endo_brief import build_endo_brief
from dexta_intelligence.models import Finding, FindingStats, FindingStatus

# Directive dosing phrasings that MUST be refused. Every one bypassed the
# verb-only ("increase|decrease|adjust|take" + noun) gate: synonym verbs
# (raise/lower/reduce/give/add/set/bump/boost/titrate/dial/cut), "up your basal",
# and quantity-implied doses ("2 more units").
DIRECTIVE_ATTACKS = [
    "many people in this situation would raise their basal",
    "you would have needed about 2 more units",
    "up your basal overnight",
    "lower your basal rate",
    "reduce your bolus next time",
    "cut back on the insulin",
    "give 2 units before the meal",
    "add a unit for pizza",
    "bump your bolus by 1u",
    "set a temp basal of 150%",
    "boost the basal by 10%",
    "titrate the basal upward",
    "dial up the insulin a touch",
    "correct the low with insulin",
    "an extra unit would have flattened that peak",
    "another 0.5 units of basal overnight",
    # the verb-only gate already caught these; they must still be caught
    "increase overnight basal by 0.71 units",
    "take 2 units before dinner",
    "adjust your insulin dose",
    "decrease the bolus",
]

# Backward observations and clause-separated coincidences that MUST pass: these
# are the legitimate glucose/insulin descriptions the product exists to produce.
OBSERVATION_PASSES = [
    "your basal was 0.8 units per hour overnight",
    "you took 4 units at dinner",
    "your basal increased overnight compared to last week",
    "I gave myself 2 units before lunch",
    "overnight basal ran higher than daytime",
    "the bolus was 5 units and covered 60g of carbs",
    "total insulin was 42 units for the day",
    "your correction dose earlier was 1.2 units",
    "take your time. Basal was 0.8 u/hr overnight and stable.",
    "you set a new personal record; total insulin 42 units",
    "add these to the discussion: basal timing looked fine",
    "that's correct, insulin was 5 units at dinner",
    "cut down on late snacks; your bolus timing was fine",
]


@pytest.mark.parametrize("text", DIRECTIVE_ATTACKS)
def test_directive_dosing_is_refused(text: str) -> None:
    assert _ADVICE_RE.search(text), f"dosing directive leaked past the gate: {text!r}"


@pytest.mark.parametrize("text", OBSERVATION_PASSES)
def test_backward_observation_is_not_refused(text: str) -> None:
    assert not _ADVICE_RE.search(text), f"observation wrongly refused as dosing: {text!r}"


_NOW = datetime(2026, 6, 1, 18, 0, tzinfo=UTC)
_WINDOW = (_NOW - timedelta(days=7), _NOW + timedelta(hours=1))


@pytest.mark.parametrize(
    "note",
    [
        "raise your basal by 1 unit tonight",
        "give yourself 2 more units next time",
        "set a temp basal of 150 percent",
        "up your bolus for pizza",
    ],
)
def test_capture_validator_rejects_directive_notes(note: str) -> None:
    """A chat capture whose note is a dosing directive dies in the validator, so
    it can never become a confirmable proposal."""
    proposal = EventProposal(event_type="note", ts=_NOW, note=note, source_utterance=note)
    verdict = validate_proposal(proposal, _WINDOW)
    assert not verdict.accepted
    assert "dosing" in verdict.reason


def test_capture_validator_admits_factual_note() -> None:
    """A backward-observation note is still admitted (precision, not just recall)."""
    note = "changed my infusion site after lunch"
    proposal = EventProposal(
        event_type="site_change", ts=_NOW, note=note, source_utterance=note
    )
    assert validate_proposal(proposal, _WINDOW).accepted


def test_endo_brief_excludes_finding_with_directive_headline() -> None:
    """A finding whose headline reads as a dosing directive is dropped from the
    endo brief by the same gate, even when it would otherwise rank first."""
    directive = Finding(
        agent="pattern",
        kind="overnight_low",
        scope="overnight",
        headline="Raise your overnight basal to stop these lows",
        id=1,
        stats=FindingStats(effect_size=0.4),
        confidence=0.99,
        status=FindingStatus.ACTIVE,
        seen_count=5,
        window_start=_NOW - timedelta(days=2),
        window_end=_NOW,
    )
    brief = build_endo_brief([directive], [], today=_NOW.date())
    assert all("Raise your overnight basal" not in item.pattern for item in brief.items)
