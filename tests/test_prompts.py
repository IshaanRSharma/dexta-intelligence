"""Prompt registry: file loading, user override, and the locked safety rail."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dexta_intelligence.agents import chat, prompts

if TYPE_CHECKING:
    from pathlib import Path

# The exact chat system prompt, to prove the file migration changed nothing.
_EXPECTED_CHAT = (
    "You are dexta, a continuous health-intelligence assistant for one Type-1 "
    "diabetes patient. You reason over their real data using the tools provided - "
    "you never compute statistics yourself, you call a tool. Decide which tools "
    "(if any) a question needs; a question about framing or what you already know "
    "may need none.\n\n"
    "Hard rules:\n"
    "- Observation and discussion only. NEVER give dosing, insulin, carb-ratio, or "
    "medication advice. If asked, say that is for their care team and offer to show "
    "the relevant pattern instead.\n"
    "- Every number you state must come from a tool result you actually called.\n"
    "- If the data cannot answer, say so plainly and say what would be needed.\n"
    "Be concise and specific. Cite the n behind any comparison."
)


def test_load_returns_chat_prompt() -> None:
    text = prompts.load("chat_system")
    assert "NEVER give dosing" in text
    assert text.startswith("You are dexta")


def test_chat_system_unchanged_by_migration() -> None:
    assert chat._SYSTEM == _EXPECTED_CHAT


def test_with_safety_is_noop_when_rail_present() -> None:
    text = "Do the analysis. NEVER give dosing advice."
    assert prompts.with_safety(text) == text


def test_with_safety_reapplies_rail_when_missing() -> None:
    out = prompts.with_safety("A custom prompt with the rail stripped out.")
    assert prompts.SAFETY_RAIL in out
    assert "NEVER give dosing" in out


def test_override_dir_takes_precedence(tmp_path: Path) -> None:
    (tmp_path / "chat_system.md").write_text("Overridden body, no rail here.\n")
    loaded = prompts.load("chat_system", overrides_dir=tmp_path)
    assert loaded == "Overridden body, no rail here."
    # the override lost the rail, but with_safety puts it back
    assert "NEVER give dosing" in prompts.with_safety(loaded)
