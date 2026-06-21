"""Prompt registry: agent prompts as version-controlled markdown.

Each prompt is a ``<name>.md`` file in this package, loaded by name. A user can
override any of them with a directory of same-named files (``[prompts] dir``).
The dosing/observation-only rail is a code constant re-applied by
:func:`with_safety`, so an override can never remove it.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

__all__ = ["SAFETY_RAIL", "load", "with_safety"]

SAFETY_RAIL = (
    "Observation and discussion only. NEVER give dosing, insulin, carb-ratio, "
    "or medication advice; that is for the care team. Offer the relevant "
    "pattern instead."
)
_RAIL_SENTINEL = "NEVER give dosing"


def load(name: str, *, overrides_dir: Path | None = None) -> str:
    """Return prompt ``name``, preferring a user override when one exists."""
    if overrides_dir is None:
        overrides_dir = _configured_overrides_dir()
    if overrides_dir is not None:
        override = Path(overrides_dir).expanduser() / f"{name}.md"
        if override.is_file():
            return override.read_text(encoding="utf-8").strip()
    return resources.files(__name__).joinpath(f"{name}.md").read_text(encoding="utf-8").strip()


def with_safety(text: str) -> str:
    """Guarantee the dosing rail is present, even on an overridden prompt."""
    if _RAIL_SENTINEL in text:
        return text
    return f"{text.rstrip()}\n\n{SAFETY_RAIL}"


def _configured_overrides_dir() -> Path | None:
    try:
        from dexta_intelligence.config import load_config  # noqa: PLC0415

        return load_config().prompts.dir
    except Exception:
        return None
