"""Parse a JSON object from an LLM response, tolerant of markdown code fences.

Shared by the agents that ask the model for JSON. ``content`` may be a string or
LangChain content-parts. ``context`` labels a parse-failure warning; an empty
context stays silent.
"""

from __future__ import annotations

import json
import logging
from typing import Any

__all__ = ["parse_json"]

logger = logging.getLogger(__name__)


def parse_json(content: Any, *, context: str = "") -> dict[str, Any] | None:
    """Return the JSON object in ``content``, or ``None`` if it is not JSON."""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    else:
        return None
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        if context:
            logger.warning("%s: non-JSON response: %s", context, text[:200])
        return None
    return parsed if isinstance(parsed, dict) else None
