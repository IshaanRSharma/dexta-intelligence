"""Numeric-faithfulness guard — no model may introduce numbers the data lacks.

The core safety property of the platform: **every number in LLM-authored
prose must trace to the deterministic evidence pool**, within a small
relative tolerance. Untraceable output is rejected and the caller falls back
to deterministic text. The model ranks and explains; it cannot invent.

This is a hardened port of the guard that shipped in the donor codebase's
clinical brief (where it ran in production against real endocrinologist
review). Improvements over the donor:

- Reports **all** violations with surrounding context, not a boolean — so
  false rejections are debuggable and measurable (eval E1's
  false-rejection-rate metric depends on this).
- The allowed-constants set is explicit, documented, and per-call
  extensible instead of a module-level mystery set.
- Percent-of-pool matching and the absolute floor are tunable per surface.

Honest limits (state these, never oversell): this is set-membership
checking, not semantic verification. A number can match the pool while
being cited in the wrong context (eval E2's consistency invariants exist
for exactly that class), and prose with *no* numbers passes trivially.
It is a guardrail against fabricated figures, not a proof of correctness.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

__all__ = [
    "DEFAULT_ALLOWED_CONSTANTS",
    "FaithfulnessReport",
    "Violation",
    "audit",
    "extract_numbers",
]

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

#: Integers that may appear in clinical prose without tracing to evidence.
#: Each entry is a convention, not data: clock arithmetic (0-24, 30, 60, 90),
#: consensus glucose thresholds (54, 70, 100, 180, 250), percentage anchors
#: (0, 100), and guideline citation years.
DEFAULT_ALLOWED_CONSTANTS: frozenset[int] = frozenset(
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 15, 20, 24, 30, 60, 90, 70, 100, 180, 54, 250, 2019}
)

_CONTEXT_CHARS = 40


@dataclass(frozen=True, slots=True)
class Violation:
    """One untraceable number, with enough context to debug the rejection."""

    number: float
    context: str
    nearest_pool_value: float | None

    def __str__(self) -> str:
        nearest = (
            f"nearest evidence value {self.nearest_pool_value}"
            if self.nearest_pool_value is not None
            else "evidence pool empty"
        )
        return f"{self.number} not traceable ({nearest}) in: …{self.context}…"


@dataclass(frozen=True, slots=True)
class FaithfulnessReport:
    """Outcome of one audit. ``ok`` is the gate; violations are the evidence."""

    ok: bool
    violations: tuple[Violation, ...]
    n_numbers_checked: int

    def __bool__(self) -> bool:
        return self.ok


def extract_numbers(obj: Any) -> list[float]:
    """Recursively collect every numeric value reachable in ``obj``.

    Walks dicts/lists, takes ints and floats directly (bools excluded —
    they are not citable figures), and parses numbers embedded in strings.
    This builds the evidence pool the guard checks prose against.
    """
    out: list[float] = []
    _extract_into(obj, out)
    return out


def _extract_into(obj: Any, out: list[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, dict):
        for value in obj.values():
            _extract_into(value, out)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _extract_into(value, out)
    elif isinstance(obj, (int, float)):
        if math.isfinite(obj):
            out.append(float(obj))
    elif isinstance(obj, str):
        for match in _NUMBER_RE.findall(obj):
            out.append(float(match))


def audit(
    texts: str | list[str],
    evidence: Any,
    *,
    rel_tolerance: float = 0.05,
    abs_floor: float = 1.0,
    allowed_constants: frozenset[int] = DEFAULT_ALLOWED_CONSTANTS,
) -> FaithfulnessReport:
    """Audit prose against an evidence pool.

    A cited number ``c`` passes if any pool value ``p`` satisfies
    ``|c - p| <= max(rel_tolerance * |p|, abs_floor)`` (magnitudes compared
    absolutely, so sign formatting like "-0.4%" vs "0.4" never causes a
    false rejection), or if ``c`` is an allowed clinical/clock constant.

    Args:
        texts: one string or a list of prose strings to audit.
        evidence: any structure; its numbers become the pool via
            :func:`extract_numbers`.
        rel_tolerance: relative slack for rounding differences (default 5%).
        abs_floor: absolute slack floor so tiny pool values don't demand
            impossible precision.
        allowed_constants: integers exempt from tracing.

    Returns:
        A :class:`FaithfulnessReport`; falsy when any number is untraceable.
    """
    text = texts if isinstance(texts, str) else "\n".join(texts)
    pool = [abs(p) for p in extract_numbers(evidence)]

    violations: list[Violation] = []
    n_checked = 0
    for match in _NUMBER_RE.finditer(text):
        cited = abs(float(match.group()))
        n_checked += 1
        if cited.is_integer() and int(cited) in allowed_constants:
            continue
        if any(abs(cited - p) <= max(rel_tolerance * p, abs_floor) for p in pool):
            continue
        start, end = match.span()
        context = text[max(0, start - _CONTEXT_CHARS) : end + _CONTEXT_CHARS].replace("\n", " ")
        nearest = min(pool, key=lambda p: abs(cited - p)) if pool else None
        violations.append(
            Violation(number=float(match.group()), context=context, nearest_pool_value=nearest)
        )

    return FaithfulnessReport(
        ok=not violations, violations=tuple(violations), n_numbers_checked=n_checked
    )
