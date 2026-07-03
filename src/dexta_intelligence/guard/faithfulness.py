"""Numeric-faithfulness guard - no model may introduce numbers the data lacks.

The core safety property: every number in LLM-authored prose must trace to the
deterministic evidence pool, within a small relative tolerance. Untraceable
output is rejected and the caller falls back to deterministic text. The model
discovers, ranks, and explains, but every number it cites must trace to the pool.

Design notes:

- Reports all violations with surrounding context, not a boolean, so false
  rejections are debuggable and measurable.
- The allowed-constants set is explicit, documented, and per-call extensible.
- Percent-of-pool matching and the absolute floor are tunable per surface.

Two layers:

- **Membership** (always on): every cited number must appear in the evidence pool
  within tolerance, or be an allowed constant. Guards against fabricated figures.
- **Provenance** (opt-in, ``check_provenance=True``): a number cited next to a
  metric name must match *that metric's* value, not merely exist somewhere in the
  pool. This catches "right number, wrong metric" - e.g. citing the standard
  deviation as the coefficient of variation - the exact error the LLM-CGM
  benchmark found a code-execution agent still makes. It uses the metric ontology
  in :mod:`dexta_intelligence.guard.metrics` to bind numbers to metrics.

Honest limits: membership is set-membership, not semantic verification, and prose
with no numbers passes trivially. Provenance only fires when the ontology can
resolve both the cited metric and a conflicting one, so it is precise but not
exhaustive. It is a guardrail, not a proof of correctness.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from dexta_intelligence.guard.metrics import is_percent, metric_for_key, metrics_in_context

__all__ = [
    "DEFAULT_ALLOWED_CONSTANTS",
    "FaithfulnessReport",
    "ProvenanceViolation",
    "Violation",
    "audit",
    "build_provenance",
    "extract_numbers",
]

#: A number, tolerating thousands separators so "25,920" parses as one figure
#: (25920) rather than splitting into 25 and 920. The grouped alternative is
#: tried first; a bare run of digits is the fallback.
_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")


def _to_float(token: str) -> float:
    """Parse a numeric token, dropping thousands separators (``25,920`` -> 25920)."""
    return float(token.replace(",", ""))

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
class ProvenanceViolation:
    """A number cited for the wrong metric: it matches ``matched_metric`` in the
    evidence, not the ``claimed_metric`` named in its sentence."""

    number: float
    claimed_metric: str
    matched_metric: str
    context: str
    expected: tuple[float, ...]

    def __str__(self) -> str:
        exp = ", ".join(f"{v:g}" for v in self.expected) or "no value"
        return (
            f"{self.number:g} cited as {self.claimed_metric} (expected {exp}) but matches "
            f"{self.matched_metric} in: …{self.context}…"
        )


@dataclass(frozen=True, slots=True)
class FaithfulnessReport:
    """Outcome of one audit. ``ok`` is the gate; violations are the evidence."""

    ok: bool
    violations: tuple[Violation, ...]
    n_numbers_checked: int
    provenance_violations: tuple[ProvenanceViolation, ...] = field(default_factory=tuple)

    def __bool__(self) -> bool:
        return self.ok


def extract_numbers(obj: Any) -> list[float]:
    """Recursively collect every numeric value reachable in ``obj``.

    Walks dicts/lists, takes ints and floats directly (bools excluded -
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
            out.append(_to_float(match))


def build_provenance(evidence: Any) -> dict[str, list[float]]:
    """Bind evidence numbers to the metric they describe.

    Walks the evidence structure; a numeric leaf under a key the ontology knows
    (``mean``, ``cv_pct``, ...) is recorded under that canonical metric. Parent
    keys propagate through lists, so ``{"daily_means": [130, 140]}`` maps both to
    ``mean`` if ``daily_means`` resolves. Keys the ontology does not know are
    ignored - those numbers are still covered by the membership check.
    """
    out: dict[str, list[float]] = {}
    _provenance_into(evidence, None, out)
    return out


def _provenance_into(obj: Any, key: str | None, out: dict[str, list[float]]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, dict):
        for k, value in obj.items():
            _provenance_into(value, str(k), out)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _provenance_into(value, key, out)
    elif isinstance(obj, (int, float)) and math.isfinite(obj):
        metric = metric_for_key(key) if key else None
        if metric is not None:
            out.setdefault(metric, []).append(float(obj))


def _matches_metric(cited: float, values: list[float], percent: bool,
                    rel_tolerance: float) -> bool:
    """Whether ``cited`` equals one of ``values`` (a metric's evidence values).

    Tolerance is scaled to the compared magnitudes (``rel_tolerance * max(|c|,|v|)``)
    rather than a flat floor, so a fraction (0.287) is not spuriously "close" to a
    percent (28.7). For ratio/percent metrics a value stored as a fraction still
    matches a percent citation and vice versa, so unit convention never misfires.
    """
    candidates = [cited]
    if percent:
        candidates += [cited / 100.0, cited * 100.0]
    return any(
        abs(c - v) <= rel_tolerance * max(abs(c), abs(v)) + 1e-9
        for v in values for c in candidates
    )


def audit(
    texts: str | list[str],
    evidence: Any,
    *,
    rel_tolerance: float = 0.05,
    abs_floor: float = 1.0,
    allowed_constants: frozenset[int] = DEFAULT_ALLOWED_CONSTANTS,
    check_provenance: bool = False,
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
        check_provenance: also flag numbers cited for the wrong metric (a number
            that matches a *different* metric than the one its sentence names).

    Returns:
        A :class:`FaithfulnessReport`; falsy when any number is untraceable or
        (when ``check_provenance``) cited for the wrong metric.
    """
    text = texts if isinstance(texts, str) else "\n".join(texts)
    pool = [abs(p) for p in extract_numbers(evidence)]
    provenance = build_provenance(evidence) if check_provenance else {}

    violations: list[Violation] = []
    prov_violations: list[ProvenanceViolation] = []
    n_checked = 0
    for match in _NUMBER_RE.finditer(text):
        if _is_clock_fragment(text, match):
            continue
        cited = abs(_to_float(match.group()))
        n_checked += 1
        if cited.is_integer() and int(cited) in allowed_constants:
            continue
        start, end = match.span()
        context = text[max(0, start - _CONTEXT_CHARS) : end + _CONTEXT_CHARS].replace("\n", " ")
        traceable = any(abs(cited - p) <= max(rel_tolerance * p, abs_floor) for p in pool)
        if not traceable:
            nearest = min(pool, key=lambda p: abs(cited - p)) if pool else None
            violations.append(Violation(
                number=_to_float(match.group()), context=context, nearest_pool_value=nearest
            ))
            continue
        if provenance:
            pv = _check_provenance_at(cited, context, provenance, rel_tolerance)
            if pv is not None:
                prov_violations.append(pv)

    return FaithfulnessReport(
        ok=not violations and not prov_violations,
        violations=tuple(violations),
        n_numbers_checked=n_checked,
        provenance_violations=tuple(prov_violations),
    )


def _check_provenance_at(
    cited: float, context: str, provenance: dict[str, list[float]],
    rel_tolerance: float,
) -> ProvenanceViolation | None:
    """Flag a traceable number that is cited for the wrong metric.

    Fires only when the sentence names a metric we hold a value for, the number
    does not match that metric, and it does match a different metric we hold.
    That triple keeps it precise: ambiguous or unresolvable context never fires.
    """
    claimed = metrics_in_context(context)
    claimed_present = [m for m in claimed if m in provenance]
    if not claimed_present:
        return None
    if any(_matches_metric(cited, provenance[m], is_percent(m), rel_tolerance)
           for m in claimed_present):
        return None
    matched = next(
        (o for o, vals in provenance.items()
         if o not in claimed and _matches_metric(cited, vals, is_percent(o), rel_tolerance)),
        None,
    )
    if matched is None:
        return None
    expected = tuple(provenance[claimed_present[0]])
    return ProvenanceViolation(
        number=cited, claimed_metric=claimed_present[0], matched_metric=matched,
        context=context, expected=expected,
    )


def _is_clock_fragment(text: str, match: re.Match[str]) -> bool:
    """True when ``match`` is part of a clock time (``8:43``, ``10:51 AM``), not a claim.

    The guard audits citable figures, not time-of-day formatting. Without this,
    bolus tables and prose like "8:43 AM" produce spurious ``43`` / ``10`` rejects
    because ISO timestamps are not flattened into the numeric evidence pool.
    """
    start, end = match.span()
    if start > 0 and text[start - 1] == ":":
        return True
    return end < len(text) and text[end] == ":"
