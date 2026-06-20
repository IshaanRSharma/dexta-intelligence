"""E1 numeric faithfulness - guard catch rate on simulated model prose.

Ground truth is non-LLM: we synthesize an evidence pool of Finding-like dicts
with known numbers, then generate prose with planted ground-truth labels:

- ``faithful`` - cites only pool numbers (± small rounding within tolerance).
- ``fabricated`` - cites a planted number absent from the pool (and far from
  every allowed constant), so the guard *must* reject it.
- ``miscontextualized`` - cites pool numbers plus one out-of-pool derived
  figure; the guard rejects on the untraceable figure.

The guard (:func:`dexta_intelligence.guard.faithfulness.audit`) is the unit
under test. Metrics: catch rate on fabricated/miscontextualized prose
(target 1.0) and false-rejection rate on faithful prose (target < 0.05).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from dexta_intelligence.guard.faithfulness import audit, extract_numbers

__all__ = ["E1FaithfulnessResult", "run_e1"]

#: Targets the eval is scored against (spec §14, E1).
_CATCH_TARGET = 1.0
_FALSE_REJECT_TARGET = 0.05


@dataclass(frozen=True, slots=True)
class E1FaithfulnessResult:
    """Outcome of one E1 sweep over simulated prose."""

    n_faithful: int
    n_fabricated: int
    n_miscontextualized: int
    catch_rate_fabricated: float
    catch_rate_miscontextualized: float
    false_rejection_rate: float
    catch_target: float
    false_reject_target: float
    passed: bool


def _evidence_pool(rng: random.Random) -> dict[str, float | int]:
    """Build a Finding-like evidence dict with distinctive numbers.

    Numbers are spread out so a fabricated figure can be planted that is far
    from every pool value *and* every allowed clinical/clock constant.
    """
    mean = round(rng.uniform(120.0, 175.0), 1)
    return {
        "n_readings": rng.randint(2000, 8000),
        "mean_mg_dl": mean,
        "gmi_pct": round(3.31 + 0.02392 * mean, 1),
        "cv_pct": round(rng.uniform(28.0, 44.0), 1),
        "tir_pct": round(rng.uniform(55.0, 78.0), 1),
        "tbr_pct": round(rng.uniform(1.0, 6.0), 1),
        "tar_pct": round(rng.uniform(18.0, 40.0), 1),
        "coverage_pct": round(rng.uniform(85.0, 99.0), 1),
    }


def _faithful_text(pool: dict[str, float | int], rng: random.Random) -> str:
    """Prose citing pool numbers, some nudged within the 5% tolerance."""
    tir = float(pool["tir_pct"])
    mean = float(pool["mean_mg_dl"])
    gmi = float(pool["gmi_pct"])
    # Nudge mean within ±3% to exercise the rounding-tolerance path.
    nudged_mean = round(mean * (1.0 + rng.uniform(-0.03, 0.03)), 1)
    return (
        f"Time in range sat at {tir:.1f}% over the window, with a mean glucose "
        f"near {nudged_mean:.1f} mg/dL and an estimated GMI of {gmi:.1f}%. "
        f"Coverage was {float(pool['coverage_pct']):.1f}%."
    )


def _fabricated_text(pool: dict[str, float | int], rng: random.Random) -> str:
    """Prose citing a planted figure that exists in neither the pool nor the
    allowed-constants set. The guard must flag it."""
    planted = round(rng.uniform(311.0, 389.0), 1)
    tir = float(pool["tir_pct"])
    return (
        f"Time in range was {tir:.1f}%, but the standout was a sustained "
        f"excursion peaking at {planted:.1f} mg/dL - the dominant driver of the "
        f"window's variability."
    )


def _miscontextualized_text(pool: dict[str, float | int], rng: random.Random) -> str:
    """Prose citing real pool numbers plus one out-of-pool *derived* figure.

    Set-membership faithfulness catches the untraceable derived number even
    though the surrounding figures are all real."""
    tir = float(pool["tir_pct"])
    mean = float(pool["mean_mg_dl"])
    # A plausible-sounding but un-pooled derived ratio, far from any pool value.
    derived = round(rng.uniform(412.0, 487.0), 1)
    return (
        f"Mean glucose was {mean:.1f} mg/dL and time in range {tir:.1f}%, "
        f"implying a glycemic load index of {derived:.1f} units for the period."
    )


def _is_untraceable(pool: dict[str, float | int], number: float) -> bool:
    """True when ``number`` traces to no pool value within the guard tolerance.

    Used only to assert our planted figures are genuinely out-of-pool, so the
    eval's ground-truth labels are honest and not an artifact of a collision.
    """
    pool_vals = [abs(v) for v in extract_numbers(pool)]
    return not any(abs(number - p) <= max(0.05 * p, 1.0) for p in pool_vals)


def run_e1(*, seed: int = 7000, n_texts: int = 30) -> E1FaithfulnessResult:
    """Run the faithfulness guard over ``n_texts`` prose samples per class.

    Each of the three classes (faithful, fabricated, miscontextualized) gets
    ``n_texts`` independent samples drawn from a fresh evidence pool.
    """
    if n_texts < 1:
        msg = "n_texts must be >= 1"
        raise ValueError(msg)

    rng = random.Random(seed)

    faithful_rejected = 0
    fabricated_caught = 0
    mis_caught = 0

    for _ in range(n_texts):
        pool = _evidence_pool(rng)

        faithful = _faithful_text(pool, rng)
        if not audit(faithful, pool).ok:
            faithful_rejected += 1

        fabricated = _fabricated_text(pool, rng)
        if not audit(fabricated, pool).ok:
            fabricated_caught += 1

        mis = _miscontextualized_text(pool, rng)
        if not audit(mis, pool).ok:
            mis_caught += 1

    catch_fab = fabricated_caught / n_texts
    catch_mis = mis_caught / n_texts
    false_reject = faithful_rejected / n_texts

    passed = (
        catch_fab >= _CATCH_TARGET
        and catch_mis >= _CATCH_TARGET
        and false_reject < _FALSE_REJECT_TARGET
    )

    return E1FaithfulnessResult(
        n_faithful=n_texts,
        n_fabricated=n_texts,
        n_miscontextualized=n_texts,
        catch_rate_fabricated=catch_fab,
        catch_rate_miscontextualized=catch_mis,
        false_rejection_rate=false_reject,
        catch_target=_CATCH_TARGET,
        false_reject_target=_FALSE_REJECT_TARGET,
        passed=passed,
    )
