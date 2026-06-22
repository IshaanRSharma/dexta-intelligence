"""Eval runner CLI - produces the reportable benchmark table."""

from __future__ import annotations

import argparse
import sys

from eval.metrics.e1_faithfulness import run_e1
from eval.metrics.e2_power import run_e2_power
from eval.metrics.e3_accuracy import run_e3_accuracy
from eval.metrics.e4_null_fdr import run_e4_null_fdr
from eval.metrics.e5_perturbation import run_e5
from eval.metrics.e6_attribution import run_e6_attribution
from eval.metrics.e6_faithfulness import run_e6_faithfulness
from eval.metrics.e6_safety import run_e6_safety
from eval.metrics.e7_reasoning import run_e7_reasoning
from eval.metrics.e_consensus import run_e_consensus
from eval.report import (
    e1_row,
    e2_row,
    e3_row,
    e4_row,
    e5_row,
    e6_attribution_row,
    e6_faithfulness_row,
    e6_safety_row,
    e7_reasoning_row,
    e_consensus_row,
    render_json,
    render_markdown,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="dexta-intelligence eval harness")
    sub = parser.add_subparsers(dest="command", required=True)

    e4 = sub.add_parser("e4-null", help="E4 null-set FDR calibration")
    e4.add_argument("--datasets", type=int, default=20, help="Number of null sets")
    e4.add_argument("--days", type=int, default=90, help="Days per synthetic set")
    e4.add_argument("--format", choices=("md", "json"), default="md")

    e1 = sub.add_parser("e1", help="E1 numeric-faithfulness guard eval")
    e1.add_argument("--texts", type=int, default=30, help="Prose samples per class")
    e1.add_argument("--seed", type=int, default=7000, help="RNG seed")
    e1.add_argument("--format", choices=("md", "json"), default="md")

    e5 = sub.add_parser("e5", help="E5 perturbation robustness eval")
    e5.add_argument("--days", type=int, default=90, help="Days of scenario data")
    e5.add_argument("--seed", type=int, default=5000, help="RNG seed")
    e5.add_argument("--format", choices=("md", "json"), default="md")

    e2 = sub.add_parser("e2", help="E2 statistical-power true-discovery eval")
    e2.add_argument("--seeds", type=int, default=5, help="Seeds per (effect, span) cell")
    e2.add_argument("--seed-base", type=int, default=7700, help="Base RNG seed")
    e2.add_argument("--format", choices=("md", "json"), default="md")

    e3 = sub.add_parser("e3", help="E3 clinical-accuracy (error grid + MARD) eval")
    e3.add_argument("--days", type=int, default=30, help="Days of synthetic data")
    e3.add_argument("--seed", type=int, default=8800, help="RNG seed")
    e3.add_argument("--format", choices=("md", "json"), default="md")

    ec = sub.add_parser("consensus", help="E_consensus rollup-formula agreement eval")
    ec.add_argument("--days", type=int, default=14, help="Days of synthetic data")
    ec.add_argument("--seed", type=int, default=9100, help="RNG seed")
    ec.add_argument("--format", choices=("md", "json"), default="md")

    e6 = sub.add_parser("e6", help="E6 end-to-end agentic eval (needs a model)")
    e6.add_argument("--format", choices=("md", "json"), default="md")

    e7 = sub.add_parser("reasoning", help="E7 reasoning-process eval (needs a model)")
    e7.add_argument("--format", choices=("md", "json"), default="md")

    args = parser.parse_args(argv)
    handler = _HANDLERS.get(args.command)
    if handler is None:
        return 2
    return handler(args)


def _emit(data: dict[str, object], fmt: str, summary: str) -> None:
    if fmt == "json":
        sys.stdout.write(render_json(data) + "\n")
    else:
        sys.stdout.write(render_markdown(data) + "\n" + summary + "\n")


def _cmd_e4(args: argparse.Namespace) -> int:
    result = run_e4_null_fdr(n_datasets=args.datasets, n_days=args.days)
    status = "PASS" if result.passed else "FAIL"
    _emit(
        e4_row(result),
        args.format,
        f"\n**{status}** - empirical FDR {result.empirical_fdr:.1%} (target ≤ {result.alpha:.0%})",
    )
    return 0 if result.passed else 1


def _cmd_e1(args: argparse.Namespace) -> int:
    result = run_e1(seed=args.seed, n_texts=args.texts)
    status = "PASS" if result.passed else "FAIL"
    _emit(
        e1_row(result),
        args.format,
        f"\n**{status}** - fabricated catch {result.catch_rate_fabricated:.0%}, "
        f"false-rejection {result.false_rejection_rate:.1%} "
        f"(targets {result.catch_target:.0%} / < {result.false_reject_target:.0%})",
    )
    return 0 if result.passed else 1


def _cmd_e5(args: argparse.Namespace) -> int:
    result = run_e5(seed=args.seed, n_days=args.days)
    status = "PASS" if result.passed else "FAIL"
    _emit(
        e5_row(result),
        args.format,
        f"\n**{status}** - min Jaccard {result.min_jaccard:.2f}, "
        f"{result.total_new_kinds} corruption-induced new kinds "
        f"(targets ≥ {result.jaccard_target:.1f} / 0)",
    )
    return 0 if result.passed else 1


def _cmd_e2(args: argparse.Namespace) -> int:
    result = run_e2_power(n_seeds=args.seeds, seed_base=args.seed_base)
    status = "PASS" if result.passed else "FAIL"
    _emit(
        e2_row(result),
        args.format,
        f"\n**{status}** - best recall {result.best_recall:.0%} "
        f"(target >= {result.recall_target:.0%})",
    )
    return 0 if result.passed else 1


def _cmd_e3(args: argparse.Namespace) -> int:
    result = run_e3_accuracy(seed=args.seed, n_days=args.days)
    _emit(
        e3_row(result),
        args.format,
        f"\n**REPORT** - MARD {result.mard_pct:.1f}%, "
        f"Clarke A+B {result.clarke_ab_pct:.1f}%, "
        f"Parkes A+B {result.parkes_ab_pct:.1f}% "
        f"over {result.n_pairs} forecast pairs",
    )
    return 0


def _cmd_consensus(args: argparse.Namespace) -> int:
    result = run_e_consensus(seed=args.seed, n_days=args.days)
    status = "PASS" if result.passed else "FAIL"
    _emit(
        e_consensus_row(result),
        args.format,
        f"\n**{status}** - {result.n_disagreements} disagreements "
        f"in {result.n_checks} metric checks",
    )
    return 0 if result.passed else 1


def _cmd_e6(args: argparse.Namespace) -> int:
    """End-to-end agentic eval: attribution, faithfulness, safety. Needs a model.

    Skippable without a key (returns 2 with a clear message) so the deterministic
    evals stay key-free while E6 grades the real agent when a provider is set.
    """
    from dexta_intelligence.cli._common import discovery_model  # noqa: PLC0415
    from dexta_intelligence.config import load_config  # noqa: PLC0415

    model = discovery_model(load_config(None))
    if model is None:
        sys.stdout.write("E6 needs a language model. Set a provider and an API key, then retry.\n")
        return 2

    attribution = run_e6_attribution(model)
    faithfulness = run_e6_faithfulness(model)
    safety = run_e6_safety(model)
    rows: dict[str, object] = {
        **e6_attribution_row(attribution),
        **e6_faithfulness_row(faithfulness),
        **e6_safety_row(safety),
    }
    passed = attribution.passed and faithfulness.passed and safety.passed
    status = "PASS" if passed else "FAIL"
    _emit(
        rows,
        args.format,
        f"\n**{status}** attribution {attribution.accuracy:.0%} "
        f"(target >= {attribution.accuracy_target:.0%}), "
        f"faithful {faithfulness.faithful_rate:.0%} "
        f"(target >= {faithfulness.faithful_target:.0%}), "
        f"dosing violations {safety.violations}/{safety.n_prompts} (target 0)",
    )
    return 0 if passed else 1


def _cmd_reasoning(args: argparse.Namespace) -> int:
    """Reasoning-process eval: coverage, efficiency, gap-handling, soundness.

    Needs a model. Skippable without a key (returns 2) like E6, so the
    deterministic evals stay key-free while this grades the real agent's path.
    """
    from dexta_intelligence.cli._common import discovery_model  # noqa: PLC0415
    from dexta_intelligence.config import load_config  # noqa: PLC0415

    model = discovery_model(load_config(None))
    if model is None:
        sys.stdout.write("E7 needs a language model. Set a provider and an API key, then retry.\n")
        return 2

    result = run_e7_reasoning(model)
    status = "PASS" if result.passed else "FAIL"
    _emit(
        e7_reasoning_row(result),
        args.format,
        f"\n**{status}** - coverage {result.mean_coverage:.0%} "
        f"(target >= {result.coverage_target:.0%}), "
        f"soundness {result.soundness_rate:.0%} "
        f"(target >= {result.soundness_target:.0%}), "
        f"efficiency {result.mean_efficiency:.0%}, "
        f"gap-handling {result.gap_handling_rate:.0%}",
    )
    return 0 if result.passed else 1


_HANDLERS = {
    "e4-null": _cmd_e4,
    "e1": _cmd_e1,
    "e5": _cmd_e5,
    "e2": _cmd_e2,
    "e3": _cmd_e3,
    "consensus": _cmd_consensus,
    "e6": _cmd_e6,
    "reasoning": _cmd_reasoning,
}


if __name__ == "__main__":
    raise SystemExit(main())
