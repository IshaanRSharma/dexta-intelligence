"""Eval runner CLI — produces the reportable benchmark table."""

from __future__ import annotations

import argparse
import sys

from eval.metrics.e1_faithfulness import run_e1
from eval.metrics.e4_null_fdr import run_e4_null_fdr
from eval.metrics.e5_perturbation import run_e5
from eval.report import e1_row, e4_row, e5_row, render_json, render_markdown


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

    args = parser.parse_args(argv)

    if args.command == "e4-null":
        e4_result = run_e4_null_fdr(n_datasets=args.datasets, n_days=args.days)
        e4_data = e4_row(e4_result)
        if args.format == "json":
            sys.stdout.write(render_json(e4_data) + "\n")
        else:
            sys.stdout.write(render_markdown(e4_data) + "\n")
            status = "PASS" if e4_result.passed else "FAIL"
            sys.stdout.write(f"\n**{status}** — empirical FDR {e4_result.empirical_fdr:.1%} "
                             f"(target ≤ {e4_result.alpha:.0%})\n")
        return 0 if e4_result.passed else 1

    if args.command == "e1":
        e1_result = run_e1(seed=args.seed, n_texts=args.texts)
        e1_data = e1_row(e1_result)
        if args.format == "json":
            sys.stdout.write(render_json(e1_data) + "\n")
        else:
            sys.stdout.write(render_markdown(e1_data) + "\n")
            status = "PASS" if e1_result.passed else "FAIL"
            sys.stdout.write(
                f"\n**{status}** — fabricated catch "
                f"{e1_result.catch_rate_fabricated:.0%}, false-rejection "
                f"{e1_result.false_rejection_rate:.1%} "
                f"(targets {e1_result.catch_target:.0%} / "
                f"< {e1_result.false_reject_target:.0%})\n"
            )
        return 0 if e1_result.passed else 1

    if args.command == "e5":
        e5_result = run_e5(seed=args.seed, n_days=args.days)
        e5_data = e5_row(e5_result)
        if args.format == "json":
            sys.stdout.write(render_json(e5_data) + "\n")
        else:
            sys.stdout.write(render_markdown(e5_data) + "\n")
            status = "PASS" if e5_result.passed else "FAIL"
            sys.stdout.write(
                f"\n**{status}** — min Jaccard {e5_result.min_jaccard:.2f}, "
                f"{e5_result.total_new_kinds} corruption-induced new kinds "
                f"(targets ≥ {e5_result.jaccard_target:.1f} / 0)\n"
            )
        return 0 if e5_result.passed else 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
