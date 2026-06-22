"""Argument parser and command dispatch for the dexta CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from dexta_intelligence.cli._common import (
    init_config_path,
    resolve_config_path,
)
from dexta_intelligence.cli.analysis import cmd_analyze, cmd_investigate
from dexta_intelligence.cli.daemon import cmd_daemon
from dexta_intelligence.cli.data import cmd_doctor, cmd_init, cmd_sync, cmd_upload
from dexta_intelligence.cli.intelligence import (
    cmd_ask,
    cmd_brief,
    cmd_demo,
    cmd_explain,
    cmd_goals,
    cmd_monitor,
    cmd_timing,
    cmd_wiki,
)
from dexta_intelligence.cli.research import cmd_nof1
from dexta_intelligence.cli.serve import cmd_serve
from dexta_intelligence.config import load_config

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dexta",
        description="Continuous health intelligence for Type 1 diabetes.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to dexta.toml (default: ./dexta.toml if present, else ~/.dexta/dexta.toml)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Override the SQLite database path from config",
    )

    sub = parser.add_subparsers(dest="command", required=False)

    init_p = sub.add_parser("init", help="Write starter dexta.toml and create the SQLite database")
    init_p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing config file",
    )

    sub.add_parser("doctor", help="Connectivity, auth, and coverage checks")
    sub.add_parser("sync", help="Pull from configured data sources")
    analyze_p = sub.add_parser("analyze", help="Run the agent harness on stored data")
    analyze_p.add_argument(
        "--lens",
        default="analyze",
        help="Named agent route (builtin: analyze, watch, why, insulin; or a [lens.*] entry)",
    )
    investigate_p = sub.add_parser(
        "investigate",
        help="Deep investigation: the coordinator plans and runs investigations for a goal",
    )
    investigate_p.add_argument(
        "goal", nargs="?", default=None, help="Goal to investigate (omit for a whole-record pass)"
    )

    sub.add_parser("wiki", help="Regenerate the markdown knowledge base from stored findings")
    sub.add_parser("brief", help="Render a physician-visit brief from accumulated findings")

    ask_p = sub.add_parser("ask", help="Ask a question; the model reasons over your data + memory")
    ask_p.add_argument("question", help="Natural-language question about your data")
    ask_p.add_argument(
        "--seek",
        action="store_true",
        help="Goal-seeking mode: reflect and re-scope across rounds until answered",
    )

    explain_p = sub.add_parser(
        "explain", help="Explain a spike: deterministic treatment-aware investigation"
    )
    explain_p.add_argument(
        "when",
        help="ISO date (locates the day's largest excursion) or ISO datetime of the event",
    )

    goals_p = sub.add_parser("goals", help="Goal-directed background agents (add/list/tick)")
    goals_p.add_argument("action", choices=("add", "list", "tick"))
    goals_p.add_argument("statement", nargs="?", help="Goal text (for 'add')")
    goals_p.add_argument(
        "--target", type=float, default=None, help="Numeric success target for the goal metric"
    )

    upload_p = sub.add_parser(
        "upload", help="Import glucose history from a CSV or Tidepool JSON export"
    )
    upload_p.add_argument(
        "file",
        type=Path,
        help="Dexcom Clarity, LibreView CSV, or Tidepool JSON export",
    )
    upload_p.add_argument(
        "--format",
        dest="csv_format",
        choices=("auto", "clarity", "libreview", "tidepool"),
        default="auto",
        help="Export format (default: auto-detect from header or .json extension)",
    )
    upload_p.add_argument(
        "--tz",
        default="UTC",
        help="IANA timezone for device-local timestamps (default: UTC)",
    )

    research_p = sub.add_parser(
        "research",
        help="Pre-register a hypothesis and run a rigorous single-subject (n-of-1) test",
    )
    research_p.add_argument(
        "statement",
        nargs="?",
        default=None,
        help='Free-text hypothesis, e.g. "weekends run higher than weekdays"',
    )
    research_p.add_argument(
        "--compare",
        default=None,
        help="Comparison to register (weekend, sleep, workout, meal_carbs)",
    )
    research_p.add_argument(
        "--metric",
        default="mean_glucose",
        help="Outcome metric for metric-aware comparisons (mean_glucose, tir)",
    )
    research_p.add_argument(
        "--save", action="store_true", help="Persist the result as a kind='nof1' finding"
    )
    research_p.add_argument(
        "--seed", type=int, default=1729, help="Permutation seed (default: 1729)"
    )

    sub.add_parser(
        "demo", help="Run dexta end-to-end on a synthetic patient (no data or API key needed)"
    )
    sub.add_parser("monitor", help="Scan recent data for anomalies (lows/highs/cliffs/gaps)")

    timing_p = sub.add_parser(
        "timing", help="Observation-only briefing for a time bucket (no dosing, no API key)"
    )
    timing_p.add_argument(
        "--bucket",
        default="dinner",
        help="Preset (overnight/breakfast/lunch/dinner/bedtime) or hour range like 17-22",
    )
    timing_p.add_argument(
        "--intent",
        choices=("general", "meal", "basal"),
        default="general",
        help="general (default), meal (adds timing cards), or basal (adds drift card)",
    )

    daemon_p = sub.add_parser(
        "daemon", help="Run the cadence driver: sync + monitor + goal ticks + periodic deep pass"
    )
    daemon_p.add_argument(
        "--interval", type=float, default=5.0, help="Minutes between cycles (default: 5)"
    )
    daemon_p.add_argument(
        "--deep-every",
        type=int,
        default=12,
        help="Cycles between deep coordinator passes (default: 12)",
    )
    daemon_p.add_argument(
        "--once", action="store_true", help="Run a single cycle and exit (no sleep)"
    )

    serve_p = sub.add_parser("serve", help="Run the local web GUI (needs the [gui] extra)")
    serve_p.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1; pass 0.0.0.0 to expose on the LAN - no auth)",
    )
    serve_p.add_argument(
        "--port",
        type=int,
        default=8787,
        help="Port to listen on (default: 8787)",
    )
    serve_p.add_argument(
        "--sync-every",
        type=int,
        default=None,
        metavar="MIN",
        help="Re-sync data sources every MIN minutes in the background "
        "(default: the [server] auto_sync_minutes config, or off)",
    )
    serve_p.add_argument(
        "--demo",
        action="store_true",
        help="Seed a synthetic patient into an empty database first (no data or API key needed)",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:  # noqa: PLR0911, PLR0912
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "demo":
        return cmd_demo(out=sys.stdout)

    config_path = resolve_config_path(args.config)
    config = load_config(config_path if config_path.is_file() else args.config)

    if args.command == "init":
        target = init_config_path(args.config)
        return cmd_init(
            config_path=target,
            db_path=args.db,
            force=args.force,
            out=sys.stdout,
        )

    if args.command == "doctor":
        return cmd_doctor(
            config=config,
            db_path=args.db,
            out=sys.stdout,
        )

    if args.command == "sync":
        return cmd_sync(
            config=config,
            db_path=args.db,
            out=sys.stdout,
        )

    if args.command == "analyze":
        try:
            return cmd_analyze(
                config=config,
                db_path=args.db,
                out=sys.stdout,
                lens=args.lens,
            )
        except ValueError as exc:
            sys.stdout.write(f"{exc}\n")
            return 2

    if args.command == "ask":
        return cmd_ask(
            question=args.question,
            config=config,
            db_path=args.db,
            out=sys.stdout,
            seek=args.seek,
        )

    if args.command == "explain":
        return cmd_explain(
            when=args.when,
            config=config,
            db_path=args.db,
            out=sys.stdout,
        )

    if args.command == "investigate":
        return cmd_investigate(
            goal=args.goal,
            config=config,
            db_path=args.db,
            out=sys.stdout,
        )

    if args.command == "research":
        return cmd_nof1(
            config=config,
            db_path=args.db,
            out=sys.stdout,
            statement=args.statement,
            compare=args.compare,
            metric=args.metric,
            save=args.save,
            seed=args.seed,
        )

    if args.command == "monitor":
        return cmd_monitor(
            config=config,
            db_path=args.db,
            out=sys.stdout,
        )

    if args.command == "timing":
        return cmd_timing(
            bucket=args.bucket,
            intent=args.intent,
            config=config,
            db_path=args.db,
            out=sys.stdout,
        )

    if args.command == "daemon":
        return cmd_daemon(
            config=config,
            db_path=args.db,
            out=sys.stdout,
            interval=args.interval,
            deep_every=args.deep_every,
            once=args.once,
        )

    if args.command == "goals":
        return cmd_goals(
            action=args.action,
            statement=args.statement,
            target=args.target,
            config=config,
            db_path=args.db,
            out=sys.stdout,
        )

    if args.command == "brief":
        return cmd_brief(
            config=config,
            db_path=args.db,
            out=sys.stdout,
        )

    if args.command == "wiki":
        return cmd_wiki(
            config=config,
            db_path=args.db,
            out=sys.stdout,
        )

    if args.command == "serve":
        return cmd_serve(
            config=config,
            db_path=args.db,
            out=sys.stdout,
            host=args.host,
            port=args.port,
            config_path=args.config,
            sync_every=args.sync_every,
            demo=args.demo,
        )

    if args.command == "upload":
        return cmd_upload(
            path=args.file,
            config=config,
            db_path=args.db,
            csv_format=args.csv_format,
            tz=args.tz,
            out=sys.stdout,
        )

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
