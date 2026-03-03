from __future__ import annotations

import argparse
import json
import sys

from sim.app.runner import run
from sim.features.run_explorer.service import RunExplorerService


def _print_table(rows: list[dict], cols: list[str]) -> None:
    if not rows:
        print("(no rows)")
        return

    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    sep = "  ".join("-" * widths[c] for c in cols)
    print(header)
    print(sep)
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="marketing-sim")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- run ----
    p_run = sub.add_parser("run", help="Run the simulation")
    p_run.add_argument("--config", default="config/simulation.yaml")

    # ---- inspect ----
    p_ins = sub.add_parser("inspect", help="Inspect DuckDB outputs for a run_id")
    p_ins.add_argument("--db", default="data/sim.duckdb")
    p_ins.add_argument("--run-id", default=None)
    p_ins.add_argument(
        "--what",
        choices=["runs", "summary", "counts", "head", "head-json"],
        default="summary",
        help="Query to run",
    )
    p_ins.add_argument("--limit", type=int, default=50)

    args = parser.parse_args(argv)

    if args.cmd == "run":
        result = run(args.config)
        print(f"run_id={result.ctx.run_id} duckdb={result.duckdb_path}")
        return 0

    if args.cmd == "inspect":
        ex = RunExplorerService(duckdb_path=args.db)

        if args.what == "runs":
            runs = ex.list_run_ids(limit=args.limit)
            for r in runs:
                print(r)
            return 0

        if args.run_id is None:
            # sensible default: most recent run_id (by sort order)
            runs = ex.list_run_ids(limit=1)
            if not runs:
                print("No runs found in DuckDB.")
                return 1
            args.run_id = runs[0]

        if args.what == "summary":
            s = ex.summary(run_id=args.run_id)
            print(f"run_id: {s.run_id}")
            print(f"events: {s.events}")
            print(f"users: {s.users}")
            print(f"sessions: {s.sessions}")
            print(f"conversions: {s.conversions}")
            return 0

        if args.what == "counts":
            rows = ex.event_counts(run_id=args.run_id)
            _print_table(rows, cols=["event_type", "channel", "n"])
            return 0

        if args.what in ("head", "head-json"):
            rows = ex.head_events(run_id=args.run_id, limit=args.limit)
            if args.what == "head-json":
                print(json.dumps(rows, default=str, indent=2))
            else:
                # print a compact table with the most useful columns if present
                preferred = [
                    "ts_utc",
                    "sim_time_s",
                    "event_type",
                    "user_id",
                    "session_id",
                    "channel",
                    "page",
                    "intent_source",
                ]
                cols = [c for c in preferred if rows and c in rows[0]]
                cols = cols or list(rows[0].keys())
                _print_table(rows, cols=cols)
            return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
