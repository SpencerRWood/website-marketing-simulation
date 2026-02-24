from __future__ import annotations

import argparse
import sys

from sim.app.runner import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="marketing-sim")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run the simulation")
    p_run.add_argument("--config", default="config/simulation.yaml")

    args = parser.parse_args(argv)

    if args.cmd == "run":
        result = run(args.config)
        # minimal stdout signal
        print(f"run_id={result.ctx.run_id} duckdb={result.duckdb_path}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
