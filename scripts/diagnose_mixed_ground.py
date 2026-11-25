"""Reconstruct mixed_ground maps and inspect scripted routes and missed coins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_platformer.benchmark.coin_diagnostics import diagnose_mixed_ground


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", type=Path, default=Path("game_content/generation_suites/small_v1.json")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--route-seconds", type=float, default=15.0)
    parser.add_argument("--route-expansions", type=int, default=25_000)
    args = parser.parse_args()
    report = diagnose_mixed_ground(
        args.suite, args.output_dir,
        route_seconds=args.route_seconds, route_expansions=args.route_expansions,
    )
    print(json.dumps(
        [{"level_id": row["level_id"], "route": row["route"],
          "agents": {name: {"outcome": result["outcome"],
                            "coins_collected": result["coins_collected"],
                            "missed_ids": result["missed_ids"]}
                     for name, result in row["agents"].items()}}
         for row in report["maps"]], indent=2,
    ))


if __name__ == "__main__":
    main()
