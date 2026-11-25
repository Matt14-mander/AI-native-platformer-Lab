"""Freeze a geometry-deduplicated development/holdout benchmark manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_platformer.benchmark.generalization import freeze_plan, load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("config/generalization_v2_plan.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("game_content/benchmark_suites/generalization_v2.json")
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    reference_path = Path(plan["reference_suite"])
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    expected = freeze_plan(plan, reference)
    if args.verify:
        if load_manifest(args.output) != expected:
            parser.error("frozen manifest differs from plan or reference suite")
    else:
        if args.output.exists():
            parser.error("output exists; use --verify or a new path")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    counts = {split: sum(row["split"] == split for row in expected["entries"])
              for split in ("development", "holdout")}
    print(json.dumps({"suite": expected["suite_id"], "counts": counts,
                      "reference_unique_geometries": expected["reference_unique_geometries"]}))


if __name__ == "__main__":
    main()
