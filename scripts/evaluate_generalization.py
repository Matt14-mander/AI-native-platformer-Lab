"""Evaluate a frozen map split with scripted baselines and optionally PPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_platformer.benchmark.stratified import evaluate_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("game_content/benchmark_suites/generalization_v2.json"),
    )
    parser.add_argument("--split", choices=("development", "holdout"), default="development")
    parser.add_argument("--open-holdout", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--policy-seed", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.split == "holdout" and not args.open_holdout:
        parser.error("holdout evaluation requires --open-holdout; results become exposed")
    if args.output.exists():
        parser.error("output exists; preserve earlier results")
    report = evaluate_manifest(
        args.manifest, split=args.split, policy_seed=args.policy_seed,
        model_path=args.model,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["by_agent"], indent=2))


if __name__ == "__main__":
    main()
