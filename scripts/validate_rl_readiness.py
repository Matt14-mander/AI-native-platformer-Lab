"""Run SB3 compatibility, random stability, and reward exploit gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_platformer.benchmark import readiness_report
from ai_platformer.content.curriculum import TrainingLevelRepository


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, help="override configured episode count")
    parser.add_argument("--max-steps", type=int, help="override per-episode step limit")
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    parser.add_argument("--manifest", help="course manifest used by all readiness checks")
    parser.add_argument(
        "--courses", action="store_true", help="also check all train/validation courses"
    )
    parser.add_argument(
        "--environment-id",
        choices=("PlatformerState-v0", "PlatformerState-v1"),
        default="PlatformerState-v0",
    )
    args = parser.parse_args()
    if args.episodes is not None and args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")

    root = Path(__file__).resolve().parents[1]
    with (root / "config" / "rl_readiness_v0.json").open(encoding="utf-8") as stream:
        config = json.load(stream)
    levels = None
    if args.courses:
        repo = TrainingLevelRepository(args.manifest)
        levels = [
            level
            for splits in repo.manifest["splits"].values()
            for suite in ("train", "validation")
            for level in splits[suite]
        ]
    report = readiness_report(
        episodes=args.episodes or int(config["episodes"]),
        max_steps_per_episode=args.max_steps or int(config["max_steps_per_episode"]),
        base_seed=int(config["base_seed"]),
        level_ids=levels,
        environment={
            "environment_id": args.environment_id,
            **({"curriculum_manifest": args.manifest} if args.manifest else {}),
        },
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
