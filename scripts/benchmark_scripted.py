"""Run fixed-seed scripted baselines for PlatformerState-v0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_platformer.agents.scripted import SCRIPTED_AGENTS
from ai_platformer.benchmark import evaluate_scripted_agent
from ai_platformer.envs.factory import EnvironmentFactory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", choices=("train", "validation", "unseen", "test"), default="validation"
    )
    parser.add_argument("--config", type=Path, default=Path("config/benchmark_v0.json"))
    parser.add_argument(
        "--task", choices=("flat", "obstacle", "gap", "mixed", "full"), default="full"
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        choices=tuple(SCRIPTED_AGENTS),
        default=list(SCRIPTED_AGENTS),
    )
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="optional benchmark-only episode cap",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    path = args.config if args.config.is_absolute() else root / args.config
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    suite = "unseen" if args.suite == "test" else args.suite
    seeds = config["suites"][suite]
    environment = dict(config.get("environment", {}))
    environment["environment_id"] = config["environment_id"]
    environment.setdefault("action_repeat", int(config["action_repeat"]))
    environment.setdefault("episode_step_limit", 1024)
    if args.max_steps is not None:
        environment["episode_step_limit"] = args.max_steps
    factory = EnvironmentFactory(environment)
    levels = (
        [factory.level_id]
        if args.task == "full"
        else factory.repository.split(args.task, "test" if suite == "unseen" else suite)
    )
    results = [
        evaluate_scripted_agent(
            name,
            SCRIPTED_AGENTS[name],
            seeds,
            action_repeat=int(config["action_repeat"]),
            environment=environment,
            level_ids=levels,
        ).to_dict()
        for name in args.agents
    ]
    report = {
        "schema_version": 1,
        "environment_id": config["environment_id"],
        "suite": args.suite,
        "seeds": seeds,
        "level_ids": levels,
        "protocol": factory.protocol(levels),
        "results": results,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
