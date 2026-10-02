"""Evaluate a versioned checkpoint without training or selecting on final test data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO

from ai_platformer.agents.ppo.training import checkpoint_metadata, evaluate_policy
from ai_platformer.envs.factory import EnvironmentFactory, protocol_hash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--task", choices=("flat", "obstacle", "gap", "mixed", "full"), required=True
    )
    parser.add_argument("--suite", choices=("validation", "test"), default="validation")
    parser.add_argument("--seeds", type=int, nargs="+", help="explicit evaluation seeds")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = args.model.with_suffix(".zip")
    metadata = checkpoint_metadata(path)
    config = metadata["config"]
    factory = EnvironmentFactory(config["environment"])
    previous = metadata["signature"]["protocol"]
    if protocol_hash(factory.protocol(list(previous["levels"]))) != protocol_hash(previous):
        parser.error("checkpoint content/protocol has changed; restore its manifest/levels")
    if args.output.exists():
        parser.error("output already exists; choose a new report path")
    seeds = args.seeds or (
        config["evaluation"]["seeds"] if args.suite == "validation" else [1000, 1001, 1002, 1003]
    )
    if not seeds or any(seed < 0 for seed in seeds):
        parser.error("evaluation seeds must be nonnegative")
    if args.suite == "test" and set(seeds) & set(
        config["train_seeds"] + config["evaluation"]["seeds"]
    ):
        parser.error("test seeds overlap training or validation")
    levels = (
        [factory.level_id]
        if args.task == "full"
        else factory.repository.split(args.task, args.suite)
    )
    model = PPO.load(path, device="cpu")
    report = {
        "model": str(path.resolve()),
        "task": args.task,
        "suite": args.suite,
        "note": "full level is a transfer/regression task, not unseen geometry"
        if args.task == "full"
        else None,
        "protocol": factory.protocol(levels),
        "evaluation": evaluate_policy(
            model,
            seeds=seeds,
            environment=config["environment"],
            level_ids=levels,
            trace_path=args.output.with_suffix(".failure.json"),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["evaluation"]["summary"], indent=2))


if __name__ == "__main__":
    main()
