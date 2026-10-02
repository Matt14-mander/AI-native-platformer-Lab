"""Run deterministic, sampled and fixed-observation policy diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from ai_platformer.agents.ppo.diagnostics import MODES, diagnose_policy
from ai_platformer.agents.ppo.training import checkpoint_metadata, write_json
from ai_platformer.envs.factory import EnvironmentFactory, protocol_hash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--task", choices=("flat", "obstacle", "gap", "mixed"), default="gap")
    parser.add_argument(
        "--suite", choices=("train", "validation", "test", "ood"), default="validation"
    )
    parser.add_argument(
        "--manifest", help="optional diagnostic content; source protocol is still verified"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(200, 210)))
    parser.add_argument("--modes", choices=MODES, nargs="+", default=list(MODES))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = args.model.with_suffix(".zip")
    try:
        metadata = checkpoint_metadata(path)
        config = metadata["config"]
        factory = EnvironmentFactory(config["environment"])
        previous = metadata["signature"]["protocol"]
        if protocol_hash(factory.protocol(list(previous["levels"]))) != protocol_hash(previous):
            raise ValueError("checkpoint content/protocol has changed; restore its manifest/levels")
        if args.output.exists():
            raise FileExistsError("output already exists; choose a new report path")
        if args.suite == "test" and set(args.seeds) & set(
            config["train_seeds"] + config["evaluation"]["seeds"]
        ):
            raise ValueError("test seeds overlap training or validation")
        if args.manifest:
            factory = EnvironmentFactory(
                {**config["environment"], "curriculum_manifest": args.manifest}
            )
        levels = factory.repository.split(args.task, args.suite)
        model = PPO.load(path, device="cpu")
        report = diagnose_policy(
            model, factory=factory, level_ids=levels, seeds=args.seeds, modes=tuple(args.modes)
        )
        report.update(
            model=str(path.resolve()),
            task=args.task,
            suite=args.suite,
            source_signature_hash=metadata["signature_hash"],
            source_model_sha256=metadata.get("model_sha256"),
            source_timesteps=metadata["saved_timesteps"],
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, report)
    except (ValueError, KeyError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))
    for mode, result in report["modes"].items():
        print(
            f"{mode}: success={result['summary']['success_rate']:.1%}, "
            f"jumps={result['behavior']['mean_effective_jumps']:.2f}, "
            f"landed_after_gap={result['behavior']['landed_after_gap_rate']:.1%}"
        )


if __name__ == "__main__":
    main()
