"""Frozen checkpoint transfer to v5 composition cells; never evaluate test/OOD."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from stable_baselines3 import PPO

from ai_platformer.agents.ppo.diagnostics import diagnose_policy
from ai_platformer.agents.ppo.training import checkpoint_metadata, evaluate_policy, write_json
from ai_platformer.agents.scripted import MoveRightAgent, RuleJumpAgent
from ai_platformer.benchmark.scripted import evaluate_scripted_agent
from ai_platformer.content.curriculum_v5 import PREFIX, composition_cell
from ai_platformer.envs.factory import EnvironmentFactory, protocol_hash


def stratify(evaluation: dict, specs: dict) -> dict:
    """Aggregate cells and each geometric axis without weighting by episode length."""
    groups = {}
    for episode in evaluation["episodes"]:
        height, width, recovery = composition_cell(specs[episode["level_id"]])
        keys = (
            f"cell/{height}/{width}/{recovery}",
            f"height/{height}",
            f"gap_width/{width}",
            f"recovery/{recovery}",
        )
        for key in keys:
            group = groups.setdefault(
                key, {"episodes": 0, "successes": 0, "deaths": 0, "timeouts": 0, "levels": set()}
            )
            group["episodes"] += 1
            group["successes"] += int(episode["outcome"] == "success")
            group["deaths"] += int(episode["outcome"] == "death")
            group["timeouts"] += int(episode["outcome"] == "time_limit")
            group["levels"].add(episode["level_id"])
    for group in groups.values():
        group["unique_layouts"] = len(group.pop("levels"))
        group["success_rate"] = group["successes"] / group["episodes"]
    return dict(sorted(groups.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", default="config/curriculum_v5.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sampling-seeds", type=int, nargs="+", default=[200, 201, 202])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; select a new report path")
    if (
        not args.sampling_seeds
        or len(set(args.sampling_seeds)) != len(args.sampling_seeds)
        or any(seed < 0 for seed in args.sampling_seeds)
    ):
        parser.error("sampling seeds must be distinct and nonnegative")
    torch.set_num_threads(1)
    path = args.model.with_suffix(".zip")
    metadata = checkpoint_metadata(path)
    source = EnvironmentFactory(metadata["config"]["environment"])
    previous = metadata["signature"]["protocol"]
    if protocol_hash(source.protocol(list(previous["levels"]))) != protocol_hash(previous):
        parser.error("source checkpoint protocol/content has changed")
    config = {**metadata["config"]["environment"], "curriculum_manifest": args.manifest}
    factory = EnvironmentFactory(config)
    if factory.repository.manifest["generator_version"] != 5:
        parser.error("requires the v5 composition manifest")
    model = PPO.load(path, device="cpu")
    if model.num_timesteps != metadata["saved_timesteps"]:
        parser.error("checkpoint timestep differs from sidecar")
    cohorts = {
        "legacy_validation": [
            x for x in factory.repository.split("mixed", "validation") if not x.startswith(PREFIX)
        ],
        "new_train_probe": [
            x for x in factory.repository.split("mixed", "train") if x.startswith(PREFIX)
        ],
        "new_composition_validation": [
            x for x in factory.repository.split("mixed", "validation") if x.startswith(PREFIX)
        ],
    }
    if any(not levels for levels in cohorts.values()):
        parser.error("missing legacy or new composition cohorts")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "model": str(path.resolve()),
        "model_sha256": metadata["model_sha256"],
        "saved_timesteps": metadata["saved_timesteps"],
        "source_signature_hash": metadata["signature_hash"],
        "source_protocol_verified": True,
        "target_environment": config,
        "target_protocol": factory.protocol([x for levels in cohorts.values() for x in levels]),
        "deterministic_seed": 100,
        "sampling_seeds": args.sampling_seeds,
        "interpretation": "Frozen policy; no training, tuning or checkpoint selection. "
        "For a v4-trained model this is zero-shot transfer from isolated skills, not "
        "post-v5-training generalization. Sampling seeds are not independent training runs. "
        "Test/OOD are excluded. A source trained on these cells requires a new holdout design.",
        "cohorts": {},
    }
    specs = factory.repository.manifest["levels"]
    for name, levels in cohorts.items():
        evaluation = evaluate_policy(
            model,
            seeds=[100],
            environment=config,
            level_ids=levels,
            trace_path=args.output.with_name(f"{args.output.stem}.{name}.failure.json"),
        )
        controls = {
            agent: evaluate_scripted_agent(
                agent, agent_type, [100], environment=config, level_ids=levels
            ).to_dict()
            for agent, agent_type in (("rule-jump", RuleJumpAgent), ("move-right", MoveRightAgent))
        }
        report["cohorts"][name] = {
            "level_ids": levels,
            "evaluation": evaluation,
            "strata": stratify(evaluation, specs),
            "controls": controls,
        }
        write_json(args.output, report)
        print(name, evaluation["summary"], flush=True)
    levels = cohorts["new_composition_validation"]
    sampled = diagnose_policy(
        model, factory=factory, level_ids=levels, seeds=args.sampling_seeds, modes=("stochastic",)
    )
    report["sampling_diagnostics"] = sampled
    report["sampling_strata"] = stratify(sampled["modes"]["stochastic"], specs)
    write_json(args.output, report)
    print("sampled", sampled["modes"]["stochastic"]["summary"], flush=True)


if __name__ == "__main__":
    main()
