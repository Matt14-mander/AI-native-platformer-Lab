"""Aggregate teacher labels on BC learner states, then refit the actor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from ai_platformer.agents.ppo.pcg_experiment import (
    collect_dagger_data, evaluate_model, file_hash, fit_actor, training_requests,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=Path("config/generalization_v2_plan.json"))
    parser.add_argument("--manifest", type=Path, default=Path("game_content/benchmark_suites/generalization_v2.json"))
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--epochs-per-round", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new run directory")
    if args.rounds < 1:
        parser.error("rounds must be positive")
    base = json.loads((args.bc_run / "report.json").read_text(encoding="utf-8"))
    if base["method"] != "behavior_cloning" or base["model_sha256"] != file_hash(args.bc_run / "model.zip"):
        parser.error("BC source model or report is invalid")
    if base["demonstrations_sha256"] != file_hash(args.bc_run / "demonstrations.npz"):
        parser.error("BC demonstrations changed")
    if base["manifest_sha256"] != file_hash(args.manifest) or base["plan_sha256"] != file_hash(args.plan):
        parser.error("source and target map protocol differ")
    requests = training_requests(args.plan, base["train_seeds"], args.manifest)
    with np.load(args.bc_run / "demonstrations.npz") as data:
        observations, actions = data["observations"], data["actions"]
    torch.set_num_threads(1)
    model = PPO.load(args.bc_run / "model.zip", device="cpu")
    args.output.mkdir(parents=True)
    baseline = evaluate_model(model, args.manifest)
    best = (baseline["targets_met"], baseline["successes"], baseline["mean_coin_ratio"])
    selected_round = 0
    model.save(args.output / "model")
    rounds = []
    for index in range(args.rounds):
        new_obs, new_labels, maps = collect_dagger_data(model, requests, policy_seed=base["seed"])
        observations = np.concatenate((observations, new_obs))
        actions = np.concatenate((actions, new_labels))
        history = fit_actor(model, observations, actions, epochs=args.epochs_per_round,
                            batch_size=256, learning_rate=args.learning_rate, seed=base["seed"] + index + 1)
        development = evaluate_model(model, args.manifest)
        score = (development["targets_met"], development["successes"], development["mean_coin_ratio"])
        if score > best:
            best = score
            selected_round = index + 1
            model.save(args.output / "model")
        rounds.append({"round": index + 1, "new_samples": len(new_labels),
                       "total_samples": len(actions), "learner_maps": maps,
                       "history": history, "development": development})
    np.savez_compressed(args.output / "aggregated.npz", observations=observations, actions=actions)
    report = {"schema_version": 1, "method": "dagger", "source_bc_sha256": base["model_sha256"],
              "source_dataset_sha256": base["demonstrations_sha256"], "seed": base["seed"],
              "train_seeds": base["train_seeds"], "plan_sha256": file_hash(args.plan),
              "manifest_sha256": file_hash(args.manifest), "rounds": rounds,
              "baseline_development": baseline, "selected_round": selected_round,
              "selection_rule": "maximize (targets_met, successes, mean_coin_ratio) on development",
              "epochs_per_round": args.epochs_per_round, "learning_rate": args.learning_rate,
              "model_sha256": file_hash(args.output / "model.zip"),
              "aggregated_sha256": file_hash(args.output / "aggregated.npz"),
              "scope": "DAgger labels learner states on training maps; development only for selection; holdout unopened."}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected_round": selected_round,
                      "baseline_targets": baseline["targets_met"],
                      "last_round_targets": rounds[-1]["development"]["targets_met"]}, indent=2))


if __name__ == "__main__":
    main()
