"""Compare uniform and failure-prioritized map sampling with fixed PPO budgets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO

from ai_platformer.agents.ppo.pcg_experiment import evaluate_model, file_hash, training_requests
from ai_platformer.agents.ppo.pcg_training import initialize_actor, make_ppo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=Path("config/generalization_v2_plan.json"))
    parser.add_argument("--manifest", type=Path, default=Path("game_content/benchmark_suites/generalization_v2.json"))
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[17, 23, 31])
    parser.add_argument("--timesteps", type=int, default=16384)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new run directory")
    if args.timesteps < 512 or args.timesteps % 512 or len(set(args.training_seeds)) != len(args.training_seeds):
        parser.error("timesteps must be a multiple of 512 and training seeds unique")
    base = json.loads((args.bc_run / "report.json").read_text(encoding="utf-8"))
    if base["method"] != "behavior_cloning" or base["model_sha256"] != file_hash(args.bc_run / "model.zip"):
        parser.error("BC source model or report is invalid")
    if base["plan_sha256"] != file_hash(args.plan) or base["manifest_sha256"] != file_hash(args.manifest):
        parser.error("map protocol differs from BC source")
    requests = training_requests(args.plan, base["train_seeds"], args.manifest)
    bc = PPO.load(args.bc_run / "model.zip", device="cpu")
    args.output.mkdir(parents=True)
    rows = []
    for seed in args.training_seeds:
        for mode in ("uniform", "failure_prioritized"):
            model = make_ppo(requests, seed=seed, adaptive=mode == "failure_prioritized")
            try:
                initialize_actor(model, bc)
                model.learn(total_timesteps=args.timesteps)
                stem = args.output / f"{mode}_seed_{seed}"
                model.save(stem)
                development = evaluate_model(model, args.manifest)
                workers = model.get_env().envs
                counts = {request.level_id: sum(int(worker.sample_counts[i]) for worker in workers)
                          for i, request in enumerate(requests)}
                rows.append({"mode": mode, "training_seed": seed,
                             "timesteps": model.num_timesteps,
                             "model_sha256": file_hash(stem.with_suffix(".zip")),
                             "sample_counts": counts, "development": development})
                report = {"schema_version": 1, "method": "generated_map_sampling_comparison",
                          "source_bc_sha256": base["model_sha256"],
                          "train_map_seeds": base["train_seeds"], "training_seeds": args.training_seeds,
                          "requested_timesteps_per_run": args.timesteps,
                          "adaptive_rule": "EWMA of failure or coin deficit; 50% uniform mixture; training episodes only",
                          "plan_sha256": file_hash(args.plan), "manifest_sha256": file_hash(args.manifest),
                          "runs": rows, "scope": "Development selection only; holdout unopened."}
                (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                print(f"{mode} seed={seed}: targets={development['targets_met']}/12", flush=True)
            finally:
                model.get_env().close()


if __name__ == "__main__":
    main()
