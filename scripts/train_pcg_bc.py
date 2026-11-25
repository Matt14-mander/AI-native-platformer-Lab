"""Train a behavior-cloning student from observation-only generated-map demonstrations."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from ai_platformer.agents.ppo.pcg_experiment import (
    build_model, collect_teacher_data, evaluate_model, factory_for_request,
    file_hash, fit_actor, training_requests,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=Path("config/generalization_v2_plan.json"))
    parser.add_argument("--manifest", type=Path, default=Path("game_content/benchmark_suites/generalization_v2.json"))
    parser.add_argument("--train-seeds", type=int, nargs="+", default=list(range(20261200, 20261208)))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new run directory")
    requests = training_requests(args.plan, args.train_seeds, args.manifest)
    args.output.mkdir(parents=True)
    observations, actions, maps = collect_teacher_data(requests, policy_seed=args.seed)
    np.savez_compressed(args.output / "demonstrations.npz", observations=observations, actions=actions)
    with tempfile.TemporaryDirectory(prefix="platformer-bc-") as directory:
        factory, _ = factory_for_request(requests[0], Path(directory) / "level.json")
        model = build_model(factory, seed=args.seed)
        try:
            history = fit_actor(model, observations, actions, epochs=args.epochs,
                                batch_size=256, learning_rate=1e-3, seed=args.seed)
            model.save(args.output / "model")
            development = evaluate_model(model, args.manifest)
        finally:
            model.get_env().close()
    report = {
        "schema_version": 1, "method": "behavior_cloning", "seed": args.seed,
        "train_seeds": args.train_seeds, "plan_sha256": file_hash(args.plan),
        "manifest_sha256": file_hash(args.manifest), "samples": len(actions),
        "teacher_maps": maps, "history": history, "development": development,
        "model_sha256": file_hash(args.output / "model.zip"),
        "demonstrations_sha256": file_hash(args.output / "demonstrations.npz"),
        "scope": "Training on generated train seeds only; frozen development evaluated; holdout unopened.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: development[key] for key in ("maps", "successes", "targets_met", "mean_coin_ratio")}, indent=2))


if __name__ == "__main__":
    main()
