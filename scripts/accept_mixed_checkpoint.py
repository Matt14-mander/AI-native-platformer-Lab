"""Freeze an accepted mixed checkpoint, then evaluate reserved test/OOD once."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
from stable_baselines3 import PPO

from ai_platformer.agents.ppo.diagnostics import diagnose_policy
from ai_platformer.agents.ppo.training import checkpoint_metadata, evaluate_policy, write_json
from ai_platformer.benchmark.acceptance import (
    acceptance_gate,
    reserved_pools,
    validate_reserved_seeds,
)
from ai_platformer.benchmark.scripted import BenchmarkResult, EpisodeResult
from ai_platformer.content.curriculum_v5 import PREFIX
from ai_platformer.envs.factory import EnvironmentFactory, protocol_hash
from scripts.evaluate_mixed_generalization import stratify
from scripts.verify_stage_entry import verify_stage_entry


def subset(evaluation, levels):
    return BenchmarkResult(
        evaluation["summary"]["agent"],
        tuple(
            EpisodeResult(**episode)
            for episode in evaluation["episodes"]
            if episode["level_id"] in levels
        ),
    ).to_dict()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002, 1003])
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory is nonempty; acceptance reports cannot be overwritten")
    torch.set_num_threads(1)
    path = args.model.with_suffix(".zip")
    meta = checkpoint_metadata(path)
    config = meta["config"]
    validate_reserved_seeds(args.seeds, config)
    factory = EnvironmentFactory(config["environment"])
    protocol = meta["signature"]["protocol"]
    if protocol_hash(factory.protocol(list(protocol["levels"]))) != protocol_hash(protocol):
        parser.error("checkpoint content/protocol changed")
    verification = verify_stage_entry(path, "full")
    if not verification["accepted"]:
        parser.error("checkpoint has not passed automatic mixed promotion plus reload gate")
    stages = config["curriculum"]["stages"][: meta["curriculum_state"]["stage_index"]]
    if stages[-1]["task"] != "mixed":
        parser.error("requires the checkpoint stopped immediately after mixed")
    pools = reserved_pools(factory.repository, stages)
    plan = {
        "schema_version": 1,
        "model": str(path.resolve()),
        "model_sha256": meta["model_sha256"],
        "sidecar_sha256": hashlib.sha256(path.with_suffix(".json").read_bytes()).hexdigest(),
        "signature_hash": meta["signature_hash"],
        "saved_timesteps": meta["saved_timesteps"],
        "seeds": args.seeds,
        "pool_ids": pools,
        "criteria": {
            stage["task"]: {key: stage[key] for key in ("success_threshold", "max_success_steps")}
            for stage in stages
        },
        "reserved_protocols": {
            suite: factory.protocol([level for ids in tasks.values() for level in ids])
            for suite, tasks in pools.items()
        },
        "policy_modes": ["deterministic", "stochastic"],
        "selection": "Final automatic mixed promotion checkpoint; fixed before holdout rolls. "
        "No selection/retraining based on these outcomes. Each mode must pass every layout's "
        "success-rate and successful-step criteria. OOD uses the same declared criteria.",
        "interpretation": "Environment seeds do not add geometry; stochastic seeds sample "
        "actions. This is one training lineage, not a multi-training-seed claim. Full level "
        "is excluded because it is not unseen geometry. No flat/obstacle OOD pools exist.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "validation_reload.json", verification)
    write_json(args.output_dir / "frozen_plan.json", plan)
    model = PPO.load(path, device="cpu")
    if model.num_timesteps != plan["saved_timesteps"]:
        parser.error("checkpoint timestep differs from sidecar")
    report = {"schema_version": 1, "plan": plan, "suites": {}}
    for suite, tasks in pools.items():
        levels = [level for ids in tasks.values() for level in ids]
        deterministic = evaluate_policy(
            model,
            seeds=args.seeds,
            environment=config["environment"],
            level_ids=levels,
            trace_path=args.output_dir / f"{suite}.failure.json",
        )
        modes = {"deterministic": deterministic}
        gate = acceptance_gate(stages, tasks, deterministic)
        write_json(
            args.output_dir / f"{suite}.deterministic.json",
            {"evaluation": deterministic, "gate": gate},
        )
        diagnostic = diagnose_policy(
            model, factory=factory, level_ids=levels, seeds=args.seeds, modes=("stochastic",)
        )
        sampled = diagnostic["modes"]["stochastic"]
        modes["stochastic"] = sampled
        write_json(args.output_dir / f"{suite}.stochastic.json", diagnostic)
        results = {}
        for mode, evaluation in modes.items():
            gate = acceptance_gate(stages, tasks, evaluation)
            mixed = tasks.get("mixed", [])
            cohorts = {
                "legacy_mixed": [x for x in mixed if not x.startswith((PREFIX, "v6_mixed_"))],
                "v5_mixed": [x for x in mixed if x.startswith(PREFIX)],
                "v6_mixed": [x for x in mixed if x.startswith("v6_mixed_")],
            }
            results[mode] = {
                "summary": evaluation["summary"],
                "gate": gate,
                "tasks": {task: subset(evaluation, ids)["summary"] for task, ids in tasks.items()},
                "mixed_cohorts": {
                    name: subset(evaluation, ids)["summary"] for name, ids in cohorts.items() if ids
                },
                "mixed_strata": stratify(
                    subset(evaluation, mixed), factory.repository.manifest["levels"]
                )
                if mixed
                else {},
            }
            print(suite, mode, "passed", gate["passed"], evaluation["summary"], flush=True)
        report["suites"][suite] = results
        write_json(args.output_dir / "report.json", report)
    if (
        hashlib.sha256(path.read_bytes()).hexdigest() != plan["model_sha256"]
        or hashlib.sha256(path.with_suffix(".json").read_bytes()).hexdigest()
        != plan["sidecar_sha256"]
    ):
        parser.error("checkpoint changed during acceptance; results are invalid")
    report["checkpoint_unchanged"] = True
    report["passed"] = all(
        result["gate"]["passed"] for modes in report["suites"].values() for result in modes.values()
    )
    write_json(args.output_dir / "report.json", report)
    print("acceptance:", report["passed"], flush=True)
    if not report["passed"]:
        parser.exit(1, "reserved test/OOD acceptance failed; inspect the report\n")


if __name__ == "__main__":
    main()
