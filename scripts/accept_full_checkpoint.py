"""Reload a completed full curriculum, then assess full playback and map regression."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
from stable_baselines3 import PPO

from ai_platformer.agents.ppo.diagnostics import diagnose_policy
from ai_platformer.agents.ppo.gates import assess_prerequisites
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
    stages = config["curriculum"]["stages"]
    state = meta["curriculum_state"]
    history = state.get("history", [])
    rounds = history[-1].get("validation_timesteps", []) if history else []
    if (
        state["status"] != "completed"
        or state["stage_index"] != len(stages) - 1
        or stages[-1]["task"] != "full"
        or rounds != sorted(set(rounds))
        or not rounds
        or rounds[-1] != meta["saved_timesteps"]
        or not history
        or history[-1].get("task") != "full"
        or history[-1].get("outcome") != "mastered"
        or history[-1].get("end") != meta["saved_timesteps"]
        or len(history[-1].get("validation_timesteps", []))
        != config["curriculum"]["required_evaluations"]
    ):
        parser.error("requires final automatic full completion checkpoint")
    model = PPO.load(path, device="cpu")
    validation = evaluate_policy(
        model,
        seeds=config["evaluation"]["seeds"],
        environment=config["environment"],
        level_ids=[level for stage in stages for level in stage["validation_levels"]],
    )
    verification = {"evaluation": validation, "gate": assess_prerequisites(stages, validation)}
    if not verification["gate"]["passed"]:
        parser.error("reloaded full checkpoint fails validation")
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
            stage["task"]: {
                key: stage[key]
                for key in (
                    "success_threshold",
                    "max_success_steps",
                    "min_coin_ratio",
                    "coin_ratio_thresholds",
                )
                if key in stage
            }
            for stage in stages
        },
        "reserved_protocols": {
            suite: factory.protocol([level for ids in tasks.values() for level in ids])
            for suite, tasks in pools.items()
        },
        "policy_modes": ["deterministic", "stochastic"],
        "selection": "Final automatic full completion policy, frozen before reserved rolls.",
        "interpretation": "Known level_1 and previously exposed mixed pools are regression. "
        "Generated full test/OOD maps are fresh only on first frozen cohort evaluation. "
        "Environment seeds do not add geometry. Report each training seed separately.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "validation_reload.json", verification)
    write_json(args.output_dir / "frozen_plan.json", plan)
    model = PPO.load(path, device="cpu")
    if model.num_timesteps != plan["saved_timesteps"]:
        parser.error("checkpoint timestep differs from sidecar")
    full_levels = stages[-1]["validation_levels"]
    full = diagnose_policy(
        model,
        factory=factory,
        level_ids=full_levels,
        seeds=args.seeds,
        modes=("deterministic", "stochastic"),
    )
    write_json(args.output_dir / "full_modes.json", full)
    report = {
        "schema_version": 1,
        "plan": plan,
        "suites": {},
        "full": {
            mode: {
                "summary": result["summary"],
                "by_level": result["by_level"],
                "gate": assess_prerequisites([stages[-1]], result),
            }
            for mode, result in full["modes"].items()
        },
    }
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
                "full_cohorts": {
                    prefix: {
                        "summary": subset(evaluation, ids)["summary"],
                        "gate": acceptance_gate(stages, {"full": ids}, subset(evaluation, ids)),
                    }
                    for prefix in sorted(
                        {
                            level.split("_full_")[0] + "_full_"
                            for level in tasks.get("full", [])
                            if "_full_" in level
                        }
                    )
                    if (
                        ids := [
                            level for level in tasks.get("full", []) if level.startswith(prefix)
                        ]
                    )
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
    report["passed"] = all(x["gate"]["passed"] for x in report["full"].values()) and all(
        result["gate"]["passed"] for modes in report["suites"].values() for result in modes.values()
    )
    write_json(args.output_dir / "report.json", report)
    print("acceptance:", report["passed"], flush=True)
    if not report["passed"]:
        parser.exit(1, "reserved test/OOD acceptance failed; inspect the report\n")


if __name__ == "__main__":
    main()
