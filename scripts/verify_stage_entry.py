"""Reload a checkpoint and verify the recorded automatic stage-entry decision."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from ai_platformer.agents.ppo.gates import assess_prerequisites
from ai_platformer.agents.ppo.training import checkpoint_metadata, evaluate_policy, write_json
from ai_platformer.envs.factory import EnvironmentFactory, protocol_hash


def verify_stage_entry(path: Path, target: str) -> dict:
    path = path.with_suffix(".zip")
    metadata = checkpoint_metadata(path)
    config = metadata["config"]
    stages = config["curriculum"]["stages"]
    tasks = [stage["task"] for stage in stages]
    if target not in tasks or tasks.index(target) == 0:
        raise ValueError("target must be configured and have prerequisite stages")
    index = tasks.index(target)
    factory = EnvironmentFactory(config["environment"])
    previous = metadata["signature"]["protocol"]
    if protocol_hash(factory.protocol(list(previous["levels"]))) != protocol_hash(previous):
        raise ValueError("checkpoint environment/content protocol has changed")
    levels = [level for stage in stages[:index] for level in stage["validation_levels"]]
    evaluation = evaluate_policy(
        PPO.load(path, device="cpu"),
        seeds=config["evaluation"]["seeds"],
        environment=config["environment"],
        level_ids=levels,
    )
    gate = assess_prerequisites(stages[:index], evaluation)
    state = metadata["curriculum_state"]
    evidence = state.get("readiness", {})
    history = state.get("history", [])
    rounds = evidence.get("evaluation_timesteps", [])
    recorded = (
        state["status"] == "ready_for_stage"
        and state["stage_index"] == index
        and evidence.get("target") == target
        and evidence.get("timesteps") == metadata["saved_timesteps"]
        and evidence.get("required_evaluations") == config["curriculum"]["required_evaluations"]
        and evidence.get("prerequisites", {}).get("passed", False)
        and len(rounds) == config["curriculum"]["required_evaluations"]
        and rounds == sorted(set(rounds))
        and rounds[-1] == metadata["saved_timesteps"]
        and bool(history)
        and history[-1].get("outcome") == "mastered"
        and history[-1].get("task") == stages[index - 1]["task"]
        and history[-1].get("end") == metadata["saved_timesteps"]
    )
    return {
        "schema_version": 1,
        "target": target,
        "accepted": bool(recorded and gate["passed"]),
        "automatic_promotion_recorded": bool(recorded),
        "model": str(path.resolve()),
        "model_sha256": metadata["model_sha256"],
        "signature_hash": metadata["signature_hash"],
        "saved_timesteps": metadata["saved_timesteps"],
        "gate": gate,
        "recorded_evidence": evidence,
        "evaluation": evaluation,
        "note": "Exact saved policy reloaded; validation only. Repeated environment seeds are not independent training runs.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--target", choices=("obstacle", "gap", "mixed", "full"), default="mixed")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise FileExistsError("output already exists")
        report = verify_stage_entry(args.model, args.target)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, report)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))
    print(f"{args.target}: {'accepted' if report['accepted'] else 'rejected'}")
    for task, gate in report["gate"]["tasks"].items():
        print(f"{task}: {gate['qualified_layouts']}/{gate['layouts']} qualified")
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
