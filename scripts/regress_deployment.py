"""Check source PPO and both TinyInfer paths on complete deterministic episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ai_platformer.agents.ppo.training import checkpoint_metadata, write_json
from ai_platformer.deployment.backends import load_backend
from ai_platformer.deployment.package import file_hash, load_actor_metadata


def regression_levels(factory) -> list[str]:
    """Validation representatives plus all full reserved layouts (already exposed)."""
    levels = [factory.level_id]
    for task in ("flat", "obstacle", "gap", "mixed"):
        ids = factory.repository.split(task, "validation")
        levels.extend(ids[index] for index in sorted({0, len(ids) // 2, len(ids) - 1}))
    full = factory.repository.manifest["splits"].get("full", {})
    for suite in ("validation", "test", "ood"):
        levels.extend(full.get(suite, []))
    return list(dict.fromkeys(levels))


def regress(checkpoint: Path, actor: Path, library: Path, *, seeds: list[int]) -> dict:
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("distinct nonnegative regression seeds required")
    checkpoint = checkpoint.with_suffix(".zip")
    meta = load_actor_metadata(actor)
    source = checkpoint_metadata(checkpoint)
    if source["model_sha256"] != meta["source"]["model_sha256"]:
        raise ValueError("checkpoint does not match actor export source")
    torch.set_num_threads(1)
    backends = {}
    try:
        backends["sb3"] = load_backend("sb3", checkpoint)
        backends["original"] = load_backend("tinyinfer", actor, library=library)
        backends["fused_relu"] = load_backend("tinyinfer", actor, library=library, fuse_relu=True)
        levels = regression_levels(backends["sb3"].factory)
        report = {
            "schema_version": 1,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": file_hash(checkpoint),
            "actor_sha256": file_hash(actor),
            "metadata_sha256": file_hash(actor.with_suffix(".json")),
            "library_sha256": file_hash(library),
            "levels": levels,
            "seeds": seeds,
            "backends": list(backends),
            "episodes": [],
            "mismatch": None,
            "note": "Deployment parity regression on already exposed maps, not fresh policy generalization. "
            "Every step compares action, observation, reward, termination and full WorldSnapshot.",
        }
        for level in levels:
            for seed in seeds:
                envs, observations = {}, {}
                try:
                    for name, backend in backends.items():
                        env = backend.factory.make(level_id=level)
                        envs[name] = env
                        observations[name], _ = env.reset(seed=seed)
                    trace = hashlib.sha256()
                    steps = 0
                    total = 0.0
                    while True:
                        values = {}
                        for name, backend in backends.items():
                            action = int(
                                np.asarray(
                                    backend.policy.predict(observations[name], deterministic=True)[
                                        0
                                    ]
                                ).item()
                            )
                            obs, reward, terminated, truncated, info = envs[name].step(action)
                            observations[name] = obs
                            values[name] = {
                                "action": action,
                                "observation": obs.tolist(),
                                "reward": reward,
                                "terminated": terminated,
                                "truncated": truncated,
                                "state": asdict(envs[name].core.state),
                                "info": info,
                            }
                        reference = values["sb3"]
                        if any(value != reference for value in values.values()):
                            report["mismatch"] = {
                                "level": level,
                                "seed": seed,
                                "step": steps,
                                "values": values,
                            }
                            report["passed"] = False
                            return report
                        trace.update(json.dumps(reference, sort_keys=True).encode())
                        steps += 1
                        total += reference["reward"]
                        if reference["terminated"] or reference["truncated"]:
                            break
                    info = reference["info"]
                    report["episodes"].append(
                        {
                            "level_id": level,
                            "seed": seed,
                            "steps": steps,
                            "outcome": info["outcome"],
                            "ticks": info["tick"],
                            "coins_collected": info["coins_collected"],
                            "coins_total": info["coins_total"],
                            "return": total,
                            "trace_sha256": trace.hexdigest(),
                        }
                    )
                finally:
                    for env in envs.values():
                        env.close()
        report["unchanged"] = (
            file_hash(checkpoint) == report["checkpoint_sha256"]
            and file_hash(actor) == report["actor_sha256"]
            and file_hash(actor.with_suffix(".json")) == report["metadata_sha256"]
            and file_hash(library) == report["library_sha256"]
        )
        report["passed"] = report["unchanged"] and all(
            ep["outcome"] == "success" for ep in report["episodes"]
        )
        return report
    finally:
        for backend in backends.values():
            backend.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="exported actor.onnx")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[100])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise FileExistsError("regression report exists; choose a new path")
        report = regress(args.checkpoint, args.model, args.library, seeds=args.seeds)
        write_json(args.output, report)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f"deployment regression failed: {error}\n")
    print(f"parity: {report['passed']}; {len(report['episodes'])} episodes across three backends")
    if not report["passed"]:
        parser.exit(1, "deployment trajectory regression failed; inspect report\n")


if __name__ == "__main__":
    main()
