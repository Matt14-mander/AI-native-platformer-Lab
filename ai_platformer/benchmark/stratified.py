"""Stratified evaluation over a precommitted geometry manifest."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import defaultdict
from pathlib import Path

from ai_platformer.agents.scripted import MoveRightAgent, RuleJumpAgent
from ai_platformer.agents.scripted.coin_jump import CoinJumpAgent
from ai_platformer.agents.scripted.ground_coin_jump import GroundCoinJumpAgent
from ai_platformer.benchmark.coin_diagnostics import run_agent
from ai_platformer.benchmark.generalization import load_manifest
from ai_platformer.content.generation_request import GenerationRequest
from ai_platformer.content.level_spec import geometry_hash
from ai_platformer.content.pcg import generate_level
from ai_platformer.envs.factory import EnvironmentFactory


BASELINES = {
    "move_right": MoveRightAgent,
    "rule_jump": RuleJumpAgent,
    "coin_jump": CoinJumpAgent,
    "ground_coin_jump": GroundCoinJumpAgent,
}


def summarize(rows: list[dict]) -> dict:
    count = len(rows)
    if not count:
        raise ValueError("cannot summarize an empty group")
    return {
        "maps": count,
        "successes": sum(row["outcome"] == "success" for row in rows),
        "collection_targets_met": sum(row["collection_target_met"] for row in rows),
        "deaths": sum(row["outcome"] == "death" for row in rows),
        "timeouts": sum(row["outcome"] == "time_limit" for row in rows),
        "mean_progress": sum(row["progress"] for row in rows) / count,
        "mean_coin_ratio": sum(row["coin_ratio"] for row in rows) / count,
        "mean_steps": sum(row["steps"] for row in rows) / count,
        "mean_return": sum(row["return"] for row in rows) / count,
    }


def evaluate_manifest(
    manifest_path: Path,
    *,
    split: str = "development",
    policy_seed: int = 100,
    model_path: Path | None = None,
) -> dict:
    if split not in {"development", "holdout"}:
        raise ValueError("unknown split")
    manifest = load_manifest(manifest_path)
    selected = [entry for entry in manifest["entries"] if entry["split"] == split]
    if not selected:
        raise ValueError("split has no maps")
    loaded = None
    if model_path is not None:
        from ai_platformer.deployment.backends import load_backend

        loaded = load_backend("sb3", model_path)
        if loaded.factory.environment_id != "PlatformerState-v2":
            loaded.close()
            raise ValueError("this suite requires a PlatformerState-v2 checkpoint")
    entries = []
    try:
        with tempfile.TemporaryDirectory(prefix="platformer-generalization-") as temp:
            path = Path(temp) / "level.json"
            for entry in selected:
                request = GenerationRequest.model_validate(entry["request"])
                generated = generate_level(request)
                if geometry_hash(generated.level) != entry["geometry_sha256"]:
                    raise ValueError("frozen geometry changed")
                path.write_text(json.dumps(generated.level.model_dump()) + "\n")
                if loaded is None:
                    factory = EnvironmentFactory(
                        {
                            "environment_id": "PlatformerState-v2",
                            "level_spec": str(path),
                            "physics": generated.report["physics"],
                            "action_repeat": 4,
                            "episode_step_limit": 768,
                        }
                    )
                else:
                    factory = loaded.factory.for_level_spec(path)
                    from dataclasses import asdict

                    if asdict(factory.settings.physics) != generated.report["physics"]:
                        raise ValueError("checkpoint physics differ from generator profile")
                agents = {}
                for name, agent_type in BASELINES.items():
                    result = run_agent(factory, agent_type, seed=policy_seed)
                    result.pop("trace")
                    agents[name] = _metrics(result, request.min_coin_ratio)
                if loaded is not None:
                    from ai_platformer.agents.llm.evaluation import evaluate_ppo

                    result, _ = evaluate_ppo(
                        loaded, factory, seed=policy_seed,
                        request_target=request.min_coin_ratio,
                        expected_target=request.min_coin_ratio,
                    )
                    agents["ppo"] = _metrics(
                        {
                            "outcome": result["outcome"],
                            "steps": result["steps"],
                            "coins_collected": result["coins_collected"],
                            "coins_total": result["coins_total"],
                            "episode_return": result["return"],
                            "progress": result["progress"],
                        },
                        request.min_coin_ratio,
                    )
                entries.append(
                    {
                        "level_id": request.level_id,
                        "family": entry["family"],
                        "generation_seed": request.seed,
                        "geometry_sha256": entry["geometry_sha256"],
                        "collection_target": request.min_coin_ratio,
                        "agents": agents,
                    }
                )
    finally:
        if loaded is not None:
            loaded.close()
    agent_names = tuple(entries[0]["agents"])
    groups = defaultdict(list)
    for entry in entries:
        groups[entry["family"]].append(entry)
    return {
        "schema_version": 1,
        "suite_id": manifest["suite_id"],
        "split": split,
        "policy_seed": policy_seed,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "model_sha256": _file_sha256(model_path) if model_path is not None else None,
        "unique_geometries": len({entry["geometry_sha256"] for entry in entries}),
        "note": "One policy seed per static map; it does not increase the geometry sample count.",
        "by_agent": {
            name: {
                "overall": summarize([entry["agents"][name] for entry in entries]),
                "by_family": {
                    family: summarize([entry["agents"][name] for entry in group])
                    for family, group in sorted(groups.items())
                },
            }
            for name in agent_names
        },
        "maps": entries,
    }


def _metrics(result: dict, target: float) -> dict:
    total = result["coins_total"]
    ratio = result["coins_collected"] / total if total else 1.0
    return {
        "outcome": result["outcome"],
        "steps": result["steps"],
        "progress": result["progress"],
        "coins_collected": result["coins_collected"],
        "coins_total": total,
        "coin_ratio": ratio,
        "collection_target_met": result["outcome"] == "success" and ratio + 1e-12 >= target,
        "return": result["episode_return"],
    }


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
