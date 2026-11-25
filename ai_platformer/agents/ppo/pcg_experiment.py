"""Reproducible imitation experiments on generated PlatformerState-v2 maps."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ai_platformer.agents.scripted.ground_coin_jump import GroundCoinJumpAgent
from ai_platformer.benchmark.generalization import load_manifest
from ai_platformer.content.generation_request import GenerationRequest
from ai_platformer.content.level_spec import geometry_hash
from ai_platformer.content.pcg import generate_level
from ai_platformer.envs.factory import EnvironmentFactory


def training_requests(plan_path: Path, seeds: list[int], frozen_path: Path) -> list[GenerationRequest]:
    """Reject overlapping seeds and geometry before collecting a single sample."""
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("training seeds must be nonempty and unique")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    frozen = load_manifest(frozen_path)
    forbidden_seeds = set(sum(plan["splits"].values(), []))
    if forbidden_seeds.intersection(seeds):
        raise ValueError("training seeds overlap frozen evaluation seeds")
    seen = {entry["geometry_sha256"] for entry in frozen["entries"]}
    requests = []
    for seed in seeds:
        for family in plan["families"]:
            request = GenerationRequest.model_validate(
                {"schema_version": 1, "level_id": f"imitation_{family['id']}_{seed}",
                 "seed": seed, **{key: value for key, value in family.items() if key != "id"}}
            )
            digest = geometry_hash(generate_level(request).level)
            if digest in seen:
                raise ValueError("training geometry overlaps another map or frozen evaluation")
            seen.add(digest)
            requests.append(request)
    return requests


def factory_for_request(request: GenerationRequest, path: Path) -> tuple[EnvironmentFactory, str]:
    generated = generate_level(request)
    path.write_text(json.dumps(generated.level.model_dump()) + "\n", encoding="utf-8")
    return EnvironmentFactory(
        {"environment_id": "PlatformerState-v2", "level_spec": str(path),
         "physics": generated.report["physics"], "action_repeat": 4,
         "episode_step_limit": 768}
    ), geometry_hash(generated.level)


def rollout(factory: EnvironmentFactory, agent, *, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    env = factory.make()
    observations, actions = [], []
    agent.reset(seed=seed)
    try:
        observation, _ = env.reset(seed=seed)
        while True:
            action = int(agent.act(observation))
            observations.append(observation.copy())
            actions.append(action)
            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.int64), {
            "outcome": info["outcome"], "steps": len(actions),
            "coins_collected": info["coins_collected"], "coins_total": info["coins_total"],
            "progress": info["progress"], "return": info["episode_return"],
        }
    finally:
        env.close()


class ModelAgent:
    def __init__(self, model):
        self.model = model

    def reset(self, *, seed: int) -> None:
        del seed

    def act(self, observation) -> int:
        return int(self.model.predict(observation, deterministic=True)[0])


def build_model(factory: EnvironmentFactory, *, seed: int):
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    return PPO(
        "MlpPolicy", factory.make(), seed=seed, device="cpu", verbose=0,
        learning_rate=3e-4, n_steps=256, batch_size=256, n_epochs=5,
        policy_kwargs={"activation_fn": torch.nn.ReLU,
                       "net_arch": {"pi": [64, 64], "vf": [64, 64]}},
    )


def fit_actor(model, observations: np.ndarray, actions: np.ndarray, *, epochs: int,
              batch_size: int, learning_rate: float, seed: int) -> list[dict]:
    if epochs < 1 or len(actions) == 0:
        raise ValueError("nonempty demonstrations and positive epochs required")
    rng = np.random.default_rng(seed)
    parameters = list(model.policy.mlp_extractor.policy_net.parameters()) + list(
        model.policy.action_net.parameters()
    )
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    obs = torch.as_tensor(observations, dtype=torch.float32, device=model.device)
    labels = torch.as_tensor(actions, dtype=torch.long, device=model.device)
    history = []
    model.policy.set_training_mode(True)
    try:
        for epoch in range(epochs):
            losses = []
            order = rng.permutation(len(actions))
            for start in range(0, len(actions), batch_size):
                indices = order[start:start + batch_size]
                logits = model.policy.get_distribution(obs[indices]).distribution.logits
                loss = F.cross_entropy(logits, labels[indices])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            history.append({"epoch": epoch + 1, "mean_loss": sum(losses) / len(losses)})
    finally:
        model.policy.set_training_mode(False)
    return history


def collect_teacher_data(requests: list[GenerationRequest], *, policy_seed: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    all_obs, all_actions, maps = [], [], []
    with tempfile.TemporaryDirectory(prefix="platformer-teacher-") as directory:
        path = Path(directory) / "level.json"
        for request in requests:
            factory, digest = factory_for_request(request, path)
            observations, actions, result = rollout(factory, GroundCoinJumpAgent(), seed=policy_seed)
            all_obs.append(observations)
            all_actions.append(actions)
            maps.append({"level_id": request.level_id, "geometry_sha256": digest, **result})
    return np.concatenate(all_obs), np.concatenate(all_actions), maps


def evaluate_model(model, manifest_path: Path, *, split: str = "development", seed: int = 100,
                   open_holdout: bool = False) -> dict:
    if split not in {"development", "holdout"} or (split == "holdout" and not open_holdout):
        raise ValueError("holdout requires explicit open_holdout=True")
    manifest = load_manifest(manifest_path)
    rows = []
    with tempfile.TemporaryDirectory(prefix="platformer-eval-") as directory:
        path = Path(directory) / "level.json"
        for entry in manifest["entries"]:
            if entry["split"] != split:
                continue
            request = GenerationRequest.model_validate(entry["request"])
            factory, digest = factory_for_request(request, path)
            if digest != entry["geometry_sha256"]:
                raise ValueError("frozen geometry changed")
            _, _, result = rollout(factory, ModelAgent(model), seed=seed)
            ratio = result["coins_collected"] / result["coins_total"] if result["coins_total"] else 1.0
            rows.append({"level_id": request.level_id, "family": entry["family"],
                         "geometry_sha256": digest, "coin_ratio": ratio,
                         "target_met": result["outcome"] == "success" and ratio >= request.min_coin_ratio,
                         **result})
    return {"split": split, "maps": len(rows),
            "successes": sum(row["outcome"] == "success" for row in rows),
            "targets_met": sum(row["target_met"] for row in rows),
            "mean_coin_ratio": sum(row["coin_ratio"] for row in rows) / len(rows),
            "rows": rows}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
