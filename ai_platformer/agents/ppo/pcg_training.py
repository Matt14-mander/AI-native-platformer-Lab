"""Generated-map PPO training with explicit episode-boundary level sampling."""

from __future__ import annotations

from dataclasses import asdict

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from ai_platformer.content.level_spec import geometry_hash, to_level_definition
from ai_platformer.content.pcg import SEGMENTS_V1_PHYSICS, generate_level
from ai_platformer.envs.factory import EnvironmentFactory
from ai_platformer.envs.platformer_state_v2 import PlatformerStateEnvV2


class GeneratedMapSampler(gym.Wrapper):
    def __init__(self, requests, *, rng_seed: int, weights: dict[str, float] | None = None):
        generated = [generate_level(request) for request in requests]
        self.levels = {item.level.level_id: to_level_definition(item.level) for item in generated}
        self.ids = [request.level_id for request in requests]
        self.hashes = {item.level.level_id: geometry_hash(item.level) for item in generated}
        if len(self.levels) != len(requests) or len(set(self.hashes.values())) != len(requests):
            raise ValueError("training maps must have distinct IDs and geometry")
        settings = EnvironmentFactory(
            {"environment_id": "PlatformerState-v2", "physics": asdict(SEGMENTS_V1_PHYSICS)}
        ).settings
        env = PlatformerStateEnvV2(
            level_id=self.ids[0], settings=settings, level_loader=self.levels.__getitem__,
            action_repeat=4, episode_step_limit=768,
        )
        super().__init__(env)
        self.rng = np.random.default_rng(rng_seed)
        self.weights = None
        if weights is not None:
            values = np.asarray([weights.get(item, 0.0) for item in self.ids], dtype=float)
            if not np.all(np.isfinite(values)) or np.any(values < 0) or values.sum() <= 0:
                raise ValueError("invalid level weights")
            self.weights = values / values.sum()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        index = int(self.rng.choice(len(self.ids), p=self.weights))
        episode_seed = int(self.rng.integers(2**31))
        return self.env.reset(seed=episode_seed, options={"level_id": self.ids[index]})


def make_ppo(requests, *, seed: int, n_envs: int = 4, weights=None) -> PPO:
    torch.set_num_threads(1)
    env = DummyVecEnv([
        lambda rank=rank: GeneratedMapSampler(requests, rng_seed=seed + rank, weights=weights)
        for rank in range(n_envs)
    ])
    return PPO(
        "MlpPolicy", env, seed=seed, device="cpu", verbose=0,
        learning_rate=3e-4, n_steps=128, batch_size=256, n_epochs=5,
        policy_kwargs={"activation_fn": torch.nn.ReLU,
                       "net_arch": {"pi": [64, 64], "vf": [64, 64]}},
    )


def initialize_actor(model: PPO, bc_model: PPO) -> None:
    """Copy only the policy branch; leave value estimates and optimizer fresh."""
    model.policy.mlp_extractor.policy_net.load_state_dict(
        bc_model.policy.mlp_extractor.policy_net.state_dict(), strict=True
    )
    model.policy.action_net.load_state_dict(bc_model.policy.action_net.state_dict(), strict=True)
