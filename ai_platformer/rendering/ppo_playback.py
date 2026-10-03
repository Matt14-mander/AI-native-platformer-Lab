"""PPO playback uses the training environment as its sole simulation clock."""

from __future__ import annotations

from time import perf_counter
from typing import Protocol

import numpy as np

from ai_platformer.core import Action
from ai_platformer.envs.factory import EnvironmentFactory


class PlaybackPolicy(Protocol):
    def predict(self, observation: np.ndarray, *, deterministic: bool): ...


class PlaybackSession:
    def __init__(
        self,
        policy: PlaybackPolicy,
        factory: EnvironmentFactory,
        *,
        level_id: str,
        seed: int,
        deterministic: bool = True,
    ) -> None:
        self.policy = policy
        self.env = factory.make(level_id=level_id, seed=seed)
        self.seed = seed
        self.deterministic = deterministic
        self.reset()

    def reset(self) -> None:
        # Seed policy sampling too: replaying the same level/seed is reproducible.
        from stable_baselines3.common.utils import set_random_seed

        set_random_seed(self.seed)
        self.observation, self.info = self.env.reset(seed=self.seed)
        self.done = False
        self.action = Action.NOOP
        self.inference_ms = 0.0
        self.episode_return = 0.0

    def step(self) -> None:
        if self.done:
            raise RuntimeError("episode is finished; reset before playback")
        started = perf_counter()
        action, _ = self.policy.predict(self.observation, deterministic=self.deterministic)
        self.inference_ms = (perf_counter() - started) * 1000
        self.action = Action(int(np.asarray(action).item()))
        self.observation, reward, terminated, truncated, self.info = self.env.step(int(self.action))
        self.episode_return += reward
        self.done = terminated or truncated

    def summary(self) -> dict:
        return {
            "level_id": self.env.level_id,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "outcome": self.info.get("outcome"),
            "steps": self.info["episode_step"],
            "ticks": self.info["tick"],
            "progress": self.info["progress"],
            "return": self.episode_return,
        }


def solids_for_playback(factory: EnvironmentFactory, level_id: str) -> dict:
    repository = factory.repository
    if level_id not in repository.levels:
        return repository.legacy.load_solids_by_group(level_id)
    level = repository.load(level_id)
    # Curriculum courses have floor segments and optional raised obstacles.
    return {
        "ground": tuple(s for s in level.solids if s.y == level.spawn_bottom),
        "pipe": tuple(s for s in level.solids if s.y != level.spawn_bottom),
        "step": (),
    }


def create_renderer(factory: EnvironmentFactory, session: PlaybackSession, screen_size: tuple):
    from ai_platformer.rendering.pygame_level import PygameLevelRenderer

    level = session.env.level
    return PygameLevelRenderer(
        screen_size=screen_size,
        level_width=level.width,
        goal_x=level.goal_x,
        solids_by_kind=solids_for_playback(factory, level.level_id),
        player_width=factory.settings.physics.player_width,
        player_height=factory.settings.physics.player_height,
    )
