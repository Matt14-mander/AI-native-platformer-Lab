"""Gymnasium state-vector environment backed by the shared game core."""

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ai_platformer.content.legacy import LegacyLevelRepository
from ai_platformer.core import Action, BasicPlatformerCore, SolidRect, WorldSnapshot
from ai_platformer.core.level import LevelDefinition
from ai_platformer.settings import GameplaySettings, load_gameplay_settings

from .reward import RewardConfig, compose_reward


class ObservationIndex(IntEnum):
    """Stable indices of the ``PlatformerState-v0`` observation contract."""

    PLAYER_X = 0
    PLAYER_Y = 1
    VELOCITY_X = 2
    VELOCITY_Y = 3
    GROUNDED = 4
    FACING = 5
    PROGRESS = 6
    GROUND_EDGE_DISTANCE = 7
    OBSTACLE_DISTANCE = 8
    OBSTACLE_HEIGHT = 9
    COIN_DELTA_X = 10
    COIN_DELTA_Y = 11
    COIN_RATIO = 12
    REMAINING_TIME = 13


OBSERVATION_SIZE = len(ObservationIndex)


class PlatformerStateEnv(gym.Env[np.ndarray, int]):
    """Deterministic state observation environment for training and evaluation."""

    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(
        self,
        *,
        level_id: str | None = None,
        seed: int | None = None,
        action_repeat: int = 4,
        sensor_range: float = 240.0,
        episode_step_limit: int | None = None,
        settings: GameplaySettings | None = None,
        reward_config: RewardConfig | None = None,
        level_loader: Callable[[str], LevelDefinition] | None = None,
    ) -> None:
        super().__init__()
        if action_repeat <= 0:
            raise ValueError("action_repeat must be positive")
        if sensor_range <= 0:
            raise ValueError("sensor_range must be positive")
        if episode_step_limit is not None and episode_step_limit <= 0:
            raise ValueError("episode_step_limit must be positive")

        self.settings = settings or load_gameplay_settings()
        self.level_id = level_id or self.settings.level_id
        self.default_seed = self.settings.seed if seed is None else seed
        self.action_repeat = action_repeat
        self.sensor_range = sensor_range
        self.episode_step_limit = episode_step_limit
        self.reward_config = reward_config or RewardConfig()
        self.repository = LegacyLevelRepository()
        self._level_loader = level_loader or self.repository.load
        self.level = self._level_loader(self.level_id)
        self.core = BasicPlatformerCore(self._level_loader, config=self.settings.physics)
        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self._episode_done = True
        self._episode_return = 0.0
        self._episode_steps = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        actual_seed = self.default_seed if seed is None else seed
        requested_level = (
            self.level_id if options is None else options.get("level_id", self.level_id)
        )
        self.level_id = str(requested_level)
        self.level = self._level_loader(self.level_id)
        state = self.core.reset(seed=actual_seed, level_id=self.level_id)
        self._episode_done = False
        self._episode_return = 0.0
        self._episode_steps = 0
        return self._observation(state), self._info(state, core_ticks=0)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._episode_done:
            raise RuntimeError("episode is finished; call reset() before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"unsupported action id: {action}")

        previous = self.core.state
        previous_coins = int(previous.metadata.get("coins_collected", 0))
        terminated = False
        truncated = False
        core_ticks = 0
        core_info: dict[str, Any] = {}

        for _ in range(self.action_repeat):
            result = self.core.step(Action(int(action)))
            core_ticks += 1
            core_info = dict(result.info)
            terminated = result.terminated
            truncated = result.truncated
            if terminated or truncated:
                break

        state = self.core.state
        coin_delta = int(state.metadata.get("coins_collected", 0)) - previous_coins
        self._episode_steps += 1
        if (
            not terminated
            and not truncated
            and self.episode_step_limit is not None
            and self._episode_steps >= self.episode_step_limit
        ):
            truncated = True
            core_info["outcome"] = "time_limit"
        reward, reward_parts = compose_reward(
            previous_progress=previous.progress,
            current_progress=state.progress,
            coin_delta=coin_delta,
            outcome=core_info.get("outcome"),
            config=self.reward_config,
        )
        self._episode_return += reward
        self._episode_done = terminated or truncated

        info = self._info(state, core_ticks=core_ticks)
        info.update(core_info)
        info["reward_components"] = reward_parts
        info["episode_return"] = self._episode_return
        return self._observation(state), reward, terminated, truncated, info

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None

    def _observation(self, state: WorldSnapshot) -> np.ndarray:
        player = state.player
        config = self.settings.physics
        obstacle_distance, obstacle_height = self._obstacle(state)
        coin_dx, coin_dy = self._nearest_coin_delta(state)
        coins_total = int(state.metadata.get("coins_total", 0))
        coins_collected = int(state.metadata.get("coins_collected", 0))
        return np.array(
            [
                self._unit(player.x / self.level.width),
                self._unit(player.y / self.level.height),
                self._signed(player.velocity_x / config.max_run_speed),
                self._signed(
                    player.velocity_y
                    / max(abs(config.jump_velocity), config.max_fall_speed)
                ),
                1.0 if player.grounded else 0.0,
                float(player.facing),
                self._unit(state.progress),
                self._unit(self._ground_edge_distance(state) / self.sensor_range),
                self._unit(obstacle_distance / self.sensor_range),
                self._unit(obstacle_height / (config.player_height * 4.0)),
                self._signed(coin_dx / self.sensor_range),
                self._signed(coin_dy / self.sensor_range),
                self._unit(coins_collected / coins_total if coins_total else 0.0),
                self._remaining_time(state),
            ],
            dtype=np.float32,
        )

    def _ground_edge_distance(self, state: WorldSnapshot) -> float:
        player = state.player
        feet = player.y + self.settings.physics.player_height
        cursor = player.x + self.settings.physics.player_width
        active_blocks = [
            SolidRect(entity.x, entity.y, entity.width, entity.height)
            for entity in state.entities
            if entity.active and entity.kind in {"brick", "box"}
        ]
        intervals = sorted(
            (solid.x, solid.right)
            for solid in (*self.level.solids, *active_blocks)
            if abs(solid.y - feet) <= 1e-5 and solid.right >= cursor
        )
        limit = cursor
        for start, end in intervals:
            if start > limit + 1e-5:
                break
            limit = max(limit, end)
        return min(self.sensor_range, max(0.0, limit - cursor))

    def _obstacle(self, state: WorldSnapshot) -> tuple[float, float]:
        player = state.player
        player_right = player.x + self.settings.physics.player_width
        player_bottom = player.y + self.settings.physics.player_height
        active_entities = [
            entity
            for entity in state.entities
            if entity.active and entity.kind in {"brick", "box", "enemy"}
        ]
        candidates = list(self.level.solids) + [
            SolidRect(entity.x, entity.y, entity.width, entity.height)
            for entity in active_entities
        ]
        candidates = [
            solid
            for solid in candidates
            if solid.right >= player_right
            and solid.x >= player_right - 1e-5
            and solid.y < player_bottom - 1e-5
            and solid.bottom > player.y
        ]
        if not candidates:
            return self.sensor_range, 0.0
        obstacle = min(candidates, key=lambda solid: solid.x)
        return (
            min(self.sensor_range, max(0.0, obstacle.x - player_right)),
            max(0.0, player_bottom - obstacle.y),
        )

    def _nearest_coin_delta(self, state: WorldSnapshot) -> tuple[float, float]:
        active = [entity for entity in state.entities if entity.active and entity.kind == "coin"]
        if not active:
            return self.sensor_range, 0.0
        player = state.player
        coin = min(active, key=lambda item: (item.x - player.x) ** 2 + (item.y - player.y) ** 2)
        return coin.x - player.x, coin.y - player.y

    def _info(self, state: WorldSnapshot, *, core_ticks: int) -> dict[str, Any]:
        return {
            "level_id": state.level_id,
            "seed": state.seed,
            "tick": state.tick,
            "core_ticks": core_ticks,
            "episode_step": self._episode_steps,
            "progress": state.progress,
            "score": int(state.metadata.get("score", 0)),
            "coins_collected": int(state.metadata.get("coins_collected", 0)),
            "coins_total": int(state.metadata.get("coins_total", 0)),
            "shield_charges": int(state.metadata.get("shield_charges", 0)),
            "enemies_defeated": int(state.metadata.get("enemies_defeated", 0)),
            "blocks_broken": int(state.metadata.get("blocks_broken", 0)),
        }

    def _remaining_time(self, state: WorldSnapshot) -> float:
        if self.episode_step_limit is not None:
            return self._unit(1.0 - self._episode_steps / self.episode_step_limit)
        return self._unit(1.0 - state.tick / self.settings.physics.max_episode_steps)

    @staticmethod
    def _unit(value: float) -> float:
        return max(0.0, min(1.0, value))

    @staticmethod
    def _signed(value: float) -> float:
        return max(-1.0, min(1.0, value))
