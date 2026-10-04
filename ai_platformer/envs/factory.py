"""One resolved environment protocol for training and all evaluations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from math import isfinite
from typing import Any

from ai_platformer.content.curriculum import TrainingLevelRepository, content_hash
from ai_platformer.core import PhysicsConfig
from ai_platformer.settings import AI_GAMEPLAY_PATH, load_gameplay_settings

from .platformer_state import OBSERVATION_SIZE, ObservationIndex, PlatformerStateEnv
from .platformer_state_v1 import JUMP_HELD_INDEX, PlatformerStateEnvV1
from .platformer_state_v2 import PlatformerStateEnvV2
from .reward import RewardConfig


class EnvironmentFactory:
    def __init__(self, config: dict[str, Any]) -> None:
        allowed = {
            "action_repeat",
            "episode_step_limit",
            "sensor_range",
            "physics",
            "reward",
            "curriculum_manifest",
            "level_id",
            "environment_id",
            "level_spec",
        }
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(f"unknown environment options: {sorted(unknown)}")
        self.environment_id = config.get("environment_id", "PlatformerState-v0")
        if self.environment_id not in {
            "PlatformerState-v0",
            "PlatformerState-v1",
            "PlatformerState-v2",
        }:
            raise ValueError("unsupported environment_id")
        self.settings = load_gameplay_settings(
            AI_GAMEPLAY_PATH if self.environment_id == "PlatformerState-v2" else None
        )
        physics = asdict(self.settings.physics)
        physics.update(config.get("physics", {}))
        for key, value in physics.items():
            if not isfinite(value) or (value >= 0 if key == "jump_velocity" else value <= 0):
                raise ValueError(f"invalid physics {key}")
        self.settings = replace(self.settings, physics=PhysicsConfig(**physics))
        self.reward = RewardConfig(**config.get("reward", {}))
        if any(not isfinite(value) for value in asdict(self.reward).values()):
            raise ValueError("reward values must be finite")
        if "level_spec" in config:
            if "curriculum_manifest" in config:
                raise ValueError("choose level_spec or curriculum_manifest, not both")
            from ai_platformer.content.level_repository import LevelSpecRepository

            self.repository = LevelSpecRepository(
                config["level_spec"], physics=self.settings.physics
            )
            default_level = self.repository.spec.level_id
        else:
            self.repository = TrainingLevelRepository(config.get("curriculum_manifest"))
            default_level = self.settings.level_id
        self.level_id = str(config.get("level_id", default_level))
        self.action_repeat = config.get("action_repeat", 4)
        self.step_limit = config.get("episode_step_limit", 1024)
        self.sensor_range = float(config.get("sensor_range", 240.0))
        if (
            not isinstance(self.action_repeat, int)
            or isinstance(self.action_repeat, bool)
            or self.action_repeat <= 0
        ):
            raise ValueError("action_repeat must be a positive integer")
        if not isfinite(self.sensor_range) or self.sensor_range <= 0:
            raise ValueError("action_repeat and sensor_range must be positive")
        if self.step_limit is not None and (
            not isinstance(self.step_limit, int)
            or isinstance(self.step_limit, bool)
            or self.step_limit <= 0
        ):
            raise ValueError("episode_step_limit must be positive")

    def make(self, *, level_id: str | None = None, seed: int | None = None) -> PlatformerStateEnv:
        env_type = {
            "PlatformerState-v0": PlatformerStateEnv,
            "PlatformerState-v1": PlatformerStateEnvV1,
            "PlatformerState-v2": PlatformerStateEnvV2,
        }[self.environment_id]
        return env_type(
            level_id=level_id or self.level_id,
            seed=seed,
            action_repeat=self.action_repeat,
            episode_step_limit=self.step_limit,
            sensor_range=self.sensor_range,
            settings=self.settings,
            reward_config=self.reward,
            level_loader=self.repository.load,
        )

    def protocol(self, level_ids: list[str]) -> dict[str, Any]:
        return {
            "environment_id": self.environment_id,
            "action_version": 1,
            "observation_version": int(self.environment_id != "PlatformerState-v0"),
            "reward_version": 1,
            "observation_size": OBSERVATION_SIZE + int(self.environment_id != "PlatformerState-v0"),
            "observation_indices": {
                **{item.name: int(item) for item in ObservationIndex},
                **(
                    {"JUMP_HELD": JUMP_HELD_INDEX}
                    if self.environment_id != "PlatformerState-v0"
                    else {}
                ),
            },
            "action_repeat": self.action_repeat,
            "episode_step_limit": self.step_limit,
            "sensor_range": self.sensor_range,
            "physics": asdict(self.settings.physics),
            "reward": asdict(self.reward),
            "levels": {
                item: content_hash(self.repository.load(item)) for item in sorted(level_ids)
            },
            "curriculum_generator_version": self.repository.manifest["generator_version"],
            "curriculum_manifest_hash": protocol_hash(self.repository.manifest),
        }

    def for_level_spec(self, path) -> EnvironmentFactory:
        """Explore a candidate using the same resolved policy/environment settings."""
        return EnvironmentFactory(
            {
                "environment_id": self.environment_id,
                "level_spec": str(path),
                "action_repeat": self.action_repeat,
                "episode_step_limit": self.step_limit,
                "sensor_range": self.sensor_range,
                "physics": asdict(self.settings.physics),
                "reward": asdict(self.reward),
            }
        )


def protocol_hash(protocol: dict) -> str:
    return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
