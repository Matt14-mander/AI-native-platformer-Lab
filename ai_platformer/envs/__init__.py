"""Gymnasium adapters around :mod:`ai_platformer.core`."""

from gymnasium.envs.registration import register, registry

from .platformer_state import OBSERVATION_SIZE, ObservationIndex, PlatformerStateEnv
from .platformer_state_v1 import JUMP_HELD_INDEX, PlatformerStateEnvV1
from .reward import RewardConfig, compose_reward

ENV_ID = "PlatformerState-v0"

if ENV_ID not in registry:
    register(id=ENV_ID, entry_point="ai_platformer.envs:PlatformerStateEnv")
if "PlatformerState-v1" not in registry:
    register(id="PlatformerState-v1", entry_point="ai_platformer.envs:PlatformerStateEnvV1")

__all__ = [
    "ENV_ID",
    "JUMP_HELD_INDEX",
    "OBSERVATION_SIZE",
    "ObservationIndex",
    "PlatformerStateEnv",
    "PlatformerStateEnvV1",
    "RewardConfig",
    "compose_reward",
]
