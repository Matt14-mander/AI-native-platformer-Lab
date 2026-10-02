"""State v1 exposes the jump latch needed for deterministic repeated jumping."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from ai_platformer.core import Action, control_for

from .platformer_state import OBSERVATION_SIZE, PlatformerStateEnv

JUMP_HELD_INDEX = OBSERVATION_SIZE


class PlatformerStateEnvV1(PlatformerStateEnv):
    def __init__(self, **kwargs):
        self._jump_held = False
        super().__init__(**kwargs)
        self.observation_space = spaces.Box(
            -1.0, 1.0, shape=(OBSERVATION_SIZE + 1,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        self._jump_held = False
        return super().reset(seed=seed, options=options)

    def step(self, action):
        if not self._episode_done and self.action_space.contains(action):
            self._jump_held = control_for(Action(int(action))).jump
        return super().step(action)

    def _observation(self, state):
        return np.append(super()._observation(state), np.float32(self._jump_held))
