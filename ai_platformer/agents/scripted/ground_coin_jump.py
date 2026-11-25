"""Collection teacher that uses a short jump over a low obstacle near ground coins."""

from ai_platformer.agents.scripted.coin_jump import CoinJumpAgent
from ai_platformer.core import Action
from ai_platformer.envs import ObservationIndex


class GroundCoinJumpAgent(CoinJumpAgent):
    """Preserve the coin teacher's gap and high-obstacle jumps."""

    def __init__(self, *, short_hold_steps: int = 3):
        super().__init__()
        if short_hold_steps < 1:
            raise ValueError("short_hold_steps must be positive")
        self.short_hold_steps = short_hold_steps
        self.short_remaining = 0
        self.short_airborne = False

    def reset(self, *, seed: int) -> None:
        super().reset(seed=seed)
        self.short_remaining = 0
        self.short_airborne = False

    def act(self, observation) -> int:
        grounded = observation[ObservationIndex.GROUNDED] > 0.5
        if self.short_remaining:
            self.short_remaining -= 1
            return int(Action.RIGHT_RUN_JUMP)
        if self.short_airborne and not grounded:
            return int(Action.RIGHT_RUN)
        if grounded:
            self.short_airborne = False
        obstacle = observation[ObservationIndex.OBSTACLE_DISTANCE]
        edge = observation[ObservationIndex.GROUND_EDGE_DISTANCE]
        height = observation[ObservationIndex.OBSTACLE_HEIGHT]
        coin_y = observation[ObservationIndex.COIN_DELTA_Y]
        if (
            grounded
            and obstacle <= self.trigger_distance
            and edge > self.trigger_distance
            and 0 < height <= 0.75
            and coin_y >= -0.1
        ):
            self.coin_jump = False
            self.short_airborne = True
            self.short_remaining = self.short_hold_steps - 1
            return int(Action.RIGHT_RUN_JUMP)
        return super().act(observation)

    def observe_executed_action(self, label: int, executed: int) -> None:
        """Discard a planned jump when a DAgger learner chose another action."""
        if label != executed:
            self.short_remaining = 0
            self.short_airborne = False
            self.coin_jump = False
