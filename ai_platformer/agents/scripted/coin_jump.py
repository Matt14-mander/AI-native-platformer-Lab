"""Observation-only collection teacher with hazard jumps taking priority."""

from ai_platformer.agents.scripted.baselines import RuleJumpAgent
from ai_platformer.core import Action


class CoinJumpAgent(RuleJumpAgent):
    """Use short walking jumps for reachable coins; keep high running hazard jumps."""

    def __init__(self):
        super().__init__()
        self.coin_jump = False

    def reset(self, *, seed: int) -> None:
        self.coin_jump = False

    def act(self, observation) -> int:
        grounded = observation[4] > 0.5
        rising = observation[3] < 0
        dx, dy = observation[10:12]
        if grounded:
            self.coin_jump = False
            if len(observation) > 14 and observation[14] > 0.5:
                return int(Action.RIGHT_RUN)
            if min(observation[7], observation[8]) <= self.trigger_distance:
                return int(Action.RIGHT_RUN_JUMP)
            if 0.02 < dx < 0.4 and -0.65 < dy < -0.1:
                self.coin_jump = True
                return int(Action.RIGHT_JUMP)
            return int(Action.RIGHT_RUN)
        if self.coin_jump:
            if rising and dy < -0.1 and dx > -0.15:
                return int(Action.RIGHT_JUMP)
            return int(Action.RIGHT)
        return super().act(observation)
