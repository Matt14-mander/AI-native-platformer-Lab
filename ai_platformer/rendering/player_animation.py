"""Tick-driven visual state; never changes simulation or collision geometry."""

from __future__ import annotations

from dataclasses import dataclass
from math import sin, tau

from ai_platformer.core import WorldSnapshot


@dataclass(frozen=True)
class AnimationPose:
    name: str = "idle"
    frame: int = 0
    height_delta: int = 0
    angle: int = 0


class PlayerAnimator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.previous = None
        self.pose = AnimationPose()
        self.phase = 0.0
        self.landing_until = -1
        self.hurt_until = -1

    def update(self, state: WorldSnapshot) -> AnimationPose:
        previous = self.previous
        if previous is not None and (
            state.episode_id != previous.episode_id or state.tick < previous.tick
        ):
            self.reset()
            previous = None
        if previous is not None and state.tick == previous.tick:
            return self.pose
        player = state.player
        speed = abs(player.velocity_x)
        braking = False
        if previous is not None:
            old = previous.player
            if player.grounded and not old.grounded and old.velocity_y > 0:
                self.landing_until = state.tick + 8
            if state.metadata.get("invulnerable_ticks", 0) > previous.metadata.get("invulnerable_ticks", 0):
                self.hurt_until = state.tick + 10
            braking = player.grounded and speed > 1.5 and (
                player.velocity_x * player.facing < 0
                or abs(old.velocity_x) - speed > 0.5
            )
            if player.grounded and old.grounded:
                self.phase += abs(player.x - old.x) / 22.0

        if not player.alive:
            pose = AnimationPose("hurt", angle=25 * player.facing)
        elif state.tick < self.hurt_until:
            pose = AnimationPose("hurt", angle=8 * player.facing)
        elif not player.grounded:
            name = "apex" if abs(player.velocity_y) <= 1.5 else (
                "jump" if player.velocity_y < 0 else "fall"
            )
            pose = AnimationPose(name)
            self.landing_until = -1
        elif state.tick < self.landing_until:
            pose = AnimationPose("land", height_delta=round(4 * (state.tick - self.landing_until + 8) / 8))
        elif braking:
            pose = AnimationPose("brake")
        elif speed >= 0.35:
            pose = AnimationPose("run" if speed > 6 else "walk", int(self.phase) % 4)
        else:
            self.phase = 0.0
            pose = AnimationPose("idle", height_delta=round(sin(state.tick * tau / 120)))
        self.previous = state
        self.pose = pose
        return pose
