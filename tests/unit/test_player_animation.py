import unittest
from dataclasses import replace

from ai_platformer.core import PlayerSnapshot, WorldSnapshot
from ai_platformer.rendering.player_animation import PlayerAnimator


def snapshot(tick, **values):
    player = replace(PlayerSnapshot(100, 200, 0, 0, True), **values)
    return WorldSnapshot("test:0", tick, "test", 0, player)


class PlayerAnimationTests(unittest.TestCase):
    def test_air_land_and_jump_interrupt(self):
        animator = PlayerAnimator()
        for tick, vy, expected in ((1, -8, "jump"), (2, -1, "apex"), (3, 8, "fall")):
            self.assertEqual(animator.update(snapshot(tick, grounded=False, velocity_y=vy)).name, expected)
        self.assertEqual(animator.update(snapshot(4)).name, "land")
        self.assertEqual(animator.update(snapshot(5, grounded=False, velocity_y=-8)).name, "jump")
        self.assertEqual(animator.update(snapshot(20)).name, "idle")

    def test_stride_follows_distance_and_pause_does_not_advance(self):
        animator = PlayerAnimator()
        animator.update(snapshot(0, velocity_x=4))
        state = snapshot(1, x=122, velocity_x=4)
        pose = animator.update(state)
        self.assertEqual((pose.name, pose.frame), ("walk", 1))
        for _ in range(10):
            self.assertEqual(animator.update(state), pose)
        self.assertEqual(animator.update(snapshot(2, x=144, velocity_x=10)).name, "run")

    def test_braking_turning_and_reset(self):
        animator = PlayerAnimator()
        animator.update(snapshot(0, velocity_x=12))
        self.assertEqual(animator.update(snapshot(1, velocity_x=11.1)).name, "brake")
        self.assertEqual(animator.update(snapshot(2, velocity_x=9, facing=-1)).name, "brake")
        self.assertEqual(animator.update(snapshot(0)).name, "idle")
        self.assertEqual(animator.phase, 0)

    def test_hurt_then_recovery_and_death(self):
        animator = PlayerAnimator()
        animator.update(snapshot(0))
        hurt = replace(snapshot(1), metadata={"invulnerable_ticks": 60})
        self.assertEqual(animator.update(hurt).name, "hurt")
        self.assertEqual(animator.update(replace(snapshot(12), metadata={"invulnerable_ticks": 49})).name, "idle")
        self.assertEqual(animator.update(snapshot(13, alive=False)).angle, 25)


if __name__ == "__main__":
    unittest.main()
