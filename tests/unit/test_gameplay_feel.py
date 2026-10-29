import unittest
from dataclasses import replace
from pathlib import Path

from ai_platformer.core import Action, BasicPlatformerCore, LevelDefinition, PhysicsConfig, SolidRect
from ai_platformer.settings import load_gameplay_settings


def responsive_config(**overrides):
    path = Path(__file__).resolve().parents[2] / "config" / "gameplay_player.json"
    return replace(load_gameplay_settings(path).physics, **overrides)


def make_core(config=None, *, solids=None, spawn_bottom=300):
    level = LevelDefinition(
        "feel", 10000, 1000, 10, spawn_bottom, 9900,
        (SolidRect(0, 300, 10000, 700),) if solids is None else solids,
    )
    core = BasicPlatformerCore(lambda _: level, config=config or responsive_config())
    core.reset(seed=1, level_id="feel")
    return core


class GameplayFeelTests(unittest.TestCase):
    def test_acceleration_braking_and_shift_release(self):
        core = make_core()
        for _ in range(20):
            core.step(Action.RIGHT_RUN)
        self.assertAlmostEqual(core.state.player.velocity_x, 12)
        start = core.state.player.x
        for _ in range(14):
            core.step(Action.NOOP)
        self.assertEqual(core.state.player.velocity_x, 0)
        self.assertLess(core.state.player.x - start, 80)
        for _ in range(20):
            core.step(Action.RIGHT_RUN)
        core.step(Action.RIGHT)
        self.assertAlmostEqual(core.state.player.velocity_x, 11.1)
        for _ in range(7):
            core.step(Action.RIGHT)
        self.assertAlmostEqual(core.state.player.velocity_x, 6)

    def test_turn_and_air_control(self):
        core = make_core()
        for _ in range(20):
            core.step(Action.RIGHT_RUN)
        for _ in range(11):
            core.step(Action.LEFT_RUN)
        self.assertLess(core.state.player.velocity_x, 0)
        core = make_core(spawn_bottom=100)
        core.step(Action.RIGHT)
        self.assertAlmostEqual(core.state.player.velocity_x, 0.25)
        core.step(Action.NOOP)
        self.assertAlmostEqual(core.state.player.velocity_x, 0.21)

    def leave_ledge(self):
        core = make_core(solids=(SolidRect(0, 300, 45, 700),))
        for _ in range(30):
            core.step(Action.RIGHT_RUN)
            if not core.state.player.grounded:
                return core
        self.fail("did not leave ledge")

    def test_coyote_window_and_no_second_jump(self):
        core = self.leave_ledge()
        for _ in range(4):
            core.step(Action.NOOP)
        core.step(Action.JUMP)
        self.assertLess(core.state.player.velocity_y, -9)
        self.assertTrue(core.jump_started_this_tick)
        core.step(Action.NOOP)
        self.assertFalse(core.jump_started_this_tick)
        core.step(Action.JUMP)
        self.assertGreater(core.state.player.velocity_y, -9)
        self.assertFalse(core.jump_started_this_tick)
        core = self.leave_ledge()
        for _ in range(7):
            core.step(Action.NOOP)
        core.step(Action.JUMP)
        self.assertGreater(core.state.player.velocity_y, 0)

    def test_press_before_landing_is_consumed_once(self):
        core = make_core(spawn_bottom=296)
        core.step(Action.JUMP)
        core.step(Action.JUMP)
        landed = core.step(Action.JUMP).state
        self.assertTrue(core.jump_started_this_tick)
        self.assertLess(landed.player.velocity_y, 0)
        self.assertFalse(landed.player.grounded)
        for _ in range(100):
            core.step(Action.JUMP)
        self.assertTrue(core.state.player.grounded)
        self.assertEqual(core.state.player.velocity_y, 0)
        self.assertFalse(core.jump_started_this_tick)

    def test_early_press_expires(self):
        core = make_core(spawn_bottom=100)
        core.step(Action.JUMP)
        for _ in range(60):
            core.step(Action.NOOP)
        self.assertTrue(core.state.player.grounded)
        self.assertEqual(core.state.player.y, 268)

    def test_release_cuts_jump_and_repress_does_not_restore_float(self):
        a, b = make_core(), make_core()
        a.step(Action.JUMP)
        b.step(Action.JUMP)
        a.step(Action.NOOP)
        b.step(Action.NOOP)
        for _ in range(3):
            a.step(Action.JUMP)
            b.step(Action.NOOP)
            self.assertEqual(a.state.player.y, b.state.player.y)
        high = make_core()
        low = make_core()
        high_y = low_y = 268
        for tick in range(100):
            high.step(Action.JUMP)
            low.step(Action.JUMP if tick == 0 else Action.NOOP)
            high_y = min(high_y, high.state.player.y)
            low_y = min(low_y, low.state.player.y)
        self.assertGreater(268 - high_y, 170)
        self.assertGreater(268 - low_y, 32)
        self.assertLess(268 - low_y, 60)

    def test_reset_clears_pending_jump(self):
        core = make_core(spawn_bottom=296)
        core.step(Action.JUMP)
        core.reset(seed=1, level_id="feel")
        for _ in range(5):
            core.step(Action.NOOP)
        self.assertTrue(core.state.player.grounded)


if __name__ == "__main__":
    unittest.main()
