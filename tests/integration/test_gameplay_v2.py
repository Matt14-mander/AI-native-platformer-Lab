"""Fixed jump semantics, shared human/agent physics and legacy compatibility."""

import unittest

import gymnasium as gym
import numpy as np

from ai_platformer.benchmark.gameplay import audit_settings, measure_jump
from ai_platformer.benchmark.readiness import run_reward_exploit_audit
from ai_platformer.core import Action, BasicPlatformerCore, LevelDefinition, SolidRect
from ai_platformer.envs.factory import EnvironmentFactory, protocol_hash
from ai_platformer.settings import AI_GAMEPLAY_PATH, load_gameplay_settings


class GameplayV2Tests(unittest.TestCase):
    def setUp(self):
        self.settings = load_gameplay_settings(AI_GAMEPLAY_PATH)
        self.level = LevelDefinition(
            "flat", 20000, 600, 200, 540, 19000, (SolidRect(0, 540, 20000, 60),)
        )

    def core(self):
        core = BasicPlatformerCore(lambda _: self.level, config=self.settings.physics)
        core.reset(seed=123, level_id="flat")
        return core

    def test_release_and_repress_in_air_do_not_change_trajectory(self):
        held, tapping = self.core(), self.core()
        for tick in range(45):
            expected = held.step(Action.JUMP).state
            action = Action.JUMP if tick in (0, 3, 7, 12, 25) else Action.NOOP
            actual = tapping.step(action).state
            self.assertEqual(actual.player, expected.player)
        self.assertTrue(held.state.player.grounded)

    def test_hold_cannot_retrigger_and_release_enables_next_jump(self):
        report = audit_settings(self.settings)
        self.assertEqual(report["continuous_hold"]["launches"], 1)
        self.assertTrue(all(report["checks"].values()))
        self.assertTrue(report["fixed_height"])
        heights = [measure_jump(self.settings, count)["height_px"] for count in (1, 4, 8, 300)]
        np.testing.assert_allclose(heights, 182.7)

    def test_gym_v2_and_factory_share_profile_and_observation_contract(self):
        env = gym.make("PlatformerState-v2", episode_step_limit=1024)
        factory = EnvironmentFactory({"environment_id": "PlatformerState-v2"})
        reference = factory.make()
        try:
            expected, _ = reference.reset(seed=123)
            actual, _ = env.reset(seed=123)
            np.testing.assert_array_equal(actual, expected)
            for action in (Action.RIGHT_RUN, Action.RIGHT_RUN_JUMP, Action.RIGHT_RUN):
                expected, *_ = reference.step(int(action))
                actual, *_ = env.step(int(action))
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(reference.core.state, env.unwrapped.core.state)
            self.assertEqual(env.observation_space.shape, (15,))
            self.assertEqual(env.action_space.n, 10)
        finally:
            env.close()
            reference.close()

    def test_legacy_protocol_keeps_old_physics_and_new_version_is_distinct(self):
        legacy = EnvironmentFactory({"environment_id": "PlatformerState-v1"})
        candidate = EnvironmentFactory({"environment_id": "PlatformerState-v2"})
        self.assertEqual(legacy.settings.physics.jump_velocity, -10.5)
        self.assertEqual(legacy.settings.physics.rising_gravity, 0.3)
        self.assertEqual(legacy.settings.physics.falling_gravity, 1.0)
        self.assertNotEqual(
            protocol_hash(legacy.protocol([])), protocol_hash(candidate.protocol([]))
        )
        self.assertEqual(
            legacy.protocol([])["observation_indices"],
            candidate.protocol([])["observation_indices"],
        )

    def test_terminal_audit_rejects_inverted_rewards_with_new_physics(self):
        report = run_reward_exploit_audit(
            environment={
                "environment_id": "PlatformerState-v2",
                "reward": {"death_penalty": 10.0, "success_reward": -10.0},
            }
        )
        self.assertFalse(report.passed)
        self.assertFalse(report.checks["death_is_penalized"])
        self.assertFalse(report.checks["success_is_rewarded"])


if __name__ == "__main__":
    unittest.main()
