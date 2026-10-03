"""Variable jump semantics, collectible rewards and legacy compatibility."""

import unittest

import gymnasium as gym
import numpy as np

from ai_platformer.benchmark.gameplay import audit_settings, measure_jump
from ai_platformer.benchmark.readiness import run_reward_exploit_audit
from ai_platformer.content.legacy import LegacyLevelRepository
from ai_platformer.core import Action, BasicPlatformerCore, LevelDefinition, SolidRect
from ai_platformer.envs.factory import EnvironmentFactory, protocol_hash
from ai_platformer.envs.platformer_state_v2 import PlatformerStateEnvV2
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

    def test_release_reduces_height_and_repress_does_not_start_air_jump(self):
        held, released = self.core(), self.core()
        held.step(Action.JUMP)
        released.step(Action.JUMP)
        held.step(Action.JUMP)
        released.step(Action.NOOP)
        self.assertGreater(released.state.player.velocity_y, held.state.player.velocity_y)
        before = released.state.player.velocity_y
        released.step(Action.JUMP)
        self.assertGreater(released.state.player.velocity_y, before)
        self.assertFalse(released.state.player.grounded)

    def test_hold_cannot_retrigger_and_release_enables_next_jump(self):
        report = audit_settings(self.settings)
        self.assertEqual(report["continuous_hold"]["launches"], 1)
        self.assertTrue(all(report["checks"].values()))
        self.assertFalse(report["fixed_height"])
        heights = [measure_jump(self.settings, count)["height_px"] for count in (1, 4, 8, 300)]
        np.testing.assert_allclose(heights, [56.0, 83.2, 114.4, 183.6])
        self.assertEqual(measure_jump(self.settings, 600)["height_px"], heights[-1])

    def test_short_jump_can_collect_intro_acorns_with_one_shot_reward(self):
        coins = tuple(
            coin
            for coin in LegacyLevelRepository().load("level_1").collectibles
            if coin.entity_id.startswith("intro-coin")
        )
        level = LevelDefinition(
            "acorn_probe",
            2000,
            600,
            460,
            538,
            1900,
            (SolidRect(0, 538, 2000, 62),),
            collectibles=coins,
        )
        env = PlatformerStateEnvV2(level_id=level.level_id, level_loader=lambda _: level)
        try:
            env.reset(seed=0)
            collection_reward = 0.0
            for action in [Action.RIGHT] * 5 + [Action.RIGHT_JUMP] + [Action.RIGHT] * 23:
                _, _, _, _, info = env.step(int(action))
                collection_reward += info["reward_components"]["coin"]
            self.assertEqual(env.core.state.metadata["coins_collected"], 5)
            self.assertEqual(env.core.state.metadata["score"], 500)
            self.assertEqual(collection_reward, 5 * env.reward_config.coin_reward)
            for _ in range(10):
                _, _, _, _, info = env.step(int(Action.LEFT))
                self.assertEqual(info["reward_components"]["coin"], 0.0)
        finally:
            env.close()

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
