"""Playback must preserve the evaluation environment's observation/tick contract."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ai_platformer.core import Action
from ai_platformer.envs.factory import EnvironmentFactory, protocol_hash
from ai_platformer.rendering.ppo_playback import PlaybackSession, solids_for_playback


class SequencePolicy:
    def __init__(self):
        self.index = 0
        self.observations = []

    def predict(self, observation, *, deterministic):
        self.observations.append(observation.copy())
        action = (Action.RIGHT_RUN_JUMP, Action.RIGHT_RUN)[self.index % 2]
        self.index += 1
        return np.array(action), None


class PlaybackTests(unittest.TestCase):
    def setUp(self):
        self.factory = EnvironmentFactory(
            {
                "environment_id": "PlatformerState-v1",
                "curriculum_manifest": "config/curriculum_v4.json",
                "episode_step_limit": 80,
                "action_repeat": 4,
            }
        )
        self.level = self.factory.repository.split("gap", "validation")[0]

    def test_playback_matches_direct_environment_for_entire_episode(self):
        policy = SequencePolicy()
        session = PlaybackSession(policy, self.factory, level_id=self.level, seed=123)
        reference = self.factory.make(level_id=self.level, seed=123)
        observation, _ = reference.reset(seed=123)
        total_reward = 0.0
        try:
            while not session.done:
                session.step()
                np.testing.assert_array_equal(policy.observations[-1], observation)
                observation, reward, terminated, truncated, info = reference.step(
                    int(session.action)
                )
                total_reward += reward
                np.testing.assert_array_equal(session.observation, observation)
                self.assertEqual(session.env.core.state, reference.core.state)
                self.assertEqual(session.done, terminated or truncated)
                self.assertEqual(session.info, info)
            self.assertEqual(session.summary()["return"], total_reward)
            with self.assertRaises(RuntimeError):
                session.step()
            session.reset()
            observation, _ = reference.reset(seed=123)
            np.testing.assert_array_equal(session.observation, observation)
            self.assertEqual(session.env.core.state, reference.core.state)
            self.assertFalse(session.done)
            self.assertEqual(session.episode_return, 0)
        finally:
            session.env.close()
            reference.close()

    def test_course_and_legacy_render_groups_preserve_all_collision_solids(self):
        for task in ("flat", "obstacle", "gap", "mixed", "full"):
            level_id = (
                self.factory.level_id
                if task == "full"
                else self.factory.repository.split(task, "validation")[0]
            )
            with self.subTest(task=task):
                groups = solids_for_playback(self.factory, level_id)
                rendered = [solid for group in groups.values() for solid in group]
                self.assertCountEqual(rendered, self.factory.repository.load(level_id).solids)

    def test_window_controls_pause_replay_next_and_speed_without_changing_repeat(self):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        import pygame

        from scripts.play_ppo import main

        levels = self.factory.repository.split("gap", "validation")
        protocol = self.factory.protocol(levels)
        metadata = {
            "config": {
                "environment": {
                    "environment_id": "PlatformerState-v1",
                    "curriculum_manifest": "config/curriculum_v4.json",
                    "episode_step_limit": 80,
                    "action_repeat": 4,
                },
                "evaluation": {"seeds": [123]},
            },
            "signature": {"protocol": protocol},
            "saved_timesteps": 0,
        }
        probe = self.factory.make(level_id=levels[0])
        policy = SimpleNamespace(
            num_timesteps=0,
            observation_space=probe.observation_space,
            action_space=probe.action_space,
            predict=lambda observation, deterministic: (np.array(Action.RIGHT_RUN), None),
        )
        probe.close()
        sessions = []
        snapshots = []
        keys = {
            1: pygame.K_p,
            3: pygame.K_SPACE,
            4: pygame.K_r,
            5: pygame.K_n,
            6: pygame.K_RIGHTBRACKET,
            8: pygame.K_ESCAPE,
        }

        def create_session(*args, **kwargs):
            session = PlaybackSession(*args, **kwargs)
            sessions.append(session)
            return session

        def events():
            session = sessions[-1]
            frame = len(snapshots)
            snapshots.append((session.env.level_id, session.env.core.state.tick))
            return [pygame.event.Event(pygame.KEYDOWN, key=keys[frame])] if frame in keys else []

        with (
            patch("sys.argv", ["play_ppo", "--model", "unused.zip"]),
            patch("ai_platformer.agents.ppo.training.checkpoint_metadata", return_value=metadata),
            patch("stable_baselines3.PPO.load", return_value=policy),
            patch(
                "ai_platformer.rendering.ppo_playback.PlaybackSession", side_effect=create_session
            ),
            patch("pygame.time.Clock", return_value=SimpleNamespace(tick=lambda fps: 67)),
            patch("pygame.event.get", side_effect=events),
        ):
            main()
        self.assertEqual(protocol_hash(protocol), protocol_hash(self.factory.protocol(levels)))
        self.assertEqual([tick for _, tick in snapshots], [0, 4, 4, 4, 4, 0, 0, 0, 8])
        self.assertEqual(snapshots[5][0], levels[0])
        self.assertEqual(snapshots[6][0], levels[1])
        self.assertTrue(all(session.env.action_repeat == 4 for session in sessions))


if __name__ == "__main__":
    unittest.main()
