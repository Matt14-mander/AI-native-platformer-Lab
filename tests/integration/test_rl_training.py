"""Regress protocol fairness, stage transitions, and real PPO persistence."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

try:
    import stable_baselines3  # noqa: F401 -- verify the optional training runtime imports
except ImportError:
    TRAINING_AVAILABLE = False
else:
    TRAINING_AVAILABLE = True
    from ai_platformer.agents.ppo.configuration import resolve_training_config
    from ai_platformer.agents.ppo.curriculum import CourseSampler, CurriculumState
    from ai_platformer.agents.ppo.training import evaluate_policy, train_ppo
    from ai_platformer.agents.scripted import MoveRightAgent, RuleJumpAgent
    from ai_platformer.benchmark.scripted import evaluate_scripted_agent
    from ai_platformer.content.curriculum import TrainingLevelRepository, content_hash
    from ai_platformer.envs.factory import EnvironmentFactory

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(TRAINING_AVAILABLE, "PPO training dependencies are not installed")
class CourseProtocolTests(unittest.TestCase):
    def test_v1_exposes_jump_latch_and_preserves_v0_features(self):
        import gymnasium as gym
        import numpy as np
        from gymnasium.utils.env_checker import check_env

        factory = EnvironmentFactory(
            {"environment_id": "PlatformerState-v1", "episode_step_limit": 128}
        )
        env = gym.make(
            "PlatformerState-v1",
            level_id="course_flat_00",
            level_loader=factory.repository.load,
            episode_step_limit=128,
        ).unwrapped
        old = EnvironmentFactory({"episode_step_limit": 128}).make(level_id="course_flat_00")
        try:
            check_env(env)
            obs, _ = env.reset(seed=7)
            old_obs, _ = old.reset(seed=7)
            self.assertEqual(obs.shape, (15,))
            self.assertEqual(obs.dtype, np.float32)
            self.assertEqual(obs[14], 0)
            np.testing.assert_array_equal(obs[:14], old_obs)
            for _ in range(25):
                obs, *_ = env.step(3)
                old_obs, *_ = old.step(3)
                np.testing.assert_array_equal(obs[:14], old_obs)
            self.assertEqual(obs[14], 1)
            self.assertEqual(obs[4], 1)  # Holding jump after landing cannot retrigger it.
            obs, *_ = env.step(0)
            self.assertEqual(obs[14], 0)
            obs, *_ = env.step(3)
            self.assertEqual(obs[14], 1)
            self.assertLess(obs[3], 0)  # Release then press starts a new jump.
            obs, _ = env.reset(seed=7)
            self.assertEqual(obs[14], 0)
        finally:
            env.close()
            old.close()

    def test_course_splits_are_unique_and_all_geometry_is_playable(self):
        repo = TrainingLevelRepository()
        hashes = [content_hash(level) for level in repo.levels.values()]
        self.assertEqual(len(hashes), len(set(hashes)))
        for task, splits in repo.manifest["splits"].items():
            self.assertEqual(
                [len(splits[key]) for key in ("train", "validation", "test")], [8, 4, 4]
            )
            levels = [level for ids in splits.values() for level in ids]
            rule = evaluate_scripted_agent(
                "rule", RuleJumpAgent, [100], max_steps=256, level_ids=levels
            )
            right = evaluate_scripted_agent(
                "right", MoveRightAgent, [100], max_steps=256, level_ids=levels
            )
            self.assertEqual(rule.summary()["success_rate"], 1.0, task)
            self.assertEqual(right.summary()["success_rate"], float(task == "flat"), task)

    def test_scripted_and_policy_share_observation_and_timeout(self):
        class TimeAwareAgent:
            def reset(self, *, seed):
                pass

            def act(self, obs):
                return 6 if obs[13] > 0.5 else 0

        class FakeModel:
            def predict(self, obs, deterministic):
                self_outer.assertTrue(deterministic)
                return TimeAwareAgent().act(obs), None

        self_outer = self
        environment = {"episode_step_limit": 20, "action_repeat": 2, "sensor_range": 180}
        levels = ["course_obstacle_00"]
        scripted = evaluate_scripted_agent(
            "scripted", TimeAwareAgent, [100], environment=environment, level_ids=levels
        ).to_dict()
        policy = evaluate_policy(
            FakeModel(), seeds=[100], environment=environment, level_ids=levels
        )
        self.assertEqual(scripted["episodes"], policy["episodes"])
        self.assertEqual(policy["episodes"][0]["steps"], 20)
        self.assertEqual(policy["episodes"][0]["outcome"], "time_limit")

    def test_sampler_changes_level_only_at_reset_and_uses_training_split(self):
        repo = TrainingLevelRepository()
        stages = [{"train_levels": repo.split(task, "train")} for task in ("flat", "gap")]
        factory = EnvironmentFactory({"episode_step_limit": 128})
        sampler = CourseSampler(
            factory.make(level_id=stages[0]["train_levels"][0]),
            stages=stages,
            seeds=[0, 1],
            rng_seed=7,
            replay_fraction=0.2,
        )
        _, old = sampler.reset()
        sampler.stage_index = 1
        self.assertEqual(sampler.unwrapped.core.state.level_id, old["level_id"])
        selected = set()
        for _ in range(100):
            _, info = sampler.reset()
            selected.add(info["level_id"])
            self.assertIn(info["seed"], [0, 1])
        self.assertTrue(selected & set(stages[0]["train_levels"]))
        self.assertTrue(selected & set(stages[1]["train_levels"]))
        self.assertTrue(selected <= {level for stage in stages for level in stage["train_levels"]})
        sampler.close()

    def test_promotion_requires_consecutive_passes_and_retention(self):
        stages = [
            {
                "task": task,
                "success_threshold": 0.8,
                "validation_levels": [task],
                "max_timesteps": 100,
            }
            for task in ("flat", "gap")
        ]
        state = CurriculumState(stages, required_evaluations=2)
        good = {"by_level": {"flat": {"success_rate": 1}, "gap": {"success_rate": 1}}}
        bad = deepcopy(good)
        bad["by_level"]["flat"]["success_rate"] = 0
        self.assertFalse(state.observe(good, 10))
        self.assertFalse(state.observe(bad, 20))
        self.assertFalse(state.observe(good, 30))
        self.assertTrue(state.observe(good, 40))
        self.assertEqual(state.data["stage_index"], 1)
        self.assertFalse(state.observe(bad, 50))
        self.assertFalse(state.observe(good, 60))
        self.assertTrue(state.observe(good, 70))
        self.assertEqual(state.data["status"], "completed")

    def test_manual_stage_selection_is_audited_without_claiming_mastery(self):
        stages = [{"task": task} for task in ("flat", "obstacle", "gap")]
        state = CurriculumState(stages, required_evaluations=2)
        state.data["consecutive_passes"] = 1
        state.select_stage("gap", 8192)
        self.assertEqual(state.data["stage_index"], 2)
        self.assertEqual(state.data["stage_start"], 8192)
        self.assertEqual(state.data["consecutive_passes"], 0)
        self.assertEqual(
            state.data["history"],
            [
                {
                    "outcome": "manual_stage_change",
                    "from": "flat",
                    "to": "gap",
                    "timesteps": 8192,
                    "reason": "explicit_start_stage",
                }
            ],
        )
        with self.assertRaises(ValueError):
            state.select_stage("unknown", 9000)
        self.assertEqual(state.data["stage_index"], 2)

    def test_reject_invalid_seed_pools_and_rollout_configuration(self):
        config = json.loads((ROOT / "config/ppo_state_v0.json").read_text())
        for patch in (
            {"evaluation": {"seeds": []}},
            {"train_seeds": [100]},
            {"total_timesteps": 0},
        ):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                resolve_training_config({**config, **patch})
        config["algorithm"]["batch_size"] = 300
        with self.assertRaises(ValueError):
            resolve_training_config(config)


@unittest.skipUnless(TRAINING_AVAILABLE, "PPO training dependencies are not installed")
class CheckpointIntegrationTests(unittest.TestCase):
    def test_curriculum_budget_stops_training_before_next_stage(self):
        config = json.loads((ROOT / "config/ppo_curriculum_v1.json").read_text())
        config.update(total_timesteps=128, n_envs=2, verbose=0)
        config["algorithm"].update(n_steps=32, batch_size=32, n_epochs=1)
        config["environment"]["episode_step_limit"] = 16
        config["curriculum"]["stages"][0]["max_timesteps"] = 8
        with tempfile.TemporaryDirectory() as directory:
            report = train_ppo(config, Path(directory))
            self.assertEqual(report["trained_timesteps"], 8)
            self.assertEqual(report["curriculum_state"]["status"], "budget_exhausted")
            self.assertEqual(report["curriculum_state"]["stage_index"], 0)
            with self.assertRaisesRegex(ValueError, "requires"):
                train_ppo(config, Path(directory) / "no_resume", start_stage="gap")
            self.assertFalse((Path(directory) / "no_resume").exists())
            config["total_timesteps"] = 64
            continued = Path(directory) / "gap"
            result = train_ppo(
                config, continued, resume=Path(directory) / "model.zip", start_stage="gap"
            )
            self.assertEqual(result["initial_timesteps"], 8)
            self.assertEqual(result["curriculum_state"]["stage_index"], 2)
            self.assertEqual(result["curriculum_state"]["stage_start"], 8)
            self.assertEqual(result["requested_start_stage"], "gap")
            self.assertNotIn(
                "mastered", [item["outcome"] for item in result["curriculum_state"]["history"]]
            )
            logs = "\n".join(path.read_text() for path in (continued / "monitor").glob("*.csv"))
            self.assertIn("course_gap_", logs)

    def test_real_training_evaluation_reload_resume_and_compatibility(self):
        from stable_baselines3 import PPO

        config = json.loads((ROOT / "config/ppo_state_v0.json").read_text())
        config.update(total_timesteps=128, n_envs=2, verbose=0, checkpoint_every_timesteps=64)
        config["algorithm"].update(n_steps=32, batch_size=32, n_epochs=1)
        config["environment"].update(level_id="course_obstacle_00", episode_step_limit=16)
        config["evaluation"] = {"seeds": [100], "every_timesteps": 64}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "first"
            report = train_ppo(config, output)
            self.assertEqual(report["trained_timesteps"], 128)
            self.assertGreaterEqual(len(report["evaluation_history"]), 2)
            self.assertTrue((output / "checkpoints/best_baseline.zip").is_file())
            self.assertTrue((output / "checkpoints/step_64.json").is_file())
            model = PPO.load(output / "model.zip", device="cpu")
            less_evaluation = deepcopy(config)
            less_evaluation["evaluation"]["every_timesteps"] = 1000
            comparison = Path(directory) / "less_evaluation"
            train_ppo(less_evaluation, comparison)
            comparison_model = PPO.load(comparison / "model.zip", device="cpu")
            import torch

            for key, weights in model.policy.state_dict().items():
                self.assertTrue(
                    torch.equal(weights, comparison_model.policy.state_dict()[key]), key
                )
            reloaded = evaluate_policy(
                model,
                seeds=[100],
                environment=report["config"]["environment"],
                level_ids=["course_obstacle_00"],
            )
            self.assertEqual(reloaded, report["evaluation"])
            resumed = train_ppo(
                report["config"], Path(directory) / "resumed", resume=output / "model.zip"
            )
            self.assertEqual(resumed["initial_timesteps"], 128)
            self.assertEqual(resumed["trained_timesteps"], 256)
            incompatible = deepcopy(config)
            incompatible["environment"]["action_repeat"] = 2
            with self.assertRaisesRegex(ValueError, "incompatible"):
                train_ppo(incompatible, Path(directory) / "invalid", resume=output / "model.zip")
            self.assertFalse((Path(directory) / "invalid").exists())
            incompatible = deepcopy(config)
            incompatible["environment_id"] = "PlatformerState-v1"
            with self.assertRaisesRegex(ValueError, "incompatible"):
                train_ppo(
                    incompatible, Path(directory) / "wrong_version", resume=output / "model.zip"
                )
            with self.assertRaises(FileExistsError):
                train_ppo(config, output)
