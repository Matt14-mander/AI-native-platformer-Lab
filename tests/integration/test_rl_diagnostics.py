"""Regress policy controls, fresh weight transfer and held-out course contracts."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from tests.integration.test_rl_training import ROOT, TRAINING_AVAILABLE

if TRAINING_AVAILABLE:
    import numpy as np
    import torch
    from stable_baselines3 import PPO

    from ai_platformer.agents.ppo.curriculum import CurriculumState
    from ai_platformer.agents.ppo.diagnostics import DiagnosticPolicyAgent, diagnose_policy
    from ai_platformer.agents.ppo.training import checkpoint_metadata, evaluate_policy, train_ppo
    from ai_platformer.agents.scripted import MoveRightAgent, RuleJumpAgent
    from ai_platformer.benchmark.scripted import evaluate_scripted_agent
    from ai_platformer.content.curriculum import TrainingLevelRepository, content_hash
    from ai_platformer.content.curriculum_v2 import build_manifest
    from ai_platformer.envs.factory import EnvironmentFactory


@unittest.skipUnless(TRAINING_AVAILABLE, "PPO training dependencies are not installed")
class DiagnosticContractTests(unittest.TestCase):
    def test_v2_reproducibility_disjoint_geometry_and_rule_reachability(self):
        repo = TrainingLevelRepository("config/curriculum_v2.json")
        self.assertEqual(repo.manifest, build_manifest())
        hashes = [content_hash(level) for level in repo.levels.values()]
        self.assertEqual(len(hashes), len(set(hashes)))
        environment = {
            "curriculum_manifest": "config/curriculum_v2.json",
            "environment_id": "PlatformerState-v1",
        }
        for task in ("gap", "mixed"):
            reference_strata = None
            for suite in ("train", "validation", "test", "ood"):
                levels = repo.split(task, suite)
                self.assertEqual(len(levels), 12 if suite == "ood" else 36)
                rule = evaluate_scripted_agent(
                    "rule",
                    RuleJumpAgent,
                    [100],
                    environment=environment,
                    max_steps=256,
                    level_ids=levels,
                )
                self.assertEqual(rule.summary()["success_rate"], 1, (task, suite))
                if suite != "ood":
                    offset = repo.manifest["distribution"]["split_offsets"][suite]
                    strata = set()
                    for level in levels:
                        spec = repo.manifest["levels"][level]
                        strata.add(
                            (
                                spec["spawn_x"] - offset[0],
                                spec["gap_x"] - spec["spawn_x"] - offset[1],
                                spec["gap_width"] - offset[2],
                            )
                        )
                    self.assertEqual(len(strata), 36)
                    if reference_strata is None:
                        reference_strata = strata
                    self.assertEqual(strata, reference_strata)

    def test_jump_events_inside_action_repeat_and_landing_are_calibrated(self):
        environment = {
            "environment_id": "PlatformerState-v1",
            "episode_step_limit": 256,
            "curriculum_manifest": "config/curriculum_v2.json",
        }
        factory = EnvironmentFactory(environment)
        levels = ["v2_gap_train_00", "v2_gap_train_11"]
        diagnostic = diagnose_policy(
            None,
            factory=factory,
            level_ids=levels,
            seeds=[100],
            modes=("deterministic",),
            agent_factory=RuleJumpAgent,
        )["modes"]["deterministic"]
        ordinary = evaluate_scripted_agent(
            "rule", RuleJumpAgent, [100], environment=environment, level_ids=levels
        ).to_dict()
        self.assertEqual(diagnostic["episodes"], ordinary["episodes"])
        self.assertEqual(diagnostic["behavior"]["landed_after_gap_rate"], 1)
        for item in diagnostic["diagnostics"]:
            self.assertGreaterEqual(len(item["jumps"]), 1)
            self.assertGreater(item["first_jump_edge_distance"], 0)
            self.assertLessEqual(item["first_jump_edge_distance"], 120)
            self.assertGreater(item["landings"][0]["tick"], item["jumps"][0]["tick"])
        walk = diagnose_policy(
            None,
            factory=factory,
            level_ids=levels,
            seeds=[100],
            modes=("deterministic",),
            agent_factory=MoveRightAgent,
        )
        self.assertEqual(walk["modes"]["deterministic"]["behavior"]["mean_effective_jumps"], 0)

    def test_ground_edge_sensor_matches_geometry_before_and_during_jump(self):
        factory = EnvironmentFactory(
            {
                "curriculum_manifest": "config/curriculum_v2.json",
                "environment_id": "PlatformerState-v1",
            }
        )
        env = factory.make(level_id="v2_gap_train_00")
        try:
            obs, _ = env.reset(seed=100)
            for _ in range(12):
                distance = (
                    env.level.solids[0].right
                    - env.core.state.player.x
                    - factory.settings.physics.player_width
                )
                self.assertAlmostEqual(float(obs[7]), min(240, max(0, distance)) / 240, places=6)
                obs, *_ = env.step(6)
            obs, *_ = env.step(9)
            self.assertEqual(obs[4], 0)
            self.assertEqual(
                obs[7], 0
            )  # v1 reports support at current feet height, not projected ground.
        finally:
            env.close()

    def test_policy_sampling_is_paired_reproducible_and_does_not_use_global_rng(self):
        environment = {"environment_id": "PlatformerState-v1", "episode_step_limit": 64}
        factory = EnvironmentFactory(environment)
        env = factory.make(level_id="course_gap_00")
        try:
            model = PPO("MlpPolicy", env, n_steps=16, batch_size=16, seed=7, device="cpu")
            observation, _ = env.reset(seed=100)
            greedy = DiagnosticPolicyAgent(model, "deterministic")
            greedy.reset(seed=200)
            expected, _ = model.predict(observation, deterministic=True)
            self.assertEqual(greedy.act(observation), int(expected))
            fixed = DiagnosticPolicyAgent(model, "fixed_observation")
            reference = DiagnosticPolicyAgent(model, "stochastic")
            fixed.reset(seed=200)
            reference.reset(seed=200)
            global_rng = torch.get_rng_state().clone()
            for i in range(20):
                changed = observation if i == 0 else np.zeros_like(observation)
                self.assertEqual(fixed.act(changed), reference.act(observation))
            self.assertTrue(torch.equal(global_rng, torch.get_rng_state()))
            self.assertEqual(set(fixed.total_variations), {0})
            report = diagnose_policy(
                model, factory=factory, level_ids=["course_gap_00"], seeds=[200, 201]
            )
            self.assertEqual(
                report,
                diagnose_policy(
                    model, factory=factory, level_ids=["course_gap_00"], seeds=[200, 201]
                ),
            )
            self.assertEqual(
                report["modes"]["deterministic"]["episodes"],
                evaluate_policy(
                    model, environment=environment, level_ids=["course_gap_00"], seeds=[200, 201]
                )["episodes"],
            )
        finally:
            env.close()

    def test_speed_gate_rejects_slow_success_and_checks_retention(self):
        stages = [
            {
                "task": task,
                "success_threshold": 0.8,
                "validation_levels": [task],
                "max_success_steps": 160,
            }
            for task in ("obstacle", "gap")
        ]
        state = CurriculumState(stages, required_evaluations=1)
        state.select_stage("gap", 0)
        report = {
            "by_level": {task: {"success_rate": 1} for task in ("obstacle", "gap")},
            "episodes": [
                {"level_id": task, "outcome": "success", "steps": steps}
                for task, steps in (("obstacle", 928), ("gap", 64))
            ],
        }
        self.assertFalse(state.observe(report, 100))
        report["episodes"][0]["steps"] = 64
        self.assertTrue(state.observe(report, 200))

    def test_init_from_copies_weights_but_resets_optimizer_steps_and_curriculum(self):
        config = json.loads((ROOT / "config/ppo_curriculum_v1.json").read_text())
        config.update(total_timesteps=128, n_envs=2, verbose=0)
        config["algorithm"].update(n_steps=32, batch_size=32, n_epochs=1)
        config["environment"]["episode_step_limit"] = 16
        config["evaluation"].update(seeds=[100], every_timesteps=1000)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            train_ppo(config, source, start_stage="gap")
            source_model = PPO.load(source / "model.zip", device="cpu")
            self.assertTrue(source_model.policy.optimizer.state)
            target = deepcopy(config)
            target.update(total_timesteps=64, seed=99)
            target["environment"]["curriculum_manifest"] = "config/curriculum_v2.json"
            target["algorithm"]["ent_coef"] = 0.02
            learn = PPO.learn
            inspected = []

            def inspect_initialization(model, *args, **kwargs):
                self.assertEqual(model.num_timesteps, 0)
                self.assertFalse(model.policy.optimizer.state)
                for key, weights in source_model.policy.state_dict().items():
                    self.assertTrue(torch.equal(weights, model.policy.state_dict()[key]), key)
                inspected.append(True)
                return learn(model, *args, **kwargs)

            with patch.object(PPO, "learn", inspect_initialization):
                result = train_ppo(
                    target, root / "initialized", init_from=source / "model.zip", start_stage="gap"
                )
            self.assertEqual(inspected, [True])
            self.assertEqual(result["initial_timesteps"], 0)
            self.assertEqual(result["added_timesteps"], 64)
            self.assertEqual(result["curriculum_state"]["stage_start"], 0)
            self.assertEqual(result["curriculum_state"]["history"][0]["timesteps"], 0)
            self.assertEqual(result["initialization"]["source_timesteps"], 128)
            saved = checkpoint_metadata(root / "initialized/model.zip")
            self.assertEqual(saved["initialization"], result["initialization"])
            continued = train_ppo(
                result["config"], root / "continued", resume=root / "initialized/model.zip"
            )
            self.assertEqual(continued["initialization"], result["initialization"])
            for name, options in (
                ("both", {"resume": source / "model.zip", "init_from": source / "model.zip"}),
                ("strict_resume", {"resume": source / "model.zip"}),
            ):
                with self.assertRaises(ValueError):
                    train_ppo(target, root / name, **options)
                self.assertFalse((root / name).exists())
            target["environment"]["sensor_range"] = 180
            with self.assertRaisesRegex(ValueError, "incompatible"):
                train_ppo(target, root / "incompatible", init_from=source / "model.zip")
            self.assertFalse((root / "incompatible").exists())
