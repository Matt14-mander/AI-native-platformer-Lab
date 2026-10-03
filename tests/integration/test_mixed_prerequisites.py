"""Mixed entry must retain earlier tasks and save exactly the accepted policy."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from tests.integration.test_rl_training import ROOT, TRAINING_AVAILABLE

if TRAINING_AVAILABLE:
    from stable_baselines3 import PPO

    from ai_platformer.agents.ppo.configuration import resolve_training_config
    from ai_platformer.agents.ppo.curriculum import CourseSampler, CurriculumState
    from ai_platformer.agents.ppo.gates import assess_prerequisites, joint_selection_score
    from ai_platformer.agents.ppo.training import train_ppo
    from ai_platformer.agents.scripted import RuleJumpAgent
    from ai_platformer.benchmark.scripted import evaluate_scripted_agent
    from ai_platformer.content.curriculum import TrainingLevelRepository
    from ai_platformer.content.curriculum_v3 import build_manifest
    from ai_platformer.content.curriculum_v4 import build_manifest as build_long_gap_manifest
    from ai_platformer.envs.factory import EnvironmentFactory


@unittest.skipUnless(TRAINING_AVAILABLE, "training dependencies missing")
class MixedPrerequisiteTests(unittest.TestCase):
    def test_imitation_updates_only_actor_and_preserves_rng_and_ppo_optimizer(self):
        import torch

        from ai_platformer.agents.ppo.imitation import warm_start_policy

        factory = EnvironmentFactory(
            {"environment_id": "PlatformerState-v1", "episode_step_limit": 64}
        )
        env = factory.make(level_id="course_flat_00")
        try:
            model = PPO("MlpPolicy", env, n_steps=16, batch_size=16, seed=7)
            before = {key: value.clone() for key, value in model.policy.state_dict().items()}
            rng = torch.get_rng_state().clone()
            options = {
                "rounds": 2,
                "epochs": 2,
                "batch_size": 32,
                "learning_rate": 0.001,
                "max_steps_per_episode": 32,
            }
            report = warm_start_policy(
                model,
                factory=factory,
                stages=[{"train_levels": ["course_flat_00", "course_obstacle_00"]}],
                options=options,
                seeds=[0],
                seed=7,
            )
            self.assertEqual(report["levels"], ["course_flat_00", "course_obstacle_00"])
            self.assertEqual(model.num_timesteps, 0)
            self.assertFalse(model.policy.optimizer.state)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            for key, value in before.items():
                if "value_net" in key:
                    self.assertTrue(torch.equal(value, model.policy.state_dict()[key]), key)
            self.assertTrue(
                any(
                    not torch.equal(value, model.policy.state_dict()[key])
                    for key, value in before.items()
                    if "action_net" in key
                )
            )
        finally:
            env.close()

    def test_imitation_provenance_survives_resume_without_repeating_supervision(self):
        from unittest.mock import patch

        config = json.loads((ROOT / "config/ppo_state_v0.json").read_text())
        config.update(total_timesteps=16, n_envs=1, verbose=0)
        config["algorithm"].update(n_steps=16, batch_size=16, n_epochs=1)
        config["environment"].update(level_id="course_flat_00", episode_step_limit=16)
        config["imitation"] = {
            "rounds": 1,
            "epochs": 1,
            "batch_size": 32,
            "learning_rate": 0.001,
            "max_steps_per_episode": 8,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = train_ppo(config, root / "first")
            self.assertEqual(report["imitation"]["levels"], ["course_flat_00"])
            with patch(
                "ai_platformer.agents.ppo.training.warm_start_policy",
                side_effect=AssertionError("repeated"),
            ):
                resumed = train_ppo(
                    report["config"], root / "resumed", resume=root / "first/model.zip"
                )
            self.assertEqual(resumed["imitation"], report["imitation"])
            invalid = deepcopy(report["config"])
            invalid["imitation"]["rounds"] = 2
            with self.assertRaisesRegex(ValueError, "incompatible"):
                train_ppo(invalid, root / "invalid", resume=root / "first/model.zip")
            self.assertFalse((root / "invalid").exists())

    def test_joint_ranking_rejects_forgetting_and_slow_success(self):
        stages = [
            {
                "task": task,
                "validation_levels": [task],
                "success_threshold": 0.8,
                "max_success_steps": 100,
            }
            for task in ("flat", "obstacle", "gap")
        ]
        report = {
            "by_level": {
                t: {"success_rate": 1, "mean_progress": 1} for t in ("flat", "obstacle", "gap")
            },
            "episodes": [
                {"level_id": t, "outcome": "success", "steps": 30}
                for t in ("flat", "obstacle", "gap")
            ],
        }
        good = joint_selection_score(stages, report)
        slow = deepcopy(report)
        slow["episodes"][1]["steps"] = 928
        self.assertFalse(assess_prerequisites(stages, slow)["passed"])
        self.assertGreater(good, joint_selection_score(stages, slow))
        forgotten = deepcopy(report)
        forgotten["by_level"]["obstacle"]["success_rate"] = 0
        self.assertGreater(good, joint_selection_score(stages, forgotten))
        state = CurriculumState(stages + [{"task": "mixed"}], required_evaluations=2)
        state.select_stage("gap", 0)
        self.assertFalse(state.observe(report, 10))
        self.assertFalse(state.observe(forgotten, 20))
        self.assertFalse(state.observe(report, 30))
        self.assertTrue(state.observe(report, 40))
        self.assertEqual(state.stage["task"], "mixed")
        self.assertEqual(state.data["history"][-1]["validation_timesteps"], [30, 40])

    def test_consecutive_acceptance_evidence_survives_resume(self):
        stages = [
            {"task": "gap", "validation_levels": ["gap"], "success_threshold": 1},
            {"task": "mixed"},
        ]
        report = {"by_level": {"gap": {"success_rate": 1}}}
        state = CurriculumState(stages, required_evaluations=2)
        self.assertFalse(state.observe(report, 8192))
        restored = CurriculumState(stages, required_evaluations=2, saved=deepcopy(state.data))
        self.assertTrue(restored.observe(report, 16384))
        self.assertEqual(restored.data["history"][-1]["validation_timesteps"], [8192, 16384])

    def test_weighted_sampler_uses_only_allowed_training_courses(self):
        config = json.loads((ROOT / "config/ppo_mixed_prerequisites_v3.json").read_text())
        _, factory, stages = resolve_training_config(config)
        sampler = CourseSampler(
            factory.make(), stages=stages, seeds=[0, 1], rng_seed=7, replay_fraction=0.2
        )
        sampler.stage_index = 2
        counts = {task: 0 for task in ("flat", "obstacle", "gap")}
        allowed = {level: stage["task"] for stage in stages[:3] for level in stage["train_levels"]}
        for _ in range(1000):
            _, info = sampler.reset()
            self.assertIn(info["level_id"], allowed)
            counts[allowed[info["level_id"]]] += 1
        sampler.close()
        self.assertTrue(50 < counts["flat"] < 150, counts)
        self.assertTrue(380 < counts["obstacle"] < 520, counts)
        self.assertTrue(380 < counts["gap"] < 520, counts)
        for weights in ({"mixed": 1, "gap": 1}, {"gap": 0}, {"gap": float("nan")}, {"obstacle": 1}):
            invalid = deepcopy(config)
            invalid["curriculum"]["stages"][2]["sampling_weights"] = weights
            with self.assertRaises(ValueError):
                resolve_training_config(invalid)

    def test_v3_preserves_all_heldout_assignments_and_is_playable(self):
        old = TrainingLevelRepository("config/curriculum_v2.json")
        new = TrainingLevelRepository("config/curriculum_v3.json")
        self.assertEqual(new.manifest, build_manifest())
        for task, suites in old.manifest["splits"].items():
            for suite, ids in suites.items():
                self.assertTrue(set(ids) <= set(new.split(task, suite)))
                for level in ids:
                    self.assertEqual(old.load(level), new.load(level))
        levels = [level for ids in new.manifest["splits"]["obstacle"].values() for level in ids]
        result = evaluate_scripted_agent(
            "rule",
            RuleJumpAgent,
            [100],
            environment={"curriculum_manifest": "config/curriculum_v3.json"},
            level_ids=levels,
            max_steps=256,
        )
        self.assertEqual(result.summary()["success_rate"], 1)

    def test_long_gap_bridge_preserves_ood_and_heldout_splits(self):
        old = TrainingLevelRepository("config/curriculum_v3.json")
        new = TrainingLevelRepository("config/curriculum_v4.json")
        self.assertEqual(new.manifest, build_long_gap_manifest())
        for task, suites in old.manifest["splits"].items():
            for suite, ids in suites.items():
                self.assertTrue(set(ids) <= set(new.split(task, suite)))
                for level in ids:
                    self.assertEqual(old.load(level), new.load(level))
        self.assertEqual(old.split("gap", "ood"), new.split("gap", "ood"))
        train_distances = [
            new.manifest["levels"][level]["gap_x"]
            - new.manifest["levels"][level].get("spawn_x", 48)
            for level in new.split("gap", "train")
        ]
        ood_distances = [
            new.manifest["levels"][level]["gap_x"] - new.manifest["levels"][level]["spawn_x"]
            for level in new.split("gap", "ood")
        ]
        self.assertLess(max(train_distances), min(ood_distances))
        levels = [level for level in new.levels if level.startswith("v4_")]
        result = evaluate_scripted_agent(
            "rule",
            RuleJumpAgent,
            [100],
            environment={"curriculum_manifest": "config/curriculum_v4.json"},
            level_ids=levels,
            max_steps=256,
        )
        self.assertEqual(result.summary()["success_rate"], 1)

    def test_stop_at_promotion_preserves_policy_and_can_resume(self):
        from unittest.mock import patch

        config = json.loads((ROOT / "config/ppo_mixed_prerequisites_v3.json").read_text())
        config.update(total_timesteps=64, n_envs=1)
        config["algorithm"].update(n_steps=16, batch_size=16, n_epochs=1)
        config["evaluation"].update(every_timesteps=16, seeds=[100])
        config["environment"]["episode_step_limit"] = 16

        # Isolate the lifecycle from learning ability: deterministic synthetic acceptance report.
        def accepted(model, *, level_ids, **kwargs):
            return {
                "by_level": {level: {"success_rate": 1, "mean_progress": 1} for level in level_ids},
                "episodes": [
                    {"level_id": level, "outcome": "success", "steps": 10, "progress": 1}
                    for level in level_ids
                ],
            }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "ready"
            with patch("ai_platformer.agents.ppo.training.evaluate_policy", accepted):
                result = train_ppo(config, output, start_stage="gap", stop_before_stage="mixed")
            self.assertEqual(result["trained_timesteps"], 32)
            self.assertEqual(result["curriculum_state"]["status"], "ready_for_stage")
            self.assertEqual(result["curriculum_state"]["stage_index"], 3)
            self.assertEqual(result["curriculum_state"]["readiness"]["target"], "mixed")
            from scripts.verify_stage_entry import verify_stage_entry

            with patch("scripts.verify_stage_entry.evaluate_policy", accepted):
                verification = verify_stage_entry(output / "model.zip", "mixed")
            self.assertTrue(verification["accepted"])
            self.assertEqual(verification["recorded_evidence"]["evaluation_timesteps"], [16, 32])
            self.assertTrue(all("mixed" not in level for level in result["evaluation"]["by_level"]))
            model = PPO.load(output / "model.zip", device="cpu")
            promoted = PPO.load(output / "checkpoints/stage_32.zip", device="cpu")
            import torch

            for key, value in model.policy.state_dict().items():
                self.assertTrue(torch.equal(value, promoted.policy.state_dict()[key]))
            with patch("ai_platformer.agents.ppo.training.evaluate_policy", accepted):
                continued = train_ppo(
                    result["config"], Path(directory) / "continued", resume=output / "model.zip"
                )
            self.assertGreater(continued["trained_timesteps"], 32)
            with patch("scripts.verify_stage_entry.evaluate_policy", accepted):
                self.assertFalse(
                    verify_stage_entry(Path(directory) / "continued/model.zip", "mixed")["accepted"]
                )
            for change in ("selection", "sampling"):
                incompatible = deepcopy(result["config"])
                if change == "selection":
                    incompatible["evaluation"]["selection"] = "current_task"
                else:
                    incompatible["curriculum"]["stages"][2]["sampling_weights"]["gap"] = 0.6
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    train_ppo(incompatible, Path(directory) / change, resume=output / "model.zip")
                self.assertFalse((Path(directory) / change).exists())
            with self.assertRaises(ValueError):
                train_ppo(
                    config,
                    Path(directory) / "bypass",
                    start_stage="mixed",
                    stop_before_stage="mixed",
                )
            self.assertFalse((Path(directory) / "bypass").exists())
