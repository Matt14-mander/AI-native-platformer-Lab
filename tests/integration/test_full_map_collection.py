"""Full-map splits, collection gates and backward-compatible curriculum resolution."""

import json
import unittest
from copy import deepcopy
from pathlib import Path

from ai_platformer.agents.ppo.configuration import resolve_training_config
from ai_platformer.agents.ppo.gates import assess_prerequisites, joint_selection_score
from ai_platformer.agents.scripted.coin_jump import CoinJumpAgent
from ai_platformer.content.curriculum import TrainingLevelRepository, content_hash
from ai_platformer.content.curriculum_v7 import build_manifest
from ai_platformer.content.curriculum_v8 import build_manifest as build_v8_manifest
from ai_platformer.content.curriculum_v9 import build_manifest as build_v9_manifest
from ai_platformer.envs.factory import EnvironmentFactory


class FullCollectionTests(unittest.TestCase):
    def test_manifest_preserves_old_content_and_full_pools_are_disjoint(self):
        old = TrainingLevelRepository("config/curriculum_v6.json")
        new = TrainingLevelRepository("config/curriculum_v7.json")
        self.assertEqual(new.manifest, build_manifest())
        for level in old.levels:
            self.assertEqual(content_hash(old.load(level)), content_hash(new.load(level)))
        pools = new.manifest["splits"]["full"]
        self.assertEqual(
            [len(pools[s]) for s in ("train", "validation", "test", "ood")], [12, 4, 4, 4]
        )
        self.assertEqual(len({x for ids in pools.values() for x in ids}), 24)
        for ids in pools.values():
            for level in ids:
                self.assertEqual(len(new.load(level).collectibles), 12)
                self.assertGreater(new.load(level).width, 6000)

    def test_v8_retains_exposed_geometry_and_adds_disjoint_fresh_pools(self):
        old = TrainingLevelRepository("config/curriculum_v7.json")
        new = TrainingLevelRepository("config/curriculum_v8.json")
        self.assertEqual(new.manifest, build_v8_manifest())
        for level in old.levels:
            self.assertEqual(content_hash(old.load(level)), content_hash(new.load(level)))
        pools = new.manifest["splits"]["full"]
        self.assertEqual(
            [len(pools[s]) for s in ("train", "validation", "test", "ood")], [24, 10, 8, 8]
        )
        train_orders = {
            tuple(seg["height"] for seg in new.manifest["levels"][level]["segments"])
            for level in pools["train"]
            if level.startswith("v8_")
        }
        self.assertEqual(len(train_orders), 6)

    def test_main_area_filters_only_unreachable_coins_and_preserves_old_checkpoint_content(self):
        old = TrainingLevelRepository("config/curriculum_v8.json")
        new = TrainingLevelRepository("config/curriculum_v9.json")
        self.assertEqual(new.manifest, build_v9_manifest())
        for level in old.levels:
            self.assertEqual(content_hash(old.load(level)), content_hash(new.load(level)))
        legacy = new.load("level_1")
        main = new.load("level_1_main")
        self.assertEqual(content_hash(legacy), content_hash(old.load("level_1")))
        self.assertEqual(len(legacy.collectibles), 25)
        self.assertEqual(len(main.collectibles), 5)
        self.assertEqual(legacy.solids, main.solids)
        self.assertEqual(legacy.goal_x, main.goal_x)
        self.assertTrue(all(coin.x < main.goal_x for coin in main.collectibles))
        config = json.loads(Path("config/ppo_full_collection_v9.json").read_text())
        _, _, stages = resolve_training_config(config)
        self.assertEqual(stages[-1]["train_levels"].count("level_1_main"), 1)
        invalid = deepcopy(config)
        invalid["imitation"]["level_repeats"] = {"v9_full_test_00": 8}
        with self.assertRaises(ValueError):
            resolve_training_config(invalid)

    def test_collection_gate_rejects_rushing_and_failed_rollouts(self):
        stage = {
            "task": "full",
            "validation_levels": ["map"],
            "success_threshold": 0.9,
            "max_success_steps": 768,
            "min_coin_ratio": 0.75,
        }
        episode = {
            "level_id": "map",
            "outcome": "success",
            "steps": 200,
            "coins_collected": 8,
            "coins_total": 12,
        }
        report = {
            "by_level": {"map": {"success_rate": 1, "mean_progress": 1}},
            "episodes": [episode],
        }
        self.assertIn(
            "coin_ratio", assess_prerequisites([stage], report)["tasks"]["full"]["failures"]["map"]
        )
        episode["coins_collected"] = 9
        self.assertTrue(assess_prerequisites([stage], report)["passed"])
        first = joint_selection_score([stage], report)
        episode["coins_collected"] = 12
        episode["steps"] = 250
        self.assertGreater(joint_selection_score([stage], report), first)
        report["by_level"]["map"]["success_rate"] = 0.5
        self.assertFalse(assess_prerequisites([stage], report)["passed"])
        del episode["coins_total"]
        self.assertFalse(assess_prerequisites([stage], report)["passed"])

    def test_full_resolution_preserves_legacy_and_uses_explicit_full_splits(self):
        config = json.loads(Path("config/ppo_full_train_correction.json").read_text())
        _, _, old = resolve_training_config(config)
        self.assertEqual(old[-1]["train_levels"], ["level_1"])
        config["environment"]["curriculum_manifest"] = "config/curriculum_v7.json"
        config["curriculum"]["stages"][-1].update(
            min_coin_ratio=0.75, coin_ratio_thresholds={"level_1": 0.2}
        )
        config["imitation"]["teacher"] = "coin_jump_v1"
        config["imitation"]["augment_global_features"] = True
        _, _, stages = resolve_training_config(config)
        self.assertEqual(len(stages[-1]["train_levels"]), 13)
        self.assertEqual(len(stages[-1]["validation_levels"]), 5)
        self.assertEqual(
            set(stages[-1]["train_levels"]) & set(stages[-1]["validation_levels"]), {"level_1"}
        )
        bad_augmentation = deepcopy(config)
        bad_augmentation["imitation"]["augment_global_features"] = "true"
        with self.assertRaises(ValueError):
            resolve_training_config(bad_augmentation)
        invalid = deepcopy(config)
        invalid["curriculum"]["stages"][-1]["coin_ratio_thresholds"] = {"test": 0.1}
        with self.assertRaises(ValueError):
            resolve_training_config(invalid)

    def test_collection_teacher_improves_known_training_route_without_death(self):
        config = json.loads(Path("config/ppo_full_train_correction.json").read_text())[
            "environment"
        ]
        env = EnvironmentFactory(config).make()
        agent = CoinJumpAgent()
        obs, info = env.reset(seed=0)
        for _ in range(768):
            obs, _, terminated, truncated, info = env.step(agent.act(obs))
            if terminated or truncated:
                break
        env.close()
        self.assertEqual(info["outcome"], "success")
        self.assertGreaterEqual(info["coins_collected"], 5)
