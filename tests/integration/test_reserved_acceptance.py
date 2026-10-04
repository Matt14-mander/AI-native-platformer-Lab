"""Reserved acceptance rejects leakage and evaluates layouts, not pooled averages."""

import unittest
from types import SimpleNamespace

from ai_platformer.benchmark.acceptance import (
    acceptance_gate,
    reserved_pools,
    validate_reserved_seeds,
)
from ai_platformer.content.curriculum import TrainingLevelRepository


class ReservedAcceptanceTests(unittest.TestCase):
    def test_pools_skip_absent_ood_without_losing_reserved_geometry(self):
        repo = TrainingLevelRepository("config/curriculum_v5.json")
        stages = [
            {
                "task": task,
                "train_levels": repo.split(task, "train"),
                "validation_levels": repo.split(task, "validation"),
            }
            for task in ("flat", "obstacle", "gap", "mixed")
        ]
        pools = reserved_pools(repo, stages)
        self.assertEqual(sum(map(len, pools["test"].values())), 154)
        self.assertEqual(sum(map(len, pools["ood"].values())), 36)
        self.assertEqual(set(pools["ood"]), {"gap", "mixed"})

    def test_overlapping_pools_and_nonreserved_seeds_are_rejected(self):
        repo = SimpleNamespace(manifest={"splits": {"mixed": {"test": ["train"], "ood": ["ood"]}}})
        stages = [{"task": "mixed", "train_levels": ["train"], "validation_levels": ["val"]}]
        with self.assertRaises(ValueError):
            reserved_pools(repo, stages)
        config = {"train_seeds": [0, 1], "evaluation": {"seeds": [100]}}
        for seeds in ([0], [100], [1000, 1000], [-1]):
            with self.assertRaises(ValueError):
                validate_reserved_seeds(seeds, config)
        validate_reserved_seeds([1000, 1001], config)

    def test_bad_layout_cannot_hide_behind_high_aggregate_success(self):
        stages = [{"task": "mixed", "success_threshold": 0.8, "max_success_steps": 256}]
        episodes = [{"level_id": "a", "outcome": "success", "steps": 50}] * 9
        report = {
            "by_level": {"a": {"success_rate": 1}, "b": {"success_rate": 0}},
            "episodes": episodes + [{"level_id": "b", "outcome": "death", "steps": 20}],
        }
        gate = acceptance_gate(stages, {"mixed": ["a", "b"]}, report)
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["tasks"]["mixed"]["qualified_layouts"], 1)


if __name__ == "__main__":
    unittest.main()
