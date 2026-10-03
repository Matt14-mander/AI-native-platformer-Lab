"""Composition holdouts and legacy preservation must survive generation."""

import unittest

from ai_platformer.agents.scripted import RuleJumpAgent
from ai_platformer.benchmark.scripted import evaluate_scripted_agent
from ai_platformer.content.curriculum import TrainingLevelRepository, content_hash
from ai_platformer.content.curriculum_v5 import PREFIX, build_manifest, composition_cell


class MixedGeometryV5Tests(unittest.TestCase):
    def test_strata_count_sampling_episodes_without_inflating_layout_count(self):
        from scripts.evaluate_mixed_generalization import stratify

        specs = {
            "sample": {"obstacle_height": 64, "gap_width": 130, "obstacle_x": 300, "gap_x": 924}
        }
        report = {
            "episodes": [
                {"level_id": "sample", "outcome": "success"},
                {"level_id": "sample", "outcome": "death"},
                {"level_id": "sample", "outcome": "time_limit"},
            ]
        }
        group = stratify(report, specs)["cell/64/130/560"]
        self.assertEqual(group["episodes"], 3)
        self.assertEqual(group["unique_layouts"], 1)
        self.assertEqual(group["success_rate"], 1 / 3)
        self.assertEqual((group["deaths"], group["timeouts"]), (1, 1))

    def test_manifest_is_reproducible_and_keeps_every_legacy_assignment(self):
        old = TrainingLevelRepository("config/curriculum_v4.json")
        new = TrainingLevelRepository("config/curriculum_v5.json")
        self.assertEqual(new.manifest, build_manifest())
        for task, suites in old.manifest["splits"].items():
            for suite, ids in suites.items():
                self.assertTrue(set(ids) <= set(new.split(task, suite)))
                for level in ids:
                    self.assertEqual(content_hash(old.load(level)), content_hash(new.load(level)))

    def test_cells_are_disjoint_and_components_are_seen_in_training(self):
        repo = TrainingLevelRepository("config/curriculum_v5.json")
        cells = {}
        for suite in ("train", "validation", "test", "ood"):
            cells[suite] = {
                composition_cell(repo.manifest["levels"][level])
                for level in repo.split("mixed", suite)
                if level.startswith(PREFIX)
            }
        for a, a_cells in cells.items():
            for b, b_cells in cells.items():
                if a != b:
                    self.assertFalse(a_cells & b_cells)
        for dimension in range(3):
            train_values = {cell[dimension] for cell in cells["train"]}
            for suite in ("validation", "test"):
                self.assertTrue({cell[dimension] for cell in cells[suite]} <= train_values)
        self.assertEqual(len(set.union(*(cells[s] for s in ("train", "validation", "test")))), 27)

    def test_all_new_geometry_has_a_rule_witness_in_v2(self):
        repo = TrainingLevelRepository("config/curriculum_v5.json")
        levels = [level for level in repo.levels if level.startswith(PREFIX)]
        result = evaluate_scripted_agent(
            "rule",
            RuleJumpAgent,
            [100],
            environment={
                "environment_id": "PlatformerState-v2",
                "curriculum_manifest": "config/curriculum_v5.json",
                "episode_step_limit": 256,
            },
            level_ids=levels,
        )
        self.assertEqual(result.summary()["success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
