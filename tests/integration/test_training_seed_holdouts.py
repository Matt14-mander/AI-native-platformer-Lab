"""Fresh cohort holdouts must not change the established training experiment."""

import unittest

from ai_platformer.content.curriculum import TrainingLevelRepository, content_hash
from ai_platformer.content.curriculum_v6 import build_manifest


class TrainingSeedHoldoutTests(unittest.TestCase):
    def test_fresh_geometry_is_reproducible_and_only_reserved_pools_expand(self):
        old = TrainingLevelRepository("config/curriculum_v5.json")
        new = TrainingLevelRepository("config/curriculum_v6.json")
        self.assertEqual(new.manifest, build_manifest())
        for task, suites in old.manifest["splits"].items():
            for suite, ids in suites.items():
                if suite in ("train", "validation"):
                    self.assertEqual(ids, new.split(task, suite))
                else:
                    self.assertTrue(set(ids) <= set(new.split(task, suite)))
                for level in ids:
                    self.assertEqual(content_hash(old.load(level)), content_hash(new.load(level)))
        for suite, count in (("test", 20), ("ood", 12)):
            self.assertEqual(sum(x.startswith("v6_") for x in new.split("mixed", suite)), count)
        self.assertEqual(len(new.levels), len(old.levels) + 32)


if __name__ == "__main__":
    unittest.main()
