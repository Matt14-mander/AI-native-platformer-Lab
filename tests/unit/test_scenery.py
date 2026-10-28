import unittest

from ai_platformer.core.level import SolidRect
from ai_platformer.rendering.scenery import exposed_top, LakesideScenery


class SceneryRulesTests(unittest.TestCase):
    def test_grass_and_lip_skip_covered_platform_tops(self):
        floor = SolidRect(0, 100, 300, 100)
        step = SolidRect(100, 60, 40, 40)
        self.assertEqual(exposed_top(floor, (floor, step)), [(0, 100), (140, 300)])
        self.assertEqual(exposed_top(step, (floor, step)), [(100, 140)])

    def test_stacked_steps_and_adjacent_platforms(self):
        lower = SolidRect(0, 100, 40, 40)
        upper = SolidRect(0, 60, 40, 40)
        adjacent = SolidRect(40, 100, 40, 40)
        self.assertEqual(exposed_top(lower, (lower, upper, adjacent)), [])
        self.assertEqual(exposed_top(adjacent, (lower, upper, adjacent)), [(40, 80)])

    def test_parallax_depth_order(self):
        speeds = LakesideScenery.SPEEDS
        self.assertTrue(0 < speeds["sky"] < speeds["mountains"] < speeds["lake"] < speeds["shore"] < 1)


if __name__ == "__main__":
    unittest.main()
