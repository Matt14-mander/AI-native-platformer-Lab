import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import pygame

from ai_platformer.rendering.scenery import LakesideScenery
from ai_platformer.core.level import SolidRect


class SceneryRenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()
        pygame.display.set_mode((800, 600))
        cls.scenery = LakesideScenery((800, 600), {"ground": (SolidRect(0, 538, 2000, 62),)})

    def render(self, camera, tick):
        surface = pygame.Surface((800, 600))
        self.scenery.draw_background(surface, camera, tick)
        self.scenery.draw_edges(surface, camera)
        self.scenery.draw_ambient(surface, camera, tick)
        return pygame.image.tobytes(surface, "RGB")

    def test_repeated_paused_draw_is_identical_but_time_animates(self):
        self.assertEqual(self.render(123, 40), self.render(123, 40))
        self.assertNotEqual(self.render(123, 40), self.render(123, 80))

    def test_camera_moves_layers_and_large_offsets_are_covered(self):
        self.assertNotEqual(self.render(0, 0), self.render(300, 0))
        for camera in (0, 900, 8000, 100000):
            self.assertEqual(len(self.render(camera, 0)), 800 * 600 * 3)

    def test_mirrored_tile_edges_match(self):
        for normal, mirrored in (self.scenery.sky, self.scenery.mountains, self.scenery.lake, self.scenery.shore):
            for y in range(0, normal.get_height(), 7):
                self.assertEqual(normal.get_at((normal.get_width()-1, y)), mirrored.get_at((0, y)))


if __name__ == "__main__":
    unittest.main()
