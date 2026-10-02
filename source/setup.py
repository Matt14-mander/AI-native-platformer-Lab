"""Pygame display setup with legacy sprite sheets loaded only on demand."""

from pathlib import Path

import pygame

from source import constants as C
from source import tools


pygame.init()
SCREEN = pygame.display.set_mode((C.SCREEN_W, C.SCREEN_H))


class _LazyLegacyGraphics(dict):
    def __init__(self):
        super().__init__()
        self._loaded = False

    def __missing__(self, key):
        if not self._loaded:
            self._loaded = True
            graphics_dir = Path(__file__).resolve().parents[1] / "resources" / "graphics"
            self.update(tools.load_graphics(graphics_dir))
        return dict.__getitem__(self, key)


GRAPHICS = _LazyLegacyGraphics()
