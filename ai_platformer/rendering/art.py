"""Load and size original art without modifying its source PNGs."""

from __future__ import annotations

from pathlib import Path

import pygame


ASSET_ROOT = Path(__file__).resolve().parents[2] / "assets"


def load_art(relative_path: str, *, crop: bool = True) -> pygame.Surface:
    path = ASSET_ROOT / relative_path
    image = pygame.image.load(str(path)).convert_alpha()
    if crop:
        bounds = image.get_bounding_rect(min_alpha=8)
        if bounds.width == 0 or bounds.height == 0:
            raise ValueError(f"empty artwork: {path}")
        image = image.subsurface(bounds).copy()
    return image


def fit_height(image: pygame.Surface, height: int) -> pygame.Surface:
    width = max(1, round(image.get_width() * height / image.get_height()))
    return pygame.transform.smoothscale(image, (width, height))


def fit_rect(image: pygame.Surface, width: int, height: int) -> pygame.Surface:
    return pygame.transform.smoothscale(image, (max(1, width), max(1, height)))
