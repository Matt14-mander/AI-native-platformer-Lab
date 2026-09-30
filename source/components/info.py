"""Compact HUD for the original lakeside art theme."""

from __future__ import annotations

import pygame

from ai_platformer.rendering.art import fit_rect, load_art


class Info:
    def __init__(self, state: str):
        self.state = state
        self.font = pygame.font.Font(None, 29)
        self.small_font = pygame.font.Font(None, 22)
        self.acorn = fit_rect(load_art("tilesets/amber-acorn-collectible.png"), 21, 29)

    def update(self):
        pass

    def draw(self, surface: pygame.Surface, game_state=None):
        if self.state != "level" or game_state is None:
            return
        width = surface.get_width()
        panel = pygame.Surface((width - 32, 70), pygame.SRCALPHA)
        pygame.draw.rect(panel, (30, 40, 43, 194), panel.get_rect(), border_radius=14)
        surface.blit(panel, (16, 12))

        cream = (255, 239, 210)
        soft = (210, 225, 215)
        score = int(game_state.metadata.get("score", 0))
        collected = int(game_state.metadata.get("coins_collected", 0))
        surface.blit(self.font.render("PUPPY TRAIL", True, cream), (34, 24))
        surface.blit(self.small_font.render(f"SCORE  {score:06d}", True, soft), (35, 53))
        surface.blit(self.acorn, (276, 30))
        surface.blit(self.font.render(f"x {collected:02d}", True, cream), (303, 34))
        surface.blit(self.small_font.render("JOURNEY", True, soft), (width - 181, 25))
        bar = pygame.Rect(width - 181, 53, 147, 9)
        pygame.draw.rect(surface, (96, 111, 104), bar, border_radius=5)
        fill = bar.copy()
        fill.width = max(0, round(bar.width * game_state.progress))
        if fill.width:
            pygame.draw.rect(surface, (238, 205, 148), fill, border_radius=5)
