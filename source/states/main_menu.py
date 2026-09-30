"""Opening menu for the lakeside platformer."""

import pygame

from ai_platformer.rendering.art import fit_height, fit_rect, load_art


class MainMenu:
    def __init__(self):
        self.finished = False
        self.next = "load_screen"
        self.background = fit_height(
            load_art("backgrounds/lakeside-day.png", crop=False), 600
        )
        self.hero = fit_height(load_art("characters/child-with-puppy-idle.png"), 250)
        self.ground = fit_rect(load_art("tilesets/moss-earth-ground.png"), 800, 60)
        self.title_font = pygame.font.Font(None, 72)
        self.body_font = pygame.font.Font(None, 30)
        self.small_font = pygame.font.Font(None, 25)

    def enter(self):
        self.finished = False
        self.next = "load_screen"

    def is_finished(self):
        return self.finished

    def update(self, surface, keys):
        if keys[pygame.K_RETURN]:
            self.finished = True

        x = (surface.get_width() - self.background.get_width()) // 2
        surface.blit(self.background, (x, 0))
        surface.blit(self.ground, (0, 540))
        surface.blit(self.hero, (96, 290))

        panel = pygame.Surface((432, 300), pygame.SRCALPHA)
        pygame.draw.rect(panel, (31, 42, 45, 205), panel.get_rect(), border_radius=24)
        surface.blit(panel, (327, 157))

        cream = (255, 237, 204)
        soft = (216, 227, 219)
        surface.blit(self.title_font.render("PUPPY TRAIL", True, cream), (360, 204))
        surface.blit(self.body_font.render("A lakeside adventure", True, soft), (365, 279))
        surface.blit(self.body_font.render("PRESS ENTER TO PLAY", True, cream), (365, 354))
        surface.blit(
            self.small_font.render("Move: arrows   Jump: Space   Run: Shift", True, soft),
            (351, 414),
        )
