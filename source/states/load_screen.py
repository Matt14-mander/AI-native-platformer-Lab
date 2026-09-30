"""Short transition screens using the same original visual theme."""

import pygame

from ai_platformer.rendering.art import fit_height, load_art


class LoadScreen:
    title = "THE JOURNEY BEGINS"
    subtitle = "Hold your puppy close."
    duration = 1200

    def __init__(self):
        self.finished = False
        self.next = "level"
        self.timer = 0
        self.background = fit_height(
            load_art("backgrounds/lakeside-day.png", crop=False), 600
        )
        self.hero = fit_height(load_art("characters/child-with-puppy-idle.png"), 180)
        self.title_font = pygame.font.Font(None, 56)
        self.body_font = pygame.font.Font(None, 29)

    def enter(self):
        self.finished = False
        self.timer = 0

    def is_finished(self):
        return self.finished

    def update(self, surface, keys):
        self.draw(surface)
        now = pygame.time.get_ticks()
        if self.timer == 0:
            self.timer = now
        elif now - self.timer > self.duration:
            self.finished = True
            self.timer = 0

    def draw(self, surface):
        x = (surface.get_width() - self.background.get_width()) // 2
        surface.blit(self.background, (x, 0))
        shade = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        shade.fill((22, 35, 44, 105))
        surface.blit(shade, (0, 0))
        surface.blit(self.hero, self.hero.get_rect(midbottom=(400, 475)))
        title = self.title_font.render(self.title, True, (255, 239, 210))
        subtitle = self.body_font.render(self.subtitle, True, (236, 229, 212))
        surface.blit(title, title.get_rect(center=(400, 135)))
        surface.blit(subtitle, subtitle.get_rect(center=(400, 185)))


class GameOver(LoadScreen):
    title = "THE TRAIL ENDS HERE"
    subtitle = "The journey can begin again."
    duration = 2500

    def __init__(self):
        super().__init__()
        self.next = "main_menu"


class LevelComplete(LoadScreen):
    title = "JOURNEY COMPLETE"
    subtitle = "A lovely walk together."
    duration = 2500

    def __init__(self):
        super().__init__()
        self.next = "main_menu"
