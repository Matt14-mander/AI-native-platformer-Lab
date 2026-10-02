"""Pygame view for renderer-independent world snapshots."""

from __future__ import annotations

from collections.abc import Mapping

import pygame

from ai_platformer.core import WorldSnapshot
from ai_platformer.core.level import SolidRect
from ai_platformer.rendering.art import fit_height, fit_rect, load_art


class PygameLevelRenderer:
    def __init__(
        self,
        *,
        screen_size: tuple[int, int],
        level_width: float,
        goal_x: float,
        solids_by_kind: Mapping[str, tuple[SolidRect, ...]],
        player_width: float = 24.0,
        player_height: float = 32.0,
    ) -> None:
        self.screen_width, self.screen_height = screen_size
        self.level_width = level_width
        self.goal_x = goal_x
        self.solids_by_kind = solids_by_kind
        self.player_width = player_width
        self.player_height = player_height
        self.camera_x = 0.0

        backdrop = load_art("backgrounds/lakeside-day.png", crop=False)
        backdrop_width = round(backdrop.get_width() * self.screen_height / backdrop.get_height())
        self.background = fit_rect(backdrop, backdrop_width, self.screen_height)
        ground_art = load_art("tilesets/moss-earth-ground.png")
        edge_trim = round(ground_art.get_width() * 0.018)
        ground_art = ground_art.subsurface(
            (edge_trim, 0, ground_art.get_width() - 2 * edge_trim, ground_art.get_height())
        ).copy()
        self.ground_tile = fit_rect(ground_art, 384, 64)
        self.stump = load_art("tilesets/mossy-stump-obstacle.png")
        self.step = load_art("tilesets/mossy-step-block.png")
        self.acorn = fit_rect(load_art("tilesets/amber-acorn-collectible.png"), 18, 26)
        self.brick = fit_rect(load_art("tilesets/cracked-clay-brick.png"), 40, 40)
        self.box = fit_rect(load_art("tilesets/acorn-reward-box.png"), 40, 40)
        self.used_box = self.box.copy()
        self.used_box.fill((145, 145, 145, 255), special_flags=pygame.BLEND_RGBA_MULT)
        self.enemy = fit_rect(load_art("enemies/shadow-hedgehog.png"), 28, 28)
        self.enemy_right = pygame.transform.flip(self.enemy, True, False)
        self.shield_berry = fit_rect(load_art("items/blue-ward-berry.png"), 24, 24)
        self.player_frames = {
            "idle": (self._player_frame("child-with-puppy-idle.png"),),
            "run": (
                self._player_frame("child-with-puppy-run-1.png"),
                self._player_frame("child-with-puppy-run-2.png"),
            ),
            "jump": (self._player_frame("child-with-puppy-jump.png"),),
            "fall": (self._player_frame("child-with-puppy-fall.png"),),
        }
        self.pause_font = pygame.font.Font(None, 54)
        self._stump_cache: dict[tuple[int, int], pygame.Surface] = {}
        self._step_cache: dict[tuple[int, int], pygame.Surface] = {}

    @staticmethod
    def _player_frame(filename: str) -> pygame.Surface:
        return fit_height(load_art(f"characters/{filename}"), 52)

    def reset(self) -> None:
        self.camera_x = 0.0

    def draw(self, surface: pygame.Surface, state: WorldSnapshot) -> None:
        self._update_camera(state)
        self._draw_background(surface)
        self._draw_solids(surface)
        self._draw_goal(surface)
        for entity in state.entities:
            if not entity.active:
                continue
            x = round(entity.x - self.camera_x)
            if x < -50 or x > self.screen_width + 50:
                continue
            y = round(entity.y)
            if entity.kind == "coin":
                sprite = self.acorn
            elif entity.kind == "brick":
                sprite = self.brick
            elif entity.kind == "box":
                sprite = self.used_box if entity.state == "used" else self.box
            elif entity.kind == "enemy":
                sprite = self.enemy if entity.facing < 0 else self.enemy_right
            elif entity.kind == "powerup-shield":
                sprite = self.shield_berry
            else:
                continue
            surface.blit(sprite, (x, y))
        self._draw_player(surface, state)

    def _draw_background(self, surface: pygame.Surface) -> None:
        width = self.background.get_width()
        offset = round(self.camera_x * 0.18) % width
        for x in range(-offset, self.screen_width, width):
            surface.blit(self.background, (x, 0))

    def _draw_solids(self, surface: pygame.Surface) -> None:
        camera = self.camera_x
        visible_right = camera + self.screen_width
        for solid in self.solids_by_kind.get("ground", ()):
            if solid.right <= camera or solid.x >= visible_right:
                continue
            x = round(solid.x - camera)
            y = round(solid.y)
            width = round(solid.width)
            height = round(solid.height)
            ground_rect = pygame.Rect(x, y, width, height)
            pygame.draw.rect(surface, (61, 53, 44), ground_rect)
            old_clip = surface.get_clip()
            surface.set_clip(old_clip.clip(pygame.Rect(x, y - 4, width, height + 4)))
            tile_width = self.ground_tile.get_width()
            first = max(0, int((camera - solid.x) // tile_width))
            tile_x = solid.x + first * tile_width
            while tile_x < min(solid.right, visible_right):
                surface.blit(self.ground_tile, (round(tile_x - camera), y - 4))
                tile_x += tile_width
            surface.set_clip(old_clip)

        for solid in self.solids_by_kind.get("pipe", ()):
            if solid.right <= camera or solid.x >= visible_right:
                continue
            size = (round(solid.width), round(solid.height))
            sprite = self._stump_cache.get(size)
            if sprite is None:
                sprite = fit_rect(self.stump, *size)
                self._stump_cache[size] = sprite
            surface.blit(sprite, (round(solid.x - camera), round(solid.y)))

        for solid in self.solids_by_kind.get("step", ()):
            if solid.right <= camera or solid.x >= visible_right:
                continue
            width = round(solid.width)
            bottom = round(solid.bottom)
            top = round(solid.y)
            while bottom > top:
                height = min(44, bottom - top)
                size = (width, height)
                sprite = self._step_cache.get(size)
                if sprite is None:
                    sprite = fit_rect(self.step, *size)
                    self._step_cache[size] = sprite
                bottom -= height
                surface.blit(sprite, (round(solid.x - camera), bottom))

    def _draw_goal(self, surface: pygame.Surface) -> None:
        x = round(self.goal_x - self.camera_x)
        if not -40 <= x <= self.screen_width + 40:
            return
        pygame.draw.line(surface, (53, 49, 44), (x, 160), (x, 538), 5)
        pygame.draw.circle(surface, (239, 211, 162), (x, 158), 8)
        pygame.draw.polygon(
            surface,
            (238, 220, 178),
            ((x + 3, 174), (x + 55, 188), (x + 3, 204)),
        )

    def _draw_player(self, surface: pygame.Surface, state: WorldSnapshot) -> None:
        if state.metadata.get("invulnerable_ticks", 0) and state.tick % 6 < 3:
            return
        player = state.player
        if not player.grounded:
            pose = "jump" if player.velocity_y < 0 else "fall"
        elif abs(player.velocity_x) >= 0.35:
            pose = "run"
        else:
            pose = "idle"
        frames = self.player_frames[pose]
        frame = frames[(state.tick // 7) % len(frames)]
        if player.facing < 0:
            frame = pygame.transform.flip(frame, True, False)
        center_x = player.x + self.player_width / 2 - self.camera_x
        feet_y = player.y + self.player_height
        surface.blit(frame, (round(center_x - frame.get_width() / 2), round(feet_y - frame.get_height())))

    def _update_camera(self, state: WorldSnapshot) -> None:
        follow_line = self.screen_width / 3
        player_screen_x = state.player.x - self.camera_x
        if player_screen_x > follow_line:
            self.camera_x = state.player.x - follow_line
        max_camera = max(0.0, self.level_width - self.screen_width)
        self.camera_x = max(0.0, min(max_camera, self.camera_x))

    def draw_pause(self, surface: pygame.Surface) -> None:
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((22, 29, 36, 145))
        surface.blit(overlay, (0, 0))
        label = self.pause_font.render("PAUSED", True, (255, 240, 211))
        rect = label.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
        surface.blit(label, rect)
