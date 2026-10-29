"""Playable Pygame state backed by the deterministic shared core."""

import pygame

from ai_platformer.content.legacy import LegacyLevelRepository
from ai_platformer.core import BasicPlatformerCore
from ai_platformer.rendering.keyboard import PygameKeyboardController
from ai_platformer.rendering.pygame_level import PygameLevelRenderer
from ai_platformer.settings import GameplaySettings, load_gameplay_settings
from source.components import info
from source.ui import pause_screen, text, panel


class Level:
    def __init__(self, settings: GameplaySettings | None = None):
        self.settings = settings or load_gameplay_settings()
        self.finished = False
        self.next = "game_over"
        self.info = info.Info("level")
        self.repository = LegacyLevelRepository()
        self.level_definition = self.repository.load(self.settings.level_id)
        self.core = BasicPlatformerCore(
            self.repository.load,
            config=self.settings.physics,
        )
        self.controller = PygameKeyboardController()
        self.renderer = self._create_renderer()
        self.state = None
        self.outcome = None
        self.terminal_timer = None
        self.paused = False
        self.enter()

    def _create_renderer(self):
        return PygameLevelRenderer(
            screen_size=pygame.display.get_surface().get_size(),
            level_width=self.level_definition.width,
            goal_x=self.level_definition.goal_x,
            solids_by_kind=self.repository.load_solids_by_group(self.settings.level_id),
            player_width=self.settings.physics.player_width,
            player_height=self.settings.physics.player_height,
        )

    def enter(self):
        """Start a fresh episode every time the state becomes active."""
        self.finished = False
        self.next = "game_over"
        self.outcome = None
        self.terminal_timer = None
        self.paused = False
        self.state = self.core.reset(
            seed=self.settings.seed,
            level_id=self.settings.level_id,
        )
        self.renderer.reset()
        if pygame.mixer.get_init():
            pygame.mixer.unpause()

    def result_summary(self):
        metadata = self.state.metadata
        return {"score": metadata.get("score", 0),
                "coins": metadata.get("coins_collected", 0) + metadata.get("box_coins", 0),
                "ticks": self.state.tick, "enemies": metadata.get("enemies_defeated", 0),
                "outcome": self.outcome}

    def leave(self):
        """Stop this run's feedback before entering another screen."""
        self.renderer.feedback.reset()
        if pygame.mixer.get_init():
            pygame.mixer.unpause()

    def is_finished(self):
        return self.finished

    def update(self, surface, keys):
        current_time = pygame.time.get_ticks()
        was_pressed = getattr(keys, "was_pressed", lambda _key: False)
        if getattr(keys, "focus_lost", False) and self.outcome is None:
            self.paused = True
            if pygame.mixer.get_init():
                pygame.mixer.pause()
            self.draw(surface)
            return
        if self.outcome == "success" and was_pressed(pygame.K_RETURN):
            self.next, self.finished = "level_complete", True
            return
        if was_pressed(pygame.K_r) or (self.outcome is not None and was_pressed(pygame.K_RETURN)):
            self.enter()
            self.info.update()
            self.draw(surface)
            return
        if (self.paused or self.outcome is not None) and (
            was_pressed(pygame.K_m) or (self.outcome is not None and was_pressed(pygame.K_ESCAPE))
        ):
            self.renderer.feedback.reset()
            if pygame.mixer.get_init():
                pygame.mixer.unpause()
            self.next, self.finished = "main_menu", True
            return
        if self.outcome is None and (was_pressed(pygame.K_p) or was_pressed(pygame.K_ESCAPE)
                                     or (self.paused and was_pressed(pygame.K_RETURN))):
            self.paused = not self.paused
            if pygame.mixer.get_init():
                if self.paused:
                    pygame.mixer.pause()
                else:
                    pygame.mixer.unpause()

        if not self.paused:
            self.renderer.feedback.advance()

        if self.outcome is None and not self.paused:
            action = self.controller.action(keys)
            result = self.core.step(action, jump_pressed=self.controller.jump_pressed(keys))
            self.renderer._update_camera(result.state)
            self.renderer.feedback.consume(self.state, result, self.renderer.camera_x,
                                           (self.settings.physics.player_width, self.settings.physics.player_height))
            self.state = result.state
            if result.terminated or result.truncated:
                self.outcome = result.info.get("outcome", "time_limit")
                self.terminal_timer = current_time
        elif (
            self.outcome is not None
            and current_time - self.terminal_timer >= (self.settings.terminal_display_ms if self.outcome == "success" else 250)
        ):
            self.next = "level_complete" if self.outcome == "success" else "game_over"
            self.finished = True

        self.info.update()
        self.draw(surface)

    def draw(self, surface):
        self.renderer.draw(surface, self.state)
        self.info.draw(surface, self.state)
        self.renderer.feedback.draw_overlay(surface, self.renderer.acorn)
        if self.paused:
            pause_screen(surface)
        elif self.outcome is not None:
            panel(surface, pygame.Rect(160, 205, 480, 140))
            title = {"success": "抵达终点！", "time_limit": "本次旅程时间已到"}.get(self.outcome, "再试一次就好")
            text(surface, title, 225, 30)
            text(surface, "Enter 查看结算    R 再玩一次" if self.outcome == "success" else "Enter / R 立即重试    Esc 返回菜单", 286, 20)
        elif self.state.tick < 360:
            panel(surface, pygame.Rect(145, 95, 510, 45), 195)
            text(surface, "← → 移动   空格跳跃   Shift 奔跑   Esc 暂停", 105, 18)
