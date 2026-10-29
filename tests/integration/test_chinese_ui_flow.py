import os
import unittest
from dataclasses import replace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import pygame
from source.input import KeyboardInput
from source.states.level import Level
from source.states.main_menu import MainMenu
from source.states.load_screen import GameOver, LevelComplete
from source.tools import Game
from source.ui import font


class ChineseFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()
        cls.surface = pygame.display.set_mode((800, 600))

    def keys(self, key=None):
        keys = KeyboardInput()
        if key is not None:
            keys.handle_event(pygame.event.Event(pygame.KEYDOWN, key=key))
        return keys

    def test_chinese_glyphs_and_menu_help(self):
        self.assertTrue(all(metric is not None for metric in font(24).metrics("湖岸旅程暂停重试通关")))
        menu = MainMenu()
        keys = self.keys(pygame.K_h)
        menu.update(self.surface, keys)
        self.assertTrue(menu.show_help)
        keys.end_frame()
        menu.update(self.surface, keys)
        self.assertTrue(menu.show_help)
        menu.update(self.surface, self.keys(pygame.K_RETURN))
        self.assertTrue(menu.finished)
        self.assertEqual(menu.next, "level")

    def test_pause_freezes_and_enter_resumes_and_menu_exits(self):
        level = Level()
        level.update(self.surface, self.keys(pygame.K_ESCAPE))
        frozen = level.state
        level.update(self.surface, self.keys())
        self.assertEqual(level.state, frozen)
        level.update(self.surface, self.keys(pygame.K_RETURN))
        self.assertFalse(level.paused)
        self.assertEqual(level.state.tick, 1)
        level.update(self.surface, self.keys(pygame.K_ESCAPE))
        level.update(self.surface, self.keys(pygame.K_m))
        self.assertEqual(level.next, "main_menu")
        self.assertTrue(level.finished)

    def test_death_can_retry_before_result_transition(self):
        level = Level()
        level.outcome = "death"
        level.terminal_timer = pygame.time.get_ticks()
        level.update(self.surface, self.keys(pygame.K_RETURN))
        self.assertIsNone(level.outcome)
        self.assertEqual(level.state.tick, 0)
        self.assertFalse(level.finished)

    def test_result_payload_and_held_enter_does_not_skip(self):
        level = Level()
        result = LevelComplete()
        level.state = replace(level.state, tick=3720, metadata={"score": 750, "coins_collected": 3, "box_coins": 1, "enemies_defeated": 1})
        level.outcome = "success"
        level.terminal_timer = pygame.time.get_ticks() - 2000
        game = Game({"level": level, "level_complete": result}, "level")
        game.advance(1/60)
        self.assertIs(game.state, result)
        self.assertEqual(result.summary, {"score": 750, "coins": 4, "ticks": 3720, "enemies": 1, "outcome": "success"})
        game.advance(1)
        self.assertIs(game.state, result)
        game.input.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN))
        game.advance(1/60)
        self.assertIs(game.state, level)
        self.assertEqual(level.state.tick, 0)

    def test_death_result_waits_for_choice(self):
        result = GameOver()
        for _ in range(10):
            result.update(self.surface, self.keys())
        self.assertFalse(result.finished)
        result.update(self.surface, self.keys(pygame.K_ESCAPE))
        self.assertEqual(result.next, "main_menu")

    def test_success_enter_opens_settlement_without_resetting_score(self):
        level = Level()
        level.outcome = "success"
        level.terminal_timer = pygame.time.get_ticks()
        level.state = replace(level.state, metadata={"score": 500})
        level.update(self.surface, self.keys(pygame.K_RETURN))
        self.assertEqual(level.next, "level_complete")
        self.assertEqual(level.result_summary()["score"], 500)

    def test_focus_loss_pauses_and_focus_gain_requires_explicit_resume(self):
        level = Level()
        keys = self.keys(pygame.K_RIGHT)
        keys.handle_event(pygame.event.Event(pygame.WINDOWFOCUSLOST))
        keys.sync({})  # Unfocused polling must not restore any held keys.
        level.update(self.surface, keys)
        self.assertTrue(level.paused)
        self.assertEqual(level.state.tick, 0)
        self.assertFalse(keys[pygame.K_RIGHT])
        keys.end_frame()
        keys.handle_event(pygame.event.Event(pygame.WINDOWFOCUSGAINED))
        level.update(self.surface, keys)
        self.assertTrue(level.paused)
        self.assertEqual(level.state.tick, 0)
        level.update(self.surface, self.keys(pygame.K_RETURN))
        self.assertFalse(level.paused)

    def test_transition_draws_new_screen_without_reusing_enter(self):
        from unittest.mock import patch
        level, menu = Level(), MainMenu()
        game = Game({"main_menu": menu, "level": level}, "main_menu")
        game.input.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN))
        with patch.object(level, "draw", wraps=level.draw) as draw:
            game.advance(1/60)
            draw.assert_called_once()
        self.assertIs(game.state, level)
        self.assertEqual(level.state.tick, 0)
        game.advance(1/60)
        self.assertFalse(level.paused)
        self.assertEqual(level.state.tick, 1)

    def test_time_limit_is_preserved_and_results_clear_on_reentry(self):
        level, result = Level(), GameOver()
        level.outcome = "time_limit"
        level.terminal_timer = pygame.time.get_ticks() - 1000
        game = Game({"level": level, "game_over": result}, "level")
        game.advance(1/60)
        self.assertEqual(result.summary["outcome"], "time_limit")
        result.enter()
        self.assertEqual(result.summary, {})


if __name__ == "__main__":
    unittest.main()
