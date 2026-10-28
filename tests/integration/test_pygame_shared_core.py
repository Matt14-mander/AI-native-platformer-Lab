import os
import unittest
from collections import defaultdict

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

try:
    import pygame
except ModuleNotFoundError:  # Local headless-only environments may omit Pygame.
    pygame = None


@unittest.skipIf(pygame is None, "Pygame is not installed")
class SharedCorePygameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from source import setup
        from source.states.level import Level

        cls.surface = setup.SCREEN
        cls.level_type = Level

    @classmethod
    def tearDownClass(cls) -> None:
        pygame.quit()

    def keys(self, *pressed_keys):
        values = defaultdict(bool)
        for key in pressed_keys:
            values[key] = True
        return values

    def test_keyboard_drives_shared_core(self) -> None:
        level = self.level_type()
        start = level.state

        for _ in range(60):
            level.update(self.surface, self.keys(pygame.K_RIGHT))

        self.assertGreater(level.state.player.x, start.player.x)
        self.assertEqual(level.state.tick, 60)

    def test_short_tap_between_ticks_still_jumps(self):
        from source.input import KeyboardInput
        from pathlib import Path
        from ai_platformer.settings import load_gameplay_settings
        level = self.level_type(load_gameplay_settings(Path("config/gameplay_player.json")))
        keys = KeyboardInput()
        keys.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
        keys.handle_event(pygame.event.Event(pygame.KEYUP, key=pygame.K_SPACE))
        level.update(self.surface, keys)
        self.assertLess(level.state.player.velocity_y, 0)
        self.assertFalse(level.state.player.grounded)

    def test_fixed_clock_preserves_edges_until_a_tick_and_consumes_once(self):
        from source.tools import Game

        class CounterState:
            finished = False
            def __init__(self):
                self.ticks = 0
                self.presses = 0
            def update(self, surface, keys):
                self.ticks += 1
                self.presses += int(keys.was_pressed(pygame.K_SPACE))

        for fps in (30, 60, 144):
            state = CounterState()
            game = Game({"play": state}, "play", render_fps=fps)
            for _ in range(fps):
                game.advance(1 / fps)
            self.assertEqual(state.ticks, 60)
        state = CounterState()
        game = Game({"play": state}, "play")
        game.input.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
        game.input.handle_event(pygame.event.Event(pygame.KEYUP, key=pygame.K_SPACE))
        game.advance(0.001)
        self.assertEqual(state.presses, 0)
        game.advance(0.10)
        self.assertEqual(state.presses, 1)
        self.assertLessEqual(game.advance(5.0), 15)

    def test_v2_keyboard_and_gym_follow_same_trajectory(self) -> None:
        from ai_platformer.core import Action
        from ai_platformer.envs.factory import EnvironmentFactory

        factory = EnvironmentFactory({"environment_id": "PlatformerState-v2"})
        level = self.level_type(factory.settings)
        env = factory.make()
        try:
            env.reset(seed=factory.settings.seed)
            sequence = [
                (Action.RIGHT_RUN_JUMP, (pygame.K_RIGHT, pygame.K_SPACE, pygame.K_LSHIFT)),
                (Action.RIGHT_RUN, (pygame.K_RIGHT, pygame.K_LSHIFT)),
                (Action.RIGHT_RUN_JUMP, (pygame.K_RIGHT, pygame.K_SPACE, pygame.K_LSHIFT)),
                (Action.NOOP, ()),
            ] * 8
            for action, pressed in sequence:
                env.step(int(action))
                for _ in range(factory.action_repeat):
                    level.update(self.surface, self.keys(*pressed))
                self.assertEqual(level.state, env.core.state)
        finally:
            env.close()

    def test_jump_and_run_are_mapped_to_core_action(self) -> None:
        level = self.level_type()
        start_y = level.state.player.y

        level.update(
            self.surface,
            self.keys(pygame.K_RIGHT, pygame.K_SPACE, pygame.K_LSHIFT),
        )

        self.assertLess(level.state.player.y, start_y)
        self.assertLess(level.state.player.velocity_y, 0.0)
        self.assertGreater(level.state.player.velocity_x, 0.0)

    def test_enter_starts_a_fresh_episode(self) -> None:
        level = self.level_type()
        for _ in range(60):
            level.update(self.surface, self.keys(pygame.K_RIGHT))
        self.assertGreater(level.state.tick, 0)
        self.assertGreater(level.renderer.camera_x, 0.0)

        level.enter()

        self.assertEqual(level.state.tick, 0)
        self.assertEqual(level.state.player.x, 110.0)
        self.assertEqual(level.renderer.camera_x, 0.0)
        self.assertIsNone(level.outcome)

    def test_terminal_outcome_routes_to_correct_screen(self) -> None:
        level = self.level_type()
        level.outcome = "success"
        level.terminal_timer = (
            pygame.time.get_ticks() - level.settings.terminal_display_ms
        )

        level.update(self.surface, self.keys())

        self.assertTrue(level.finished)
        self.assertEqual(level.next, "level_complete")

    def test_pause_and_restart_controls(self) -> None:
        from source.input import KeyboardInput

        level = self.level_type()
        keys = KeyboardInput()
        keys.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p))
        level.update(self.surface, keys)
        self.assertTrue(level.paused)
        self.assertEqual(level.state.tick, 0)

        keys.end_frame()
        keys.handle_event(pygame.event.Event(pygame.KEYUP, key=pygame.K_p))
        keys.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))
        level.update(self.surface, keys)
        self.assertFalse(level.paused)
        self.assertEqual(level.state.tick, 0)


if __name__ == "__main__":
    unittest.main()
