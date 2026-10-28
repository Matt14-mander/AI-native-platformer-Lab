import os
import unittest
from dataclasses import replace
from pathlib import Path
import wave

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import pygame

from ai_platformer.core import Action, BasicPlatformerCore, BlockSpawn, EnemySpawn, PowerupSpawn
from ai_platformer.rendering.feedback import InteractionFeedback, FeedbackAudio
from ai_platformer.core.level import CollectibleSpawn
from tests.unit.test_interactive_content import small_level


class SilentAudio:
    def __init__(self):
        self.played = []
    def stop(self):
        self.played.clear()
    def play(self, name):
        self.played.append(name)


class InteractionFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.audio = SilentAudio()
        self.feedback = InteractionFeedback(self.audio)

    def step_until(self, level, action, event):
        core = BasicPlatformerCore(lambda _: level)
        previous = core.reset(seed=1, level_id=level.level_id)
        for _ in range(60):
            result = core.step(action)
            self.feedback.advance()
            self.feedback.consume(previous, result, 0)
            if any(name == event for name, _ in result.info.get("events", ())):
                return previous, result
            previous = result.state
        self.fail(f"missing event: {event}")

    def test_box_bumps_and_reward_flies_once_then_pulses(self):
        level = small_level(blocks=(BlockSpawn("box", "box", 8, 20, reward="coin"),))
        previous, result = self.step_until(level, Action.JUMP, "box_used")
        self.feedback.consume(previous, result, 0)
        self.assertEqual([e.kind for e in self.feedback.effects], ["box", "acorn"])
        self.assertEqual(self.audio.played.count("box"), 1)
        for _ in range(7):
            self.feedback.advance()
        self.assertEqual(self.feedback.box_offset("box"), -8)
        for _ in range(29):
            self.feedback.advance()
        self.assertEqual(self.feedback.hud_pulse, 12)
        self.assertEqual(self.feedback.effects, [])
        self.feedback.reset()
        self.assertEqual(self.feedback.hud_pulse, 0)

    def test_brick_fragments_expire_and_drawing_is_read_only(self):
        self.step_until(small_level(blocks=(BlockSpawn("brick", "brick", 8, 20),)), Action.JUMP, "brick_broken")
        self.assertEqual(len(self.feedback.effects), 8)
        surface = pygame.Surface((320, 200), pygame.SRCALPHA)
        brick = pygame.Surface((40, 40), pygame.SRCALPHA)
        enemy = pygame.Surface((28, 28), pygame.SRCALPHA)
        for _ in range(10):
            self.feedback.draw_world(surface, 0, brick, enemy)
        self.assertTrue(all(e.age == 0 for e in self.feedback.effects))
        for _ in range(36):
            self.feedback.advance()
        self.assertEqual(self.feedback.effects, [])
        self.assertEqual(self.audio.played, ["brick"])

    def test_world_pickup_creates_one_camera_anchored_flight(self):
        level = replace(small_level(), collectibles=(CollectibleSpawn("coin", "coin", 10, 70),))
        core = BasicPlatformerCore(lambda _: level)
        previous = core.reset(seed=1, level_id=level.level_id)
        result = core.step(Action.NOOP)
        self.feedback.consume(previous, result, 5)
        self.assertEqual(len(self.feedback.effects), 1)
        self.assertEqual(self.feedback.effects[0].kind, "acorn")
        self.assertEqual(self.feedback.effects[0].x, 13)
        self.assertEqual(self.audio.played, ["acorn"])

    def test_stomp_keeps_physical_bounce_and_has_visual_impact(self):
        enemy = EnemySpawn("enemy", 10, 80, patrol_left=10, patrol_right=10)
        _, result = self.step_until(small_level(enemies=(enemy,), spawn_bottom=40), Action.NOOP, "enemy_stomped")
        self.assertLess(result.state.player.velocity_y, 0)
        self.assertEqual({e.kind for e in self.feedback.effects}, {"stomp", "squash"})
        self.assertIn("stomp", self.audio.played)

    def test_shield_shatters_at_player_and_reset_clears_audio(self):
        enemy = EnemySpawn("side", 60, 72, patrol_left=60, patrol_right=60)
        berry = PowerupSpawn("berry", "shield", 35, 72)
        _, result = self.step_until(small_level(enemies=(enemy,), powerups=(berry,)), Action.RIGHT_RUN, "shield_absorbed")
        effect = next(e for e in self.feedback.effects if e.kind == "shield_break")
        self.assertEqual(effect.x, result.state.player.x + 12)
        self.assertEqual(result.state.metadata["shield_charges"], 0)
        self.assertIn("shield_break", self.audio.played)
        self.feedback.reset()
        self.assertEqual(self.feedback.effects, [])
        self.assertEqual(self.audio.played, [])

    def test_audio_files_are_short_valid_pcm_and_missing_device_is_safe(self):
        root = Path(__file__).resolve().parents[2] / "assets/audio"
        for name in FeedbackAudio.NAMES:
            with wave.open(str(root / f"{name}.wav")) as sound:
                self.assertEqual(sound.getsampwidth(), 2)
                self.assertEqual(sound.getnchannels(), 1)
                self.assertLessEqual(sound.getnframes() / sound.getframerate(), 0.5)
                self.assertGreater(sound.getnframes(), 1000)
        from unittest.mock import patch
        with patch("pygame.mixer.get_init", return_value=None):
            audio = FeedbackAudio()
            audio.play("box")
            self.assertEqual(audio.sounds, {})


if __name__ == "__main__":
    unittest.main()
