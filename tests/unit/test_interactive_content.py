"""Interactive blocks, enemies, and powerups share deterministic core rules."""

import unittest

from ai_platformer.content.legacy import LegacyLevelRepository
from ai_platformer.core import (
    Action,
    BasicPlatformerCore,
    BlockSpawn,
    EnemySpawn,
    LevelDefinition,
    PowerupSpawn,
    SolidRect,
)


def small_level(*, blocks=(), enemies=(), powerups=(), spawn_bottom=100.0):
    return LevelDefinition(
        level_id="interactive",
        width=300.0,
        height=200.0,
        spawn_x=10.0,
        spawn_bottom=spawn_bottom,
        goal_x=280.0,
        solids=(SolidRect(0.0, 100.0, 300.0, 100.0),) if spawn_bottom == 100.0 else (),
        blocks=blocks,
        enemies=enemies,
        powerups=powerups,
    )


class InteractiveContentTests(unittest.TestCase):
    def test_breakable_brick_is_removed_by_head_hit_and_reset_restores_it(self):
        brick = BlockSpawn("brick-a", "brick", x=8.0, y=20.0)
        level = small_level(blocks=(brick,))
        core = BasicPlatformerCore(lambda _: level)
        core.reset(seed=1, level_id=level.level_id)

        hit = core.step(Action.JUMP)

        self.assertEqual(hit.info["events"], (("brick_broken", "brick-a"),))
        self.assertFalse(next(e for e in hit.state.entities if e.entity_id == "brick-a").active)
        self.assertEqual(hit.state.metadata["score"], 50)
        restored = core.reset(seed=1, level_id=level.level_id)
        self.assertTrue(next(e for e in restored.entities if e.entity_id == "brick-a").active)

    def test_reward_boxes_are_one_shot_and_shield_box_spawns_a_pickup(self):
        coin_box = BlockSpawn("coin-box", "box", x=8.0, y=20.0, reward="coin")
        level = small_level(blocks=(coin_box,))
        core = BasicPlatformerCore(lambda _: level)
        core.reset(seed=1, level_id=level.level_id)
        first = core.step(Action.JUMP)
        self.assertEqual(first.state.metadata["score"], 100)
        self.assertEqual(first.state.metadata["box_coins"], 1)
        self.assertEqual(next(e for e in first.state.entities if e.entity_id == "coin-box").state, "used")
        for _ in range(10):
            core.step(Action.NOOP)
        second = core.step(Action.JUMP)
        self.assertEqual(second.state.metadata["score"], 100)
        self.assertEqual(second.state.metadata["box_coins"], 1)

        shield_box = BlockSpawn("ward-box", "box", x=8.0, y=20.0, reward="shield")
        ward_level = small_level(blocks=(shield_box,))
        ward_core = BasicPlatformerCore(lambda _: ward_level)
        ward_core.reset(seed=1, level_id=ward_level.level_id)
        spawned = ward_core.step(Action.JUMP)
        self.assertIn(("powerup_spawned", "ward-box"), spawned.info["events"])
        self.assertTrue(
            next(
                e for e in spawned.state.entities
                if e.entity_id == "ward-box-shield"
            ).active
        )

    def test_stomp_defeats_enemy_and_side_contact_kills_without_shield(self):
        enemy = EnemySpawn("hedgehog", x=10.0, y=80.0, patrol_left=10.0, patrol_right=10.0)
        falling_level = small_level(enemies=(enemy,), spawn_bottom=40.0)
        core = BasicPlatformerCore(lambda _: falling_level)
        core.reset(seed=7, level_id=falling_level.level_id)
        for _ in range(20):
            result = core.step(Action.NOOP)
            if ("enemy_stomped", "hedgehog") in result.info.get("events", ()):
                break
        self.assertFalse(next(e for e in result.state.entities if e.entity_id == "hedgehog").active)
        self.assertEqual(result.state.metadata["score"], 150)
        self.assertLess(result.state.player.velocity_y, 0)

        side_enemy = EnemySpawn("side", x=60.0, y=72.0, patrol_left=60.0, patrol_right=60.0)
        side_level = small_level(enemies=(side_enemy,))
        side_core = BasicPlatformerCore(lambda _: side_level)
        side_core.reset(seed=7, level_id=side_level.level_id)
        for _ in range(30):
            result = side_core.step(Action.RIGHT_RUN)
            if result.terminated:
                break
        self.assertEqual(result.info["outcome"], "death")
        self.assertFalse(result.state.player.alive)

    def test_shield_absorbs_one_enemy_hit_and_reset_clears_it(self):
        enemy = EnemySpawn("side", x=60.0, y=72.0, patrol_left=60.0, patrol_right=60.0)
        berry = PowerupSpawn("berry", "shield", x=35.0, y=72.0)
        level = small_level(enemies=(enemy,), powerups=(berry,))
        core = BasicPlatformerCore(lambda _: level)
        core.reset(seed=7, level_id=level.level_id)
        events = []
        for _ in range(30):
            result = core.step(Action.RIGHT_RUN)
            events.extend(result.info.get("events", ()))
            if ("shield_absorbed", "side") in events:
                break
        self.assertIn(("shield_collected", "berry"), events)
        self.assertIn(("shield_absorbed", "side"), events)
        self.assertTrue(result.state.player.alive)
        self.assertEqual(result.state.metadata["shield_charges"], 0)
        self.assertGreater(result.state.metadata["invulnerable_ticks"], 0)
        reset = core.reset(seed=7, level_id=level.level_id)
        self.assertEqual(reset.metadata["invulnerable_ticks"], 0)
        self.assertEqual(reset.metadata["shield_charges"], 0)
        self.assertTrue(next(e for e in reset.entities if e.entity_id == "berry").active)

    def test_enemy_patrol_and_interactions_replay_identically(self):
        enemy = EnemySpawn("walker", x=140.0, y=72.0, patrol_left=120.0, patrol_right=160.0)
        berry = PowerupSpawn("berry", "shield", x=35.0, y=72.0)
        level = small_level(enemies=(enemy,), powerups=(berry,))

        def replay():
            core = BasicPlatformerCore(lambda _: level)
            states = [core.reset(seed=99, level_id=level.level_id)]
            states.extend(core.step(Action.RIGHT).state for _ in range(10))
            return states

        self.assertEqual(replay(), replay())

    def test_first_level_contains_playable_intro_entities(self):
        level = LegacyLevelRepository().load("level_1")
        self.assertEqual(len(level.blocks), 5)
        self.assertEqual(len(level.enemies), 1)
        self.assertEqual(len(level.powerups), 1)
        self.assertEqual(level.enemies[0].entity_id, "intro-shadow-hedgehog")


if __name__ == "__main__":
    unittest.main()
