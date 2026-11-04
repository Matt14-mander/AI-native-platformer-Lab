"""Deterministic movement and interactive level simulation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .actions import Action, Control, control_for
from .engine import StepResult
from .level import BlockSpawn, EnemySpawn, LevelDefinition, PowerupSpawn, SolidRect
from .state import EntitySnapshot, PlayerSnapshot, WorldSnapshot


@dataclass(frozen=True, slots=True)
class PhysicsConfig:
    player_width: float = 24.0
    player_height: float = 32.0
    walk_acceleration: float = 0.15
    run_acceleration: float = 0.3
    turn_acceleration: float = 0.35
    max_walk_speed: float = 6.0
    max_run_speed: float = 12.0
    jump_velocity: float = -10.5
    rising_gravity: float = 0.3
    falling_gravity: float = 1.0
    max_fall_speed: float = 11.0
    max_episode_steps: int = 36_000
    responsive_controls: bool = False
    ground_braking: float = 0.9
    air_acceleration: float = 0.25
    air_braking: float = 0.04
    coyote_ticks: int = 0
    jump_buffer_ticks: int = 0
    jump_cut_velocity: float | None = None


@dataclass(slots=True)
class _EnemyRuntime:
    x: float
    direction: int


class BasicPlatformerCore:
    """One fixed simulation clock shared by keyboard play and agents."""

    def __init__(
        self,
        level_loader: Callable[[str], LevelDefinition],
        *,
        config: PhysicsConfig | None = None,
    ) -> None:
        self._level_loader = level_loader
        self.config = config or PhysicsConfig()
        self._level: LevelDefinition | None = None
        self._state: WorldSnapshot | None = None
        self._x = 0.0
        self._y = 0.0
        self._velocity_x = 0.0
        self._velocity_y = 0.0
        self._grounded = False
        self._alive = True
        self._facing = 1
        self._tick = 0
        self._seed = 0
        self._episode_id = ""
        self._jump_was_pressed = False
        self.jump_started_this_tick = False
        self._coyote_remaining = 0
        self._jump_buffer_remaining = 0
        self._jump_cut = False
        self._collected_ids: set[str] = set()
        self._collected_powerups: set[str] = set()
        self._broken_blocks: set[str] = set()
        self._used_boxes: set[str] = set()
        self._defeated_enemies: set[str] = set()
        self._enemy_runtime: dict[str, _EnemyRuntime] = {}
        self._spawned_powerups: list[PowerupSpawn] = []
        self._shield_charges = 0
        self._invulnerable_ticks = 0
        self._box_coins = 0
        self._score = 0
        self._events: list[tuple[str, str]] = []

    @property
    def state(self) -> WorldSnapshot:
        if self._state is None:
            raise RuntimeError("reset() must be called before reading state")
        return self._state

    def reset(self, *, seed: int, level_id: str) -> WorldSnapshot:
        self.jump_started_this_tick = False
        self._level = self._level_loader(level_id)
        self._seed = seed
        self._episode_id = f"{level_id}:{seed}"
        self._tick = 0
        self._x = self._level.spawn_x
        self._y = self._level.spawn_bottom - self.config.player_height
        self._velocity_x = 0.0
        self._velocity_y = 0.0
        self._alive = True
        self._facing = 1
        self._jump_was_pressed = False
        self._coyote_remaining = 0
        self._jump_buffer_remaining = 0
        self._jump_cut = False
        self._collected_ids.clear()
        self._collected_powerups.clear()
        self._broken_blocks.clear()
        self._used_boxes.clear()
        self._defeated_enemies.clear()
        self._enemy_runtime = {
            enemy.entity_id: _EnemyRuntime(enemy.x, enemy.direction)
            for enemy in self._level.enemies
        }
        self._spawned_powerups.clear()
        self._shield_charges = 0
        self._invulnerable_ticks = 0
        self._box_coins = 0
        self._score = 0
        self._events.clear()
        self._grounded = self._has_support()
        self._state = self._snapshot()
        return self._state

    def step(self, action: Action, *, jump_pressed: bool | None = None) -> StepResult:
        self.jump_started_this_tick = False
        level = self._require_level()
        if not self._alive or self.state.progress >= 1.0:
            raise RuntimeError("episode is finished; call reset() before step()")

        control = control_for(action)
        was_grounded = self._grounded
        previous_progress = self.state.progress
        previous_bottom = self._y + self.config.player_height
        self._tick += 1
        self._events.clear()
        self._invulnerable_ticks = max(0, self._invulnerable_ticks - 1)
        self._update_horizontal_velocity(control)
        self._start_jump(control, jump_pressed=jump_pressed)
        self._move_horizontal()
        self._apply_vertical_acceleration(control)
        self._move_vertical()
        # A press just before landing is consumed on the landing tick itself.
        if self.config.jump_buffer_ticks > 0 and self._grounded and self._jump_buffer_remaining > 0:
            self._try_jump(control)
        collected = self._collect_overlapping_items()
        self._collect_powerups()
        self._advance_enemies()
        enemy_outcome = self._resolve_enemy_contacts(previous_bottom)

        outcome: str | None = enemy_outcome
        if self._y > level.height:
            self._alive = False
            outcome = "death"
        if self._alive and self._progress() >= 1.0:
            outcome = "success"

        truncated = self._tick >= self.config.max_episode_steps and outcome is None
        if truncated:
            outcome = "time_limit"
        self._jump_was_pressed = control.jump
        self._jump_buffer_remaining = max(0, self._jump_buffer_remaining - 1)
        if not was_grounded:
            self._coyote_remaining = max(0, self._coyote_remaining - 1)
        self._state = self._snapshot()
        reward = self._state.progress - previous_progress + 0.05 * len(collected)
        if outcome == "success":
            reward += 1.0
        elif outcome == "death":
            reward -= 1.0

        info: dict[str, object] = {
            "score": self._score,
            "coins_collected": len(self._collected_ids),
            "shield_charges": self._shield_charges,
        }
        if outcome:
            info["outcome"] = outcome
        if collected:
            info["collected"] = collected
        if self._events:
            info["events"] = tuple(self._events)
        return StepResult(
            state=self._state,
            reward=reward,
            terminated=outcome in {"success", "death"},
            truncated=truncated,
            info=info,
        )

    def _require_level(self) -> LevelDefinition:
        if self._level is None:
            raise RuntimeError("reset() must be called before reading state")
        return self._level

    def _collision_solids(self) -> list[tuple[SolidRect, BlockSpawn | None]]:
        level = self._require_level()
        solids = [(solid, None) for solid in level.solids]
        solids.extend(
            (block.rect, block)
            for block in level.blocks
            if block.entity_id not in self._broken_blocks
        )
        return solids

    def _update_horizontal_velocity(self, control: Control) -> None:
        if self.config.responsive_controls:
            if control.horizontal:
                self._facing = control.horizontal
            max_speed = self.config.max_run_speed if control.run else self.config.max_walk_speed
            target = control.horizontal * max_speed
            if not self._grounded:
                rate = self.config.air_acceleration if control.horizontal else self.config.air_braking
            elif not control.horizontal or (
                self._velocity_x * target > 0 and abs(self._velocity_x) > abs(target)
            ):
                rate = self.config.ground_braking
            elif self._velocity_x * target < 0:
                rate = self.config.turn_acceleration
            else:
                rate = self.config.run_acceleration if control.run else self.config.walk_acceleration
            delta = target - self._velocity_x
            self._velocity_x += max(-rate, min(rate, delta))
            return
        if control.horizontal == 0:
            acceleration = self.config.run_acceleration if control.run else self.config.walk_acceleration
            if self._velocity_x > 0:
                self._velocity_x = max(0.0, self._velocity_x - acceleration)
            elif self._velocity_x < 0:
                self._velocity_x = min(0.0, self._velocity_x + acceleration)
            return
        self._facing = control.horizontal
        max_speed = self.config.max_run_speed if control.run else self.config.max_walk_speed
        normal_acceleration = self.config.run_acceleration if control.run else self.config.walk_acceleration
        is_turning = self._velocity_x != 0 and (self._velocity_x > 0) != (control.horizontal > 0)
        acceleration = self.config.turn_acceleration if is_turning else normal_acceleration
        self._velocity_x += control.horizontal * acceleration
        self._velocity_x = max(-max_speed, min(max_speed, self._velocity_x))

    def _start_jump(self, control: Control, *, jump_pressed: bool | None = None) -> None:
        if self._grounded:
            self._coyote_remaining = self.config.coyote_ticks
        pressed = control.jump and not self._jump_was_pressed
        if jump_pressed is not None:
            pressed = jump_pressed
        if pressed:
            self._jump_buffer_remaining = max(1, self.config.jump_buffer_ticks)
        self._try_jump(control)

    def _try_jump(self, control: Control) -> None:
        if self._jump_buffer_remaining <= 0:
            return
        if not (self._grounded or self._coyote_remaining > 0):
            return
        self._velocity_y = self.config.jump_velocity
        # Observer-only fact; not part of snapshot/reward/replay contracts.
        self.jump_started_this_tick = True
        self._grounded = False
        self._jump_buffer_remaining = 0
        self._coyote_remaining = 0
        self._jump_cut = False
        self._cut_jump(control)

    def _cut_jump(self, control: Control) -> None:
        if self.config.jump_cut_velocity is not None and not control.jump and self._velocity_y < 0:
            self._velocity_y = max(self._velocity_y, self.config.jump_cut_velocity)
            self._jump_cut = True

    def _apply_vertical_acceleration(self, control: Control) -> None:
        self._cut_jump(control)
        gravity = self.config.rising_gravity if self._velocity_y < 0 and control.jump and not self._jump_cut else self.config.falling_gravity
        self._velocity_y = min(self.config.max_fall_speed, self._velocity_y + gravity)

    def _move_horizontal(self) -> None:
        level = self._require_level()
        self._x += self._velocity_x
        self._x = max(0.0, min(self._x, level.width - self.config.player_width))
        for solid, _block in self._collision_solids():
            if not self._overlaps(solid):
                continue
            if self._velocity_x > 0:
                self._x = solid.x - self.config.player_width
            elif self._velocity_x < 0:
                self._x = solid.right
            self._velocity_x = 0.0

    def _move_vertical(self) -> None:
        previous_y = self._y
        self._y += self._velocity_y
        self._grounded = False
        for solid, block in self._collision_solids():
            if not self._overlaps(solid):
                continue
            previous_bottom = previous_y + self.config.player_height
            if self._velocity_y >= 0 and previous_bottom <= solid.y:
                self._y = solid.y - self.config.player_height
                self._velocity_y = 0.0
                self._grounded = True
            elif self._velocity_y < 0 and previous_y >= solid.bottom:
                self._y = solid.bottom
                self._velocity_y = 0.0
                if block is not None:
                    self._hit_block(block)
        if not self._grounded:
            self._grounded = self._has_support()

    def _hit_block(self, block: BlockSpawn) -> None:
        if block.kind == "brick":
            self._broken_blocks.add(block.entity_id)
            self._score += 50
            self._events.append(("brick_broken", block.entity_id))
        elif block.entity_id not in self._used_boxes:
            self._used_boxes.add(block.entity_id)
            self._events.append(("box_used", block.entity_id))
            if block.reward == "coin":
                self._box_coins += 1
                self._score += 100
            elif block.reward == "shield":
                self._spawned_powerups.append(
                    PowerupSpawn(
                        entity_id=f"{block.entity_id}-shield",
                        kind="shield",
                        x=block.x + 8,
                        y=block.y - 26,
                    )
                )
                self._events.append(("powerup_spawned", block.entity_id))
        else:
            self._events.append(("box_bumped", block.entity_id))

    def _has_support(self) -> bool:
        feet = self._y + self.config.player_height
        player_right = self._x + self.config.player_width
        return any(
            abs(feet - solid.y) <= 1e-6
            and self._x < solid.right
            and player_right > solid.x
            for solid, _block in self._collision_solids()
        )

    def _overlaps(self, solid: SolidRect) -> bool:
        return (
            self._x < solid.right
            and self._x + self.config.player_width > solid.x
            and self._y < solid.bottom
            and self._y + self.config.player_height > solid.y
        )

    def _progress(self) -> float:
        level = self._require_level()
        distance = level.goal_x - level.spawn_x
        return max(0.0, min(1.0, (self._x - level.spawn_x) / distance))

    def _collect_overlapping_items(self) -> tuple[str, ...]:
        collected: list[str] = []
        for item in self._require_level().collectibles:
            if item.entity_id in self._collected_ids:
                continue
            if self._intersects(item.x, item.y, item.width, item.height):
                self._collected_ids.add(item.entity_id)
                self._score += item.score
                collected.append(item.entity_id)
        return tuple(collected)

    def _collect_powerups(self) -> None:
        powerups = self._require_level().powerups + tuple(self._spawned_powerups)
        for item in powerups:
            if item.entity_id in self._collected_powerups:
                continue
            if self._intersects(item.x, item.y, item.width, item.height):
                self._collected_powerups.add(item.entity_id)
                self._shield_charges = 1
                self._score += 200
                self._events.append(("shield_collected", item.entity_id))

    def _advance_enemies(self) -> None:
        level = self._require_level()
        for enemy in level.enemies:
            if enemy.entity_id in self._defeated_enemies:
                continue
            runtime = self._enemy_runtime[enemy.entity_id]
            next_x = runtime.x + runtime.direction * enemy.speed
            if not enemy.patrol_left <= next_x <= enemy.patrol_right or self._enemy_hits_wall(
                enemy, next_x
            ):
                runtime.direction *= -1
                next_x = runtime.x + runtime.direction * enemy.speed
            runtime.x = max(enemy.patrol_left, min(enemy.patrol_right, next_x))

    def _enemy_hits_wall(self, enemy: EnemySpawn, x: float) -> bool:
        enemy_right = x + enemy.width
        for solid, _block in self._collision_solids():
            if (
                x < solid.right
                and enemy_right > solid.x
                and enemy.y < solid.bottom
                and enemy.y + enemy.height > solid.y
            ):
                return True
        return False

    def _resolve_enemy_contacts(self, previous_bottom: float) -> str | None:
        for enemy in self._require_level().enemies:
            if enemy.entity_id in self._defeated_enemies:
                continue
            runtime = self._enemy_runtime[enemy.entity_id]
            if not self._intersects(runtime.x, enemy.y, enemy.width, enemy.height):
                continue
            if self._velocity_y > 0 and previous_bottom <= enemy.y + 6:
                self._defeated_enemies.add(enemy.entity_id)
                self._y = enemy.y - self.config.player_height
                self._velocity_y = -6.5
                self._grounded = False
                self._score += 150
                self._events.append(("enemy_stomped", enemy.entity_id))
            elif self._invulnerable_ticks > 0:
                continue
            elif self._shield_charges:
                self._shield_charges -= 1
                self._invulnerable_ticks = 90
                self._velocity_y = -5.0
                self._grounded = False
                self._events.append(("shield_absorbed", enemy.entity_id))
            else:
                self._alive = False
                self._events.append(("player_hit", enemy.entity_id))
                return "death"
        return None

    def _intersects(self, x: float, y: float, width: float, height: float) -> bool:
        return (
            self._x < x + width
            and self._x + self.config.player_width > x
            and self._y < y + height
            and self._y + self.config.player_height > y
        )

    def _snapshot(self) -> WorldSnapshot:
        level = self._require_level()
        entities: list[EntitySnapshot] = [
            EntitySnapshot(
                entity_id=item.entity_id,
                kind=item.kind,
                x=item.x,
                y=item.y,
                active=item.entity_id not in self._collected_ids,
                width=item.width,
                height=item.height,
            )
            for item in level.collectibles
        ]
        entities.extend(
            EntitySnapshot(
                entity_id=block.entity_id,
                kind=block.kind,
                x=block.x,
                y=block.y,
                active=block.entity_id not in self._broken_blocks,
                width=block.width,
                height=block.height,
                state="used" if block.entity_id in self._used_boxes else "",
            )
            for block in level.blocks
        )
        entities.extend(
            EntitySnapshot(
                entity_id=enemy.entity_id,
                kind="enemy",
                x=self._enemy_runtime[enemy.entity_id].x,
                y=enemy.y,
                active=enemy.entity_id not in self._defeated_enemies,
                width=enemy.width,
                height=enemy.height,
                facing=self._enemy_runtime[enemy.entity_id].direction,
            )
            for enemy in level.enemies
        )
        entities.extend(
            EntitySnapshot(
                entity_id=item.entity_id,
                kind="powerup-shield",
                x=item.x,
                y=item.y,
                active=item.entity_id not in self._collected_powerups,
                width=item.width,
                height=item.height,
            )
            for item in level.powerups + tuple(self._spawned_powerups)
        )
        return WorldSnapshot(
            episode_id=self._episode_id,
            tick=self._tick,
            level_id=level.level_id,
            seed=self._seed,
            player=PlayerSnapshot(
                x=self._x,
                y=self._y,
                velocity_x=self._velocity_x,
                velocity_y=self._velocity_y,
                grounded=self._grounded,
                alive=self._alive,
                facing=self._facing,
            ),
            entities=tuple(entities),
            progress=self._progress(),
            metadata={
                "score": self._score,
                "coins_collected": len(self._collected_ids),
                "coins_total": len(level.collectibles),
                "box_coins": self._box_coins,
                "shield_charges": self._shield_charges,
                "invulnerable_ticks": self._invulnerable_ticks,
                "enemies_defeated": len(self._defeated_enemies),
                "blocks_broken": len(self._broken_blocks),
            },
        )
