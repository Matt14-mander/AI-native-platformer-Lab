"""Measure jump/input response from real shared-core trajectories."""

from __future__ import annotations

from dataclasses import asdict

from ai_platformer.core import Action, BasicPlatformerCore, LevelDefinition, SolidRect
from ai_platformer.settings import GameplaySettings


def measure_jump(
    settings: GameplaySettings,
    hold_ticks: int,
    *,
    running: bool = False,
    include_trajectory: bool = False,
) -> dict:
    level = LevelDefinition(
        "mechanics_probe", 20000, 600, 200, 540, 19000, (SolidRect(0, 540, 20000, 60),)
    )
    core = BasicPlatformerCore(lambda _: level, config=settings.physics)
    core.reset(seed=0, level_id=level.level_id)
    if running:
        for _ in range(100):
            core.step(Action.RIGHT_RUN)
    initial = core.state.player
    minimum_y = initial.y
    apex_tick = 0
    trajectory = [[0, 0.0]]
    for tick in range(1, 301):
        jump = tick <= hold_ticks
        action = (
            (Action.RIGHT_RUN_JUMP if jump else Action.RIGHT_RUN)
            if running
            else (Action.JUMP if jump else Action.NOOP)
        )
        player = core.step(action).state.player
        if include_trajectory:
            trajectory.append([tick, initial.y - player.y])
        if player.y < minimum_y:
            minimum_y, apex_tick = player.y, tick
        if player.grounded:
            result = {
                "hold_ticks": hold_ticks,
                "height_px": initial.y - minimum_y,
                "apex_ticks": apex_tick,
                "airtime_ticks": tick,
                "airtime_seconds": tick / settings.render_fps,
                "distance_px": player.x - initial.x,
            }
            if include_trajectory:
                result["trajectory"] = trajectory
            return result
    raise AssertionError("jump did not land within 300 physics ticks")


def audit_settings(settings: GameplaySettings) -> dict:
    jumps = [measure_jump(settings, ticks) for ticks in (1, 4, 8, 300)]
    level = LevelDefinition(
        "input_probe", 20000, 600, 200, 540, 19000, (SolidRect(0, 540, 20000, 60),)
    )
    core = BasicPlatformerCore(lambda _: level, config=settings.physics)
    core.reset(seed=0, level_id=level.level_id)
    launches = 0
    for _ in range(600):
        before = core.state.player.grounded
        after = core.step(Action.JUMP).state.player
        launches += int(before and not after.grounded and after.velocity_y < 0)
    continuous_hold = {
        "ticks": 600,
        "launches": launches,
        "final_grounded": core.state.player.grounded,
    }
    core.step(Action.NOOP)
    second = core.step(Action.JUMP).state.player
    moving = {}
    for name, action, maximum in (
        ("walk", Action.RIGHT, settings.physics.max_walk_speed),
        ("run", Action.RIGHT_RUN, settings.physics.max_run_speed),
    ):
        core.reset(seed=0, level_id=level.level_id)
        ticks = 0
        while core.state.player.velocity_x < maximum - 1e-6:
            core.step(action)
            ticks += 1
        full_speed = core.state.player.velocity_x
        walk_switch_speed = core.step(Action.RIGHT).state.player.velocity_x
        stop_ticks = 0
        while core.state.player.velocity_x > 0:
            core.step(Action.NOOP)
            stop_ticks += 1
        moving[name] = {
            "acceleration_ticks": ticks,
            "acceleration_seconds": ticks / settings.render_fps,
            "full_speed": full_speed,
            "speed_after_walk_action": walk_switch_speed,
            "stop_ticks_after_walk_switch": stop_ticks,
        }
    return {
        "physics": asdict(settings.physics),
        "jumps": jumps,
        "full_speed_run_jump": measure_jump(settings, 300, running=True),
        "continuous_hold": continuous_hold,
        "movement": moving,
        "checks": {
            "hold_does_not_retrigger": launches == 1 and continuous_hold["final_grounded"],
            "release_and_press_retriggers": not second.grounded and second.velocity_y < 0,
            "jumps_bounded": all(item["airtime_ticks"] < 300 for item in jumps),
        },
        "fixed_height": len({round(item["height_px"], 6) for item in jumps}) == 1,
    }
