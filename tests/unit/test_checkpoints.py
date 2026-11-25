from ai_platformer.core import BasicPlatformerCore, Action
from ai_platformer.core.level import LevelDefinition, SolidRect, CollectibleSpawn


def make_core(enabled=True):
    level = LevelDefinition("checkpoints", 1200, 600, 48, 540, 1100,
                            (SolidRect(0, 540, 500, 60), SolidRect(700, 540, 500, 60)),
                            (CollectibleSpawn("coin", "coin", 200, 508),))
    core = BasicPlatformerCore(lambda _: level, checkpoints=enabled)
    core.reset(seed=7, level_id=level.level_id)
    return core


def test_checkpoint_activation_death_recovery_and_full_reset_are_deterministic():
    cores = [make_core(), make_core()]
    events = []
    for _ in range(400):
        results = [core.step(Action.RIGHT_RUN) for core in cores]
        assert results[0] == results[1]
        events.extend(results[0].info.get("events", ()))
        if results[0].state.metadata["respawns"] == 2:
            break
    state = cores[0].state
    assert state.metadata["respawns"] == 2
    assert state.player.alive and state.player.grounded
    assert state.metadata["coins_collected"] == 1 and state.metadata["score"] == 100
    assert not results[0].terminated
    assert state.tick > 0
    assert [name for name, _ in events].count("checkpoint_activated") == 1
    assert [name for name, _ in events].count("checkpoint_respawn") == 2
    for core in cores:
        reset = core.reset(seed=7, level_id="checkpoints")
        assert reset.tick == 0 and reset.player.x == 48
        assert reset.metadata["checkpoint_position"] is None
        assert reset.metadata["respawns"] == reset.metadata["score"] == 0
        assert reset.entities[0].active


def test_default_ai_core_still_ends_on_death():
    core = make_core(False)
    for _ in range(200):
        result = core.step(Action.RIGHT_RUN)
        if result.terminated:
            break
    assert result.info["outcome"] == "death"
    assert "checkpoint_stage" not in result.state.metadata


def test_respawn_does_not_extend_time_limit():
    from dataclasses import replace
    core = make_core()
    for _ in range(200):
        result = core.step(Action.RIGHT_RUN)
        if result.state.metadata["respawns"]:
            break
    core.config = replace(core.config, max_episode_steps=core.state.tick + 1)
    result = core.step(Action.NOOP)
    assert result.truncated and result.info["outcome"] == "time_limit"
