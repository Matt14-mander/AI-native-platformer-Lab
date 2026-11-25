import pygame
from tests.integration.test_human_play_records import app, tap, screen
from ai_platformer.adaptive.telemetry import read_session


def test_main_entry_activates_checkpoint_and_r_clears_it(screen, tmp_path):
    session, records, game = app(tmp_path)
    try:
        tap(game, pygame.K_RIGHT)
        tap(game, pygame.K_RETURN)
        assert game.state.core.checkpoints
        game.input.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))
        for _ in range(500):
            game.advance(1 / 60)
            if game.state.state.metadata["checkpoint_stage"]:
                break
        assert game.state.state.metadata["checkpoint_stage"] == 1
        tap(game, pygame.K_r)
        assert game.state.state.tick == 0
        assert game.state.state.metadata["checkpoint_stage"] == 0
        records.close()
        parsed = read_session(next((tmp_path / "sessions").iterdir()))
        assert parsed["complete"]
        assert any(e["type"] == "interaction" and e["payload"]["name"] == "checkpoint_activated"
                   for e in parsed["events"])
    finally:
        records.close()


def test_repeated_respawns_keep_one_valid_attempt_and_no_duplicate_coins(screen, tmp_path):
    from dataclasses import replace
    from tests.unit.test_checkpoints import make_core
    from ai_platformer.core import Action
    from ai_platformer.content.play_session import PlayRepository
    core = make_core()
    session, records, game = app(tmp_path)
    prepared = replace(session.prepared, repository=PlayRepository(core._level),
                       settings=replace(session.prepared.settings, level_id="checkpoints", physics=core.config))
    try:
        records.begin(prepared, session.entries[0], core.state)
        for _ in range(400):
            before = core.state
            result = core.step(Action.RIGHT_RUN)
            records.tick(before, Action.RIGHT_RUN, result, jump_pressed=False, jump_started=False)
            if core.state.metadata["respawns"] == 2:
                break
        records.close()
        parsed = read_session(next((tmp_path / "sessions").iterdir()))
        assert parsed["complete"] and len(parsed["attempts"]) == 1
        assert sum(e["type"] == "coin_collected" for e in parsed["events"]) == 1
        respawns = [e for e in parsed["events"] if e["type"] == "interaction"
                    and e["payload"]["name"] == "checkpoint_respawn"]
        assert len(respawns) == 2
        assert parsed["attempts"][0]["motion"]["backward_distance_px"] == 0
    finally:
        records.close()
