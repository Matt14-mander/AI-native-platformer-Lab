import json
from dataclasses import replace

import pygame

from ai_platformer.adaptive.telemetry import read_session
from tests.integration.test_human_play_records import app, tap, screen


def reach_checkpoint(game):
    tap(game, pygame.K_RIGHT)
    tap(game, pygame.K_RETURN)
    game.input.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))
    for _ in range(500):
        game.advance(1 / 60)
        if game.state.state.metadata["checkpoint_stage"]:
            game.input.handle_event(pygame.event.Event(pygame.KEYUP, key=pygame.K_RIGHT))
            return game.state.state
    raise AssertionError("checkpoint not reached")


def test_restart_application_restores_exact_checkpoint_and_valid_records(screen, tmp_path):
    session, records, game = app(tmp_path)
    try:
        checkpoint = reach_checkpoint(game)
        for _ in range(12):
            game.advance(1 / 60)
        assert game.state.state.tick > checkpoint.tick
    finally:
        records.close()
    session, records, game = app(tmp_path)
    try:
        tap(game, pygame.K_c)
        assert game.state.state == checkpoint
        assert records.resumed
        assert records.recorder.factory.environment_id.endswith("-resumed")
        game.advance(1 / 60)
        assert game.state.state.tick == checkpoint.tick + 1
        tap(game, pygame.K_r)
        assert game.state.state.tick == 0
        assert not records.resumed and records.checkpoint.data is None
        assert json.loads((tmp_path / "checkpoint.json").read_text()) is None
    finally:
        records.close()
    for directory in (tmp_path / "sessions").iterdir():
        assert read_session(directory)["complete"]


def test_failed_resume_does_not_switch_level_or_destroy_save(screen, tmp_path):
    session, records, game = app(tmp_path)
    try:
        reach_checkpoint(game)
    finally:
        records.close()
    session, records, game = app(tmp_path)
    try:
        original = session.prepared
        saved = records.checkpoint.data
        records.checkpoint.data = {**saved, "state_hash": "0" * 64}
        tap(game, pygame.K_c)
        assert session.prepared is original and session.index == 0
        assert not game.state.finished
        assert "验证失败" in game.state.error
        assert records.checkpoint.data is not None
        records.checkpoint.data = saved
        session.settings = replace(session.settings, seed=session.settings.seed + 1)
        tap(game, pygame.K_c)
        assert session.prepared is original
        assert "配置已变化" in game.state.error
        tap(game, pygame.K_RETURN)
        assert game.state.state.tick == 0 and records.checkpoint.data is None
    finally:
        records.close()


def test_old_progress_without_checkpoint_and_corrupt_checkpoint_are_safe(screen, tmp_path):
    session, records, game = app(tmp_path)
    try:
        tap(game, pygame.K_c)
        assert "暂无检查点" in game.state.error
        tap(game, pygame.K_RETURN)
    finally:
        records.close()
    (tmp_path / "checkpoint.json").write_text('{"version": 999}', encoding="utf-8")
    session, records, game = app(tmp_path)
    try:
        assert records.checkpoint.disabled
        tap(game, pygame.K_RETURN)
        assert game.state.state.tick == 0
        assert (tmp_path / "checkpoint.json").read_text() == '{"version": 999}'
    finally:
        records.close()


def test_terminal_run_clears_saved_checkpoint(screen, tmp_path):
    session, records, game = app(tmp_path)
    try:
        reach_checkpoint(game)
        assert records.checkpoint.data
        # Bring the deadline forward without changing the gameplay/update path.
        game.state.core.config = replace(game.state.core.config,
                                        max_episode_steps=game.state.state.tick + 1)
        game.advance(1 / 60)
        assert game.state.outcome == "time_limit"
        assert records.checkpoint.data is None
    finally:
        records.close()


def test_resumed_run_can_save_the_next_checkpoint(screen, tmp_path):
    session, records, game = app(tmp_path)
    try:
        first = reach_checkpoint(game)
    finally:
        records.close()
    session, records, game = app(tmp_path)
    try:
        tap(game, pygame.K_c)
        game.input.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))
        for _ in range(500):
            game.advance(1 / 60)
            if game.state.state.metadata["checkpoint_stage"] == 2:
                break
        second = game.state.state
        assert second.metadata["checkpoint_stage"] == 2 and second.tick > first.tick
    finally:
        records.close()
    session, records, game = app(tmp_path)
    try:
        tap(game, pygame.K_c)
        assert game.state.state == second
        assert game.state.state.metadata["coins_collected"] >= first.metadata["coins_collected"]
    finally:
        records.close()
    for directory in (tmp_path / "sessions").iterdir():
        assert read_session(directory)["complete"]
