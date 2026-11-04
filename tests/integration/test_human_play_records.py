import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
pygame = pytest.importorskip("pygame")

from ai_platformer.adaptive.human_play import HumanPlayRecords, progress_key
from ai_platformer.adaptive.telemetry import read_session
from ai_platformer.content.play_session import default_play_session, PlayRepository
from ai_platformer.settings import load_gameplay_settings
from source.states.level import Level
from source.states.load_screen import GameOver, LevelComplete
from source.states.main_menu import MainMenu
from source.tools import Game


@pytest.fixture
def screen():
    pygame.init()
    yield pygame.display.set_mode((800, 600))
    pygame.quit()


def tap(game, key):
    game.input.handle_event(pygame.event.Event(pygame.KEYDOWN, key=key))
    game.advance(1 / 60)
    game.input.handle_event(pygame.event.Event(pygame.KEYUP, key=key))


def app(tmp_path, settings=None, clock=None, specs=()):
    settings = settings or load_gameplay_settings(Path("config/gameplay_player.json"))
    session = default_play_session(settings, specs)
    records = HumanPlayRecords(tmp_path, clock=clock)
    session.records = records
    states = {"main_menu": MainMenu(session=session), "level": Level(session=session),
              "game_over": GameOver(session=session), "level_complete": LevelComplete(session=session)}
    return session, records, Game(states, "main_menu")


def test_pause_retry_timeout_and_short_tap_are_recorded_once(screen, tmp_path):
    now = [0.0]
    settings = load_gameplay_settings(Path("config/gameplay_player.json"))
    settings = replace(settings, render_fps=144,
                       physics=replace(settings.physics, max_episode_steps=6))
    session, records, game = app(tmp_path, settings, clock=lambda: now[0])
    try:
        assert not records.active and not (tmp_path / "sessions").exists()
        tap(game, pygame.K_RETURN)
        game.input.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
        game.input.handle_event(pygame.event.Event(pygame.KEYUP, key=pygame.K_SPACE))
        game.advance(1 / 60)
        assert game.state.state.player.velocity_y < 0
        now[0] = 2.0
        tap(game, pygame.K_p)
        frozen = game.state.state
        now[0] = 12.0
        game.advance(0.2)
        assert game.state.state == frozen
        tap(game, pygame.K_p)
        now[0] = 13.0
        tap(game, pygame.K_r)
        for _ in range(6):
            game.advance(1 / 60)
        assert game.state.outcome == "time_limit"
        records.close()
        records.close()  # Close is idempotent even after a natural terminal event.
        paths = list((tmp_path / "sessions").iterdir())
        assert len(paths) == 1
        parsed = read_session(paths[0])
        assert parsed["complete"]
        attempts = parsed["attempts"]
        assert [a["reason"] for a in attempts] == ["restart", "time_limit"]
        assert len({a["attempt_id"] for a in attempts}) == 2
        assert attempts[0]["paused_wall_seconds"] == 10
        assert attempts[0]["active_wall_seconds"] == 3
        assert attempts[1]["simulation_seconds"] == 6 / 60
        events = parsed["events"]
        assert len([e for e in events if e["type"] == "jump_requested"]) == 1
        assert next(e for e in events if e["type"] == "jump_requested")["payload"]["released_before_tick"]
        assert not any(e["type"] == "jump_released" for e in events)
        assert len([e for e in events if e["type"] == "jump_started"]) == 1
        assert parsed["manifest"]["tick_hz"] == 60
        assert parsed["manifest"]["environment"]["environment_id"] == "HumanPlatformer-v1"
        assert records.best(session.prepared)["attempts"] == 2
    finally:
        records.close()


def test_success_persists_switch_closes_record_and_menu_restores_last_entry(screen, tmp_path):
    session, records, game = app(tmp_path)
    try:
        tap(game, pygame.K_RIGHT)
        tap(game, pygame.K_RETURN)
        game.input.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))
        game.input.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LSHIFT))
        for _ in range(200):
            game.advance(1 / 60)
            if game.state.outcome:
                break
        assert game.state.outcome == "success"
        best = dict(records.best(session.prepared))
        assert best["completed"] and best["fastest_ticks"] > 0
        first_key = records.key
        game.input.handle_event(pygame.event.Event(pygame.KEYUP, key=pygame.K_RIGHT))
        tap(game, pygame.K_ESCAPE)
        tap(game, pygame.K_LEFT)
        tap(game, pygame.K_RETURN)
        assert session.index == 0 and records.key != first_key
        records.close()
        parsed = [read_session(p) for p in (tmp_path / "sessions").iterdir()]
        assert all(p["complete"] for p in parsed)
        assert sorted(a["reason"] for p in parsed for a in p["attempts"]) == ["quit", "success"]
    finally:
        records.close()
    restored = HumanPlayRecords(tmp_path)
    try:
        assert restored.store.data["levels"][first_key] == best
        assert restored.store.data["player_id"] == records.store.data["player_id"]
        # Last selection is a source identity, independent of catalog ordering.
        session.records = restored
        menu = MainMenu(session=session)
        assert menu.selection == 0
    finally:
        restored.close()


def test_failed_admission_never_creates_attempt_or_changes_progress(screen, tmp_path):
    session, records, game = app(tmp_path / "save", specs=[tmp_path / "missing.json"])
    try:
        tap(game, pygame.K_LEFT)
        tap(game, pygame.K_RETURN)
        assert isinstance(game.state, MainMenu) and game.state.error
        assert records.store.data["levels"] == {}
        assert not (tmp_path / "save" / "sessions").exists()
    finally:
        records.close()


def test_protocol_and_content_changes_have_separate_progress(tmp_path):
    settings = load_gameplay_settings(Path("config/gameplay_player.json"))
    prepared = default_play_session(settings).prepared
    changed = replace(prepared, settings=replace(settings, physics=replace(settings.physics,
                                                                          max_run_speed=11)))
    level = prepared.repository.definition
    other_map = replace(prepared, repository=PlayRepository(replace(level, goal_x=level.goal_x - 1)))
    assert len({progress_key(prepared), progress_key(changed), progress_key(other_map)}) == 3


def test_recording_failure_does_not_stop_simulation(screen, tmp_path, monkeypatch):
    session, records, game = app(tmp_path)
    try:
        tap(game, pygame.K_RETURN)
        def fail(event):
            raise OSError("disk full")
        monkeypatch.setattr(records.recorder.writer, "append", fail)
        with pytest.warns(RuntimeWarning):
            tap(game, pygame.K_SPACE)
        assert records.recorder.failed and records.notice
        tick = game.state.state.tick
        game.advance(1 / 60)
        assert game.state.state.tick == tick + 1
        records.close()
        path = next((tmp_path / "sessions").iterdir())
        assert json.loads((path / "manifest.json").read_text())["status"] == "incomplete"
    finally:
        records.close()


def test_main_closes_active_record_on_window_exit(screen, tmp_path, monkeypatch):
    from source.main import main

    def run(game):
        tap(game, pygame.K_RETURN)
        game.advance(1 / 60)
    monkeypatch.setattr(Game, "run", run)
    monkeypatch.setattr("sys.argv", ["main", "--save-dir", str(tmp_path)])
    main()
    path = next((tmp_path / "sessions").iterdir())
    parsed = read_session(path)
    assert parsed["complete"] and parsed["attempts"][0]["reason"] == "quit"
    assert parsed["manifest"]["headless"] is True
    assert not parsed["eligible_for_profile"]


def test_identical_geometry_with_new_id_keeps_recording_valid(screen, tmp_path):
    session, records, game = app(tmp_path)
    try:
        tap(game, pygame.K_RETURN)
        prepared = session.prepared
        definition = replace(prepared.repository.definition, level_id="renamed-map")
        renamed = replace(prepared, settings=replace(prepared.settings, level_id="renamed-map"),
                          repository=PlayRepository(definition))
        from ai_platformer.core import BasicPlatformerCore
        core = BasicPlatformerCore(renamed.repository.load, config=renamed.settings.physics)
        state = core.reset(seed=0, level_id="renamed-map")
        records.begin(renamed, session.entries[0], state)
        records.close()
        parsed = read_session(next((tmp_path / "sessions").iterdir()))
        assert parsed["complete"]
        assert [a["level_id"] for a in parsed["attempts"]] == ["level_1", "renamed-map"]
        assert parsed["attempts"][0]["reason"] == "level_switch"
    finally:
        records.close()


def test_closed_human_sessions_feed_existing_profile_pipeline(screen, tmp_path):
    from ai_platformer.adaptive.reporting import build_report

    settings = load_gameplay_settings(Path("config/gameplay_player.json"))
    settings = replace(settings, physics=replace(settings.physics, max_episode_steps=3))
    session, records, game = app(tmp_path, settings)
    try:
        tap(game, pygame.K_RETURN)
        for _ in range(3):
            game.advance(1 / 60)
        records.close()
        report = build_report(list((tmp_path / "sessions").iterdir()))
        assert report["metrics"]["attempts"][0]["actor_type"] == "human"
        assert report["metrics"]["attempts"][0]["profile_eligible"]
        assert len(report["profiles"]) == 1
        assert report["profiles"][0]["status"] != "ready"
    finally:
        records.close()


def test_startup_highlights_saved_local_entry_without_starting_attempt(screen, tmp_path):
    session, records, game = app(tmp_path)
    try:
        tap(game, pygame.K_RIGHT)
        tap(game, pygame.K_RETURN)
    finally:
        records.close()
    session, records, game = app(tmp_path)
    try:
        assert game.state.selection == 1
        assert session.index == 0  # Selection is not admission or an active run.
        assert not records.active
    finally:
        records.close()


def test_repress_and_release_between_ticks_keeps_previous_hold_valid(screen, tmp_path):
    session, records, game = app(tmp_path)
    try:
        tap(game, pygame.K_RETURN)
        tap(game, pygame.K_SPACE)
        game.input.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
        game.input.handle_event(pygame.event.Event(pygame.KEYUP, key=pygame.K_SPACE))
        game.advance(1 / 60)
        assert not records.recorder.failed
        records.close()
        parsed = read_session(next((tmp_path / "sessions").iterdir()))
        assert parsed["complete"]
        releases = [e for e in parsed["events"] if e["type"] == "jump_released"]
        assert [e["payload"]["held_ticks"] for e in releases] == [1]
    finally:
        records.close()
