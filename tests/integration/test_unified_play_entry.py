"""Human menu/LevelSpec admission and repeated transitions use the shared core."""

import os
from dataclasses import asdict, replace
from pathlib import Path

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
pygame = pytest.importorskip("pygame")
pytest.importorskip("gymnasium")
pytest.importorskip("pydantic")

from ai_platformer.content.play_session import PlayAdmissionError, default_play_session
from ai_platformer.core import Action, BasicPlatformerCore
from ai_platformer.settings import load_gameplay_settings
from source.states.level import Level
from source.states.load_screen import GameOver, LevelComplete
from source.states.main_menu import MainMenu
from source.tools import Game


@pytest.fixture
def settings():
    return load_gameplay_settings(Path("config/gameplay_player.json"))


@pytest.fixture
def surface():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    yield screen
    pygame.quit()


def tap(game, key):
    game.input.handle_event(pygame.event.Event(pygame.KEYDOWN, key=key))
    game.advance(1 / 60)
    game.input.handle_event(pygame.event.Event(pygame.KEYUP, key=key))


def make_game(settings, extra=()):
    session = default_play_session(settings, extra)
    states = {"main_menu": MainMenu(session=session), "level": Level(session=session),
              "game_over": GameOver(), "level_complete": LevelComplete(session=session)}
    return session, Game(states, "main_menu")


def test_menu_select_spec_retry_focus_and_return_to_legacy(surface, settings):
    session, game = make_game(settings)
    tap(game, pygame.K_RIGHT)
    tap(game, pygame.K_RETURN)
    level = game.state
    assert isinstance(level, Level)
    assert level.settings.level_id == "course_flat_08"
    assert level.settings.physics == settings.physics
    assert level.state.tick == 0
    assert level.repository is session.prepared.repository
    # A tap shorter than one physics tick must still reach the shared input path.
    game.input.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
    game.input.handle_event(pygame.event.Event(pygame.KEYUP, key=pygame.K_SPACE))
    game.advance(1 / 60)
    assert level.state.player.velocity_y < 0
    game.input.handle_event(pygame.event.Event(pygame.WINDOWFOCUSLOST))
    game.advance(1 / 60)
    frozen = level.state
    game.input.handle_event(pygame.event.Event(pygame.WINDOWFOCUSGAINED))
    game.advance(1 / 60)
    assert level.paused and level.state == frozen
    tap(game, pygame.K_RETURN)
    assert not level.paused
    level.state = replace(level.state, metadata={"score": 999})
    level.outcome = "death"
    level.terminal_timer = pygame.time.get_ticks()
    tap(game, pygame.K_RETURN)
    assert level.state.tick == 0 and level.state.metadata["score"] == 0
    assert not level.renderer.feedback.effects
    tap(game, pygame.K_ESCAPE)
    tap(game, pygame.K_m)
    assert isinstance(game.state, MainMenu)
    assert game.state.selection == 1
    tap(game, pygame.K_LEFT)
    tap(game, pygame.K_RETURN)
    assert game.state.settings.level_id == settings.level_id
    assert game.state.state.tick == 0
    assert game.state.renderer.camera_x == 0


def test_success_next_level_and_last_level_do_not_wrap(surface, settings):
    session, game = make_game(settings)
    tap(game, pygame.K_RETURN)
    level = game.state
    level.state = replace(level.state, metadata={"score": 800})
    level.outcome = "success"
    level.terminal_timer = pygame.time.get_ticks()
    tap(game, pygame.K_RETURN)
    assert isinstance(game.state, LevelComplete)
    assert game.state.summary["score"] == 800
    tap(game, pygame.K_n)
    assert isinstance(game.state, Level)
    assert session.index == 1 and game.state.state.tick == 0
    assert game.state.state.metadata["score"] == 0
    game.state.outcome = "success"
    game.state.terminal_timer = pygame.time.get_ticks()
    tap(game, pygame.K_RETURN)
    tap(game, pygame.K_n)
    assert isinstance(game.state, LevelComplete) and session.index == 1
    tap(game, pygame.K_r)
    assert game.state.settings.level_id == "course_flat_08"


def test_invalid_map_stays_in_menu_and_next_failure_preserves_result(surface, settings, tmp_path):
    bad = tmp_path / "broken.json"
    bad.write_text("{", encoding="utf-8")
    session, game = make_game(settings, [bad])
    original = session.prepared
    tap(game, pygame.K_LEFT)  # Wrap to the invalid last entry.
    tap(game, pygame.K_RETURN)
    assert isinstance(game.state, MainMenu)
    assert game.state.error and session.prepared is original
    assert session.index == 0
    session.select(1)
    result = game.state_dict["level_complete"]
    game.state = result
    result.enter()
    result.receive_result({"score": 42})
    tap(game, pygame.K_n)
    assert game.state is result and result.error
    assert result.summary["score"] == 42 and session.index == 1
    tap(game, pygame.K_r)
    assert game.state.settings.level_id == "course_flat_08"


def test_admission_route_uses_exact_human_physics_and_replays_on_game_core(settings):
    session = default_play_session(settings)
    session.select(1)
    prepared = session.prepared
    route = prepared.route
    assert route["environment"]["physics"] == asdict(settings.physics)
    assert route["environment"]["action_repeat"] == 1
    core = BasicPlatformerCore(prepared.repository.load, config=settings.physics)
    core.reset(seed=settings.seed, level_id=prepared.settings.level_id)
    for action in route["actions"]:
        result = core.step(Action(action))
    assert result.info["outcome"] == "success"
    assert result.state.tick == route["ticks"]


def test_spec_retry_uses_admitted_snapshot_not_rewritten_disk(settings, tmp_path):
    path = tmp_path / "local.json"
    path.write_text(Path("game_content/level_specs/flat_v1.json").read_text(), encoding="utf-8")
    session = default_play_session(settings, [path, path])
    assert len(session.entries) == 3
    session.select(2)
    prepared = session.prepared
    path.write_text("invalid", encoding="utf-8")
    session.select(2)
    assert session.prepared is prepared
    assert prepared.repository.load(prepared.settings.level_id).width == 768
    session.select(0)
    with pytest.raises(PlayAdmissionError):
        session.select(2)
    assert session.index == 0


def test_static_valid_but_unverified_candidate_is_not_admitted(settings, monkeypatch):
    from ai_platformer.content import routes

    monkeypatch.setattr(routes, "search_route", lambda *a, **kw: {"status": "unknown"})
    session = default_play_session(settings)
    with pytest.raises(PlayAdmissionError, match="暂未找到"):
        session.select(1)
    assert session.index == 0


def test_main_cli_wires_custom_specs_into_same_game(monkeypatch):
    from source.main import main

    observed = []

    def run(game):
        session = game.state.session
        assert game.state_dict["level"].session is session
        assert game.state_dict["level_complete"].session is session
        assert len(session.entries) == 3
        observed.append(session.entries[-1].path.name)
        tap(game, pygame.K_RIGHT)
        tap(game, pygame.K_RETURN)
        assert game.state.settings.level_id == "course_flat_08"
        pygame.quit()

    monkeypatch.setattr(Game, "run", run)
    monkeypatch.setattr("sys.argv", ["main", "--no-save", "--level-spec", "game_content/level_specs/full_v1.json"])
    main()
    assert observed == ["full_v1.json"]


def test_transition_discards_old_screen_catchup(surface, settings):
    session, game = make_game(settings)
    game.input.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN))
    game.input.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))
    game.advance(0.25)
    assert isinstance(game.state, Level)
    assert game.state.state.tick == 0
    game.advance(1 / 60)
    assert game.state.state.tick == 1


def test_failed_admission_from_help_is_visible_and_recoverable(surface, settings, tmp_path):
    session, game = make_game(settings, [tmp_path / "missing.json"])
    original = session.prepared
    tap(game, pygame.K_LEFT)
    tap(game, pygame.K_h)
    tap(game, pygame.K_RETURN)
    assert isinstance(game.state, MainMenu)
    assert game.state.error and not game.state.show_help
    assert session.prepared is original
    tap(game, pygame.K_RIGHT)
    tap(game, pygame.K_RETURN)
    assert isinstance(game.state, Level)
    assert game.state.settings.level_id == settings.level_id
