"""Camera backtracking and entrypoint-specific pause instructions."""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
pygame = pytest.importorskip("pygame")

from ai_platformer.rendering.pygame_level import PygameLevelRenderer


@pytest.mark.parametrize("width", [800, 1000])
def test_camera_follows_backtracking_without_changing_forward_framing(width):
    renderer = object.__new__(PygameLevelRenderer)
    renderer.screen_width = width
    renderer.level_width = 4000
    renderer.camera_x = 0
    state = lambda x: SimpleNamespace(player=SimpleNamespace(x=x))
    renderer._update_camera(state(1000))
    assert 1000 - renderer.camera_x == pytest.approx(width / 3)
    forward_camera = renderer.camera_x
    renderer._update_camera(state(990))
    assert renderer.camera_x == forward_camera  # No jitter inside the dead zone.
    renderer._update_camera(state(600))
    assert 600 - renderer.camera_x == pytest.approx(width / 4)
    renderer._update_camera(state(0))
    assert renderer.camera_x == 0
    renderer._update_camera(state(4000))
    assert renderer.camera_x == 4000 - width
    renderer.level_width = width / 2
    renderer._update_camera(state(100))
    assert renderer.camera_x == 0


def test_level_spec_pause_instructions_and_actual_event_loop(monkeypatch):
    pytest.importorskip("gymnasium")
    pytest.importorskip("pydantic")
    from scripts import play_level_spec
    from source import ui

    # Exercise the actual CLI loop: pause, restart while paused, resume, quit.
    batches = iter([
        [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p)],
        [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r)],
        [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p)],
        [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)],
    ])
    monkeypatch.setattr(pygame.event, "get", lambda: next(batches))
    monkeypatch.setattr("sys.argv", ["play_level_spec", "--input",
                                    "game_content/level_specs/flat_v1.json"])
    with patch.object(ui, "text", wraps=ui.text) as text:
        play_level_spec.main()
    labels = [call.args[1] for call in text.call_args_list]
    assert labels.count("游戏已暂停") == 2
    assert labels.count("P   继续播放") == 2
    assert "R   重开本关（保持暂停）" in labels
    assert "Esc   退出播放" in labels
    assert "M   返回主菜单" not in labels
    assert "Enter / Esc / P   继续旅程" not in labels


def test_main_game_retains_its_pause_controls():
    from source import ui

    pygame.font.init()
    try:
        with patch.object(ui, "text") as text:
            ui.pause_screen(pygame.Surface((800, 600)))
        labels = [call.args[1] for call in text.call_args_list]
        assert "Enter / Esc / P   继续旅程" in labels
        assert "M   返回主菜单" in labels
    finally:
        pygame.quit()


def test_fonts_are_recreated_after_each_pygame_shutdown():
    from source.ui import font

    for _ in range(3):
        pygame.font.init()
        font(24).render("暂停", True, (255, 255, 255))
        pygame.quit()
        assert font.cache_info().currsize == 0
