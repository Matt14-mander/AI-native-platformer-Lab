import argparse
from pathlib import Path

import pygame

from ai_platformer.settings import load_gameplay_settings
from ai_platformer.content.play_session import default_play_session
from source import constants as C
from source import tools
from source.states import level, load_screen, main_menu


def main():
    parser = argparse.ArgumentParser(
        description="Play Puppy Trail with a selected gameplay profile"
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "gameplay_player.json",
        help="gameplay JSON; default: responsive player controls",
    )
    parser.add_argument("--level-spec", type=Path, action="append", default=[],
                        help="add a local LevelSpec to the menu (repeatable)")
    args = parser.parse_args()
    settings = load_gameplay_settings(args.settings)
    session = default_play_session(settings, args.level_spec)
    pygame.init()
    pygame.display.set_mode((C.SCREEN_W, C.SCREEN_H))
    pygame.display.set_caption("抱着小狗去旅行")

    state_dict = {
        "main_menu": main_menu.MainMenu(session=session),
        "load_screen": load_screen.LoadScreen(),
        "level": level.Level(settings, session=session),
        "game_over": load_screen.GameOver(),
        "level_complete": load_screen.LevelComplete(session=session),
    }
    game = tools.Game(state_dict, "main_menu", render_fps=settings.render_fps)
    game.run()


if __name__ == "__main__":
    main()
