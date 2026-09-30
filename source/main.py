import pygame

from ai_platformer.settings import load_gameplay_settings
from source import constants as C
from source import tools
from source.states import main_menu, load_screen, level

def main():

    settings = load_gameplay_settings()
    pygame.init()
    pygame.display.set_mode((C.SCREEN_W, C.SCREEN_H))
    pygame.display.set_caption("Puppy Trail")

    state_dict = {
        'main_menu': main_menu.MainMenu(),
        'load_screen': load_screen.LoadScreen(),
        'level': level.Level(settings),
        'game_over': load_screen.GameOver(),
        'level_complete': load_screen.LevelComplete(),
    }
    game = tools.Game(state_dict, 'main_menu', render_fps=settings.render_fps)
    game.run()

if __name__ == '__main__':
    main()
