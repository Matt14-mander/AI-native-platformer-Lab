"""Shared Chinese typography and keyboard UI."""
from functools import lru_cache
from pathlib import Path
import pygame
CREAM = (255, 239, 210)
SOFT = (208, 224, 211)
@lru_cache(maxsize=16)
def font(size):
    candidates = [Path(__file__).resolve().parents[1] / "assets/fonts/chinese.ttf", Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf")]
    for path in candidates:
        if path.exists():
            return pygame.font.Font(str(path), size)
    match = pygame.font.match_font("notosanscjksc,wenquanyizenhei,pingfangsc,heiti")
    if match:
        return pygame.font.Font(match, size)
    raise RuntimeError("请安装中文字体，或将字体放入 assets/fonts/chinese.ttf")
def pressed(keys, key):
    edge = getattr(keys, "was_pressed", None)
    return edge(key) if edge else bool(keys[key])
def text(surface, label, y, size=24, color=CREAM, x=None):
    image = font(size).render(label, True, color)
    surface.blit(image, image.get_rect(midtop=(surface.get_width()//2 if x is None else x, y)))
def panel(surface, rect, alpha=228):
    layer = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(layer, (27,42,44,alpha), layer.get_rect(), border_radius=20)
    pygame.draw.rect(layer, (175,190,143,145), layer.get_rect(), 1, border_radius=20)
    surface.blit(layer, rect)
def pause_screen(surface):
    shade = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    shade.fill((15,28,33,125))
    surface.blit(shade,(0,0))
    panel(surface,pygame.Rect(surface.get_width()//2-240,140,480,325))
    text(surface,"歇一会儿",167,36)
    text(surface,"游戏已暂停",220,19,SOFT)
    text(surface,"Enter / Esc / P   继续旅程",270)
    text(surface,"R   重新开始本关",316)
    text(surface,"M   返回主菜单",362)
    text(surface,"← → 移动   空格跳跃   Shift 奔跑",416,18,SOFT)
