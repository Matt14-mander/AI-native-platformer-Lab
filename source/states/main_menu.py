"""Chinese opening menu with compact controls."""
import pygame
from ai_platformer.rendering.art import fit_height, fit_rect, load_art
from source.ui import panel, text, pressed, SOFT
class MainMenu:
    def __init__(self):
        self.background=fit_height(load_art("backgrounds/lakeside-day.png",crop=False),600)
        self.hero=fit_height(load_art("characters/child-with-puppy-idle.png"),230)
        self.ground=fit_rect(load_art("tilesets/moss-earth-ground.png"),800,60)
        self.enter()
    def enter(self):
        self.finished=False
        self.next="level"
        self.show_help=False
    def is_finished(self):
        return self.finished
    def update(self,surface,keys):
        if pressed(keys,pygame.K_RETURN):
            self.finished=True
        if pressed(keys,pygame.K_h):
            self.show_help=not self.show_help
        if pressed(keys,pygame.K_ESCAPE):
            if self.show_help:
                self.show_help=False
            else:
                pygame.event.post(pygame.event.Event(pygame.QUIT))
        self.draw(surface)

    def draw(self, surface):
        surface.blit(self.background,((surface.get_width()-self.background.get_width())//2,0))
        surface.blit(self.ground,(0,540))
        surface.blit(self.hero,self.hero.get_rect(midbottom=(145,540)))
        panel(surface,pygame.Rect(280,125,480,365))
        text(surface,"抱着小狗去旅行",160,34,x=520)
        text(surface,"沿着湖岸，收集旅途中的小惊喜",217,18,SOFT,x=520)
        if self.show_help:
            lines=["← → 移动   Shift / S 奔跑","空格 / A 跳跃：轻按低跳，长按高跳","Esc / P 暂停   R 快速重试","Enter 开始   H 收起提示"]
        else:
            lines=["Enter   开始旅程","H   操作说明","Esc   退出游戏","← → 移动   空格跳跃   Shift 奔跑"]
        for index,line in enumerate(lines):
            text(surface,line,280+index*45,18 if index==3 or self.show_help else 24,x=520)
