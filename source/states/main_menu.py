"""Chinese opening menu with compact controls."""
import pygame
from ai_platformer.rendering.art import fit_height, fit_rect, load_art
from source.ui import panel, text, pressed, SOFT
from ai_platformer.content.play_session import PlayAdmissionError
class MainMenu:
    def __init__(self, *, session=None):
        self.session = session
        self.background=fit_height(load_art("backgrounds/lakeside-day.png",crop=False),600)
        self.hero=fit_height(load_art("characters/child-with-puppy-idle.png"),230)
        self.ground=fit_rect(load_art("tilesets/moss-earth-ground.png"),800,60)
        self.enter()
    def enter(self):
        self.finished=False
        self.next="level"
        self.show_help=False
        self.selection = self.session.index if self.session else 0
        self.error = ""
    def is_finished(self):
        return self.finished
    def update(self,surface,keys):
        if self.session and not self.show_help:
            direction = int(pressed(keys, pygame.K_RIGHT)) - int(pressed(keys, pygame.K_LEFT))
            if direction:
                self.selection = (self.selection + direction) % len(self.session.entries)
                self.error = ""
        if pressed(keys,pygame.K_RETURN):
            try:
                if self.session:
                    self.error = "正在检查关卡，请稍候…"
                    self.draw(surface)
                    pygame.display.flip()
                    self.session.select(self.selection)
                self.error = ""
                self.finished=True
            except PlayAdmissionError as error:
                self.error = str(error)
                self.show_help = False
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
            if self.session:
                entry = self.session.entries[self.selection]
                lines = [f"←  {entry.title}  →",
                         f"第 {self.selection + 1} / {len(self.session.entries)} 关 · Enter 开始",
                         "H 操作说明   Esc 退出", self.error or "选择关卡，和小狗一起出发"]
        for index,line in enumerate(lines):
            text(surface,line,280+index*45,18 if index==3 or self.show_help else 22,x=520)
