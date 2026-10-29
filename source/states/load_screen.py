"""Persistent Chinese results, with immediate retry."""
import pygame
from ai_platformer.rendering.art import fit_height, load_art
from source.ui import panel, text, pressed, SOFT
class LoadScreen:
    def __init__(self):
        self.background=fit_height(load_art("backgrounds/lakeside-day.png",crop=False),600)
        self.summary={}
        self.enter()
    def enter(self):
        self.finished=False
        self.next="level"
    def receive_result(self,summary):
        self.summary=dict(summary)
    def is_finished(self):
        return self.finished
    def update(self,surface,keys):
        self.finished=True
    def draw(self,surface):
        surface.blit(self.background,((surface.get_width()-self.background.get_width())//2,0))
        panel(surface,pygame.Rect(surface.get_width()//2-260,110,520,390))
        text(surface,self.title,142,36)
        text(surface,self.subtitle,201,19,SOFT)
        score=self.summary.get("score",0)
        coins=self.summary.get("coins",0)
        seconds=int(self.summary.get("ticks",0)/60)
        text(surface,f"得分  {score}     橡果  {coins}",267,25)
        text(surface,f"用时  {seconds//60:02d}:{seconds%60:02d}     击败敌人  {self.summary.get('enemies',0)}",313,20,SOFT)
        text(surface,"Enter / R   再玩一次" if isinstance(self,LevelComplete) else "Enter / R   立即重试",389,25)
        text(surface,"Esc / M   返回主菜单",443,20,SOFT)
class GameOver(LoadScreen):
    title="没关系，再试一次"
    subtitle="调整起跳时机，小狗还在陪着你。"
    def update(self,surface,keys):
        if pressed(keys,pygame.K_RETURN) or pressed(keys,pygame.K_r):
            self.next,self.finished="level",True
        elif pressed(keys,pygame.K_ESCAPE) or pressed(keys,pygame.K_m):
            self.next,self.finished="main_menu",True
        self.draw(surface)
class LevelComplete(GameOver):
    title="抵达终点！"
    subtitle="这一段湖岸旅程，和小狗一起完成了。"
