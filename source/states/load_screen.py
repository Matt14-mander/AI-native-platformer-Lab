"""Persistent Chinese results, with immediate retry."""
import pygame
from ai_platformer.rendering.art import fit_height, load_art
from source.ui import panel, text, pressed, SOFT
from ai_platformer.content.play_session import PlayAdmissionError
class LoadScreen:
    title="准备出发"
    subtitle="新的湖岸旅程即将开始。"
    def __init__(self, *, session=None):
        self.session = session
        self.background=fit_height(load_art("backgrounds/lakeside-day.png",crop=False),600)
        self.summary={}
        self.error = ""
        self.enter()
    def enter(self):
        self.finished=False
        self.next="level"
        self.summary={}
        self.error = ""
    def receive_result(self,summary):
        self.summary=dict(summary)
    def is_finished(self):
        return self.finished
    def update(self,surface,keys):
        self.finished=True
    def draw(self,surface):
        surface.blit(self.background,((surface.get_width()-self.background.get_width())//2,0))
        panel(surface,pygame.Rect(surface.get_width()//2-260,110,520,390))
        timed_out = self.summary.get("outcome") == "time_limit"
        text(surface,"本次旅程时间已到" if timed_out else self.title,142,36)
        text(surface,"已经走过的路，都会成为下一次的经验。" if timed_out else self.subtitle,201,19,SOFT)
        score=self.summary.get("score",0)
        coins=self.summary.get("coins",0)
        seconds=int(self.summary.get("ticks",0)/60)
        text(surface,f"得分  {score}     橡果  {coins}",267,25)
        text(surface,f"用时  {seconds//60:02d}:{seconds%60:02d}     击败敌人  {self.summary.get('enemies',0)}",313,20,SOFT)
        text(surface,"Enter / R   再玩一次" if isinstance(self,LevelComplete) else "Enter / R   立即重试",389,25)
        if isinstance(self, LevelComplete) and self.session:
            label = self.error or ("N   前往下一关" if self.session.has_next else "已完成目录中的最后一关")
            text(surface, label, 352, 18, SOFT)
        text(surface,"Esc / M   返回主菜单",443,20,SOFT)
        if self.session and self.session.records:
            records = self.session.records
            best = records.best(self.session.prepared)
            label = records.notice
            if not label and best:
                label = f"已保存 · 最高分 {best['best_score']} · 最多橡果 {best['best_coins']}"
                if best["fastest_ticks"] is not None:
                    label += f" · 最快 {best['fastest_ticks'] / 60:.1f}秒"
            if label:
                text(surface, label, 478, 16, SOFT)
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
    def update(self, surface, keys):
        if self.session and self.session.has_next and pressed(keys, pygame.K_n):
            try:
                self.error = "正在检查下一关，请稍候…"
                self.draw(surface)
                pygame.display.flip()
                self.session.next_level()
                self.error = ""
                self.next, self.finished = "level", True
            except PlayAdmissionError as error:
                self.error = str(error)
            self.draw(surface)
            return
        super().update(surface, keys)
