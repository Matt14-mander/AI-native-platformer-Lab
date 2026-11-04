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
        self.leave()
        self.finished=False
        self.next="level"
        self.summary={}
        self.error = ""
        self.offer = None
        self.suggestion_message = "D   查看新练习关建议（可选）"
    def leave(self):
        job = getattr(self, "suggestion_job", None)
        if job:
            job.cancel()
        self.suggestion_job = None
    def receive_result(self,summary):
        self.summary=dict(summary)
    def is_finished(self):
        return self.finished
    def update(self,surface,keys):
        self.finished=True
    def draw(self,surface):
        surface.blit(self.background,((surface.get_width()-self.background.get_width())//2,0))
        panel(surface,pygame.Rect(30,80,surface.get_width()-60,500))
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
        if self.session:
            text(surface, self.suggestion_message, 515, 16, SOFT)
            if self.offer and self.offer.prepared:
                text(surface, "A   试玩已验证的新关卡    ·    R 保留原关重试", 549, 18)

    def suggestion_input(self, surface, keys):
        if not self.session:
            return False
        if self.suggestion_job and self.suggestion_job.future.done():
            job, self.suggestion_job = self.suggestion_job, None
            try:
                offer = job.future.result()
                if job.cancelled.is_set() or self.session.prepared is not offer.origin:
                    raise PlayAdmissionError("建议已失效，请重新查看。")
                self.offer = offer
                self.suggestion_message = offer.message
            except Exception:
                self.suggestion_message = "建议暂不可用，仍可重试原关或返回菜单。"
            # A key pressed during checking cannot accept a newly arriving offer.
            self.draw(surface)
            return True
        if pressed(keys, pygame.K_d):
            # One immutable offer per result screen; repeated keys do not rerun
            # expensive analysis or create duplicate candidate artifacts.
            if self.offer is None and self.suggestion_job is None:
                try:
                    from ai_platformer.adaptive.suggestion_job import start_suggestion
                    self.suggestion_job = start_suggestion(self.session)
                    self.suggestion_message = "正在后台检查… 可按 R 重试原关，M 返回菜单。"
                except (PlayAdmissionError, ImportError):
                    self.suggestion_message = "建议暂不可用或上次检查尚未结束，请稍后再试。"
            self.draw(surface)
            return True
        if pressed(keys, pygame.K_a) and self.offer and self.offer.prepared:
            try:
                from ai_platformer.adaptive.candidates import accept_offer
                accept_offer(self.session, self.offer)
                self.next, self.finished = "level", True
            except PlayAdmissionError as error:
                self.suggestion_message = str(error)
            self.draw(surface)
            return True
        return False
class GameOver(LoadScreen):
    title="没关系，再试一次"
    subtitle="调整起跳时机，小狗还在陪着你。"
    def update(self,surface,keys):
        if pressed(keys,pygame.K_RETURN) or pressed(keys,pygame.K_r):
            self.next,self.finished="level",True
        elif pressed(keys,pygame.K_ESCAPE) or pressed(keys,pygame.K_m):
            self.next,self.finished="main_menu",True
        elif self.suggestion_input(surface, keys):
            return
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
