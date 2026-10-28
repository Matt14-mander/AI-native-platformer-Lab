"""Short-lived visual/audio feedback driven by completed gameplay events."""

from dataclasses import dataclass
from math import cos, sin, pi

import pygame

from ai_platformer.rendering.art import ASSET_ROOT

ACORN_HUD_CENTER = (286, 44)


class FeedbackAudio:
    NAMES = ("box", "brick", "acorn", "stomp", "shield_break", "shield_pickup", "hit")

    def __init__(self):
        self.sounds = {}
        if pygame.mixer.get_init():
            for name in self.NAMES:
                try:
                    sound = pygame.mixer.Sound(str(ASSET_ROOT / "audio" / f"{name}.wav"))
                    sound.set_volume(0.32)
                    self.sounds[name] = sound
                except (pygame.error, FileNotFoundError):
                    pass  # Visual feedback remains usable without an audio device.

    def play(self, name):
        if name in self.sounds:
            self.sounds[name].play()

    def stop(self):
        for sound in self.sounds.values():
            sound.stop()


@dataclass
class Effect:
    kind: str
    x: float
    y: float
    duration: int
    age: int = 0
    vx: float = 0
    vy: float = 0
    index: int = 0
    entity_id: str = ""


class InteractionFeedback:
    def __init__(self, audio=None):
        self.audio = audio if audio is not None else FeedbackAudio()
        self.reset()

    def reset(self):
        self.effects = []
        self.last_event = None
        self.hud_pulse = 0
        self.audio.stop()

    def advance(self):
        self.hud_pulse = max(0, self.hud_pulse - 1)
        for effect in self.effects:
            effect.age += 1
            if effect.kind == "acorn" and effect.age == effect.duration:
                self.hud_pulse = 12
        self.effects = [e for e in self.effects if e.age < e.duration]

    def consume(self, previous, result, camera_x, player_size=(24, 32)):
        state = result.state
        token = (state.episode_id, state.tick)
        if token == self.last_event:
            return
        if self.last_event and (token[0] != self.last_event[0] or token[1] < self.last_event[1]):
            self.reset()
        self.last_event = token
        entities = {e.entity_id: e for e in previous.entities}
        entities.update({e.entity_id: e for e in state.entities})
        sounds = set()
        box_rewards = state.metadata.get("box_coins", 0) - previous.metadata.get("box_coins", 0)
        shield_boxes = {entity_id for event, entity_id in result.info.get("events", ()) if event == "powerup_spawned"}
        for event, entity_id in result.info.get("events", ()):
            entity = entities.get(entity_id)
            x = entity.x + entity.width / 2 if entity else state.player.x
            y = entity.y if entity else state.player.y
            if event in ("box_used", "box_bumped"):
                self.effects.append(Effect("box", x, y, 14, entity_id=entity_id))
                sounds.add("box")
                if box_rewards > 0 and entity_id not in shield_boxes and event == "box_used":
                    self.effects.append(Effect("acorn", x - camera_x, y, 36))
                    sounds.add("acorn")
                    box_rewards -= 1
            elif event == "brick_broken":
                for i in range(8):
                    self.effects.append(Effect("brick", x, y + 20, 36, vx=(i % 4 - 1.5) * 1.7,
                                               vy=-3.5 - (i // 4) * 1.7, index=i))
                sounds.add("brick")
            elif event == "enemy_stomped":
                self.effects.append(Effect("stomp", x, y, 18))
                self.effects.append(Effect("squash", entity.x if entity else x, y, 12))
                sounds.add("stomp")
            elif event in ("shield_absorbed", "shield_collected", "player_hit"):
                px = state.player.x + player_size[0] / 2
                py = state.player.y + player_size[1] - 26
                kind = {"shield_absorbed": "shield_break", "shield_collected": "shield_pickup", "player_hit": "hit"}[event]
                self.effects.append(Effect(kind, px, py, 28))
                sounds.add(kind)
        for entity_id in result.info.get("collected", ()):
            entity = entities[entity_id]
            self.effects.append(Effect("acorn", entity.x + entity.width / 2 - camera_x, entity.y, 36))
            sounds.add("acorn")
        # Coalesce same-tick pickups so overlapping coins do not stack loud sounds.
        for name in sorted(sounds):
            self.audio.play(name)
        self.effects = self.effects[-192:]

    def box_offset(self, entity_id):
        return next((-round(8 * sin(pi * e.age / e.duration)) for e in self.effects
                     if e.kind == "box" and e.entity_id == entity_id), 0)

    def draw_world(self, surface, camera_x, brick, enemy):
        layer = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        for e in self.effects:
            t = e.age
            p = t / e.duration
            alpha = round(255 * (1 - p))
            x, y = e.x - camera_x, e.y
            if e.kind == "brick":
                piece = brick.subsurface((e.index % 4 * 10, e.index // 4 * 20, 10, 20))
                piece = pygame.transform.rotate(piece, t * (8 if e.index % 2 else -8))
                piece.set_alpha(alpha)
                layer.blit(piece, (round(x + e.vx * t), round(y + e.vy * t + 0.15 * t * t)))
            elif e.kind == "squash":
                sprite = pygame.transform.smoothscale(enemy, (34, 10))
                sprite.set_alpha(alpha)
                layer.blit(sprite, (round(x - 3), round(y + 18)))
            elif e.kind in ("stomp", "hit", "shield_pickup"):
                color = {"stomp": (255, 224, 164), "hit": (255, 150, 120), "shield_pickup": (150, 225, 255)}[e.kind]
                radius = round(8 + 22 * p)
                pygame.draw.circle(layer, (*color, alpha), (round(x), round(y)), radius, 2)
                for i in range(6):
                    angle = i * pi / 3
                    pygame.draw.circle(layer, (*color, alpha), (round(x + cos(angle) * radius), round(y + sin(angle) * radius)), 2)
            elif e.kind == "shield_break":
                for i in range(10):
                    angle = i * pi / 5
                    radius = 22 + t * 1.4
                    sx, sy = x + cos(angle) * radius, y + sin(angle) * radius + 0.045 * t * t
                    pygame.draw.polygon(layer, (155, 226, 255, alpha),
                                        [(round(sx), round(sy - 5)), (round(sx + 4), round(sy)), (round(sx - 3), round(sy + 5))])
        surface.blit(layer, (0, 0))

    def draw_overlay(self, surface, acorn):
        for e in self.effects:
            if e.kind != "acorn":
                continue
            p = e.age / e.duration
            # Pop upward first, then curve toward the fixed HUD icon.
            u = max(0, (p - 0.18) / 0.82)
            u = u * u * (3 - 2 * u)
            x = e.x + (ACORN_HUD_CENTER[0] - e.x) * u
            y = e.y + (ACORN_HUD_CENTER[1] - e.y) * u - 38 * sin(pi * p)
            surface.blit(acorn, (round(x - acorn.get_width() / 2), round(y - acorn.get_height() / 2)))
        if self.hud_pulse:
            radius = 17 + (12 - self.hud_pulse)
            pygame.draw.circle(surface, (255, 226, 155), ACORN_HUD_CENTER, radius, 2)
