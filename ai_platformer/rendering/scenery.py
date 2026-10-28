"""Layered lakeside scenery and exposed platform edges, on the simulation clock."""

from math import sin

import pygame

from ai_platformer.rendering.art import load_art, fit_rect


def exposed_top(solid, solids):
    """Subtract any solid covering this top (including stacked step blocks)."""
    spans = [(solid.x, solid.right)]
    for other in solids:
        if other is solid or not other.y < solid.y <= other.bottom:
            continue
        remaining = []
        for left, right in spans:
            if other.right <= left or other.x >= right:
                remaining.append((left, right))
            else:
                if left < other.x:
                    remaining.append((left, other.x))
                if other.right < right:
                    remaining.append((other.right, right))
        spans = remaining
    return spans


class LakesideScenery:
    SPEEDS = {"sky": 0.04, "mountains": 0.12, "lake": 0.32, "shore": 0.62}

    def __init__(self, size, solids_by_kind):
        self.width, self.height = size
        self.layer = pygame.Surface(size, pygame.SRCALPHA)
        original = load_art("backgrounds/lakeside-day.png", crop=False)
        sky = original.subsurface((0, 0, original.get_width(), round(original.get_height() * 0.60)))
        self.sky = self._pair(fit_rect(sky, round(self.height * 2), round(self.height * 0.74)))
        self.mountains = self._pair(fit_rect(load_art("backgrounds/lakeside-mountains.png", crop=False),
                                            round(self.height * 1.5), round(self.height * 0.30)))
        self.lake = self._pair(fit_rect(load_art("backgrounds/lakeside-water.png", crop=False),
                                       round(self.height * 1.5), round(self.height * 0.28)))
        self.shore = self._pair(fit_rect(load_art("backgrounds/lakeside-nearshore.png", crop=False),
                                        round(self.height * 1.4), round(self.height * 0.15)))
        solids = tuple(s for group in solids_by_kind.values() for s in group)
        self.tops = [(solid, exposed_top(solid, solids)) for solid in solids]
        self.ground_tops = [(s, exposed_top(s, solids)) for s in solids_by_kind.get("ground", ())]

    @staticmethod
    def _pair(image):
        # Alternating mirrored tiles meet at identical edge pixels.
        return image, pygame.transform.flip(image, True, False)

    def _tile(self, surface, pair, camera, speed, y):
        width = pair[0].get_width()
        offset = round(camera * speed)
        first = offset // width
        for index in range(first, first + self.width // width + 3):
            surface.blit(pair[index % 2], (index * width - offset, y))

    def draw_background(self, surface, camera, tick):
        surface.fill((114, 145, 155))
        self._tile(surface, self.sky, camera, self.SPEEDS["sky"], 0)
        self._tile(surface, self.mountains, camera, self.SPEEDS["mountains"], round(self.height * 0.44))
        self._tile(surface, self.lake, camera, self.SPEEDS["lake"], round(self.height * 0.72))
        self.layer.fill((0, 0, 0, 0))
        for i in range(18):
            x = (i * 137 - camera * self.SPEEDS["lake"] + 3 * sin(tick / 95 + i)) % (self.width + 40) - 20
            y = self.height * (0.75 + (i % 6) * 0.025)
            light = max(0, sin(tick / 40 + i * 2.1))
            if light > 0.35:
                length = round(4 + light * 10)
                pygame.draw.line(self.layer, (255, 235, 184, round(light * 95)),
                                 (round(x), round(y)), (round(x + length), round(y)), 1)
        surface.blit(self.layer, (0, 0))
        self._tile(surface, self.shore, camera, self.SPEEDS["shore"], round(self.height * 0.74))

    def draw_edges(self, surface, camera):
        for solid, spans in self.tops:
            y = round(solid.y)
            for left, right in spans:
                x1, x2 = round(left - camera), round(right - camera) - 1
                if x2 < 0 or x1 >= self.width:
                    continue
                # The flat highlighted lip is exactly at the collision top.
                pygame.draw.polygon(surface, (73, 91, 61),
                                    [(x1, y), (x2, y), (max(x1, x2 - 2), y + 4), (min(x2, x1 + 2), y + 4)])
                pygame.draw.line(surface, (155, 165, 106), (x1, y), (x2, y), 1)

    def draw_ambient(self, surface, camera, tick):
        self.layer.fill((0, 0, 0, 0))
        for solid, spans in self.ground_tops:
            for left, right in spans:
                first = max(int(left // 64) + 1, int(camera // 64))
                last = min(int(right // 64), int((camera + self.width) // 64) + 1)
                for index in range(first, last):
                    wx = index * 64 + (index * 17 % 19)
                    if not left + 5 < wx < right - 5:
                        continue
                    x, y = round(wx - camera), round(solid.y)
                    sway = sin(tick / 32 + index) * 2
                    for blade in (-1, 0, 1):
                        pygame.draw.line(self.layer, (124, 148, 86, 190), (x + blade * 2, y - 1),
                                         (round(x + blade * 3 + sway), y - 4 - (index + blade) % 4), 1)
        # Just five small, softly pulsing lights near the shoreline, behind play objects.
        for i in range(5):
            x = (i * 211 + 43 - camera * 0.62 + 9 * sin(tick / 83 + i)) % (self.width + 80) - 40
            y = self.height * 0.84 + 12 * sin(tick / 71 + i * 2)
            alpha = round(25 + 90 * (0.5 + 0.5 * sin(tick / 44 + i)))
            pygame.draw.circle(self.layer, (235, 228, 144, alpha // 5), (round(x), round(y)), 5)
            pygame.draw.circle(self.layer, (255, 241, 172, alpha), (round(x), round(y)), 1)
        surface.blit(self.layer, (0, 0))
