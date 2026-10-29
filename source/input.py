"""Event-driven input state for the legacy Pygame application."""

from __future__ import annotations

import pygame


class KeyboardInput:
    """Track held keys without letting game objects poll global keyboard state."""

    def __init__(self) -> None:
        self._pressed: set[int] = set()
        self._just_pressed: set[int] = set()
        self.focused = True
        self.focus_lost = False

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if not self.focused:
                return
            if event.key not in self._pressed:
                self._just_pressed.add(event.key)
            self._pressed.add(event.key)
        elif event.type == pygame.KEYUP:
            self._pressed.discard(event.key)
        elif event.type == pygame.WINDOWFOCUSLOST:
            self.focused = False
            self.focus_lost = True
            self._pressed.clear()
            self._just_pressed.clear()
        elif event.type == pygame.WINDOWFOCUSGAINED:
            self.focused = True

    def sync(self, pressed) -> None:
        """Synchronize gameplay keys with the physical keyboard each frame.

        Events preserve responsiveness, while polling prevents a missed window
        event from leaving a key incorrectly pressed or released.
        """

        if not self.focused:
            return
        tracked_keys = (
            pygame.K_UP,
            pygame.K_DOWN,
            pygame.K_LEFT,
            pygame.K_RIGHT,
            pygame.K_RETURN,
            pygame.K_a,
            pygame.K_s,
            pygame.K_SPACE,
            pygame.K_LSHIFT,
            pygame.K_RSHIFT,
            pygame.K_p,
            pygame.K_r,
            pygame.K_m,
            pygame.K_h,
            pygame.K_ESCAPE,
        )
        for key in tracked_keys:
            if pressed[key]:
                if key not in self._pressed:
                    self._just_pressed.add(key)
                self._pressed.add(key)
            else:
                self._pressed.discard(key)

    def is_pressed(self, key: int) -> bool:
        return key in self._pressed

    def was_pressed(self, key: int) -> bool:
        return key in self._just_pressed

    def end_frame(self) -> None:
        """Consume edges after a simulation tick, not after a render-only frame."""
        self._just_pressed.clear()
        self.focus_lost = False

    def __getitem__(self, key: int) -> bool:
        return self.is_pressed(key)
