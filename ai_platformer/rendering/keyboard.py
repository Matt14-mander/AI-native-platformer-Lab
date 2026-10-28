"""Pygame keyboard adapter that emits engine-neutral actions."""

from __future__ import annotations

import pygame

from ai_platformer.core import Action


class PygameKeyboardController:
    """Translate current keyboard state into one discrete core action."""

    @staticmethod
    def jump_pressed(keys) -> bool | None:
        was_pressed = getattr(keys, "was_pressed", None)
        if was_pressed is None:
            return None
        return was_pressed(pygame.K_a) or was_pressed(pygame.K_SPACE)

    def action(self, keys) -> Action:
        left = keys[pygame.K_LEFT]
        right = keys[pygame.K_RIGHT]
        jump = keys[pygame.K_a] or keys[pygame.K_SPACE]
        run = keys[pygame.K_s] or keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]

        # Opposite horizontal inputs cancel one another deterministically.
        horizontal = int(bool(right)) - int(bool(left))
        if horizontal < 0:
            if jump and run:
                return Action.LEFT_RUN_JUMP
            if jump:
                return Action.LEFT_JUMP
            if run:
                return Action.LEFT_RUN
            return Action.LEFT
        if horizontal > 0:
            if jump and run:
                return Action.RIGHT_RUN_JUMP
            if jump:
                return Action.RIGHT_JUMP
            if run:
                return Action.RIGHT_RUN
            return Action.RIGHT
        if jump:
            return Action.JUMP
        return Action.NOOP
