"""Versioned gameplay settings shared by the app and simulation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ai_platformer.core import PhysicsConfig

AI_GAMEPLAY_PATH = Path(__file__).resolve().parents[1] / "config/gameplay_ai_v2.json"


@dataclass(frozen=True, slots=True)
class GameplaySettings:
    schema_version: int
    level_id: str
    seed: int
    render_fps: int
    terminal_display_ms: int
    physics: PhysicsConfig

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported gameplay settings version: {self.schema_version}")
        if self.render_fps <= 0:
            raise ValueError("render_fps must be positive")
        if self.terminal_display_ms < 0:
            raise ValueError("terminal_display_ms cannot be negative")


def load_gameplay_settings(path: Path | None = None) -> GameplaySettings:
    project_root = Path(__file__).resolve().parents[1]
    settings_path = path or project_root / "config" / "gameplay.json"
    with settings_path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    return GameplaySettings(
        schema_version=int(data["schema_version"]),
        level_id=str(data["level_id"]),
        seed=int(data["seed"]),
        render_fps=int(data["render_fps"]),
        terminal_display_ms=int(data["terminal_display_ms"]),
        physics=PhysicsConfig(**data.get("physics", {})),
    )
