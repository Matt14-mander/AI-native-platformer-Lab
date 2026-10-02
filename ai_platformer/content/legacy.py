"""Read-only adapter from the original map JSON to the new core model."""

from __future__ import annotations

import json
from pathlib import Path

from ai_platformer.core.level import (
    BlockSpawn,
    CollectibleSpawn,
    EnemySpawn,
    LevelDefinition,
    PowerupSpawn,
    SolidRect,
)


class LegacyLevelRepository:
    """Load legacy maps without leaking their JSON shape into the core."""

    def __init__(
        self,
        maps_directory: Path | None = None,
        overlay_directory: Path | None = None,
    ) -> None:
        project_root = Path(__file__).resolve().parents[2]
        self.maps_directory = maps_directory or project_root / "source" / "data" / "maps"
        self.overlay_directory = overlay_directory or project_root / "game_content" / "levels"

    def _legacy_data(self, level_id: str) -> dict:
        path = self.maps_directory / f"{level_id}.json"
        if not path.is_file():
            raise KeyError(f"unknown legacy level: {level_id}")
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)

    def _overlay_data(self, level_id: str) -> dict:
        path = self.overlay_directory / f"{level_id}.json"
        if not path.is_file():
            return {}
        with path.open(encoding="utf-8") as stream:
            data = json.load(stream)
        if data.get("schema_version") != 1:
            raise ValueError(f"unsupported content overlay version: {path}")
        return data

    def load_solids_by_group(self, level_id: str) -> dict[str, tuple[SolidRect, ...]]:
        data = self._legacy_data(level_id)
        return {
            group: tuple(
                SolidRect(
                    x=float(item["x"]),
                    y=float(item["y"]),
                    width=float(item["width"]),
                    height=float(item["height"]),
                )
                for item in data.get(group, ())
            )
            for group in ("ground", "pipe", "step")
        }

    def load(self, level_id: str) -> LevelDefinition:
        data = self._legacy_data(level_id)
        overlay = self._overlay_data(level_id)
        maps = data.get("maps")
        if not maps:
            raise ValueError(f"legacy level has no spawn metadata: {level_id}")
        initial_map = maps[0]
        width = float(initial_map["end_x"])
        groups = self.load_solids_by_group(level_id)
        solids = tuple(solid for group in ("ground", "pipe", "step") for solid in groups[group])
        flagpoles = data.get("flagpole", ())
        goal_x = float(flagpoles[0]["x"]) if flagpoles else width

        legacy_collectibles = tuple(
            CollectibleSpawn(
                entity_id=f"legacy-coin-{index}",
                kind="coin",
                x=float(item["x"]),
                y=float(item["y"]),
            )
            for index, item in enumerate(data.get("coin", ()))
        )
        overlay_collectibles = tuple(
            CollectibleSpawn(
                entity_id=str(item["id"]),
                kind=str(item.get("kind", "coin")),
                x=float(item["x"]),
                y=float(item["y"]),
                width=float(item.get("width", 16.0)),
                height=float(item.get("height", 24.0)),
                score=int(item.get("score", 100)),
            )
            for item in overlay.get("collectibles", ())
        )
        powerups = tuple(
            PowerupSpawn(
                entity_id=str(item["id"]),
                kind=str(item.get("kind", "shield")),
                x=float(item["x"]),
                y=float(item["y"]),
            )
            for item in overlay.get("powerups", ())
        )
        return LevelDefinition(
            level_id=level_id,
            width=width,
            height=600.0,
            spawn_x=float(initial_map["player_x"]),
            spawn_bottom=float(initial_map["player_y"]),
            goal_x=goal_x,
            solids=solids,
            collectibles=legacy_collectibles + overlay_collectibles,
            blocks=self._overlay_blocks(overlay),
            enemies=self._overlay_enemies(overlay),
            powerups=powerups,
        )

    @staticmethod
    def _overlay_blocks(overlay: dict) -> tuple[BlockSpawn, ...]:
        return tuple(
            BlockSpawn(
                entity_id=str(item["id"]),
                kind=str(item["kind"]),
                x=float(item["x"]),
                y=float(item["y"]),
                reward=str(item.get("reward", "none")),
            )
            for item in overlay.get("blocks", ())
        )

    @staticmethod
    def _overlay_enemies(overlay: dict) -> tuple[EnemySpawn, ...]:
        return tuple(
            EnemySpawn(
                entity_id=str(item["id"]),
                x=float(item["x"]),
                y=float(item["y"]),
                patrol_left=float(item["patrol_left"]),
                patrol_right=float(item["patrol_right"]),
                direction=int(item.get("direction", -1)),
                speed=float(item.get("speed", 1.0)),
            )
            for item in overlay.get("enemies", ())
        )
