"""Versioned training geometry, independent of legacy maps and rendering."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

from ai_platformer.content.full_courses import make_full_course
from ai_platformer.content.legacy import LegacyLevelRepository
from ai_platformer.core.level import LevelDefinition, SolidRect

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def content_hash(level: LevelDefinition) -> str:
    """Hash gameplay content without allowing different IDs to hide duplicates."""
    data = asdict(level)
    del data["level_id"]
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def make_course(level_id: str, spec: dict) -> LevelDefinition:
    kind = spec["task"]
    if kind == "full":
        if "main_area_source" in spec:
            source = LegacyLevelRepository().load(spec["main_area_source"])
            coins = tuple(coin for coin in source.collectibles if 0 <= coin.x < source.goal_x)
            return replace(source, level_id=level_id, collectibles=coins)
        return make_full_course(level_id, spec)
    if kind not in {"flat", "obstacle", "gap", "mixed"}:
        raise ValueError(f"unknown course task: {kind}")
    width = float(spec["width"])
    floor = 540.0
    spawn = float(spec.get("spawn_x", 48))
    goal = width - 100.0
    solids = []
    if kind in {"gap", "mixed"}:
        gap_x, gap_width = float(spec["gap_x"]), float(spec["gap_width"])
        if not spawn + 240 < gap_x < gap_x + gap_width < goal - 120:
            raise ValueError(f"invalid gap or landing area: {level_id}")
        solids.extend(
            [
                SolidRect(0, floor, gap_x, 60),
                SolidRect(gap_x + gap_width, floor, width - gap_x - gap_width, 60),
            ]
        )
    else:
        solids.append(SolidRect(0, floor, width, 60))
    if kind in {"obstacle", "mixed"}:
        x, height = float(spec["obstacle_x"]), float(spec["obstacle_height"])
        if not spawn + 180 < x < goal - 180 or not 0 < height <= 96:
            raise ValueError(f"invalid course obstacle: {level_id}")
        if kind == "mixed" and x + 64 + 360 >= float(spec["gap_x"]):
            raise ValueError(f"insufficient recovery distance: {level_id}")
        solids.append(SolidRect(x, floor - height, 64, height))
    return LevelDefinition(level_id, width, 600.0, spawn, floor, goal, tuple(solids))


class TrainingLevelRepository:
    """Explicit course manifest with legacy fallback for full-level transfer."""

    def __init__(self, manifest_path: str | Path | None = None) -> None:
        path = Path(manifest_path or "config/curriculum_v1.json")
        self.path = path if path.is_absolute() else PROJECT_ROOT / path
        self.manifest = json.loads(self.path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != 1:
            raise ValueError("unsupported curriculum manifest version")
        self.legacy = LegacyLevelRepository()
        self.levels = {
            level_id: make_course(level_id, spec)
            for level_id, spec in self.manifest["levels"].items()
        }
        seen_ids: set[str] = set()
        seen_hashes: set[str] = set()
        for groups in self.manifest["splits"].values():
            for ids in groups.values():
                if not ids:
                    raise ValueError("empty curriculum split")
                for level_id in ids:
                    if level_id in seen_ids:
                        raise ValueError(f"course belongs to multiple splits: {level_id}")
                    digest = content_hash(self.levels[level_id])
                    if digest in seen_hashes:
                        raise ValueError(f"duplicate geometry: {level_id}")
                    seen_ids.add(level_id)
                    seen_hashes.add(digest)
        if seen_ids != set(self.levels):
            raise ValueError("every course must belong to exactly one split")

    def load(self, level_id: str) -> LevelDefinition:
        if level_id in self.levels:
            return self.levels[level_id]
        return self.legacy.load(level_id)

    def split(self, task: str, suite: str) -> list[str]:
        return list(self.manifest["splits"][task][suite])
