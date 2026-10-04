"""Strict, renderer-independent LevelSpec v1 and lossless core adapters."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ai_platformer.core.level import (
    BlockSpawn,
    CollectibleSpawn,
    EnemySpawn,
    LevelDefinition,
    PowerupSpawn,
    SolidRect,
)

MAX_ITEMS = 10000
Number = (
    Annotated[float, Field(strict=True, allow_inf_nan=False, ge=-1_000_000, le=1_000_000)]
    | Annotated[int, Field(strict=True, ge=-1_000_000, le=1_000_000)]
)
Positive = (
    Annotated[float, Field(strict=True, allow_inf_nan=False, gt=0, le=1_000_000)]
    | Annotated[int, Field(strict=True, gt=0, le=1_000_000)]
)
Identifier = Annotated[str, Field(strict=True, min_length=1, max_length=128)]


class SpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class WorldSpec(SpecModel):
    width: Positive
    height: Positive
    units: Literal["pixels"] = "pixels"


class SpawnSpec(SpecModel):
    x: Number
    bottom: Number


class GoalSpec(SpecModel):
    x: Number


class RectSpec(SpecModel):
    x: Number
    y: Number
    width: Positive
    height: Positive


class CollectibleSpec(RectSpec):
    entity_id: Identifier
    kind: Literal["coin"] = "coin"
    score: Annotated[int, Field(strict=True, ge=0, le=1_000_000)] = 100


class BlockSpec(RectSpec):
    entity_id: Identifier
    kind: Literal["brick", "box"]
    reward: Literal["none", "coin", "shield"] = "none"


class EnemySpec(RectSpec):
    entity_id: Identifier
    patrol_left: Number
    patrol_right: Number
    direction: Literal[-1, 1] = -1
    speed: Positive = 1.0

    @field_validator("direction", mode="before")
    @classmethod
    def integer_direction(cls, value):
        if type(value) is not int:
            raise ValueError("direction must be an integer")
        return value


class PowerupSpec(RectSpec):
    entity_id: Identifier
    kind: Literal["shield"] = "shield"


class MetadataSpec(SpecModel):
    title: Annotated[str, Field(max_length=256)] = ""
    theme: Annotated[str, Field(max_length=128)] = ""
    source: Annotated[str, Field(max_length=512)] = ""
    generator_version: Annotated[str, Field(max_length=128)] = ""
    seed: Annotated[int, Field(strict=True, ge=0)] | None = None


class LevelSpec(SpecModel):
    schema_version: Literal[1]
    level_id: Identifier
    world: WorldSpec
    spawn: SpawnSpec
    goal: GoalSpec
    solids: Annotated[list[RectSpec], Field(min_length=1, max_length=MAX_ITEMS)]
    collectibles: Annotated[list[CollectibleSpec], Field(max_length=MAX_ITEMS)] = Field(
        default_factory=list
    )
    blocks: Annotated[list[BlockSpec], Field(max_length=MAX_ITEMS)] = Field(default_factory=list)
    enemies: Annotated[list[EnemySpec], Field(max_length=MAX_ITEMS)] = Field(default_factory=list)
    powerups: Annotated[list[PowerupSpec], Field(max_length=MAX_ITEMS)] = Field(
        default_factory=list
    )
    metadata: MetadataSpec = Field(default_factory=MetadataSpec)

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_version(cls, value):
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value


def from_level_definition(level: LevelDefinition, *, metadata: dict | None = None) -> LevelSpec:
    return LevelSpec.model_validate(
        {
            "schema_version": 1,
            "level_id": level.level_id,
            "world": {"width": level.width, "height": level.height},
            "spawn": {"x": level.spawn_x, "bottom": level.spawn_bottom},
            "goal": {"x": level.goal_x},
            **{
                key: [asdict(item) for item in getattr(level, key)]
                for key in ("solids", "collectibles", "blocks", "enemies", "powerups")
            },
            "metadata": metadata or {},
        }
    )


def to_level_definition(spec: LevelSpec) -> LevelDefinition:
    """Structure adapter only; callers must run geometry checks before admission."""
    return LevelDefinition(
        spec.level_id,
        spec.world.width,
        spec.world.height,
        spec.spawn.x,
        spec.spawn.bottom,
        spec.goal.x,
        **{
            key: tuple(cls(**item.model_dump()) for item in getattr(spec, key))
            for key, cls in (
                ("solids", SolidRect),
                ("collectibles", CollectibleSpawn),
                ("blocks", BlockSpawn),
                ("enemies", EnemySpawn),
                ("powerups", PowerupSpawn),
            )
        },
    )


def document_hash(spec: LevelSpec) -> str:
    """Canonical document hash, including metadata and ordered content arrays."""
    data = json.dumps(spec.model_dump(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def gameplay_hash(spec: LevelSpec) -> str:
    # Preserve the established v9 hash algorithm and entity ordering.
    from ai_platformer.content.curriculum import content_hash

    return content_hash(to_level_definition(spec))


def geometry_hash(spec: LevelSpec) -> str:
    """Deduplicate content despite different IDs or int/float spellings.

    Array ordering remains significant, as collision/entity iteration uses it.
    """

    def normalize(value):
        if isinstance(value, dict):
            return {
                key: normalize(item)
                for key, item in value.items()
                if key not in {"level_id", "entity_id"}
            }
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        if type(value) in (int, float):
            return 0.0 if value == 0 else float(value)
        return value

    data = normalize(asdict(to_level_definition(spec)))
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def parse_level_json(data: bytes) -> object:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(data, object_pairs_hook=unique)


def load_level_spec(path: Path, *, max_bytes: int = 8_000_000) -> LevelSpec:
    with Path(path).open("rb") as stream:
        data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("LevelSpec input exceeds byte limit")
    return LevelSpec.model_validate(parse_level_json(data))
