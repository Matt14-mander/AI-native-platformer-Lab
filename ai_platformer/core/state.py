"""Serializable, renderer-independent snapshots of game state."""

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class PlayerSnapshot:
    x: float
    y: float
    velocity_x: float
    velocity_y: float
    grounded: bool
    alive: bool = True
    facing: int = 1

    def __post_init__(self) -> None:
        if self.facing not in (-1, 1):
            raise ValueError("facing must be -1 or 1")


@dataclass(frozen=True, slots=True)
class EntitySnapshot:
    entity_id: str
    kind: str
    x: float
    y: float
    active: bool = True
    width: float = 0.0
    height: float = 0.0
    facing: int = 1
    state: str = ""


@dataclass(frozen=True, slots=True)
class WorldSnapshot:
    """A point-in-time state suitable for observations, replay, and rendering."""

    episode_id: str
    tick: int
    level_id: str
    seed: int
    player: PlayerSnapshot
    entities: tuple[EntitySnapshot, ...] = ()
    progress: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tick < 0:
            raise ValueError("tick cannot be negative")
        if not 0.0 <= self.progress <= 1.0:
            raise ValueError("progress must be between 0.0 and 1.0")
