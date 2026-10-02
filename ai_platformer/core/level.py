"""Minimal level-domain objects consumed by the deterministic core."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SolidRect:
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("solid dimensions must be positive")

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height


@dataclass(frozen=True, slots=True)
class CollectibleSpawn:
    entity_id: str
    kind: str
    x: float
    y: float
    width: float = 16.0
    height: float = 24.0
    score: int = 100

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("collectible dimensions must be positive")
        if self.score < 0:
            raise ValueError("collectible score cannot be negative")


@dataclass(frozen=True, slots=True)
class BlockSpawn:
    entity_id: str
    kind: str
    x: float
    y: float
    width: float = 40.0
    height: float = 40.0
    reward: str = "none"

    def __post_init__(self) -> None:
        if self.kind not in {"brick", "box"}:
            raise ValueError("block kind must be brick or box")
        if self.reward not in {"none", "coin", "shield"}:
            raise ValueError("unsupported block reward")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("block dimensions must be positive")

    @property
    def rect(self) -> SolidRect:
        return SolidRect(self.x, self.y, self.width, self.height)


@dataclass(frozen=True, slots=True)
class EnemySpawn:
    entity_id: str
    x: float
    y: float
    patrol_left: float
    patrol_right: float
    direction: int = -1
    width: float = 28.0
    height: float = 28.0
    speed: float = 1.0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.speed <= 0:
            raise ValueError("enemy dimensions and speed must be positive")
        if self.patrol_left > self.x or self.patrol_right < self.x:
            raise ValueError("enemy spawn must be inside patrol range")
        if self.direction not in (-1, 1):
            raise ValueError("enemy direction must be -1 or 1")


@dataclass(frozen=True, slots=True)
class PowerupSpawn:
    entity_id: str
    kind: str
    x: float
    y: float
    width: float = 24.0
    height: float = 24.0

    def __post_init__(self) -> None:
        if self.kind != "shield":
            raise ValueError("unsupported powerup kind")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("powerup dimensions must be positive")


@dataclass(frozen=True, slots=True)
class LevelDefinition:
    level_id: str
    width: float
    height: float
    spawn_x: float
    spawn_bottom: float
    goal_x: float
    solids: tuple[SolidRect, ...]
    collectibles: tuple[CollectibleSpawn, ...] = ()
    blocks: tuple[BlockSpawn, ...] = ()
    enemies: tuple[EnemySpawn, ...] = ()
    powerups: tuple[PowerupSpawn, ...] = ()

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("level dimensions must be positive")
        if not 0 <= self.spawn_x < self.width:
            raise ValueError("spawn_x must be inside the level")
        if not self.spawn_x < self.goal_x <= self.width:
            raise ValueError("goal_x must be after the spawn and inside the level")
