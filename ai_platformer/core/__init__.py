"""Deterministic game-domain contracts with no rendering dependency."""

from .actions import Action, Control, control_for
from .engine import GameCore, StepResult
from .level import BlockSpawn, CollectibleSpawn, EnemySpawn, LevelDefinition, PowerupSpawn, SolidRect
from .simulation import BasicPlatformerCore, PhysicsConfig
from .state import EntitySnapshot, PlayerSnapshot, WorldSnapshot

__all__ = [
    "Action",
    "BasicPlatformerCore",
    "Control",
    "BlockSpawn",
    "CollectibleSpawn",
    "EntitySnapshot",
    "EnemySpawn",
    "GameCore",
    "LevelDefinition",
    "PhysicsConfig",
    "PlayerSnapshot",
    "PowerupSpawn",
    "StepResult",
    "SolidRect",
    "WorldSnapshot",
    "control_for",
]
