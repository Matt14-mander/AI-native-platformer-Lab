"""Strict single-LevelSpec repository; never adds candidates to training splits."""

from pathlib import Path

from ai_platformer.core import PhysicsConfig

from .level_spec import document_hash, load_level_spec, to_level_definition
from .level_validation import validate_level_spec


class LevelSpecRepository:
    def __init__(self, path: str | Path, *, physics: PhysicsConfig):
        self.path = Path(path).resolve()
        self.spec = load_level_spec(self.path)
        self.validation = validate_level_spec(self.spec, physics=physics)
        if not self.validation["static_valid"]:
            errors = [i for i in self.validation["issues"] if i["severity"] == "error"]
            raise ValueError(f"LevelSpec static admission failed: {errors}")
        level = to_level_definition(self.spec)
        self.levels = {level.level_id: level}
        self.manifest = {
            "schema_version": 1,
            "generator_version": "level_spec_v1",
            "document_sha256": document_hash(self.spec),
            "splits": {},
        }

    def load(self, level_id):
        if level_id not in self.levels:
            raise KeyError(f"unknown LevelSpec level: {level_id}")
        return self.levels[level_id]

    def split(self, task, suite):
        raise ValueError("LevelSpec candidates are not assigned to training/evaluation splits")

    def solids_by_kind(self, level_id):
        level = self.load(level_id)
        return {
            "ground": tuple(s for s in level.solids if s.y == level.spawn_bottom),
            "pipe": tuple(s for s in level.solids if s.y != level.spawn_bottom),
            "step": (),
        }
