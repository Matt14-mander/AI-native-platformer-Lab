"""Versioned, bounded design requests; no free-form geometry or executable fields."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from .level_spec import Identifier, SpecModel, parse_level_json


class HeightRange(SpecModel):
    minimum: Annotated[int, Field(strict=True, ge=16, le=160)] = 48
    maximum: Annotated[int, Field(strict=True, ge=16, le=160)] = 96

    @model_validator(mode="after")
    def ordered(self):
        if self.minimum > self.maximum:
            raise ValueError("minimum must not exceed maximum")
        return self


class GapRange(SpecModel):
    minimum: Annotated[int, Field(strict=True, ge=64, le=176)] = 96
    maximum: Annotated[int, Field(strict=True, ge=64, le=176)] = 152

    @model_validator(mode="after")
    def ordered(self):
        if self.minimum > self.maximum:
            raise ValueError("minimum must not exceed maximum")
        return self


class GenerationRequest(SpecModel):
    schema_version: Literal[1]
    generator_version: Literal["segments-v1"] = "segments-v1"
    physics_profile: Literal["gameplay-ai-v2"] = "gameplay-ai-v2"
    level_id: Identifier
    seed: Annotated[int, Field(strict=True, ge=0, le=2**32 - 1)]
    obstacle_count: Annotated[int, Field(strict=True, ge=0, le=8)] = 3
    gap_count: Annotated[int, Field(strict=True, ge=0, le=8)] = 2
    obstacle_height: HeightRange = Field(default_factory=HeightRange)
    gap_width: GapRange = Field(default_factory=GapRange)
    coin_layout: Literal["none", "ground", "low_jump"] = "low_jump"
    min_coin_ratio: Annotated[float, Field(strict=True, allow_inf_nan=False, ge=0, le=1)] = 0.75
    theme: Literal["lake"] = "lake"

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_version(cls, value):
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value

    @model_validator(mode="after")
    def consistent(self):
        if self.obstacle_count + self.gap_count > 12:
            raise ValueError("at most 12 hazards per request")
        if self.coin_layout == "none" and self.min_coin_ratio > 0:
            raise ValueError("coin_layout=none requires min_coin_ratio=0")
        return self


def request_hash(request: GenerationRequest) -> str:
    data = json.dumps(request.model_dump(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def load_generation_request(path: str | Path) -> GenerationRequest:
    with Path(path).open("rb") as stream:
        raw = stream.read(1_000_001)
    if len(raw) > 1_000_000:
        raise ValueError("generation request exceeds 1 MB limit")
    return GenerationRequest.model_validate(parse_level_json(raw))
