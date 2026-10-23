"""Frozen exploratory prompt/seed suites and explicit request-level expectations."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from ai_platformer.content.generation_request import GenerationRequest
from ai_platformer.content.level_spec import SpecModel, parse_level_json

CaseID = Annotated[str, Field(strict=True, pattern=r"^[a-z][a-z0-9_]{0,47}$")]
Seed = Annotated[int, Field(strict=True, ge=0, le=2**32 - 1)]


class GenerationCase(SpecModel):
    case_id: CaseID
    prompt: Annotated[str, Field(strict=True, min_length=1, max_length=4000)]
    expected: dict[str, Any]

    @model_validator(mode="after")
    def valid_expectations(self):
        reserved = {"level_id", "seed", "schema_version"}
        if not self.expected or reserved.intersection(self.expected):
            raise ValueError("expectations must contain design fields, never caller-owned fields")
        self.expected_request(0)
        return self

    def expected_request(self, seed):
        return GenerationRequest.model_validate(
            {
                "schema_version": 1,
                "level_id": f"eval_{self.case_id}_s{seed}",
                "seed": seed,
                **self.expected,
            }
        )


class GenerationSuite(SpecModel):
    schema_version: Literal[1]
    suite_id: CaseID
    purpose: Literal["exploratory_evaluation_only"]
    generation_seeds: Annotated[list[Seed], Field(min_length=1, max_length=8)]
    policy_seed: Seed = 100
    cases: Annotated[list[GenerationCase], Field(min_length=1, max_length=16)]

    @field_validator("schema_version", mode="before")
    @classmethod
    def strict_version(cls, value):
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value

    @model_validator(mode="after")
    def bounded_unique(self):
        if len(set(self.generation_seeds)) != len(self.generation_seeds):
            raise ValueError("duplicate generation seeds")
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("duplicate case IDs")
        if len(self.cases) * len(self.generation_seeds) > 64:
            raise ValueError("suite exceeds 64 candidates")
        return self


def load_suite(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(1_000_001)
    if len(raw) > 1_000_000:
        raise ValueError("suite exceeds 1 MB limit")
    return GenerationSuite.model_validate(parse_level_json(raw))


def compare_expectations(request, expected):
    actual = request.model_dump()
    mismatches = []

    def compare(wanted, got, prefix=""):
        for key, value in wanted.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                compare(value, got[key], path)
            elif got[key] != value:
                mismatches.append({"field": path, "expected": value, "actual": got[key]})

    compare(expected, actual)
    return {"passed": not mismatches, "mismatches": mismatches}
