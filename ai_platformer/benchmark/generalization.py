"""Freeze generated geometry before evaluating a policy on it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ai_platformer.content.generation_request import GenerationRequest, request_hash
from ai_platformer.content.level_spec import geometry_hash
from ai_platformer.content.pcg import generate_level


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _request(family: dict, seed: int, *, prefix: str) -> GenerationRequest:
    family_id = family.get("id", "case")
    return GenerationRequest.model_validate(
        {
            "schema_version": 1,
            "level_id": f"{prefix}_{family_id}_{seed}",
            "seed": seed,
            **{key: value for key, value in family.items() if key != "id"},
        }
    )


def freeze_plan(plan: dict, reference_suite: dict) -> dict:
    """Build a content-addressed manifest; no policy is loaded or evaluated."""
    if plan.get("schema_version") != 1 or reference_suite.get("schema_version") != 1:
        raise ValueError("unsupported plan or reference suite version")
    if set(plan["splits"]) != {"development", "holdout"}:
        raise ValueError("expected development and holdout splits")
    if len({family["id"] for family in plan["families"]}) != len(plan["families"]):
        raise ValueError("duplicate family ID")
    seeds = [seed for pool in plan["splits"].values() for seed in pool]
    if len(seeds) != len(set(seeds)):
        raise ValueError("split seeds overlap")

    reference_hashes = set()
    for family in reference_suite["cases"]:
        for seed in reference_suite["generation_seeds"]:
            request = _request(family["expected"], seed, prefix="reference")
            reference_hashes.add(geometry_hash(generate_level(request).level))

    seen = set(reference_hashes)
    entries = []
    for split, split_seeds in plan["splits"].items():
        for family in plan["families"]:
            for seed in split_seeds:
                request = _request(family, seed, prefix=plan["suite_id"])
                geometry = geometry_hash(generate_level(request).level)
                if geometry in seen:
                    raise ValueError(f"duplicate geometry: {split}/{family['id']}/{seed}")
                seen.add(geometry)
                entries.append(
                    {
                        "split": split,
                        "family": family["id"],
                        "request": request.model_dump(),
                        "request_sha256": request_hash(request),
                        "geometry_sha256": geometry,
                    }
                )
    return {
        "schema_version": 1,
        "suite_id": plan["suite_id"],
        "purpose": "frozen_geometry_generalization; holdout excluded from development",
        "generator_version": plan["generator_version"],
        "plan_sha256": hashlib.sha256(_canonical(plan)).hexdigest(),
        "reference_suite_sha256": hashlib.sha256(_canonical(reference_suite)).hexdigest(),
        "reference_unique_geometries": len(reference_hashes),
        "entries": entries,
    }


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not manifest.get("entries"):
        raise ValueError("invalid generalization manifest")
    hashes = set()
    for entry in manifest["entries"]:
        request = GenerationRequest.model_validate(entry["request"])
        if request_hash(request) != entry["request_sha256"]:
            raise ValueError("request hash mismatch")
        geometry = geometry_hash(generate_level(request).level)
        if geometry != entry["geometry_sha256"] or geometry in hashes:
            raise ValueError("geometry hash mismatch or duplicate")
        hashes.add(geometry)
    return manifest
