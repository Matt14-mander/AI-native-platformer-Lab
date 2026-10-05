"""Versioned request strictness, geometry constraints and deterministic PCG contracts."""

import json
import random
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

from ai_platformer.content.generation_request import (
    GenerationRequest,
    load_generation_request,
    request_hash,
)
from ai_platformer.content.level_spec import document_hash, geometry_hash
from ai_platformer.content.level_validation import validate_dataset
from ai_platformer.content.pcg import SEGMENTS_V1_PHYSICS, available_generators, generate_level
from ai_platformer.settings import AI_GAMEPLAY_PATH, load_gameplay_settings


@pytest.fixture
def design_request():
    return json.loads(Path("game_content/generation_requests/mixed_v1.json").read_text())


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("generator_version", "unknown"),
        ("physics_profile", "custom"),
        ("seed", -1),
        ("seed", True),
        ("seed", 2**32),
        ("obstacle_count", "3"),
        ("gap_count", 9),
        ("theme", "file:///tmp/asset"),
        ("min_coin_ratio", float("nan")),
        ("min_coin_ratio", True),
        ("solids", []),
    ],
)
def test_strict_request_rejects_unsupported_or_unbounded_input(design_request, key, value):
    design_request[key] = value
    with pytest.raises(ValueError):
        GenerationRequest.model_validate(design_request)


def test_cross_field_constraints(design_request):
    for updates in (
        {"obstacle_count": 8, "gap_count": 8},
        {"obstacle_height": {"minimum": 96, "maximum": 48}},
        {"gap_width": {"minimum": 160, "maximum": 96}},
        {"coin_layout": "none", "min_coin_ratio": 0.75},
    ):
        with pytest.raises(ValueError):
            generate_level({**design_request, **updates})


def test_golden_map_and_versioned_dispatch(design_request):
    result = generate_level(design_request)
    assert available_generators() == ("segments-v1",)
    assert (
        document_hash(result.level)
        == "d0e6302a194088d920eaa7fa64803c97abc9c37f9be4cd6bab13721d2c4a50f2"
    )
    assert result.report["counts"] == {"obstacles": 3, "gaps": 2, "coins": 20}
    assert result.report["status"] == "candidate"
    assert result.report["validation"]["reachability"] == "not_checked"


def test_generation_is_reproducible_without_global_rng_side_effects(design_request):
    state = random.getstate()
    a = generate_level(design_request)
    b = generate_level(dict(reversed(list(design_request.items()))))
    assert a.level.model_dump() == b.level.model_dump()
    assert a.report == b.report
    assert random.getstate() == state
    assert request_hash(a.request) == request_hash(b.request)
    different = generate_level({**design_request, "seed": design_request["seed"] + 1})
    assert geometry_hash(different.level) != geometry_hash(a.level)


def test_renaming_cannot_hide_duplicate_map(design_request):
    a = generate_level(design_request)
    b = generate_level({**design_request, "level_id": "another-name"})
    assert request_hash(a.request) != request_hash(b.request)
    assert geometry_hash(a.level) == geometry_hash(b.level)
    assert not validate_dataset({"train": [a.level], "test": [b.level]}, physics=a.physics)[
        "static_valid"
    ]


def test_seed_corpus_respects_counts_dimensions_and_static_rules(design_request):
    for seed in range(32):
        r = generate_level(
            {
                **design_request,
                "seed": seed,
                "obstacle_count": 8,
                "gap_count": 4,
                "obstacle_height": {"minimum": 160, "maximum": 160},
                "gap_width": {"minimum": 176, "maximum": 176},
            }
        )
        assert r.report["validation"]["static_valid"]
        assert r.report["counts"] == {"obstacles": 8, "gaps": 4, "coins": 48}
        assert r.level.world.height == 600
        assert all(s["height"] == 160 for s in r.report["segments"] if s["kind"] == "obstacle")
        assert all(s["width"] == 176 for s in r.report["segments"] if s["kind"] == "gap")


def test_flat_and_coin_free_requests(design_request):
    flat = generate_level(
        {**design_request, "obstacle_count": 0, "gap_count": 0, "coin_layout": "ground"}
    )
    assert len(flat.level.solids) == 1
    assert flat.report["counts"] == {"obstacles": 0, "gaps": 0, "coins": 4}
    empty = generate_level({**design_request, "coin_layout": "none", "min_coin_ratio": 0.0})
    assert not empty.level.collectibles
    assert not empty.level.blocks and not empty.level.enemies and not empty.level.powerups


def test_profile_and_schema_snapshots_are_pinned():
    assert asdict(SEGMENTS_V1_PHYSICS) == asdict(load_gameplay_settings(AI_GAMEPLAY_PATH).physics)
    schema = GenerationRequest.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    assert schema == json.loads(Path("schemas/generation_request_v1.schema.json").read_text())


def test_loader_rejects_duplicate_keys_and_oversized_inputs(tmp_path):
    path = tmp_path / "request.json"
    path.write_text('{"seed":1,"seed":2}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_generation_request(path)
    path.write_bytes(b" " * 1_000_001)
    with pytest.raises(ValueError, match="1 MB"):
        load_generation_request(path)


def test_static_generation_does_not_import_training_or_environment(tmp_path):
    code = "from ai_platformer.content.pcg import generate_level; import sys; generate_level(dict(schema_version=1,level_id='flat',seed=1)); assert not {'torch','stable_baselines3','gymnasium'} & sys.modules.keys()"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_cli_packages_are_atomic_and_refuse_overwrite(tmp_path, design_request):
    path, output = tmp_path / "request.json", tmp_path / "package"
    path.write_text(json.dumps(design_request))
    command = [
        sys.executable,
        "-m",
        "scripts.generate_level",
        "--request",
        str(path),
        "--output-dir",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert {p.name for p in output.iterdir()} == {"request.json", "level.json", "report.json"}
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    assert subprocess.run(command, capture_output=True, check=False).returncode == 2
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}
    invalid = tmp_path / "invalid"
    path.write_text('{"schema_version":2}')
    command[-1] = str(invalid)
    assert subprocess.run(command, capture_output=True, check=False).returncode == 1
    assert json.loads((invalid / "report.json").read_text())["status"] == "invalid"
    assert not (invalid / "level.json").exists()
    assert not list(tmp_path.glob(".pcg-*"))
