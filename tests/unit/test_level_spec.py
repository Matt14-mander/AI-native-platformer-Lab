"""Content compatibility and strict admission diagnostics for LevelSpec v1."""

import copy
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

from ai_platformer.content.curriculum import TrainingLevelRepository, content_hash
from ai_platformer.content.level_spec import (
    LevelSpec,
    document_hash,
    from_level_definition,
    gameplay_hash,
    load_level_spec,
    to_level_definition,
)
from ai_platformer.content.level_validation import validate_dataset, validate_level_spec
from ai_platformer.core import Action, BasicPlatformerCore, PhysicsConfig


@pytest.fixture
def data():
    return json.loads(Path("game_content/level_specs/flat_v1.json").read_text())


def test_all_v9_courses_roundtrip_without_content_changes():
    repo = TrainingLevelRepository("config/curriculum_v9.json")
    for level in repo.levels.values():
        spec = from_level_definition(level)
        loaded = LevelSpec.model_validate_json(spec.model_dump_json())
        assert to_level_definition(loaded) == level
        assert gameplay_hash(loaded) == content_hash(level)


def test_generated_v9_geometry_passes_and_legacy_bounds_are_reported():
    repo = TrainingLevelRepository("config/curriculum_v9.json")
    for level_id, level in repo.levels.items():
        report = validate_level_spec(from_level_definition(level))
        assert report["reachability"] == "not_checked"
        if level_id == "level_1_main":
            assert not report["static_valid"]
            assert {i["path"] for i in report["issues"] if i["code"] == "entity_bounds"} == {
                "/solids/3",
                "/solids/10",
                "/solids/38",
            }
        else:
            assert report["static_valid"], report


@pytest.mark.parametrize(
    "path,value",
    [
        (("schema_version",), True),
        (("schema_version",), 2),
        (("world", "width"), "1000"),
        (("world", "height"), float("nan")),
        (("solids", 0, "width"), -1),
        (("spawn", "x"), True),
        (("extra",), {}),
        (("level_id",), ""),
    ],
)
def test_bad_structure_has_precise_paths(data, path, value):
    item = data
    for key in path[:-1]:
        item = item[key]
    item[path[-1]] = value
    report = validate_level_spec(data)
    assert not report["structure_valid"]
    assert report["issues"][0]["path"] == "/" + "/".join(map(str, path))


def test_duplicate_ids_and_unreachable_goal_are_errors(data):
    data["goal"]["x"] = data["world"]["width"]
    coin = {"entity_id": "same", "x": 100, "y": 500, "width": 16, "height": 24}
    data["collectibles"] = [coin, copy.deepcopy(coin)]
    report = validate_level_spec(data)
    assert {i["code"] for i in report["issues"]} >= {"goal_bounds", "duplicate_entity_id"}
    assert not report["static_valid"]


def test_spawn_collision_vs_surface_contact(data):
    assert validate_level_spec(data)["static_valid"]
    data["spawn"]["bottom"] += 1
    assert "spawn_overlap" in {i["code"] for i in validate_level_spec(data)["issues"]}


def test_collectible_diagnostics_preserve_optional_airborne_items(data):
    data["collectibles"] = [
        {"entity_id": "air", "x": 200, "y": 430, "width": 16, "height": 24},
        {"entity_id": "beyond", "x": data["goal"]["x"] + 10, "y": 500, "width": 16, "height": 24},
    ]
    r = validate_level_spec(data)
    assert r["static_valid"]
    assert [i["code"] for i in r["issues"]] == ["collectible_after_goal"]
    data["collectibles"][0]["y"] = 550
    r = validate_level_spec(data)
    assert not r["static_valid"]
    assert "collectible_embedded" in {i["code"] for i in r["issues"]}


def test_enemy_constraints_and_cross_kind_ids(data):
    data["enemies"] = [
        {
            "entity_id": "x",
            "x": 200,
            "y": 512,
            "width": 28,
            "height": 28,
            "patrol_left": 210,
            "patrol_right": 300,
        }
    ]
    assert "patrol_bounds" in {i["code"] for i in validate_level_spec(data)["issues"]}
    data["enemies"][0]["direction"] = True
    assert not validate_level_spec(data)["structure_valid"]


def test_physics_is_explicit_and_changes_spawn_admission(data):
    p = replace(PhysicsConfig(), player_height=550)
    r = validate_level_spec(data, physics=p)
    assert not r["static_valid"]
    assert r["physics"]["player_height"] == 550


def test_metadata_does_not_hide_train_test_duplicates(data):
    a = LevelSpec.model_validate(data)
    data["level_id"] = "other"
    data["metadata"]["title"] = "other title"
    b = LevelSpec.model_validate(data)
    assert document_hash(a) != document_hash(b)
    assert gameplay_hash(a) == gameplay_hash(b)
    r = validate_dataset({"train": [a], "test": [b]})
    assert not r["static_valid"]
    assert r["issues"][0]["code"] == "duplicate_gameplay"


def test_schema_snapshot_matches_model():
    actual = LevelSpec.model_json_schema()
    actual["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    assert actual == json.loads(Path("schemas/level_spec_v1.schema.json").read_text())


def test_core_replay_is_identical_after_conversion():
    original = TrainingLevelRepository("config/curriculum_v9.json").load("v8_full_validation_05")
    restored = to_level_definition(from_level_definition(original))
    cores = [BasicPlatformerCore(lambda _, level=l: level) for l in (original, restored)]
    states = [core.reset(seed=100, level_id=original.level_id) for core in cores]
    assert states[0] == states[1]
    for action in [Action.RIGHT_RUN] * 20 + [Action.RIGHT_RUN_JUMP] * 10 + [Action.RIGHT_RUN] * 10:
        steps = [core.step(action) for core in cores]
        assert steps[0] == steps[1]
        if steps[0].terminated or steps[0].truncated:
            break


def test_loader_limits_bytes(tmp_path, data):
    p = tmp_path / "level.json"
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="byte limit"):
        load_level_spec(p, max_bytes=5)


def test_cli_reports_bad_json_and_rejects_overwrite(tmp_path, data):
    level, report = tmp_path / "level.json", tmp_path / "report.json"
    level.write_text('{"schema_version":1,"schema_version":2}')
    cmd = [
        sys.executable,
        "-m",
        "scripts.validate_level_spec",
        "--input",
        str(level),
        "--output",
        str(report),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy(), check=False)
    assert result.returncode == 1
    assert json.loads(report.read_text())["issues"][0]["code"] == "input_error"
    saved = report.read_bytes()
    assert subprocess.run(cmd, capture_output=True, check=False).returncode == 2
    assert report.read_bytes() == saved
    level.write_text(json.dumps(data))
    valid = tmp_path / "valid.json"
    cmd[-1] = str(valid)
    assert subprocess.run(cmd, capture_output=True, check=False).returncode == 0
    assert json.loads(valid.read_text())["reachability"] == "not_checked"


def test_numeric_spellings_and_entity_ids_cannot_hide_duplicate_geometry(data):
    from ai_platformer.content.level_spec import geometry_hash

    a = LevelSpec.model_validate(data)
    data["level_id"] = "renamed"
    data["world"]["width"] = int(data["world"]["width"])
    b = LevelSpec.model_validate(data)
    assert gameplay_hash(a) != gameplay_hash(b)
    assert geometry_hash(a) == geometry_hash(b)
    assert not validate_dataset({"train": [a], "test": [b]})["static_valid"]


def test_loader_rejects_duplicate_json_keys(tmp_path):
    p = tmp_path / "duplicate.json"
    p.write_text('{"schema_version":1,"schema_version":2}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_level_spec(p)


def test_empty_splits_and_budget_exhaustion_are_not_admitted(data):
    assert not validate_dataset({})["static_valid"]
    assert not validate_dataset({"train": []})["static_valid"]
    data["solids"] *= 101
    data["collectibles"] = [
        {"entity_id": f"coin-{i}", "x": 100, "y": 400, "width": 16, "height": 24}
        for i in range(1000)
    ]
    report = validate_level_spec(data)
    assert not report["static_valid"]
    assert "diagnostic_budget" in {i["code"] for i in report["issues"]}
