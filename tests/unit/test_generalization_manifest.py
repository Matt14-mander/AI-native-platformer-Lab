"""The frozen benchmark must reject repeated geometry and altered requests."""

import json
from pathlib import Path

import pytest

from ai_platformer.benchmark.generalization import freeze_plan, load_manifest


ROOT = Path(__file__).resolve().parents[2]


def test_frozen_manifest_has_disjoint_geometry_and_valid_hashes():
    plan = json.loads((ROOT / "config/generalization_v2_plan.json").read_text())
    reference = json.loads((ROOT / plan["reference_suite"]).read_text())
    actual = load_manifest(ROOT / "game_content/benchmark_suites/generalization_v2.json")
    assert actual == freeze_plan(plan, reference)
    assert {row["split"] for row in actual["entries"]} == {"development", "holdout"}
    assert len(actual["entries"]) == len({row["geometry_sha256"] for row in actual["entries"]})


def test_overlap_seed_is_rejected():
    plan = json.loads((ROOT / "config/generalization_v2_plan.json").read_text())
    reference = json.loads((ROOT / plan["reference_suite"]).read_text())
    plan["splits"]["holdout"][0] = plan["splits"]["development"][0]
    with pytest.raises(ValueError, match="overlap"):
        freeze_plan(plan, reference)
