import json
from pathlib import Path

import pytest

from ai_platformer.agents.ppo.pcg_experiment import (
    collect_teacher_data, evaluate_model, training_requests,
)


PLAN = Path("config/generalization_v2_plan.json")
MANIFEST = Path("game_content/benchmark_suites/generalization_v2.json")


def test_training_pool_is_disjoint_and_teacher_uses_observations():
    requests = training_requests(PLAN, [20261200], MANIFEST)
    frozen = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert len(requests) == 6
    assert {request.seed for request in requests}.isdisjoint(
        {entry["request"]["seed"] for entry in frozen["entries"]}
    )
    observations, actions, maps = collect_teacher_data(requests, policy_seed=17)
    assert observations.shape == (len(actions), 15)
    assert len({row["geometry_sha256"] for row in maps}) == 6
    assert all(row["outcome"] == "success" for row in maps)


def test_reject_frozen_training_seed_and_holdout_without_open():
    with pytest.raises(ValueError, match="overlap"):
        training_requests(PLAN, [20261125], MANIFEST)
    with pytest.raises(ValueError, match="open_holdout"):
        evaluate_model(None, MANIFEST, split="holdout")
