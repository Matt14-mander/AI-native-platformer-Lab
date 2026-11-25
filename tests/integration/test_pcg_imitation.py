import json
from pathlib import Path

import pytest

from ai_platformer.agents.ppo.pcg_experiment import (
    collect_dagger_data, collect_teacher_data, evaluate_model, training_requests,
)
from ai_platformer.core import Action


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


def test_dagger_labels_learner_visited_states():
    class RunOnlyModel:
        def predict(self, observation, deterministic=True):
            return int(Action.RIGHT_RUN), None

    request = training_requests(PLAN, [20261200], MANIFEST)[1]
    observations, labels, maps = collect_dagger_data(RunOnlyModel(), [request], policy_seed=17)
    assert observations.shape == (len(labels), 15)
    assert int(Action.RIGHT_RUN_JUMP) in labels
    assert maps[0]["teacher_disagreements"] > 0
    assert maps[0]["learner_outcome"] != "success"


def test_generated_sampler_changes_levels_only_at_reset():
    from ai_platformer.agents.ppo.pcg_training import GeneratedMapSampler

    requests = training_requests(PLAN, [20261200], MANIFEST)
    left = GeneratedMapSampler(requests, rng_seed=17)
    right = GeneratedMapSampler(requests, rng_seed=17)
    try:
        for _ in range(8):
            _, left_info = left.reset()
            _, right_info = right.reset()
            assert left_info["level_id"] == right_info["level_id"]
            assert left_info["seed"] == right_info["seed"]
            level_before = left_info["level_id"]
            _, _, _, _, step_info = left.step(int(Action.RIGHT_RUN))
            assert step_info["level_id"] == level_before
    finally:
        left.close()
        right.close()


def test_bc_initialization_copies_actor_but_not_critic():
    import torch

    from ai_platformer.agents.ppo.pcg_training import initialize_actor, make_ppo

    requests = training_requests(PLAN, [20261200], MANIFEST)
    target = make_ppo(requests, seed=17, n_envs=2)
    source = make_ppo(requests, seed=23, n_envs=2)
    try:
        original_value = {key: value.clone() for key, value in
                          target.policy.value_net.state_dict().items()}
        initialize_actor(target, source)
        for key, value in target.policy.action_net.state_dict().items():
            assert torch.equal(value, source.policy.action_net.state_dict()[key])
        for key, value in target.policy.value_net.state_dict().items():
            assert torch.equal(value, original_value[key])
    finally:
        target.get_env().close()
        source.get_env().close()
