import json
from pathlib import Path

from ai_platformer.agents.scripted.coin_jump import CoinJumpAgent
from ai_platformer.agents.scripted.ground_coin_jump import GroundCoinJumpAgent
from ai_platformer.benchmark.coin_diagnostics import run_agent
from ai_platformer.benchmark.stratified import evaluate_manifest
from ai_platformer.content.generation_suite import load_suite
from ai_platformer.content.pcg import generate_level
from ai_platformer.envs.factory import EnvironmentFactory


def test_short_obstacle_jump_collects_old_mixed_ground_coins(tmp_path):
    suite = load_suite(Path("game_content/generation_suites/small_v1.json"))
    case = next(item for item in suite.cases if item.case_id == "mixed_ground")
    for seed in suite.generation_seeds:
        generated = generate_level(case.expected_request(seed))
        spec = tmp_path / "level.json"
        spec.write_text(json.dumps(generated.level.model_dump()))
        factory = EnvironmentFactory(
            {
                "environment_id": "PlatformerState-v2",
                "level_spec": str(spec),
                "physics": generated.report["physics"],
                "action_repeat": 4,
                "episode_step_limit": 768,
            }
        )
        baseline = run_agent(factory, CoinJumpAgent, seed=suite.policy_seed)
        teacher = run_agent(factory, GroundCoinJumpAgent, seed=suite.policy_seed)
        assert baseline["coins_collected"] == 14
        assert teacher["outcome"] == "success"
        assert teacher["coins_collected"] == teacher["coins_total"] == 16


def test_teacher_improves_frozen_development_without_regression():
    report = evaluate_manifest(
        Path("game_content/benchmark_suites/generalization_v2.json"),
        split="development",
    )
    baseline = report["by_agent"]["coin_jump"]["overall"]
    teacher = report["by_agent"]["ground_coin_jump"]["overall"]
    assert baseline["successes"] == teacher["successes"] == 12
    assert baseline["collection_targets_met"] == 10
    assert teacher["collection_targets_met"] == 12
