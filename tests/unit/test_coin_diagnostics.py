from ai_platformer.benchmark.coin_diagnostics import run_agent
from ai_platformer.agents.scripted import MoveRightAgent
from ai_platformer.envs.factory import EnvironmentFactory


def test_coin_diagnostic_tracks_collection_even_with_action_repeat():
    factory = EnvironmentFactory({"environment_id": "PlatformerState-v2",
                                  "level_id": "level_1", "episode_step_limit": 300})
    result = run_agent(factory, MoveRightAgent, seed=100)
    assert result["coins_collected"] == len(set(result["collected_ids"]))
    assert result["coins_collected"] <= result["coins_total"]
    assert result["trace"][-1]["step"] == result["steps"]
