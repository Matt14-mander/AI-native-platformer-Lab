"""Real SB3 loading and map simulation in a portable, offline two-map suite."""

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict

import pytest

pytest.importorskip("stable_baselines3")


def test_offline_batch_real_policy_route_replay_and_resume(tmp_path):
    import torch
    from stable_baselines3 import PPO

    from ai_platformer.content.generation_suite import load_suite
    from ai_platformer.content.pcg import SEGMENTS_V1_PHYSICS
    from ai_platformer.content.routes import replay_route
    from ai_platformer.core import Action
    from ai_platformer.envs.factory import EnvironmentFactory, protocol_hash

    config = {
        "environment_id": "PlatformerState-v2",
        "level_id": "course_flat_00",
        "episode_step_limit": 128,
        "action_repeat": 4,
        "physics": asdict(SEGMENTS_V1_PHYSICS),
    }
    factory = EnvironmentFactory(config)
    env = factory.make(seed=100)
    try:
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=8,
            batch_size=8,
            device="cpu",
            seed=1,
            policy_kwargs={"net_arch": [8]},
            verbose=0,
        )
        # A fixed right-running network exercises actual SB3 inference on flat maps.
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.zero_()
            model.policy.action_net.bias[int(Action.RIGHT_RUN)] = 100
        checkpoint = tmp_path / "model.zip"
        model.save(checkpoint)
    finally:
        env.close()
    signature = {"protocol": factory.protocol([factory.level_id])}
    metadata = {
        "schema_version": 2,
        "signature": signature,
        "signature_hash": protocol_hash(signature),
        "model_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "saved_timesteps": 0,
        "config": {"environment": config, "evaluation": {"seeds": [100]}},
    }
    checkpoint.with_suffix(".json").write_text(json.dumps(metadata))
    suite = load_suite("game_content/generation_suites/small_v1.json").model_dump()
    suite["cases"] = suite["cases"][:1]
    suite["generation_seeds"] = suite["generation_seeds"][:2]
    suite_file = tmp_path / "suite.json"
    suite_file.write_text(json.dumps(suite))
    output = tmp_path / "batch"
    command = [
        sys.executable,
        "-m",
        "scripts.evaluate_generated_levels",
        "--suite",
        str(suite_file),
        "--checkpoint",
        str(checkpoint),
        "--provider",
        "expected-fixture",
        "--output-dir",
        str(output),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "complete"
    assert report["summary"]["all_stage_passes"] == 2
    assert report["summary"]["llm_semantic_match_rate_over_planned"] is None
    for row in report["cases"]:
        folder = output / "cases" / row["level_id"]
        witness = json.loads((folder / "route.json").read_text())["witness"]
        assert witness["environment"]["episode_step_limit"] == 128
        assert replay_route(folder / "level.json", witness)["passed"]
        ppo = json.loads((folder / "ppo.json").read_text())
        assert ppo["outcome"] == "success"
        assert ppo["collection_ratio"] == 1
        assert len(ppo["actions"]) == ppo["steps"]
    snapshot = (output / "report.json").read_bytes()
    result = subprocess.run(command + ["--resume"], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert (output / "report.json").read_bytes() == snapshot
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == metadata["model_sha256"]
