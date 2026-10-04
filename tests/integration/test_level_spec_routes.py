"""Real-core candidate loading, bounded search and replay integrity tests."""

import copy
import json
import os
import subprocess
import sys
from collections import Counter

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("gymnasium")

from ai_platformer.content.curriculum import TrainingLevelRepository, content_hash
from ai_platformer.content.level_spec import from_level_definition
from ai_platformer.content.routes import digest, replay_route, search_route
from ai_platformer.envs.factory import EnvironmentFactory, protocol_hash
from ai_platformer.rendering.ppo_playback import solids_for_playback


@pytest.fixture(scope="module")
def candidates(tmp_path_factory):
    root = tmp_path_factory.mktemp("routes")
    repo = TrainingLevelRepository("config/curriculum_v9.json")
    result = {}
    for key, level_id in (
        ("flat", "course_flat_08"),
        ("obstacle", "course_obstacle_08"),
        ("full", "v8_full_validation_05"),
        ("legacy", "level_1_main"),
    ):
        path = root / f"{key}.json"
        path.write_text(from_level_definition(repo.load(level_id)).model_dump_json())
        result[key] = path
    return result


def candidate_factory(path):
    return EnvironmentFactory(
        {
            "environment_id": "PlatformerState-v2",
            "level_spec": str(path),
            "episode_step_limit": 768,
            "action_repeat": 4,
        }
    )


@pytest.fixture(scope="module")
def flat_route(candidates):
    report = search_route(candidate_factory(candidates["flat"]), max_seconds=10)
    assert report["status"] == "verified"
    return report["witness"]


def test_candidate_loading_matches_existing_environment_and_preserves_protocol(candidates):
    source = EnvironmentFactory(
        {"environment_id": "PlatformerState-v2", "curriculum_manifest": "config/curriculum_v9.json"}
    )
    level_id = "v8_full_validation_05"
    before = protocol_hash(source.protocol([level_id]))
    new = source.for_level_spec(candidates["full"])
    assert protocol_hash(source.protocol([level_id])) == before
    assert content_hash(source.repository.load(level_id)) == content_hash(
        new.repository.load(level_id)
    )
    assert protocol_hash(new.protocol([level_id])) != before
    a, b = source.make(level_id=level_id), new.make()
    try:
        obs_a, info_a = a.reset(seed=100)
        obs_b, info_b = b.reset(seed=100)
        assert obs_a.tolist() == obs_b.tolist() and info_a == info_b
        for action in [6] * 20 + [9] * 12 + [6] * 20:
            x, y = a.step(action), b.step(action)
            assert x[0].tolist() == y[0].tolist()
            assert x[1:] == y[1:]
            assert a.core.state == b.core.state
            if x[2] or x[3]:
                break
        assert Counter(
            s for group in solids_for_playback(new, level_id).values() for s in group
        ) == Counter(new.repository.load(level_id).solids)
    finally:
        a.close()
        b.close()
    with pytest.raises(ValueError, match="not assigned"):
        new.repository.split("full", "train")


def test_static_invalid_content_is_rejected_before_environment_creation(candidates):
    with pytest.raises(ValueError, match="static admission"):
        candidate_factory(candidates["legacy"])
    with pytest.raises(ValueError, match="not both"):
        EnvironmentFactory(
            {
                "level_spec": str(candidates["flat"]),
                "curriculum_manifest": "config/curriculum_v9.json",
            }
        )


def test_route_replays_from_fresh_environment_and_is_deterministic(candidates, flat_route):
    proof = replay_route(candidates["flat"], flat_route)
    assert proof["passed"]
    repeated = search_route(candidate_factory(candidates["flat"]), max_seconds=10)
    assert repeated["witness"] == flat_route


def test_beam_search_finds_a_real_obstacle_route(candidates):
    report = search_route(
        candidate_factory(candidates["obstacle"]),
        try_scripted=False,
        beam_width=12,
        max_expansions=15000,
        max_seconds=10,
    )
    assert report["status"] == "verified", report
    assert report["method"] == "beam_search"
    assert 9 in report["witness"]["actions"] or 5 in report["witness"]["actions"]
    assert replay_route(candidates["obstacle"], report["witness"])["passed"]


def test_full_route_can_enforce_collection_target(candidates):
    report = search_route(
        candidate_factory(candidates["full"]), min_coin_ratio=0.75, max_seconds=10
    )
    assert report["status"] == "verified", report
    w = report["witness"]
    assert w["coins_collected"] / w["coins_total"] >= 0.75
    assert replay_route(candidates["full"], w)["passed"]


def test_budget_exhaustion_is_unknown_not_unreachable(candidates):
    r = search_route(candidate_factory(candidates["flat"]), max_expansions=1)
    assert r["status"] == "unknown" and r["reason"] == "budget_exhausted"
    assert r["witness"] is None and r["expansions"] == 1


def test_interactive_search_scope_is_explicit(tmp_path, candidates):
    data = json.loads(candidates["flat"].read_text())
    data["blocks"] = [
        {"entity_id": "box", "kind": "box", "x": 200, "y": 430, "width": 40, "height": 40}
    ]
    path = tmp_path / "interactive.json"
    path.write_text(json.dumps(data))
    r = search_route(candidate_factory(path))
    assert r["status"] == "unknown" and r["reason"] == "interactive_content_unsupported"


@pytest.mark.parametrize(
    "change", ["physics", "actions", "step_trace", "initial_trace", "count", "boolean_action"]
)
def test_replay_rejects_tampered_protocol_actions_or_traces(candidates, flat_route, change):
    route = copy.deepcopy(flat_route)
    if change == "physics":
        route["environment"]["physics"]["max_run_speed"] += 1
    elif change == "actions":
        route["actions"][0] = 0
    elif change == "step_trace":
        route["step_sha256"][0] = "0" * 64
        route["trace_sha256"] = digest(route["step_sha256"])
    elif change == "initial_trace":
        route["initial_sha256"] = "0" * 64
    elif change == "count":
        route["coins_collected"] += 1
    else:
        route["actions"][0] = True
    try:
        proof = replay_route(candidates["flat"], route)
    except ValueError:
        return
    assert not proof["passed"]


def test_changed_map_cannot_reuse_old_witness(tmp_path, candidates, flat_route):
    data = json.loads(candidates["flat"].read_text())
    data["metadata"]["title"] = "changed source"
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="protocol mismatch"):
        replay_route(path, flat_route)


def test_route_cli_and_real_offscreen_playback(tmp_path, candidates, flat_route):
    pytest.importorskip("pygame")
    route = tmp_path / "route.json"
    route.write_text(json.dumps({"witness": flat_route}))
    output = tmp_path / "replay.json"
    commands = [
        [
            "scripts.replay_level_route",
            "--input",
            str(candidates["flat"]),
            "--route",
            str(route),
            "--output",
            str(output),
        ],
        [
            "scripts.play_level_spec",
            "--input",
            str(candidates["flat"]),
            "--controller",
            "replay",
            "--route",
            str(route),
            "--headless",
            "--episodes",
            "1",
            "--screenshot",
            str(tmp_path / "frame.png"),
        ],
    ]
    for command in commands:
        completed = subprocess.run(
            [sys.executable, "-m", *command],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"},
        )
        assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text())["passed"]
    assert (tmp_path / "frame.png").stat().st_size > 1000
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["steps"] == flat_route["steps"] and result["outcome"] == "success"
