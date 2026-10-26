"""Real recorded sessions exercise metric denominators, provenance and evidence gates."""

import copy
import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("pydantic")

from ai_platformer.adaptive.metrics import build_metrics
from ai_platformer.adaptive.profile import build_profiles
from ai_platformer.adaptive.reporting import build_report, markdown, source_hashes
from ai_platformer.adaptive.telemetry import TelemetryRecorder, read_session
from ai_platformer.content.level_spec import from_level_definition
from ai_platformer.core import CollectibleSpawn, LevelDefinition, SolidRect
from ai_platformer.envs.factory import EnvironmentFactory, protocol_hash


def make_factory(tmp_path, name="map_a", *, width=600, coins=True, limit=1000):
    level = LevelDefinition(
        name, width, 600, 48, 540, width-50, (SolidRect(0, 540, width, 60),),
        (CollectibleSpawn("pinecone", "coin", 48, 508),) if coins else (),
    )
    path = tmp_path / f"{name}.json"
    path.write_text(from_level_definition(level).model_dump_json(), encoding="utf-8")
    return EnvironmentFactory({"environment_id": "PlatformerState-v2", "level_spec": str(path),
                               "episode_step_limit": limit, "action_repeat": 4})


def make_session(tmp_path, factory, name, *, actor="human", player="alice", episodes=1,
                 actions=None, **kwargs):
    directory = tmp_path / name
    env = factory.make()
    recorder = TelemetryRecorder(directory, factory, actor_type=actor, player_id=player, **kwargs)
    env.telemetry = recorder
    for _ in range(episodes):
        env.reset(seed=100)
        if actions is not None:
            for action in actions:
                env.step(action)
        else:
            while True:
                _, _, term, trunc, _ = env.step(6)
                if term or trunc:
                    break
    recorder.close()
    env.close()
    return directory


def test_metrics_counts_and_motion_match_independent_physics_trace(tmp_path):
    factory = make_factory(tmp_path)
    before = protocol_hash(factory.protocol([factory.level_id]))
    actions = [0] * 125 + [6] * 8 + [7] * 10 + [9] * 3 + [6] * 5
    path = make_session(tmp_path, factory, "observed", actions=actions)
    observed = build_metrics([read_session(path)])["attempts"][0]
    baseline = factory.make(); baseline.reset(seed=100)
    ticks = []; original = baseline.core.step

    def capture(action):
        initial = baseline.core.state
        result = original(action)
        ticks.append((result.state.player.x-initial.player.x, result.state.player.grounded))
        return result

    baseline.core.step = capture
    for action in actions:
        baseline.step(action)
    motion = observed["motion"]
    assert motion["observed_ticks"] == len(ticks)
    assert motion["backward_distance_px"] == pytest.approx(sum(max(0, -dx) for dx, _ in ticks))
    assert motion["forward_distance_px"] == pytest.approx(sum(max(0, dx) for dx, _ in ticks))
    assert motion["airborne_ticks"] == sum(not grounded for _, grounded in ticks)
    assert motion["stagnation_ticks"] >= 500
    assert motion["stagnation_runs"] >= 1
    assert observed["reason"] == "quit" and not observed["valid_terminal"]
    assert observed["coins_collected"] == baseline.core.state.metadata["coins_collected"]
    assert protocol_hash(factory.protocol([factory.level_id])) == before


def test_terminal_denominators_and_retries_exclude_restarts_and_quits(tmp_path):
    factory = make_factory(tmp_path, limit=2)
    directory = tmp_path / "session"
    recorder = TelemetryRecorder(directory, factory, actor_type="human", player_id="alice")
    env = factory.make(); env.telemetry = recorder
    env.reset(seed=100)
    env.step(0); env.step(0)  # timeout
    env.reset(seed=100); env.step(0)  # failed-attempt retry, then manual restart
    env.reset(seed=100)  # no second failure, so this is not a failure retry
    recorder.close()
    metrics = build_metrics([read_session(directory)])
    summary = metrics["cohorts"][0]["summary"]
    assert summary["valid_terminal_attempts"] == 1
    assert summary["timeouts"] == 1 and summary["deaths"] == 0
    assert summary["retry_starts_after_failure"] == 1
    assert summary["observed_end_reasons"] == {"quit": 1, "restart": 1, "time_limit": 1}
    assert summary["success_rate"] == 0


def test_empty_coin_maps_and_legacy_motion_are_unknown_not_zero(tmp_path):
    path = make_session(tmp_path, make_factory(tmp_path, coins=False), "session")
    session = read_session(path)
    for event in session["events"]:
        if event["type"] == "attempt_end":
            event["payload"].pop("motion")
    for attempt in session["attempts"]:
        attempt.pop("motion")
    # Actual historical schema remains readable; no synthetic filling of motion fields.
    metrics = build_metrics([session])
    summary = metrics["cohorts"][0]["summary"]
    assert summary["collection"]["weighted_ratio"] is None
    assert summary["motion"]["backward_distance_px"] is None
    profile = build_profiles(metrics)[0]
    assert profile["dimensions"]["motion_behavior"]["status"] == "unknown"
    assert profile["dimensions"]["collection_performance"]["status"] == "unknown"


def test_readiness_requires_independent_geometry_and_uses_only_recent_window(tmp_path):
    a, b = make_factory(tmp_path), make_factory(tmp_path, "map_b", width=700)
    p1 = make_session(tmp_path, a, "a", episodes=4)
    p2 = make_session(tmp_path, b, "b", episodes=2)
    p3 = make_session(tmp_path, a, "a_later", episodes=10)
    metrics = build_metrics([read_session(p1), read_session(p2)])
    profile = build_profiles(metrics)[0]
    assert profile["status"] == "ready"
    assert len(profile["evidence"]) == 6
    assert profile["readiness"]["independent_map_count"] == 2
    for key in ("collection_preference", "jump_error_rate", "combat_skill"):
        assert profile["dimensions"][key]["status"] == "unknown"
    later = build_profiles(build_metrics([read_session(p1), read_session(p2), read_session(p3)]))[0]
    assert len(later["evidence"]) == 10
    assert later["lifetime"]["valid_attempts"] == 16
    assert later["status"] == "insufficient_data"  # recent evidence covers only one map


def test_actor_identity_environment_and_playback_are_isolated(tmp_path):
    factory = make_factory(tmp_path)
    human = make_session(tmp_path, factory, "human", episodes=5)
    baseline = build_profiles(build_metrics([read_session(human)]))
    others = [make_session(tmp_path, factory, actor, actor=actor, episodes=6)
              for actor in ("ppo", "scripted", "replay")]
    assert build_profiles(build_metrics([read_session(human), *map(read_session, others)])) == baseline
    bob = make_session(tmp_path, factory, "bob", player="bob")
    anonymous = [make_session(tmp_path, factory, f"anon_{i}", player=None) for i in range(2)]
    factory.step_limit = 900
    changed_environment = make_session(tmp_path, factory, "changed_env")
    profiles = build_profiles(build_metrics(list(map(read_session, [human, bob, *anonymous, changed_environment]))))
    assert len(profiles) == 5
    assert sum(p["identity"]["scope"] == "session_only" for p in profiles) == 2
    fast = make_session(tmp_path, factory, "fast", playback_speed=2)
    headless = make_session(tmp_path, factory, "headless", headless=True)
    assert build_profiles(build_metrics(list(map(read_session, [fast, headless])))) == []


def test_duplicates_and_source_mutation_are_rejected_and_reports_are_deterministic(tmp_path):
    path = make_session(tmp_path, make_factory(tmp_path), "session")
    before = source_hashes(path)
    first = build_report([path])
    assert build_report([path]) == first
    assert markdown(first) == markdown(build_report([path]))
    assert source_hashes(path) == before
    with pytest.raises(ValueError, match="distinct"):
        build_report([path, path])
    clone = copy.deepcopy(read_session(path))
    with pytest.raises(ValueError, match="duplicate session"):
        build_metrics([read_session(path), clone])
    for window in (0, True, 101):
        with pytest.raises(ValueError, match="window"):
            build_profiles(first["metrics"], window=window)
    events = path / "events.jsonl"
    events.write_bytes(events.read_bytes().replace(b'"success"', b'"failure"'))
    with pytest.raises(ValueError, match="digest"):
        build_report([path])


def test_summary_cli_and_offline_imports(tmp_path):
    path = make_session(tmp_path, make_factory(tmp_path), "session")
    output = tmp_path / "report"
    result = subprocess.run(
        [sys.executable, "-m", "scripts.summarize_play_sessions", "--input", str(path),
         "--output-dir", str(output)], text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["profiles"][0]["status"] == "insufficient_data"
    assert (output / "report.md").exists()
    subprocess.run([sys.executable, "-c", ("import sys; import scripts.summarize_play_sessions; "
                    "assert not any(k in sys.modules for k in ('gymnasium','pygame','torch'))")], check=True)


def test_same_geometry_alias_cannot_satisfy_two_map_gate(tmp_path):
    a, alias = make_factory(tmp_path), make_factory(tmp_path, "alias")
    paths = [make_session(tmp_path, a, "a", episodes=3),
             make_session(tmp_path, alias, "alias_sessions", episodes=3)]
    profile = build_profiles(build_metrics(list(map(read_session, paths))))[0]
    assert profile["readiness"]["valid_attempts"] == 6
    assert profile["readiness"]["independent_map_count"] == 1
    assert profile["status"] == "insufficient_data"


def test_unclosed_session_and_speed_changes_cannot_promote_profile(tmp_path):
    factory = make_factory(tmp_path, limit=1)
    recorder = TelemetryRecorder(tmp_path / "open", factory, actor_type="human", player_id="alice")
    env = factory.make(); env.telemetry = recorder
    env.reset(seed=100); env.step(0)
    metrics = build_metrics([read_session(tmp_path / "open")])
    assert metrics["cohorts"][0]["summary"]["valid_terminal_attempts"] == 0
    assert build_profiles(metrics)[0]["readiness"]["valid_attempts"] == 0
    recorder.set_speed(2)
    recorder.close()
    session = read_session(tmp_path / "open")
    assert not session["eligible_for_profile"]
    assert build_profiles(build_metrics([session])) == []


def test_death_locations_and_malformed_motion_are_handled(tmp_path):
    from ai_platformer.adaptive.telemetry import validate_event

    factory = make_factory(tmp_path)
    factory.repository.levels[factory.level_id] = LevelDefinition(
        factory.level_id, 600, 600, 48, 540, 550, (),
    )
    directory = tmp_path / "fall"
    env = factory.make()
    recorder = TelemetryRecorder(directory, factory, actor_type="human")
    env.telemetry = recorder; env.reset(seed=100)
    while True:
        _, _, term, trunc, _ = env.step(0)
        if term or trunc:
            break
    recorder.close()
    session = read_session(directory)
    summary = build_metrics([session])["cohorts"][0]["summary"]
    assert summary["deaths"] == 1 and summary["successes"] == 0
    assert summary["death_bins"] == [{"start_x": 0, "count": 1}]
    assert summary["motion"]["airborne_ratio"] == 1
    event = copy.deepcopy(next(e for e in session["events"] if e["type"] == "attempt_end"))
    event["payload"]["motion"]["observed_ticks"] += 1
    with pytest.raises(ValueError, match="motion tick"):
        validate_event(event)


def test_recorder_failure_mid_reset_does_not_break_game_or_close(tmp_path, monkeypatch):
    factory = make_factory(tmp_path)
    recorder = TelemetryRecorder(tmp_path / "broken", factory, actor_type="human")
    env = factory.make(); env.telemetry = recorder

    def broken_protocol(*args):
        raise RuntimeError("metadata unavailable")

    monkeypatch.setattr(factory, "protocol", broken_protocol)
    with pytest.warns(RuntimeWarning, match="telemetry incomplete"):
        env.reset(seed=100)
    env.step(6)
    recorder.close()
    session = read_session(tmp_path / "broken")
    assert recorder.failed and not session["complete"]
    assert build_profiles(build_metrics([session])) == []


def test_ppo_playback_cli_records_without_polluting_human_profile(tmp_path, monkeypatch):
    pytest.importorskip("pygame")
    from types import SimpleNamespace

    from ai_platformer.deployment import backends
    from scripts.play_ppo import main

    factory = make_factory(tmp_path)
    model = tmp_path / "fake.zip"; model.write_bytes(b"fake policy for entrypoint test")
    class Policy:
        def predict(self, obs, *, deterministic):
            return 6, None
    loaded = SimpleNamespace(factory=factory, policy=Policy(), default_seed=100,
                             model_path=model, label="test", close=lambda: None)
    monkeypatch.setattr(backends, "load_backend", lambda *args, **kwargs: loaded)
    monkeypatch.setattr(sys, "argv", ["play_ppo", "--model", str(model), "--level-spec",
                                     str(factory.repository.path), "--headless", "--episodes", "1",
                                     "--telemetry-dir", str(tmp_path / "ppo")])
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    main()
    session = read_session(tmp_path / "ppo")
    assert session["complete"] and session["attempts"][0]["reason"] == "success"
    assert build_profiles(build_metrics([session])) == []
    assert session["manifest"]["source"]["model_sha256"]
