"""Real-core telemetry, lifecycle, replay parity and offscreen keyboard controls."""

import copy
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace

import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("pydantic")

from ai_platformer.adaptive.telemetry import TelemetryRecorder, read_session, validate_event
from ai_platformer.content.level_spec import from_level_definition
from ai_platformer.core import BlockSpawn, CollectibleSpawn, LevelDefinition, SolidRect
from ai_platformer.envs.factory import EnvironmentFactory, protocol_hash


@pytest.fixture
def factory(tmp_path):
    level = LevelDefinition(
        "telemetry_test", width=600, height=600, spawn_x=48, spawn_bottom=540, goal_x=550,
        solids=(SolidRect(0, 540, 600, 60),),
        collectibles=(CollectibleSpawn("pinecone", "coin", 48, 508),),
        blocks=(BlockSpawn("brick", "brick", x=48, y=460),),
    )
    path = tmp_path / "level.json"
    path.write_text(from_level_definition(level).model_dump_json())
    return EnvironmentFactory({
        "environment_id": "PlatformerState-v2", "level_spec": str(path),
        "action_repeat": 4, "episode_step_limit": 100,
    })


def record(factory, directory, **kwargs):
    env = factory.make()
    recorder = TelemetryRecorder(directory, factory, actor_type="human", **kwargs)
    env.telemetry = recorder
    return env, recorder


def test_middle_tick_interactions_and_collectibles_are_not_lost(factory, tmp_path):
    env, recorder = record(factory, tmp_path / "session", trace_actions=True)
    env.reset(seed=100)
    _, _, _, _, info = env.step(3)
    assert "events" not in info  # Existing last-tick info is unchanged.
    env.step(0)
    recorder.close()
    env.close()
    session = read_session(tmp_path / "session")
    events = session["events"]
    assert session["complete"]
    assert [(e["tick"], e["payload"]["name"]) for e in events if e["type"] == "interaction"] == [
        (1, "brick_broken")
    ]
    assert [e["payload"]["entity_id"] for e in events if e["type"] == "coin_collected"] == [
        "pinecone"
    ]
    assert len([e for e in events if e["type"] == "jump_started"]) == 1
    release = next(e for e in events if e["type"] == "jump_released")
    assert release["payload"]["held_ticks"] == 4
    assert session["attempts"][0]["coins_collected"] == env.core.state.metadata["coins_collected"]


def test_jump_requests_are_distinct_from_accepted_jumps_and_holding_does_not_rejump(factory, tmp_path):
    env, recorder = record(factory, tmp_path / "session")
    env.reset(seed=100)
    for _ in range(8):
        env.step(3)  # Includes landing after the first head hit, without releasing jump.
    assert env.core.state.player.grounded
    env.step(0)
    env.step(3)
    env.step(0)
    assert not env.core.state.player.grounded
    env.step(3)  # An airborne request is not an accepted jump.
    recorder.close()
    events = read_session(tmp_path / "session")["events"]
    assert sum(e["type"] == "jump_requested" for e in events) == 3
    assert sum(e["type"] == "jump_started" for e in events) == 2
    assert any(e["type"] == "landed" for e in events)


def test_observations_rewards_states_protocol_and_action_trace_replay_unchanged(factory, tmp_path):
    before = protocol_hash(factory.protocol([factory.level_id]))
    baseline = factory.make()
    observed, recorder = record(factory, tmp_path / "session", trace_actions=True)
    assert baseline.reset(seed=42)[0].tolist() == observed.reset(seed=42)[0].tolist()
    for action in [3, 0, 6, 6, 9, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]:
        a, b = baseline.step(action), observed.step(action)
        assert a[0].tolist() == b[0].tolist()
        assert a[1:] == b[1:]
        assert baseline.core.state == observed.core.state
        if a[2] or a[3]:
            break
    recorder.close()
    assert protocol_hash(factory.protocol([factory.level_id])) == before
    session = read_session(tmp_path / "session")
    replay = factory.make()
    replay.reset(seed=42)
    actions = [e["payload"]["action"] for e in session["events"] if e["type"] == "action"]
    steps = [e["payload"] for e in session["events"] if e["type"] == "step"]
    assert len(actions) == len(steps) > 1
    for action, step in zip(actions, steps, strict=True):
        info = replay.step(action)[4]
        assert info["episode_return"] == step["episode_return"]
        assert hashlib.sha256(
            json.dumps(asdict(replay.core.state), sort_keys=True).encode()
        ).hexdigest() == step["state_sha256"]
    for env in (baseline, observed, replay):
        env.close()


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


def test_pause_restart_quit_are_exactly_once_and_seeds_do_not_identify_attempts(factory, tmp_path):
    clock = Clock()
    env, recorder = record(factory, tmp_path / "session", clock=clock)
    env.reset(seed=100)
    env.step(0)
    clock.now = 2
    recorder.set_paused(True)
    recorder.set_paused(True)
    clock.now = 7
    recorder.set_paused(False)
    clock.now = 10
    env.reset(seed=100)
    recorder.close("quit")
    recorder.close("quit")
    session = read_session(tmp_path / "session")
    assert [a["reason"] for a in session["attempts"]] == ["restart", "quit"]
    assert len({a["attempt_id"] for a in session["attempts"]}) == 2
    assert session["attempts"][0]["active_wall_seconds"] == 5
    assert session["attempts"][0]["paused_wall_seconds"] == 5
    assert session["attempts"][0]["ticks"] == 4
    assert not session["eligible_for_profile"]
    assert [e["type"] for e in session["events"]].count("pause") == 1


@pytest.mark.parametrize("reason", ["success", "death", "time_limit"])
def test_automatic_terminal_and_close_do_not_double_count(factory, tmp_path, reason):
    # Use genuine terminal transitions, including the environment's decision limit.
    if reason == "time_limit":
        factory.step_limit = 1
    elif reason == "death":
        factory.repository.levels[factory.level_id] = LevelDefinition(
            factory.level_id, 600, 600, 48, 540, 550, (),
        )
    env, recorder = record(factory, tmp_path / "session")
    env.reset(seed=100)
    for _ in range(100):
        _, _, term, trunc, info = env.step(6)
        if term or trunc:
            break
    assert info["outcome"] == reason
    recorder.close()
    session = read_session(tmp_path / "session")
    assert [a["reason"] for a in session["attempts"]] == [reason]
    assert session["eligible_for_profile"]


def test_empty_session_and_open_attempt_are_not_valid_profile_samples(factory, tmp_path):
    recorder = TelemetryRecorder(tmp_path / "empty", factory, actor_type="human")
    recorder.close()
    assert read_session(tmp_path / "empty")["complete"]
    assert not read_session(tmp_path / "empty")["eligible_for_profile"]
    env, recorder = record(factory, tmp_path / "interrupted")
    env.reset(seed=100)
    env.step(0)
    recorder.writer.stream.flush()
    assert read_session(tmp_path / "interrupted")["attempts"][0]["reason"] == "incomplete"
    # Emulate a crash while writing the next JSONL record.
    recorder.writer.stream.write('{"schema_version":')
    recorder.writer.stream.flush()
    session = read_session(tmp_path / "interrupted")
    assert session["truncated_tail"] and not session["complete"]
    assert not session["eligible_for_profile"]
    recorder.writer.close()


def test_paused_restart_carries_pause_and_level_switch_is_not_death(factory, tmp_path):
    clock = Clock()
    env, recorder = record(factory, tmp_path / "session", clock=clock)
    env.reset(seed=100)
    clock.now = 1
    recorder.set_paused(True)
    clock.now = 3
    env.reset(seed=100)
    clock.now = 7
    recorder.set_paused(False)
    old_level = factory.repository.levels[factory.level_id]
    factory.repository.levels["other"] = replace(old_level, level_id="other")
    env.reset(seed=100, options={"level_id": "other"})
    recorder.close()
    session = read_session(tmp_path / "session")
    assert [a["reason"] for a in session["attempts"]] == ["restart", "level_switch", "quit"]
    assert session["attempts"][0]["paused_wall_seconds"] == 2
    assert session["attempts"][1]["paused_wall_seconds"] == 4


def test_replay_restart_is_explicit_and_anonymous_players_stay_session_local(factory, tmp_path):
    recorder = TelemetryRecorder(tmp_path / "session", factory, actor_type="replay")
    env = factory.make(); env.telemetry = recorder
    env.reset(seed=100); env.step(0); env.reset(seed=100)
    recorder.close()
    session = read_session(tmp_path / "session")
    assert session["manifest"]["identity_scope"] == "session_only"
    assert [a["reason"] for a in session["attempts"]] == ["replay", "quit"]


@pytest.mark.parametrize("kwargs", [{"playback_speed": 2}, {"headless": True}])
def test_playback_speed_and_headless_are_not_normal_human_profile_samples(factory, tmp_path, kwargs):
    factory.step_limit = 1
    env, recorder = record(factory, tmp_path / "session", **kwargs)
    env.reset(seed=100); env.step(0); recorder.close()
    assert not read_session(tmp_path / "session")["eligible_for_profile"]


@pytest.mark.parametrize("actor", ["scripted", "ppo", "replay"])
def test_nonhuman_sessions_never_become_human_profile_data(factory, tmp_path, actor):
    factory.step_limit = 1
    env = factory.make()
    recorder = TelemetryRecorder(tmp_path / "session", factory, actor_type=actor)
    env.telemetry = recorder
    env.reset(seed=100)
    env.step(0)
    recorder.close()
    assert not read_session(tmp_path / "session")["eligible_for_profile"]


def test_write_failure_disables_recording_without_changing_game(factory, tmp_path, monkeypatch):
    baseline = factory.make()
    env, recorder = record(factory, tmp_path / "session")
    baseline.reset(seed=100)
    env.reset(seed=100)

    def failed_write(event):
        raise OSError("disk full")

    monkeypatch.setattr(recorder.writer, "append", failed_write)
    with pytest.warns(RuntimeWarning, match="telemetry incomplete"):
        result = env.step(3)
    expected = baseline.step(3)
    assert result[0].tolist() == expected[0].tolist() and result[1:] == expected[1:]
    assert env.core.state == baseline.core.state
    recorder.close()
    assert recorder.failed
    assert not read_session(tmp_path / "session")["complete"]


@pytest.mark.parametrize("change", ["version", "sequence", "identity", "payload", "hash"])
def test_malformed_events_and_changed_closed_files_are_rejected(factory, tmp_path, change):
    env, recorder = record(factory, tmp_path / "session")
    env.reset(seed=100)
    env.step(0)
    recorder.close()
    path = tmp_path / "session" / "events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    if change == "version":
        event = copy.deepcopy(events[0]); event["schema_version"] = True
        with pytest.raises(ValueError):
            validate_event(event)
        return
    if change == "payload":
        event = next(e for e in events if e["type"] == "attempt_end")
        event["payload"]["reason"] = "failure"
        with pytest.raises(ValueError):
            validate_event(event)
        return
    if change == "sequence":
        events[1]["event_seq"] += 1
    elif change == "identity":
        events[1]["session_id"] = "foreign"
    else:
        events[1]["payload"]["coins_total"] += 1
    path.write_text("".join(json.dumps(e) + "\n" for e in events))
    if change != "hash":
        # Even an unclosed file without a final digest must validate identity/sequence.
        manifest = tmp_path / "session" / "manifest.json"
        data = json.loads(manifest.read_text()); data["status"] = "recording"
        manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        read_session(tmp_path / "session")


def test_offscreen_keyboard_pause_restart_quit_entrypoint(factory, tmp_path, monkeypatch):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    pygame = pytest.importorskip("pygame")
    from scripts.play_level_spec import main

    key = lambda k: pygame.event.Event(pygame.KEYDOWN, key=k)
    frames = iter([[], [key(pygame.K_p)], [], [key(pygame.K_p)], [key(pygame.K_r)], [],
                   [pygame.event.Event(pygame.QUIT)]])
    monkeypatch.setattr(pygame.event, "get", lambda: next(frames))
    monkeypatch.setattr(sys, "argv", [
        "play_level_spec", "--input", str(factory.repository.path),
        "--telemetry-dir", str(tmp_path / "keyboard"), "--player-id", "tester",
    ])
    main()
    session = read_session(tmp_path / "keyboard")
    assert session["complete"] and session["manifest"]["actor_type"] == "human"
    assert [a["reason"] for a in session["attempts"]] == ["restart", "quit"]
    kinds = [e["type"] for e in session["events"]]
    assert "pause" in kinds and "resume" in kinds


def test_headless_cli_and_readonly_inspection(factory, tmp_path):
    session = tmp_path / "session"
    report = tmp_path / "report.json"
    for command in (
        ["scripts.play_level_spec", "--input", str(factory.repository.path),
         "--controller", "rule", "--headless", "--episodes", "1",
         "--telemetry-dir", str(session), "--trace-actions"],
        ["scripts.inspect_telemetry", "--input", str(session), "--output", str(report)],
    ):
        result = subprocess.run(
            [sys.executable, "-m", *command], capture_output=True, text=True, check=False,
            env={**os.environ, "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"},
        )
        assert result.returncode == 0, result.stderr
    data = json.loads(report.read_text())
    assert data["complete"] and data["actor_type"] == "scripted"
    assert not data["eligible_for_profile"]
    assert data["event_counts"]["attempt_end"] == 1
