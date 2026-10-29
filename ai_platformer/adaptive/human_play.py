"""Observe the human core directly, without instantiating an RL environment."""

from dataclasses import asdict, dataclass, replace
from types import SimpleNamespace
from uuid import uuid4

from ai_platformer.content.curriculum import content_hash
from .metrics import digest
from .progress import ProgressStore
from .telemetry import TelemetryRecorder


@dataclass(frozen=True)
class CoreRewardDescription:
    version: int = 1
    kind: str = "core_progress_coin_outcome"


class HumanPlayProtocol:
    environment_id = "HumanPlatformer-v1"
    action_repeat = 1
    sensor_range = 0.0
    reward = CoreRewardDescription()

    def __init__(self, prepared):
        # Main application always simulates at 60 Hz, independently of rendering.
        self.settings = replace(prepared.settings, render_fps=60)
        self.repository = prepared.repository
        self.step_limit = self.settings.physics.max_episode_steps

    def environment(self):
        return {"environment_id": self.environment_id, "action_repeat": 1,
                "episode_step_limit": self.step_limit, "sensor_range": self.sensor_range,
                "physics": asdict(self.settings.physics), "reward": asdict(self.reward)}

    def protocol(self, ids):
        return {**self.environment(), "tick_hz": 60, "input_edge_version": 1,
                "levels": {key: content_hash(self.repository.load(key)) for key in ids}}


def entry_key(entry):
    return str(entry.path.resolve()) if entry.path else f"legacy:{entry.level_id}"


def progress_key(prepared):
    protocol = HumanPlayProtocol(prepared)
    level = prepared.repository.load(prepared.settings.level_id)
    return digest({"level": content_hash(level), "environment": protocol.environment(),
                   "tick_hz": 60, "input_edge_version": 1})


class HumanPlayRecords:
    def __init__(self, directory, *, clock=None, headless=False):
        self.store = ProgressStore(directory)
        self.directory = self.store.directory
        self.clock = clock
        self.headless = headless
        self.recorder = None
        self.key = None
        self.entry = None
        self.active = False
        self.state = None
        self.total_reward = 0.0
        self.warning = ""
        self.closed = False
        self._lookup_prepared = None
        self._lookup_key = None

    @property
    def notice(self):
        return self.store.warning or self.warning

    def begin(self, prepared, entry, state):
        key = progress_key(prepared)
        source = entry_key(entry)
        same_level = (key == self.key and source == self.entry and self.state is not None
                      and self.state.level_id == state.level_id)
        self.finish("restart" if same_level else "level_switch")
        if key != self.key and self.recorder:
            self._call("close", "level_switch")
            self.recorder = None
        self.key, self.state, self.active = key, state, True
        self.entry = source
        self.total_reward = 0.0
        self.store.begin(key, state.level_id, source)
        if self.recorder is None:
            try:
                options = {"clock": self.clock} if self.clock else {}
                self.recorder = TelemetryRecorder(
                    self.directory / "sessions" / uuid4().hex, HumanPlayProtocol(prepared),
                    actor_type="human", player_id=self.store.data["player_id"],
                    headless=self.headless,
                    source={"entrypoint": "source.main", "input_edge_version": 1}, **options,
                )
            except (OSError, ValueError, TypeError):
                self.warning = "游玩记录写入失败，游戏可继续。"
        if self.recorder:
            # Same geometry can have a different LevelSpec ID; metadata stays
            # environment-bound but each attempt uses the current frozen map.
            self.recorder.factory = HumanPlayProtocol(prepared)
            self._call("set_paused", False)
            self._call("on_reset", SimpleNamespace(level=prepared.repository.load(state.level_id)), state)

    def _call(self, method, *args, **kwargs):
        if not self.recorder:
            return
        if self.recorder.failed and method != "close":
            self.warning = "游玩记录不完整，游戏可继续。"
            return
        try:
            getattr(self.recorder, method)(*args, **kwargs)
            if self.recorder.failed:
                self.warning = "游玩记录不完整，游戏可继续。"
        except (OSError, ValueError, TypeError, RuntimeError, KeyError) as error:
            self.recorder.disable(error)
            self.warning = "游玩记录不完整，游戏可继续。"

    def tick(self, before, action, result, *, jump_pressed, jump_started):
        if not self.active:
            return
        self._call("on_decision", action, result.state.tick)
        self._call("on_tick", before, action, result,
                   jump_pressed=jump_pressed, jump_started=jump_started)
        self.state = result.state
        self.total_reward += result.reward
        done = result.terminated or result.truncated
        outcome = result.info.get("outcome")
        self._call("on_step_end", {"episode_return": self.total_reward, "outcome": outcome}, done)
        if done:
            self.finish(outcome)

    def pause(self, paused):
        if self.active:
            self._call("set_paused", paused)

    def finish(self, reason):
        if not self.active:
            return
        self._call("end_attempt", reason)
        self.store.finish(self.key, self.state, reason)
        self.active = False

    def best(self, prepared):
        if self._lookup_prepared is not prepared:
            self._lookup_prepared = prepared
            self._lookup_key = progress_key(prepared)
        return self.store.data["levels"].get(self._lookup_key)

    def close(self, reason="quit"):
        if self.closed:
            return
        try:
            self.finish(reason)
            self._call("close", reason)
        finally:
            self.store.close()
            self.closed = True
