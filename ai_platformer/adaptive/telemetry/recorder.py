"""Optional per-tick observer; never changes actions, rewards or core state."""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict
from time import perf_counter
from uuid import uuid4

from ai_platformer.content.curriculum import content_hash
from ai_platformer.core import control_for

from .motion import MotionAccumulator
from .schema import ACTORS, END_REASONS
from .storage import SessionWriter


class TelemetryRecorder:
    def __init__(
        self, directory, factory, *, actor_type, player_id=None, trace_actions=False,
        playback_speed=1.0, headless=False, source=None, clock=perf_counter,
    ):
        if actor_type not in ACTORS:
            raise ValueError("unsupported actor_type")
        if player_id is not None and (not isinstance(player_id, str) or not player_id.strip()):
            raise ValueError("player_id must be a nonempty local identifier")
        if not 0.1 <= playback_speed <= 8:
            raise ValueError("invalid playback speed")
        self.factory = factory
        self.clock, self.started = clock, clock()
        self.session_id = uuid4().hex
        self.player_id = player_id
        self.actor_type = actor_type
        self.trace_actions = trace_actions
        self.writer = SessionWriter(directory)
        self.sequence = 0
        self.closed = self.failed = False
        self.attempt_id = None
        self.context = None
        self.state = None
        self.decision = 0
        self.paused = False
        self.pause_started = None
        self.attempt_started = 0.0
        self.paused_seconds = 0.0
        self.jump_held = False
        self.held_ticks = 0
        self.max_progress = 0.0
        self.summaries = []
        self.metadata = {
            "schema_version": 1, "session_id": self.session_id, "player_id": player_id,
            "actor_type": actor_type, "tick_hz": factory.settings.render_fps,
            "trace_actions": trace_actions, "playback_speed": playback_speed,
            "headless": headless, "source": source or {},
            "identity_scope": "player" if player_id is not None else "session_only",
            "environment": {
                "environment_id": factory.environment_id,
                "action_repeat": factory.action_repeat,
                "episode_step_limit": factory.step_limit,
                "sensor_range": factory.sensor_range,
                "physics": asdict(factory.settings.physics), "reward": asdict(factory.reward),
            },
        }
        self._emit("session_start", self.metadata)
        self._manifest("recording")

    def disable(self, error):
        """Fail open for gameplay, fail closed for profile eligibility."""
        if self.failed:
            return
        self.failed = True
        warnings.warn(
            f"telemetry incomplete; recording disabled ({type(error).__name__})",
            RuntimeWarning, stacklevel=2,
        )
        try:
            self.writer.manifest({**self.metadata, "status": "incomplete"})
        except OSError:
            pass

    def _manifest(self, status):
        if self.failed:
            return
        try:
            self.writer.manifest({**self.metadata, "status": status})
        except OSError as error:
            self.disable(error)

    def _emit(self, kind, payload):
        if self.closed or self.failed:
            return
        context = self.context or {
            "level_id": None, "level_sha256": None, "protocol_sha256": None, "seed": None,
        }
        event = {
            "schema_version": 1, "session_id": self.session_id,
            "attempt_id": self.attempt_id, "event_seq": self.sequence,
            "player_id": self.player_id, "actor_type": self.actor_type,
            **context, "tick": self.state.tick if self.attempt_id else None,
            "decision": self.decision if self.attempt_id else None,
            "wall_seconds": self.clock() - self.started, "type": kind, "payload": payload,
        }
        try:
            self.writer.append(event)
            self.sequence += 1
        except OSError as error:
            self.disable(error)

    def on_reset(self, env, state):
        from ai_platformer.envs.factory import protocol_hash

        if self.attempt_id:
            reason = "level_switch" if self.state.level_id != state.level_id else (
                "replay" if self.actor_type == "replay" else "restart"
            )
            self.end_attempt(reason)
        self.attempt_id = uuid4().hex
        self.state = state
        self.context = {
            "level_id": state.level_id, "seed": state.seed,
            "level_sha256": content_hash(env.level),
            "protocol_sha256": protocol_hash(self.factory.protocol([state.level_id])),
        }
        self.decision = 0
        self.jump_held = False
        self.held_ticks = 0
        self.max_progress = state.progress
        self.attempt_started = self.clock()
        self.paused_seconds = 0.0
        self.pause_started = self.attempt_started if self.paused else None
        self._emit("attempt_start", {
            "player": asdict(state.player), "coins_total": state.metadata["coins_total"],
            "initially_paused": self.paused,
        })
        self.motion = MotionAccumulator(self.metadata["tick_hz"])
        self._manifest("recording")

    def on_decision(self, action, decision):
        if self.paused:
            raise RuntimeError("simulation advanced while paused")
        self.decision = decision
        if self.trace_actions:
            self._emit("action", {"action": int(action)})

    def on_tick(self, before, action, result):
        self.motion.observe(before, result.state)
        self.state = result.state
        self.max_progress = max(self.max_progress, self.state.progress)
        control = control_for(action)
        collected = result.info.get("collected", ())
        interactions = result.info.get("events", ())
        has_event = (
            control.jump != self.jump_held
            or (not before.player.grounded and self.state.player.grounded)
            or collected or interactions
        )
        if not has_event:
            if control.jump:
                self.held_ticks += 1
            self.jump_held = control.jump
            return
        player = asdict(self.state.player)
        if control.jump and not self.jump_held:
            self.held_ticks = 0
            self._emit("jump_requested", {"player": player, "grounded_before": before.player.grounded})
        if not control.jump and self.jump_held:
            self._emit("jump_released", {"player": player, "held_ticks": self.held_ticks})
        if control.jump:
            self.held_ticks += 1
        # This is exactly the core's jump acceptance condition, even on an immediate head hit.
        if control.jump and not self.jump_held and before.player.grounded:
            self._emit("jump_started", {"player": player, "takeoff": asdict(before.player)})
        if not before.player.grounded and self.state.player.grounded:
            self._emit("landed", {"player": player})
        self.jump_held = control.jump
        entities = {item.entity_id: item for item in self.state.entities} if collected else {}
        for entity_id in collected:
            entity = entities[entity_id]
            self._emit("coin_collected", {
                "entity_id": entity_id, "x": entity.x, "y": entity.y,
                "coins_collected": self.state.metadata["coins_collected"],
                "player": player,
            })
        for name, entity_id in interactions:
            self._emit("interaction", {"name": name, "entity_id": entity_id, "player": player})

    def on_step_end(self, info, done):
        if self.trace_actions:
            self._emit("step", {
                "state_sha256": hashlib.sha256(
                    json.dumps(asdict(self.state), sort_keys=True, allow_nan=False).encode()
                ).hexdigest(),
                "episode_return": info["episode_return"],
            })
        if done:
            self.end_attempt(info["outcome"])

    def set_paused(self, paused):
        if self.closed or paused == self.paused:
            return
        now = self.clock()
        if paused:
            self.pause_started = now if self.attempt_id else None
        elif self.pause_started is not None:
            self.paused_seconds += now - self.pause_started
            self.pause_started = None
        self.paused = paused
        self._emit("pause" if paused else "resume", {})

    def set_speed(self, speed):
        if not 0.1 <= speed <= 8:
            raise ValueError("invalid playback speed")
        self._emit("speed_changed", {"speed": speed})

    def end_attempt(self, reason):
        if reason not in END_REASONS:
            raise ValueError("invalid attempt end reason")
        if self.attempt_id is None:
            return
        now = self.clock()
        paused = self.paused_seconds + (
            now - self.pause_started if self.pause_started is not None else 0.0
        )
        summary = {
            "reason": reason, "ticks": self.state.tick, "decisions": self.decision,
            "simulation_seconds": self.state.tick / self.metadata["tick_hz"],
            "active_wall_seconds": max(0.0, now - self.attempt_started - paused),
            "paused_wall_seconds": paused, "max_progress": self.max_progress,
            "player": asdict(self.state.player),
            "coins_collected": self.state.metadata["coins_collected"],
            "coins_total": self.state.metadata["coins_total"],
            "jump_held_ticks_at_end": self.held_ticks if self.jump_held else 0,
            "motion": self.motion.snapshot(),
        }
        self._emit("attempt_end", summary)
        self.summaries.append({"attempt_id": self.attempt_id, **self.context, **summary})
        self.attempt_id = self.context = None
        self.pause_started = None
        self._manifest("recording")

    def close(self, reason="quit"):
        if self.closed:
            return
        if self.failed:
            # A callback may have failed midway through reset; do not synthesize a
            # terminal summary from partially initialized or stale attempt state.
            self.attempt_id = self.context = None
        else:
            self.end_attempt(reason)
        self._emit("session_end", {"attempts": len(self.summaries)})
        try:
            from .storage import atomic_json

            atomic_json(self.writer.directory / "attempts.json", {
                "schema_version": 1, "session_id": self.session_id,
                "complete": not self.failed, "attempts": self.summaries,
            })
        except OSError as error:
            self.disable(error)
        self._manifest("closed")
        try:
            self.writer.close()
        except OSError as error:
            self.disable(error)
        self.closed = True
