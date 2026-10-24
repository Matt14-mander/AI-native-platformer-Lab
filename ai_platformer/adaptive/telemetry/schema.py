"""Telemetry v1 records facts without extending the simulation protocol."""

from __future__ import annotations

import json
import math
import re

SCHEMA_VERSION = 1
ACTORS = {"human", "scripted", "ppo", "replay"}
END_REASONS = {
    "success", "death", "time_limit", "restart", "quit", "level_switch", "replay", "error"
}
EVENT_TYPES = {
    "session_start", "session_end", "attempt_start", "attempt_end", "pause", "resume",
    "action", "step", "jump_requested", "jump_started", "jump_released", "landed",
    "coin_collected", "interaction",
}
FIELDS = {
    "schema_version", "session_id", "attempt_id", "event_seq", "player_id", "actor_type",
    "level_id", "level_sha256", "protocol_sha256", "seed", "tick", "decision",
    "wall_seconds", "type", "payload",
}


def _integer(value, *, minimum=0):
    return type(value) is int and value >= minimum


def _player(value):
    if not isinstance(value, dict) or set(value) != {
        "x", "y", "velocity_x", "velocity_y", "grounded", "alive", "facing"
    }:
        return False
    return (
        all(type(value[k]) in (int, float) and math.isfinite(value[k]) for k in (
            "x", "y", "velocity_x", "velocity_y"
        ))
        and type(value["grounded"]) is bool and type(value["alive"]) is bool
        and type(value["facing"]) is int and value["facing"] in (-1, 1)
    )


def validate_event(event):
    """Reject unknown versions, identity ambiguity and malformed event payloads."""
    if not isinstance(event, dict) or set(event) != FIELDS:
        raise ValueError("invalid telemetry fields")
    if type(event["schema_version"]) is not int or event["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported telemetry schema_version")
    if not isinstance(event["session_id"], str) or not event["session_id"]:
        raise ValueError("missing session_id")
    if event["player_id"] is not None and (
        not isinstance(event["player_id"], str) or not event["player_id"].strip()
    ):
        raise ValueError("invalid player_id")
    if event["actor_type"] not in ACTORS or event["type"] not in EVENT_TYPES:
        raise ValueError("invalid actor/event type")
    if not _integer(event["event_seq"]):
        raise ValueError("invalid event sequence")
    seconds = event["wall_seconds"]
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("invalid wall time")
    context = ("level_id", "level_sha256", "protocol_sha256", "seed", "tick", "decision")
    if event["attempt_id"] is None:
        if event["type"] not in {"session_start", "session_end", "pause", "resume"}:
            raise ValueError("attempt event without attempt_id")
        if any(event[key] is not None for key in context):
            raise ValueError("session event has attempt context")
    else:
        if not isinstance(event["attempt_id"], str) or not event["attempt_id"]:
            raise ValueError("invalid attempt_id")
        if event["type"].startswith("session_"):
            raise ValueError("session event inside attempt")
        if not isinstance(event["level_id"], str) or not event["level_id"]:
            raise ValueError("missing level_id")
        for key in ("level_sha256", "protocol_sha256"):
            if not isinstance(event[key], str) or not re.fullmatch(r"[0-9a-f]{64}", event[key]):
                raise ValueError(f"invalid {key}")
        if any(not _integer(event[key]) for key in ("seed", "tick", "decision")):
            raise ValueError("invalid attempt counters")
    payload = event["payload"]
    if not isinstance(payload, dict):
        raise TypeError("invalid payload")
    if event["type"] in {
        "attempt_start", "attempt_end", "jump_requested", "jump_started", "jump_released",
        "landed", "coin_collected", "interaction",
    } and not _player(payload.get("player")):
        raise ValueError("invalid player snapshot")
    if event["type"] == "jump_started" and not _player(payload.get("takeoff")):
        raise ValueError("invalid jump takeoff")
    if event["type"] == "jump_requested" and type(payload.get("grounded_before")) is not bool:
        raise ValueError("invalid jump request")
    if event["type"] == "session_end" and not _integer(payload.get("attempts")):
        raise ValueError("invalid session end payload")
    if event["type"] == "attempt_end":
        if payload.get("reason") not in END_REASONS:
            raise ValueError("invalid attempt end reason")
        if any(not _integer(payload.get(k)) for k in ("coins_collected", "coins_total")):
            raise ValueError("invalid collection counters")
        if payload["coins_collected"] > payload["coins_total"]:
            raise ValueError("collection exceeds total")
        if payload.get("ticks") != event["tick"] or payload.get("decisions") != event["decision"]:
            raise ValueError("summary counters disagree with event")
        for key in ("simulation_seconds", "active_wall_seconds", "paused_wall_seconds"):
            value = payload.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {key}")
        if type(payload.get("max_progress")) not in (int, float) or not (
            0 <= payload["max_progress"] <= 1
        ):
            raise ValueError("invalid max_progress")
    if event["type"] == "attempt_start" and (
        not _integer(payload.get("coins_total"))
        or type(payload.get("initially_paused")) is not bool
    ):
        raise ValueError("invalid attempt_start payload")
    if event["type"] == "action" and (
        not _integer(payload.get("action")) or payload["action"] > 9
    ):
        raise ValueError("invalid action")
    if event["type"] in {"coin_collected", "interaction"} and (
        not isinstance(payload.get("entity_id"), str) or not payload["entity_id"]
    ):
        raise ValueError("missing entity_id")
    if event["type"] == "interaction" and (
        not isinstance(payload.get("name"), str) or not payload["name"]
    ):
        raise ValueError("missing interaction name")
    if event["type"] == "jump_released" and not _integer(payload.get("held_ticks"), minimum=1):
        raise ValueError("invalid jump hold duration")
    if event["type"] == "step" and (
        not isinstance(payload.get("state_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", payload["state_sha256"])
        or type(payload.get("episode_return")) not in (int, float)
    ):
        raise ValueError("invalid action trace step")
    # Also disallow nested NaN/Infinity and non-serializable extension values.
    json.dumps(event, allow_nan=False)
    return event
