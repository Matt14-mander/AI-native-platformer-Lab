"""Append-only local events, atomic manifests and strict offline validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .schema import validate_event


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temp.replace(path)


class SessionWriter:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.stream = (self.directory / "events.jsonl").open("x", encoding="utf-8", newline="\n")
        self.count = 0
        self.digest = hashlib.sha256()

    def append(self, event):
        validate_event(event)
        line = json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n"
        self.stream.write(line)
        self.digest.update(line.encode("utf-8"))
        self.count += 1
        if self.count % 64 == 0 or event["type"] in {"attempt_end", "session_end"}:
            self.stream.flush()

    def manifest(self, value):
        self.stream.flush()
        atomic_json(self.directory / "manifest.json", {
            **value, "event_count": self.count, "events_sha256": self.digest.hexdigest(),
        })

    def close(self):
        self.stream.close()


def read_session(directory):
    """Validate identity, lifecycle, counters and integrity; never invent lost outcomes.

    A truncated final JSON line is tolerated only for an unclosed session. Other
    malformed records, changed closed files and foreign-session events are errors.
    """
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if type(manifest.get("schema_version")) is not int or (
        manifest.get("schema_version") != 1
    ) or manifest.get("status") not in {
        "recording", "closed", "incomplete"
    }:
        raise ValueError("invalid telemetry manifest")
    raw = (directory / "events.jsonl").read_bytes()
    closed = manifest["status"] == "closed"
    if closed and hashlib.sha256(raw).hexdigest() != manifest.get("events_sha256"):
        raise ValueError("telemetry event digest mismatch")
    events, attempts, active = [], [], None
    seen = set()
    previous_wall = 0.0
    paused = False
    truncated_tail = False
    lines = raw.splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            event = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if not closed and index == len(lines) - 1 and not line.endswith(b"\n"):
                truncated_tail = True
                break
            raise ValueError(f"invalid event JSON at line {index + 1}") from None
        validate_event(event)
        if event["event_seq"] != len(events):
            raise ValueError("noncontiguous event sequence")
        for key in ("session_id", "actor_type", "player_id"):
            if event[key] != manifest[key]:
                raise ValueError(f"foreign event {key}")
        if event["wall_seconds"] < previous_wall:
            raise ValueError("wall clock moved backwards")
        previous_wall = event["wall_seconds"]
        kind = event["type"]
        if not events and kind != "session_start":
            raise ValueError("missing session_start")
        if kind == "session_start" and event["payload"] != {
            k: v for k, v in manifest.items()
            if k not in {"status", "event_count", "events_sha256"}
        }:
            raise ValueError("session metadata mismatch")
        if events and (kind == "session_start" or events[-1]["type"] == "session_end"):
            raise ValueError("invalid session lifecycle")
        if kind == "session_end" and event["payload"]["attempts"] != len(attempts):
            raise ValueError("session attempt count mismatch")
        if kind in {"pause", "resume"}:
            if (kind == "pause") == paused:
                raise ValueError("invalid pause lifecycle")
            paused = kind == "pause"
        if kind == "attempt_start":
            if active is not None or event["attempt_id"] in seen or event["tick"] != 0:
                raise ValueError("invalid attempt_start")
            seen.add(event["attempt_id"])
            if event["payload"]["initially_paused"] != paused:
                raise ValueError("attempt pause state mismatch")
            active = {
                "attempt_id": event["attempt_id"], "start": event, "last": event, "coins": set(),
                "pending_action": None, "completed_decisions": 0,
            }
        elif event["attempt_id"] is not None:
            if active is None or event["attempt_id"] != active["attempt_id"]:
                raise ValueError("event outside active attempt")
            start, last = active["start"], active["last"]
            if any(event[k] != start[k] for k in (
                "level_id", "level_sha256", "protocol_sha256", "seed"
            )):
                raise ValueError("attempt identity changed")
            if event["tick"] < last["tick"] or event["decision"] < last["decision"]:
                raise ValueError("attempt counters moved backwards")
            active["last"] = event
            if kind in {"action", "step"} and not manifest.get("trace_actions"):
                raise ValueError("unexpected action trace")
            if kind == "action":
                if active["pending_action"] is not None or (
                    event["decision"] != active["completed_decisions"] + 1
                ):
                    raise ValueError("invalid action trace order")
                active["pending_action"] = event["decision"]
            elif kind == "step":
                if active["pending_action"] != event["decision"]:
                    raise ValueError("step without matching action")
                active["completed_decisions"] = event["decision"]
                active["pending_action"] = None
            if kind == "coin_collected":
                entity_id = event["payload"]["entity_id"]
                if entity_id in active["coins"]:
                    raise ValueError("duplicate coin collection")
                active["coins"].add(entity_id)
            if kind == "attempt_end":
                if manifest.get("trace_actions") and (
                    active["pending_action"] is not None
                    or active["completed_decisions"] != event["decision"]
                ):
                    raise ValueError("incomplete action trace")
                if event["payload"]["coins_collected"] != len(active["coins"]):
                    raise ValueError("collection ledger mismatch")
                if event["payload"]["coins_total"] != start["payload"]["coins_total"]:
                    raise ValueError("collection denominator changed")
                attempts.append({
                    **{k: start[k] for k in (
                        "attempt_id", "level_id", "level_sha256", "protocol_sha256", "seed"
                    )},
                    **event["payload"],
                })
                active = None
        elif active is not None:
            raise ValueError("missing active attempt context")
        events.append(event)
    complete = bool(
        closed and events and events[-1]["type"] == "session_end" and active is None
        and len(events) == manifest.get("event_count")
    )
    if closed and not complete:
        raise ValueError("closed telemetry has incomplete lifecycle")
    if active:
        attempts.append({
            **{k: active["start"][k] for k in (
                "attempt_id", "level_id", "level_sha256", "protocol_sha256", "seed"
            )},
            "reason": "incomplete",
        })
    return {
        "schema_version": 1, "manifest": manifest, "complete": complete,
        "truncated_tail": truncated_tail, "events": events, "attempts": attempts,
        "eligible_for_profile": bool(
            complete and manifest["actor_type"] == "human"
            and manifest.get("playback_speed") == 1.0 and not manifest.get("headless")
            and any(a["reason"] in {"success", "death", "time_limit"} for a in attempts)
        ),
    }
