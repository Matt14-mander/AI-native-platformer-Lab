"""Versioned offline facts reconstructed from validated telemetry sessions."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from math import floor, isclose
from statistics import mean, median

METRICS_VERSION = 1
NATURAL_ENDS = {"success", "death", "time_limit"}
DEATH_BIN_WIDTH_PX = 100


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def average(values):
    return mean(values) if values else None


def identity(manifest):
    return {
        "scope": "player" if manifest["player_id"] is not None else "session_only",
        "id": manifest["player_id"] if manifest["player_id"] is not None else manifest["session_id"],
    }


def summarize_attempts(rows):
    """Pool only within the caller's declared cohort; interrupted runs are not deaths."""
    valid = [row for row in rows if row["valid_terminal"]]
    successes = [row for row in valid if row["reason"] == "success"]
    coin_rows = [row for row in valid if row["coins_total"] > 0]
    motion_rows = [row for row in valid if row["motion"] is not None]
    motion_ticks = sum(row["ticks"] for row in motion_rows)
    holds = [ticks for row in valid for ticks in row["jump_hold_completed_ticks"]]
    interactions = Counter()
    for row in valid:
        interactions.update(row["interactions"])
    death_bins = Counter(
        floor(row["death_location"]["x"] / DEATH_BIN_WIDTH_PX)
        for row in valid if row["death_location"] is not None
    )
    return {
        "attempt_count": len(rows), "valid_terminal_attempts": len(valid),
        "excluded_attempts": len(rows) - len(valid),
        "observed_end_reasons": dict(sorted(Counter(r["reason"] for r in rows).items())),
        "successes": len(successes),
        "deaths": sum(r["reason"] == "death" for r in valid),
        "timeouts": sum(r["reason"] == "time_limit" for r in valid),
        "success_rate": len(successes) / len(valid) if valid else None,
        "mean_success_simulation_seconds": average([r["simulation_seconds"] for r in successes]),
        "mean_max_progress": average([r["max_progress"] for r in valid]),
        "mean_net_horizontal_speed_px_s": average([
            r["net_horizontal_speed_px_s"] for r in valid
            if r["net_horizontal_speed_px_s"] is not None
        ]),
        "collection": {
            "nonempty_map_attempts": len(coin_rows),
            "coins_collected": sum(r["coins_collected"] for r in coin_rows),
            "coins_total": sum(r["coins_total"] for r in coin_rows),
            "weighted_ratio": (
                sum(r["coins_collected"] for r in coin_rows)
                / sum(r["coins_total"] for r in coin_rows) if coin_rows else None
            ),
        },
        "jump_requests": sum(r["jump_requests"] for r in valid),
        "accepted_jumps": sum(r["accepted_jumps"] for r in valid),
        "landings": sum(r["landings"] for r in valid),
        "jump_hold_completed_ticks": {
            "count": len(holds), "mean": average(holds),
            "median": median(holds) if holds else None,
            "minimum": min(holds) if holds else None, "maximum": max(holds) if holds else None,
        },
        "censored_jump_holds": sum(r["jump_hold_censored_ticks"] is not None for r in valid),
        "motion": {
            "available_attempts": len(motion_rows), "missing_attempts": len(valid)-len(motion_rows),
            "status": "observed" if motion_rows else "unknown",
            "reason": None if motion_rows else "motion_summary_not_recorded",
            "airborne_ratio": (
                sum(r["motion"]["airborne_ticks"] for r in motion_rows) / motion_ticks
                if motion_ticks else None
            ),
            "backward_distance_px": (
                sum(r["motion"]["backward_distance_px"] for r in motion_rows)
                if motion_rows else None
            ),
            "stagnation_seconds": (
                sum(r["motion"]["stagnation_ticks"] / r["tick_hz"] for r in motion_rows)
                if motion_rows else None
            ),
            "stationary_seconds": (
                sum(r["motion"]["stationary_ticks"] / r["tick_hz"] for r in motion_rows)
                if motion_rows else None
            ),
        },
        "retry_starts_after_failure": sum(r["retry_after_failure"] for r in rows),
        "retry_scope": "immediately adjacent attempts in the same complete session/map/protocol",
        "death_bins": [{"start_x": key * DEATH_BIN_WIDTH_PX, "count": count}
                       for key, count in sorted(death_bins.items())],
        "interactions": dict(sorted(interactions.items())),
        "combat_skill": {"status": "unknown", "reason": "no_encounter_opportunity_denominator"},
    }


def build_metrics(sessions):
    rows, cohorts = [], defaultdict(list)
    seen_sessions, seen_attempts = set(), set()
    for input_index, session in enumerate(sessions):
        manifest = session["manifest"]
        session_id = manifest["session_id"]
        if session_id in seen_sessions:
            raise ValueError("duplicate session_id; copies cannot count as new evidence")
        seen_sessions.add(session_id)
        hz = manifest["tick_hz"]
        if type(hz) not in (int, float) or hz <= 0:
            raise ValueError("invalid tick frequency")
        events_by_attempt = defaultdict(list)
        for event in session["events"]:
            if event["attempt_id"] is not None:
                events_by_attempt[event["attempt_id"]].append(event)
        previous = None
        for attempt_index, attempt in enumerate(session["attempts"]):
            attempt_id = attempt["attempt_id"]
            if attempt_id in seen_attempts:
                raise ValueError("duplicate attempt_id")
            seen_attempts.add(attempt_id)
            events = events_by_attempt[attempt_id]
            start = events[0]
            if start["type"] != "attempt_start":
                raise ValueError("missing attempt_start")
            kinds = Counter(e["type"] for e in events)
            seconds = attempt.get("simulation_seconds")
            if seconds is not None and not isclose(seconds, attempt["ticks"] / hz, abs_tol=1e-9):
                raise ValueError("simulation duration disagrees with tick frequency")
            motion = attempt.get("motion")
            if motion is not None and motion["stagnation_min_ticks"] != max(1, round(2 * hz)):
                raise ValueError("motion threshold disagrees with tick frequency")
            valid = session["complete"] and attempt["reason"] in NATURAL_ENDS
            exclusions = []
            if not session["complete"]:
                exclusions.append("incomplete_session")
            if attempt["reason"] not in NATURAL_ENDS:
                exclusions.append("nonterminal_attempt")
            if manifest["actor_type"] != "human":
                exclusions.append("nonhuman_actor")
            if manifest.get("headless"):
                exclusions.append("headless_session")
            if manifest.get("playback_speed") != 1.0 or any(
                e["type"] == "speed_changed" and e["payload"]["speed"] != 1.0
                for e in session["events"]
            ):
                exclusions.append("nonstandard_playback_speed")
            retry = bool(
                session["complete"] and previous and previous["reason"] in {"death", "time_limit"}
                and all(previous[k] == attempt[k] for k in ("level_sha256", "protocol_sha256"))
            )
            player = attempt.get("player")
            row = {
                **{k: attempt[k] for k in (
                    "attempt_id", "level_id", "level_sha256", "protocol_sha256", "seed", "reason"
                )},
                "session_id": session_id, "actor_type": manifest["actor_type"],
                "playback_mode": {"speed": manifest.get("playback_speed"),
                                  "headless": manifest.get("headless"),
                                  "speed_changes": [e["payload"]["speed"] for e in session["events"]
                                                    if e["type"] == "speed_changed"]},
                "identity": identity(manifest),
                "controller_source_sha256": (digest(manifest.get("source", {}))
                                             if manifest["actor_type"] != "human" else None),
                "environment_sha256": digest({"environment": manifest["environment"], "tick_hz": hz}),
                "order": [input_index, attempt_index], "tick_hz": hz,
                "valid_terminal": valid, "profile_eligible": not exclusions,
                "profile_exclusion_reasons": exclusions,
                "ticks": attempt.get("ticks"), "simulation_seconds": seconds,
                "active_wall_seconds": attempt.get("active_wall_seconds"),
                "paused_wall_seconds": attempt.get("paused_wall_seconds"),
                "max_progress": attempt.get("max_progress"),
                "net_horizontal_speed_px_s": (
                    (player["x"] - start["payload"]["player"]["x"]) / seconds
                    if player is not None and seconds else None
                ),
                "coins_collected": attempt.get("coins_collected"),
                "coins_total": attempt.get("coins_total"),
                "collection_ratio": (
                    attempt["coins_collected"] / attempt["coins_total"]
                    if attempt.get("coins_total") else None
                ),
                "jump_requests": kinds["jump_requested"], "accepted_jumps": kinds["jump_started"],
                "landings": kinds["landed"],
                "jump_hold_completed_ticks": [e["payload"]["held_ticks"] for e in events
                                              if e["type"] == "jump_released"],
                "jump_hold_censored_ticks": attempt.get("jump_held_ticks_at_end") or None,
                "motion": motion, "retry_after_failure": retry,
                "death_location": ({"x": player["x"], "y": player["y"], "cause": "unknown"}
                                   if player and attempt["reason"] == "death" else None),
                "interactions": dict(sorted(Counter(e["payload"]["name"] for e in events
                                                    if e["type"] == "interaction").items())),
            }
            rows.append(row)
            cohorts[digest({k: row[k] for k in (
                "identity", "actor_type", "level_sha256", "protocol_sha256", "playback_mode",
                "controller_source_sha256",
            )})].append(row)
            previous = attempt
    return {
        "metrics_version": METRICS_VERSION,
        "definitions": {
            "natural_ends": sorted(NATURAL_ENDS), "death_bin_width_px": DEATH_BIN_WIDTH_PX,
            "ordering": "explicit input session order, then attempt_start order; no cross-session retry inference",
            "collection_denominator": "all authoritative coins on the playable map; not inferred reachable opportunities",
            "motion": "v1 recorded per physical tick; legacy logs remain unknown",
        },
        "attempts": rows,
        "cohorts": [{"cohort_id": key,
                     **{k: group[0][k] for k in (
                         "identity", "actor_type", "level_sha256", "protocol_sha256", "playback_mode",
                         "controller_source_sha256",
                     )},
                     "level_ids": sorted({r["level_id"] for r in group}),
                     "summary": summarize_attempts(group)}
                    for key, group in sorted(cohorts.items())],
    }
