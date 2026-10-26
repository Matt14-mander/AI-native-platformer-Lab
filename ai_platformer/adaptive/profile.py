"""PlayerProfile v1: explainable evidence windows, without personality inference."""

from collections import defaultdict

from .metrics import digest, summarize_attempts

PROFILE_VERSION = 1
DEFAULT_WINDOW = 10
MIN_ATTEMPTS = 5
MIN_MAPS = 2


def build_profiles(metrics, *, window=DEFAULT_WINDOW):
    if type(window) is not int or not 5 <= window <= 100:
        raise ValueError("profile window must be an integer in 5..100")
    contexts = defaultdict(list)
    # Nonhuman and nonstandard playback cannot introduce identities or environments.
    for row in metrics["attempts"]:
        if row["actor_type"] == "human" and not any(reason in row["profile_exclusion_reasons"]
                for reason in ("headless_session", "nonstandard_playback_speed")):
            key = digest({"identity": row["identity"], "environment": row["environment_sha256"]})
            contexts[key].append(row)
    profiles = []
    for key, all_rows in sorted(contexts.items()):
        valid = [r for r in all_rows if r["profile_eligible"]]
        recent = valid[-window:]
        map_count = len({r["level_sha256"] for r in recent})
        ready = len(recent) >= MIN_ATTEMPTS and map_count >= MIN_MAPS
        maps = defaultdict(list)
        for row in recent:
            maps[(row["level_sha256"], row["protocol_sha256"])].append(row)
        evidence = [{"attempt_id": r["attempt_id"], "session_id": r["session_id"]} for r in recent]
        profiles.append({
            "schema_version": 1, "profile_version": PROFILE_VERSION,
            "metrics_version": metrics["metrics_version"], "profile_id": key,
            "identity": all_rows[0]["identity"],
            "environment_sha256": all_rows[0]["environment_sha256"],
            "status": "ready" if ready else "insufficient_data",
            "readiness": {
                "valid_attempts": len(recent), "independent_map_count": map_count,
                "minimum_attempts": MIN_ATTEMPTS, "minimum_maps": MIN_MAPS,
                "meaning": "engineering sample gate, not statistical confidence or a skill rank",
                "unmet": (["need_more_attempts"] if len(recent) < MIN_ATTEMPTS else [])
                         + (["need_more_maps"] if map_count < MIN_MAPS else []),
            },
            "window": {"maximum_attempts": window, "order_basis": "explicit_input_order"},
            "evidence": evidence,
            "lifetime": {
                "scope": "all supplied sessions in this identity/environment",
                "valid_attempts": len(valid),
                "excluded_attempts": [{"attempt_id": r["attempt_id"],
                                       "reasons": r["profile_exclusion_reasons"]}
                                      for r in all_rows if not r["profile_eligible"]],
                "retry_starts_after_failure": sum(r["retry_after_failure"] for r in all_rows),
            },
            "by_map": [{
                "level_sha256": map_key[0], "protocol_sha256": map_key[1],
                "level_ids": sorted({r["level_id"] for r in group}),
                "attempt_ids": [r["attempt_id"] for r in group],
                "observed": summarize_attempts(group),
            } for map_key, group in sorted(maps.items())],
            "dimensions": {
                "completion_stability": {"status": "observed" if recent else "unknown",
                                         "basis": "by_map.observed.success_rate"},
                "progress_efficiency": {"status": "observed" if recent else "unknown",
                                        "basis": "per-map simulation time and net horizontal speed"},
                "collection_performance": {
                    "status": "observed" if any(r["coins_total"] for r in recent) else "unknown",
                    "basis": "per-map weighted collection ratio, excluding empty denominators",
                },
                "motion_behavior": {
                    "status": "observed" if any(r["motion"] is not None for r in recent) else "unknown",
                    "available_attempts": sum(r["motion"] is not None for r in recent),
                    "missing_attempts": sum(r["motion"] is None for r in recent),
                    "basis": "per-map measured airborne, backtracking and horizontal stagnation",
                },
                "jump_behavior": {
                    "status": "observed" if any(r["jump_requests"] for r in recent) else "unknown",
                    "basis": "per-map requests, accepted jumps and completed/censored hold lengths",
                },
                "retry_behavior": {
                    "status": "observed" if any(r["retry_after_failure"] for r in all_rows) else "unknown",
                    "basis": "lifetime.retry_starts_after_failure; no cross-session inference",
                },
                "collection_preference": {"status": "unknown", "reason": "performance_does_not_identify_preference"},
                "jump_error_rate": {"status": "unknown", "reason": "no_validated_error_opportunity_definition"},
                "combat_skill": {"status": "unknown", "reason": "no_encounter_opportunity_denominator"},
            },
        })
    return profiles
