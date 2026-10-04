"""Bounded real-core route search, with independent environment replay witnesses."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from math import isfinite
from time import monotonic

from ai_platformer.core import Action
from ai_platformer.envs.factory import EnvironmentFactory, protocol_hash


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def transition_digest(env, observation, info, *, action=None, reward=0.0):
    return digest(
        {
            "state": asdict(env.core.state),
            "observation": observation.tolist(),
            "info": info,
            "action": action,
            "reward": reward,
        }
    )


def resolved_config(factory):
    return {
        "environment_id": factory.environment_id,
        "action_repeat": factory.action_repeat,
        "episode_step_limit": factory.step_limit,
        "sensor_range": factory.sensor_range,
        "physics": asdict(factory.settings.physics),
        "reward": asdict(factory.reward),
    }


def collect_route(factory, actions, *, seed, min_coin_ratio, deadline=None):
    env = factory.make()
    hashes = []
    try:
        obs, info = env.reset(seed=seed)
        initial = transition_digest(env, obs, info)
        for index, action in enumerate(actions):
            if deadline is not None and monotonic() >= deadline:
                raise ValueError("replay time budget exhausted")
            obs, reward, term, trunc, info = env.step(action)
            hashes.append(transition_digest(env, obs, info, action=action, reward=reward))
            if term or trunc:
                if index != len(actions) - 1:
                    raise ValueError("route contains actions after terminal transition")
                break
        if not actions or not env._episode_done:
            raise ValueError("route does not reach a terminal state")
        ratio = info["coins_collected"] / info["coins_total"] if info["coins_total"] else 1.0
        if info.get("outcome") != "success" or ratio < min_coin_ratio:
            raise ValueError("route does not satisfy success/collection target")
        return {
            "schema_version": 1,
            "kind": "level_route_v1",
            "level_id": factory.level_id,
            "seed": seed,
            "environment": resolved_config(factory),
            "protocol_sha256": protocol_hash(factory.protocol([factory.level_id])),
            "min_coin_ratio": min_coin_ratio,
            "actions": actions,
            "initial_sha256": initial,
            "step_sha256": hashes,
            "trace_sha256": digest(hashes),
            "outcome": info["outcome"],
            "steps": len(actions),
            "ticks": info["tick"],
            "coins_collected": info["coins_collected"],
            "coins_total": info["coins_total"],
        }
    finally:
        env.close()


def replay_route(path, route, *, max_seconds=30.0):
    """Reconstruct the recorded environment, rejecting changed maps/protocols/traces."""
    if (
        not isinstance(route, dict)
        or type(route.get("schema_version")) is not int
        or route.get("schema_version") != 1
        or route.get("kind") != "level_route_v1"
    ):
        raise ValueError("unsupported route witness")
    actions = route.get("actions")
    if (
        not isinstance(actions, list)
        or not 1 <= len(actions) <= 10000
        or any(type(a) is not int or a not in set(Action) for a in actions)
    ):
        raise ValueError("route must contain 1..10000 valid integer actions")
    seed = route.get("seed")
    target = route.get("min_coin_ratio")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("invalid route seed")
    if type(target) not in (float, int) or not isfinite(target) or not 0 <= target <= 1:
        raise ValueError("invalid route collection target")
    environment = route.get("environment")
    if not isinstance(environment, dict) or set(environment) != {
        "environment_id",
        "action_repeat",
        "episode_step_limit",
        "sensor_range",
        "physics",
        "reward",
    }:
        raise ValueError("route must contain resolved environment settings")
    if (
        type(environment.get("action_repeat")) is not int
        or not 1 <= environment["action_repeat"] <= 120
    ):
        raise ValueError("route action_repeat exceeds bounds")
    if len(actions) * environment["action_repeat"] > 100000:
        raise ValueError("route tick budget exceeds 100000")
    if (
        type(max_seconds) not in (int, float)
        or not isfinite(max_seconds)
        or not 0 < max_seconds <= 300
    ):
        raise ValueError("invalid replay time budget")
    if (
        type(environment.get("episode_step_limit")) is not int
        or not 1 <= environment["episode_step_limit"] <= 10000
    ):
        raise ValueError("route step limit exceeds bounds")
    factory = EnvironmentFactory({**environment, "level_spec": str(path)})
    if route.get("level_id") != factory.level_id or route.get("protocol_sha256") != protocol_hash(
        factory.protocol([factory.level_id])
    ):
        raise ValueError("route map/environment protocol mismatch")
    expected = route.get("step_sha256")
    if (
        not isinstance(expected, list)
        or len(expected) != len(actions)
        or route.get("trace_sha256") != digest(expected)
    ):
        raise ValueError("route trace integrity mismatch")
    actual = collect_route(
        factory, actions, seed=seed, min_coin_ratio=target, deadline=monotonic() + max_seconds
    )
    fields = (
        "initial_sha256",
        "step_sha256",
        "trace_sha256",
        "outcome",
        "steps",
        "ticks",
        "coins_collected",
        "coins_total",
    )
    mismatch = next((key for key in fields if route.get(key) != actual[key]), None)
    mismatch_step = next(
        (
            i
            for i, (expected_hash, actual_hash) in enumerate(
                zip(expected, actual["step_sha256"], strict=True)
            )
            if expected_hash != actual_hash
        ),
        None,
    )
    return {
        "schema_version": 1,
        "passed": mismatch is None,
        "mismatch_field": mismatch,
        "mismatch_step": mismatch_step,
        "replay": actual,
    }


def search_route(
    factory,
    *,
    seed=100,
    max_steps=768,
    max_expansions=25000,
    beam_width=24,
    max_seconds=15.0,
    min_coin_ratio=0.0,
    try_scripted=True,
):
    """Scripted candidate probes followed by a quantized beam search.

    State pruning is heuristic; failure is always unknown, never unreachable.
    Only static solids/coins are searched in this version.
    """
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be a nonnegative uint32")
    for name, value, limit in (
        ("max_steps", max_steps, 10000),
        ("max_expansions", max_expansions, 1000000),
        ("beam_width", beam_width, 256),
    ):
        if type(value) is not int or not 1 <= value <= limit:
            raise ValueError(f"invalid {name}")
    if (
        type(max_seconds) not in (int, float)
        or not isfinite(max_seconds)
        or not 0 < max_seconds <= 300
    ):
        raise ValueError("invalid time budget")
    if (
        type(min_coin_ratio) not in (int, float)
        or not isfinite(min_coin_ratio)
        or not 0 <= min_coin_ratio <= 1
    ):
        raise ValueError("invalid coin ratio target")
    if not 1 <= factory.action_repeat <= 120:
        raise ValueError("action_repeat exceeds search bounds")
    if type(factory.step_limit) is not int or not 1 <= factory.step_limit <= 10000:
        raise ValueError("search requires a finite step limit of 1..10000")
    # No implicit change to recorded step limits when reusing a policy factory.
    max_steps = min(max_steps, factory.step_limit or max_steps)
    if max_steps * factory.action_repeat > 100000:
        raise ValueError("search route tick budget exceeds 100000")
    report = {
        "schema_version": 1,
        "status": "unknown",
        "reason": None,
        "witness": None,
        "level_id": factory.level_id,
        "algorithm": "scripted_probes_then_beam_v1",
        "budgets": {
            "max_steps": max_steps,
            "max_expansions": max_expansions,
            "beam_width": beam_width,
            "max_seconds": max_seconds,
        },
        "min_coin_ratio": min_coin_ratio,
        "expansions": 0,
        "scope": "Static solids/coins; heuristic pruning cannot prove unreachability.",
    }
    level = factory.repository.load(factory.level_id)
    if level.blocks or level.enemies or level.powerups:
        report["reason"] = "interactive_content_unsupported"
        return report
    start = monotonic()

    def budget():
        return report["expansions"] >= max_expansions or monotonic() - start >= max_seconds

    def finish(actions, method):
        witness = collect_route(
            factory, actions, seed=seed, min_coin_ratio=min_coin_ratio, deadline=monotonic() + 30
        )
        proof = replay_route(factory.repository.path, witness)
        if not proof["passed"]:
            raise RuntimeError("independent route replay failed")
        report.update(
            status="verified",
            reason="successful_route_replayed",
            witness=witness,
            method=method,
            replay_passed=True,
            elapsed_seconds=monotonic() - start,
        )
        return report

    if try_scripted:
        from ai_platformer.agents.scripted import MoveRightAgent, RuleJumpAgent
        from ai_platformer.agents.scripted.coin_jump import CoinJumpAgent

        for agent_type in (MoveRightAgent, RuleJumpAgent, CoinJumpAgent):
            env = factory.make()
            actions = []
            try:
                obs, _ = env.reset(seed=seed)
                agent = agent_type()
                agent.reset(seed=seed)
                for _ in range(max_steps):
                    if budget():
                        break
                    action = int(agent.act(obs))
                    obs, _, term, trunc, info = env.step(action)
                    actions.append(action)
                    report["expansions"] += 1
                    if term or trunc:
                        ratio = (
                            info["coins_collected"] / info["coins_total"]
                            if info["coins_total"]
                            else 1.0
                        )
                        if info.get("outcome") == "success" and ratio >= min_coin_ratio:
                            return finish(actions, agent_type.__name__)
                        break
            finally:
                env.close()
            if budget():
                break
    env = factory.make()
    try:
        env.reset(seed=seed)
        beam = [(env.core, [])]
        for _depth in range(max_steps):
            if budget() or not beam:
                break
            candidates = {}
            for core, path in beam:
                for action in Action:
                    if budget():
                        break
                    child = copy.deepcopy(
                        core,
                        {
                            id(core._level): core._level,
                            id(core._level_loader): core._level_loader,
                            id(core.config): core.config,
                            id(core._state): core._state,
                        },
                    )
                    result = None
                    for _ in range(factory.action_repeat):
                        result = child.step(action)
                        if result.terminated or result.truncated:
                            break
                    report["expansions"] += 1
                    actions = [*path, int(action)]
                    state = child.state
                    count = state.metadata["coins_collected"]
                    total = state.metadata["coins_total"]
                    ratio = count / total if total else 1.0
                    if result.terminated or result.truncated:
                        if result.info.get("outcome") == "success" and ratio >= min_coin_ratio:
                            return finish(actions, "beam_search")
                        continue
                    p = state.player
                    # Includes latch and collected IDs; equal positions can require different actions.
                    key = (
                        round(p.x / 8),
                        round(p.y / 8),
                        round(p.velocity_x / 2),
                        round(p.velocity_y / 2),
                        p.grounded,
                        child._jump_was_pressed,
                        tuple(sorted(child._collected_ids)),
                    )
                    score = p.x + (count * 80 if min_coin_ratio else 0)
                    previous = candidates.get(key)
                    if previous is None or score > previous[0]:
                        candidates[key] = (score, child, actions)
                if budget():
                    break
            beam = [
                (core, path)
                for _, core, path in sorted(candidates.values(), key=lambda n: n[0], reverse=True)[
                    :beam_width
                ]
            ]
        report.update(
            reason="budget_exhausted" if budget() else "no_witness_in_search_space",
            elapsed_seconds=monotonic() - start,
        )
        return report
    finally:
        env.close()
