"""Automated RL-readiness gates and reward exploit audits."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from math import isclose
from typing import Any

import numpy as np

from ai_platformer.agents.scripted import MoveRightAgent, RuleJumpAgent
from ai_platformer.benchmark.scripted import evaluate_scripted_agent
from ai_platformer.core import Action
from ai_platformer.envs import PlatformerStateEnv
from ai_platformer.envs.factory import EnvironmentFactory


@dataclass(frozen=True, slots=True)
class StabilityReport:
    episodes: int
    transitions: int
    max_steps_per_episode: int
    base_seed: int
    outcomes: dict[str, int]
    min_reward: float
    max_reward: float


@dataclass(frozen=True, slots=True)
class RewardAuditReport:
    passed: bool
    noop_return: float
    jump_in_place_return: float
    loop_return: float
    loop_progress_reward: float
    expected_loop_progress_reward: float
    duplicate_collectibles: tuple[str, ...]
    death_return: float
    success_return: float
    checks: dict[str, bool]


def run_sb3_checker(*, environment: dict | None = None, level_ids: list[str] | None = None) -> str:
    from stable_baselines3 import __version__ as sb3_version
    from stable_baselines3.common.env_checker import check_env

    factory = EnvironmentFactory(environment or {"episode_step_limit": 256})
    for level in level_ids or [factory.level_id]:
        env = factory.make(level_id=level)
        try:
            check_env(env, warn=True)
        finally:
            env.close()
    return sb3_version


def run_random_stability_gate(
    *,
    episodes: int = 1_000,
    max_steps_per_episode: int = 256,
    base_seed: int = 20_260_923,
    environment: dict | None = None,
    level_ids: list[str] | None = None,
) -> StabilityReport:
    if episodes <= 0 or max_steps_per_episode <= 0:
        raise ValueError("episodes and max_steps_per_episode must be positive")
    protocol = dict(environment or {})
    protocol["episode_step_limit"] = max_steps_per_episode
    factory = EnvironmentFactory(protocol)
    levels = level_ids or [factory.level_id]
    env = factory.make(level_id=levels[0])
    outcomes: Counter[str] = Counter()
    transitions = 0
    min_reward = float("inf")
    max_reward = float("-inf")
    try:
        for episode in range(episodes):
            seed = base_seed + episode
            rng = np.random.default_rng(seed)
            observation, _ = env.reset(
                seed=seed, options={"level_id": levels[episode % len(levels)]}
            )
            _assert_observation(env, observation)
            terminated = truncated = False
            info: dict[str, Any] = {}
            while not (terminated or truncated):
                action = int(rng.integers(env.action_space.n))
                observation, reward, terminated, truncated, info = env.step(action)
                transitions += 1
                _assert_observation(env, observation)
                if not np.isfinite(reward):
                    raise AssertionError("non-finite reward encountered")
                if not isclose(
                    reward,
                    sum(info["reward_components"].values()),
                    abs_tol=1e-9,
                ):
                    raise AssertionError("reward does not match component breakdown")
                min_reward = min(min_reward, reward)
                max_reward = max(max_reward, reward)
            outcomes[str(info.get("outcome", "unknown"))] += 1
    finally:
        env.close()
    return StabilityReport(
        episodes=episodes,
        transitions=transitions,
        max_steps_per_episode=max_steps_per_episode,
        base_seed=base_seed,
        outcomes=dict(sorted(outcomes.items())),
        min_reward=min_reward,
        max_reward=max_reward,
    )


def run_reward_exploit_audit(*, environment: dict | None = None) -> RewardAuditReport:
    noop = _rollout_fixed([Action.NOOP], 256, environment=environment)
    jump = _rollout_fixed([Action.JUMP, Action.NOOP], 256, environment=environment)
    loop_actions = [Action.RIGHT_RUN] * 12 + [Action.LEFT_RUN] * 24
    loop = _rollout_fixed(loop_actions, 360, environment=environment)
    expected_progress = (
        loop["final_progress"] * EnvironmentFactory(environment or {}).reward.progress_scale
    )

    death = _rollout_agent(RuleJumpAgent(trigger_distance=0.34), 1_000, environment=environment)
    success = _rollout_agent(RuleJumpAgent(), 1_000, environment=environment)
    duplicates = tuple(sorted(loop["duplicate_collectibles"]))
    checks = {
        "noop_cannot_profit": noop["return"] < 0.0,
        "jump_in_place_cannot_profit": jump["return"] < 0.0,
        "progress_is_path_independent": isclose(
            loop["components"]["progress"], expected_progress, abs_tol=1e-9
        ),
        "coins_are_one_shot": not duplicates,
        "death_is_penalized": death["outcome"] == "death"
        and death["terminal_components"]["death"] < 0.0,
        "success_is_rewarded": success["outcome"] == "success"
        and success["terminal_components"]["success"] > 0.0,
    }
    return RewardAuditReport(
        passed=all(checks.values()),
        noop_return=noop["return"],
        jump_in_place_return=jump["return"],
        loop_return=loop["return"],
        loop_progress_reward=loop["components"]["progress"],
        expected_loop_progress_reward=expected_progress,
        duplicate_collectibles=duplicates,
        death_return=death["return"],
        success_return=success["return"],
        checks=checks,
    )


def readiness_report(
    *,
    episodes: int = 1_000,
    max_steps_per_episode: int = 256,
    base_seed: int = 20_260_923,
    environment: dict | None = None,
    level_ids: list[str] | None = None,
) -> dict[str, Any]:
    gate_environment = dict(environment or {})
    gate_environment["episode_step_limit"] = max_steps_per_episode
    sb3_version = run_sb3_checker(environment=gate_environment, level_ids=level_ids)
    stability = run_random_stability_gate(
        episodes=episodes,
        max_steps_per_episode=max_steps_per_episode,
        base_seed=base_seed,
        environment=gate_environment,
        level_ids=level_ids,
    )
    reward = run_reward_exploit_audit(environment=environment)
    courses = {}
    if level_ids:
        courses = run_course_audit(level_ids, environment=gate_environment)
    factory = EnvironmentFactory(gate_environment)
    return {
        "schema_version": 1,
        "passed": reward.passed
        and stability.episodes == episodes
        and all(item["passed"] for item in courses.values()),
        "sb3_checker": {"passed": True, "version": sb3_version},
        "stability": asdict(stability),
        "reward_audit": asdict(reward),
        "course_audit": courses,
        "protocol": factory.protocol(level_ids or [factory.level_id]),
    }


def run_course_audit(level_ids: list[str], *, environment: dict | None = None) -> dict:
    """Validate reachability and potential rewards on every requested course.

    Gap courses must kill move-right; obstacles must stop it. Flat is a control.
    """
    protocol = dict(environment or {})
    protocol.setdefault("episode_step_limit", 1024)
    factory = EnvironmentFactory(protocol)
    report = {}
    for level_id in level_ids:
        spec = factory.repository.manifest["levels"].get(level_id)
        if spec is None:
            continue
        right = evaluate_scripted_agent(
            "move-right", MoveRightAgent, [100], environment=protocol, level_ids=[level_id]
        )
        rule = evaluate_scripted_agent(
            "rule-jump", RuleJumpAgent, [100], environment=protocol, level_ids=[level_id]
        )
        env = factory.make(level_id=level_id)
        try:
            env.reset(seed=100)
            initial_progress = env.core.state.progress
            parts = Counter()
            for action in [Action.RIGHT_RUN] * 12 + [Action.LEFT_RUN] * 24:
                _, _, terminated, truncated, info = env.step(int(action))
                parts.update(info["reward_components"])
                if terminated or truncated:
                    break
            conserved = isclose(
                parts["progress"],
                (env.core.state.progress - initial_progress) * factory.reward.progress_scale,
                abs_tol=1e-9,
            )
            env.reset(seed=100)
            _, noop_reward, _, _, _ = env.step(int(Action.NOOP))
        finally:
            env.close()
        flat = spec["task"] == "flat"
        checks = {
            "rule_reachable": rule.summary()["success_rate"] == 1.0,
            "move_right_calibrated": right.summary()["success_rate"] == float(flat),
            "potential_conserved": conserved,
            "noop_cannot_profit": noop_reward < 0,
        }
        report[level_id] = {
            "passed": all(checks.values()),
            "checks": checks,
            "rule_jump": rule.summary(),
            "move_right": right.summary(),
        }
    return report


def _rollout_fixed(
    actions: list[Action], steps: int, *, environment: dict | None = None
) -> dict[str, Any]:
    env = EnvironmentFactory({**(environment or {}), "episode_step_limit": steps}).make()
    observation, _ = env.reset(seed=123)
    return _rollout(
        env,
        lambda index, _: int(actions[index % len(actions)]),
        steps,
        observation,
    )


def _rollout_agent(
    agent: RuleJumpAgent, steps: int, *, environment: dict | None = None
) -> dict[str, Any]:
    env = EnvironmentFactory({**(environment or {}), "episode_step_limit": steps}).make()
    observation, _ = env.reset(seed=123)
    agent.reset(seed=123)
    return _rollout(env, lambda _, obs: agent.act(obs), steps, observation)


def _rollout(env, policy, steps: int, observation=None) -> dict[str, Any]:
    if observation is None:
        raise ValueError("an initial observation is required")
    total = 0.0
    components: Counter[str] = Counter()
    collected: set[str] = set()
    duplicates: set[str] = set()
    info: dict[str, Any] = {}
    terminal_components: dict[str, float] = {}
    try:
        for index in range(steps):
            observation, reward, terminated, truncated, info = env.step(policy(index, observation))
            total += reward
            components.update(info["reward_components"])
            for entity_id in info.get("collected", ()):
                if entity_id in collected:
                    duplicates.add(entity_id)
                collected.add(entity_id)
            if terminated or truncated:
                terminal_components = dict(info["reward_components"])
                break
    finally:
        env.close()
    return {
        "return": total,
        "components": dict(components),
        "final_progress": float(info["progress"]),
        "duplicate_collectibles": duplicates,
        "outcome": info.get("outcome"),
        "terminal_components": terminal_components,
    }


def _assert_observation(env: PlatformerStateEnv, observation: np.ndarray) -> None:
    if not env.observation_space.contains(observation):
        raise AssertionError("observation escaped declared space")
    if not np.all(np.isfinite(observation)):
        raise AssertionError("non-finite observation encountered")
