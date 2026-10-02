"""Shared episode evaluation for scripted and learned policies."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ai_platformer.agents.scripted import ScriptedAgent
from ai_platformer.envs.factory import EnvironmentFactory


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    seed: int
    episode_return: float
    steps: int
    core_ticks: int
    progress: float
    coins_collected: int
    outcome: str
    level_id: str = "level_1"


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    agent: str
    episodes: tuple[EpisodeResult, ...]

    def summary(self) -> dict[str, Any]:
        count = len(self.episodes)
        outcomes = Counter(episode.outcome for episode in self.episodes)
        successes = [item.steps for item in self.episodes if item.outcome == "success"]
        return {
            "agent": self.agent,
            "episodes": count,
            "success_rate": outcomes["success"] / count,
            "death_rate": outcomes["death"] / count,
            "truncation_rate": outcomes["time_limit"] / count,
            "mean_return": sum(item.episode_return for item in self.episodes) / count,
            "mean_progress": sum(item.progress for item in self.episodes) / count,
            "mean_steps": sum(item.steps for item in self.episodes) / count,
            "mean_coins": sum(item.coins_collected for item in self.episodes) / count,
            "mean_success_steps": sum(successes) / len(successes) if successes else None,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "episodes": [asdict(episode) for episode in self.episodes],
            "by_level": {
                level: BenchmarkResult(
                    self.agent, tuple(item for item in self.episodes if item.level_id == level)
                ).summary()
                for level in sorted({item.level_id for item in self.episodes})
            },
        }


def evaluate_scripted_agent(
    agent_name: str,
    agent_factory: Callable[[], ScriptedAgent],
    seeds: Iterable[int],
    *,
    action_repeat: int = 4,
    max_steps: int | None = None,
    environment: dict[str, Any] | None = None,
    level_ids: Iterable[str] | None = None,
) -> BenchmarkResult:
    config = dict(environment or {})
    config.setdefault("action_repeat", action_repeat)
    if max_steps is not None:
        if "episode_step_limit" in config and config["episode_step_limit"] != max_steps:
            raise ValueError("max_steps conflicts with environment episode_step_limit")
        config["episode_step_limit"] = max_steps
    return evaluate_agent(
        agent_name, agent_factory, seeds, factory=EnvironmentFactory(config), level_ids=level_ids
    )


def evaluate_agent(
    agent_name: str,
    agent_factory: Callable[[], ScriptedAgent],
    seeds: Iterable[int],
    *,
    factory: EnvironmentFactory,
    level_ids: Iterable[str] | None = None,
    trace_path: Path | None = None,
) -> BenchmarkResult:
    """Use the environment's time limit, including its remaining-time sensor.

    Save the first failed episode when requested, with memory bounded by one episode.
    """
    seed_list = list(seeds)
    levels = list(level_ids) if level_ids is not None else [factory.level_id]
    if not seed_list or not levels:
        raise ValueError("at least one evaluation seed and level are required")
    episodes = []
    env = factory.make(level_id=levels[0])
    trace_saved = False
    try:
        for level_id in levels:
            for seed in seed_list:
                agent = agent_factory()
                agent.reset(seed=seed)
                observation, info = env.reset(seed=seed, options={"level_id": level_id})
                trace = []
                episode_return = 0.0
                steps = 0
                terminated = truncated = False
                while not (terminated or truncated):
                    action = agent.act(observation)
                    before = observation.tolist()
                    observation, reward, terminated, truncated, info = env.step(action)
                    if trace_path is not None and not trace_saved:
                        trace.append(
                            {
                                "observation": before,
                                "action": int(action),
                                "reward": reward,
                                "reward_components": info["reward_components"],
                                "tick": info["tick"],
                                "progress": info["progress"],
                                "x": env.core.state.player.x,
                                "y": env.core.state.player.y,
                                "terminated": terminated,
                                "truncated": truncated,
                            }
                        )
                    episode_return += reward
                    steps += 1
                outcome = str(info.get("outcome", "unknown"))
                episodes.append(
                    EpisodeResult(
                        seed=seed,
                        episode_return=episode_return,
                        steps=steps,
                        core_ticks=int(info["tick"]),
                        progress=float(info["progress"]),
                        coins_collected=int(info["coins_collected"]),
                        outcome=outcome,
                        level_id=level_id,
                    )
                )
                if trace_path is not None and not trace_saved and outcome != "success":
                    trace_path.parent.mkdir(parents=True, exist_ok=True)
                    trace_path.write_text(
                        json.dumps(
                            {
                                "level_id": level_id,
                                "seed": seed,
                                "outcome": outcome,
                                "protocol": factory.protocol(levels),
                                "transitions": trace,
                            },
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    trace_saved = True
    finally:
        env.close()
    return BenchmarkResult(agent=agent_name, episodes=tuple(episodes))
