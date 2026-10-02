"""Paired policy controls and tick-level jump diagnostics, without changing gameplay."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from ai_platformer.benchmark.scripted import BenchmarkResult, EpisodeResult
from ai_platformer.core.actions import control_for
from ai_platformer.envs.factory import EnvironmentFactory

MODES = ("deterministic", "stochastic", "fixed_observation")


class DiagnosticPolicyAgent:
    """A private CPU sampler keeps evaluation independent of the training RNG."""

    def __init__(self, model, mode: str):
        if mode not in MODES:
            raise ValueError(f"unknown diagnostic mode: {mode}")
        self.model = model
        self.mode = mode

    def reset(self, *, seed: int) -> None:
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.initial_observation = None
        self.initial_probabilities = None
        self.probabilities = []
        self.total_variations = []

    def act(self, observation: np.ndarray) -> int:
        if self.initial_observation is None:
            self.initial_observation = observation.copy()
        actual = self.initial_observation if self.mode == "fixed_observation" else observation
        with torch.no_grad():
            tensor, _ = self.model.policy.obs_to_tensor(actual)
            probabilities = self.model.policy.get_distribution(tensor).distribution.probs[0].cpu()
        if self.initial_probabilities is None:
            self.initial_probabilities = probabilities.clone()
        self.probabilities.append(probabilities.numpy().tolist())
        self.total_variations.append(
            float(torch.abs(probabilities - self.initial_probabilities).sum()) / 2
        )
        if self.mode == "deterministic":
            return int(probabilities.argmax())
        return int(torch.multinomial(probabilities, 1, generator=self.generator))


class TickProbe:
    """Delegate unchanged core transitions and observe events inside action repeat."""

    def __init__(self, core, player_width: float, gap: tuple[float, float] | None):
        self.core = core
        self.player_width = player_width
        self.gap = gap
        self.jump_held = False
        self.jumps = []
        self.landings = []

    def __getattr__(self, name):
        return getattr(self.core, name)

    def step(self, action):
        before = self.core.state.player
        result = self.core.step(action)
        after = result.state.player
        jump = control_for(action).jump
        if before.grounded and jump and not self.jump_held and after.velocity_y < 0:
            self.jumps.append(
                {
                    "tick": result.state.tick,
                    "x": before.x,
                    "edge_distance": self.gap[0] - before.x - self.player_width
                    if self.gap
                    else None,
                }
            )
        if not before.grounded and after.grounded:
            self.landings.append({"tick": result.state.tick, "x": after.x, "y": after.y})
        self.jump_held = jump
        return result


def diagnose_policy(
    model,
    *,
    factory: EnvironmentFactory,
    level_ids: list[str],
    seeds: list[int],
    modes: tuple[str, ...] = MODES,
    agent_factory: Callable | None = None,
) -> dict[str, Any]:
    """Compare paired RNG seeds; repeated seeds are not independent geometry samples.

    An optional scripted factory calibrates event instrumentation using the same runner.
    Final-test suites should only be supplied after freezing the training design.
    """
    if (
        not level_ids
        or not seeds
        or len(set(seeds)) != len(seeds)
        or any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds)
        or not modes
        or len(set(modes)) != len(modes)
        or any(mode not in MODES for mode in modes)
    ):
        raise ValueError("diagnostics require levels, distinct nonnegative seeds and valid modes")
    reports = {}
    env = factory.make(level_id=level_ids[0])
    original_core = env.core
    try:
        for mode in modes:
            episodes, details = [], []
            context_actions = {}
            context_probabilities = {}
            for level_id in level_ids:
                spec = factory.repository.manifest["levels"].get(level_id, {})
                gap = (
                    (spec["gap_x"], spec["gap_x"] + spec["gap_width"]) if "gap_x" in spec else None
                )
                for seed in seeds:
                    env.core = original_core
                    observation, _ = env.reset(seed=seed, options={"level_id": level_id})
                    probe = TickProbe(original_core, factory.settings.physics.player_width, gap)
                    env.core = probe
                    agent = agent_factory() if agent_factory else DiagnosticPolicyAgent(model, mode)
                    agent.reset(seed=seed)
                    actions = Counter()
                    episode_return, steps = 0.0, 0
                    terminated = truncated = False
                    while not (terminated or truncated):
                        player = env.core.state.player
                        edge_distance = gap[0] - player.x - probe.player_width if gap else None
                        if not player.grounded:
                            context = "airborne"
                        elif edge_distance is None:
                            context = "grounded_no_gap"
                        elif edge_distance < 0:
                            context = "grounded_beyond_edge"
                        elif edge_distance <= 120:
                            context = "grounded_near_edge"
                        elif edge_distance <= factory.sensor_range:
                            context = "grounded_visible_edge"
                        else:
                            context = "grounded_far_edge"
                        action = agent.act(observation)
                        actions[action] += 1
                        context_actions.setdefault(context, Counter())[action] += 1
                        if isinstance(agent, DiagnosticPolicyAgent):
                            values = np.asarray(agent.probabilities[-1])
                            context_probabilities[context] = (
                                context_probabilities.get(context, np.zeros_like(values)) + values
                            )
                        observation, reward, terminated, truncated, info = env.step(action)
                        episode_return += reward
                        steps += 1
                    episodes.append(
                        EpisodeResult(
                            seed=seed,
                            episode_return=episode_return,
                            steps=steps,
                            core_ticks=info["tick"],
                            progress=info["progress"],
                            coins_collected=info["coins_collected"],
                            outcome=info["outcome"],
                            level_id=level_id,
                        )
                    )
                    details.append(
                        {
                            "level_id": level_id,
                            "seed": seed,
                            "outcome": info["outcome"],
                            "action_counts": dict(sorted(actions.items())),
                            "jumps": probe.jumps,
                            "landings": probe.landings,
                            "first_jump_edge_distance": probe.jumps[0]["edge_distance"]
                            if probe.jumps
                            else None,
                            "landed_after_gap": bool(gap)
                            and any(
                                landing["x"] + probe.player_width > gap[1]
                                for landing in probe.landings
                            ),
                            "mean_policy_tv_from_initial": float(np.mean(agent.total_variations))
                            if isinstance(agent, DiagnosticPolicyAgent)
                            else None,
                            "initial_action_probabilities": agent.probabilities[0]
                            if isinstance(agent, DiagnosticPolicyAgent)
                            else None,
                        }
                    )
            report = BenchmarkResult(mode, tuple(episodes)).to_dict()
            count = len(details)
            report["behavior"] = {
                "mean_effective_jumps": sum(len(item["jumps"]) for item in details) / count,
                "mean_landings": sum(len(item["landings"]) for item in details) / count,
                "landed_after_gap_rate": sum(item["landed_after_gap"] for item in details) / count,
                "first_jump_edge_distances": [item["first_jump_edge_distance"] for item in details],
                "action_distribution_by_context": {
                    context: {
                        "decisions": sum(counts.values()),
                        "action_counts": dict(sorted(counts.items())),
                        "mean_action_probabilities": (
                            context_probabilities[context] / sum(counts.values())
                        ).tolist()
                        if context in context_probabilities
                        else None,
                    }
                    for context, counts in sorted(context_actions.items())
                },
            }
            report["diagnostics"] = details
            reports[mode] = report
    finally:
        env.core = original_core
        env.close()
    return {
        "schema_version": 1,
        "protocol": factory.protocol(level_ids),
        "seeds": seeds,
        "sampling": "paired per-episode private torch CPU generators; all modes reset per episode",
        "interpretation": "fixed-observation sampling is an open-loop control; similar success does "
        "not establish sensor use. Environment seeds do not create new course geometry.",
        "modes": reports,
    }
