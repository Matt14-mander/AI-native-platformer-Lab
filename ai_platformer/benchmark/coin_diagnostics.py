"""Inspect ground-coin failures without requiring a learned checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

from ai_platformer.agents.scripted import MoveRightAgent, RuleJumpAgent
from ai_platformer.agents.scripted.coin_jump import CoinJumpAgent
from ai_platformer.content.generation_suite import load_suite
from ai_platformer.content.level_spec import geometry_hash
from ai_platformer.content.pcg import generate_level
from ai_platformer.content.routes import collect_route, replay_route, search_route
from ai_platformer.core import Action
from ai_platformer.envs.factory import EnvironmentFactory


AGENTS = {
    "move_right": MoveRightAgent,
    "rule_jump": RuleJumpAgent,
    "coin_jump": CoinJumpAgent,
}


class ShortJumpProbe(RuleJumpAgent):
    """Diagnostic control: change jump hold time while keeping the route rightward."""

    def __init__(self, hold_steps: int = 3):
        super().__init__()
        self.hold_steps = hold_steps
        self.remaining = 0

    def reset(self, *, seed: int) -> None:
        del seed
        self.remaining = 0

    def act(self, observation) -> int:
        if observation[4] > 0.5 and min(observation[7], observation[8]) <= self.trigger_distance:
            self.remaining = self.hold_steps
        if self.remaining:
            self.remaining -= 1
            return int(Action.RIGHT_RUN_JUMP)
        return int(Action.RIGHT_RUN)


def run_agent(factory: EnvironmentFactory, agent_type, *, seed: int) -> dict:
    """Capture collectible transitions from world state, including repeat sub-ticks."""
    env = factory.make()
    agent = agent_type()
    agent.reset(seed=seed)
    collected = []
    trace = []
    try:
        observation, _ = env.reset(seed=seed)
        previous_active = {item.entity_id for item in env.core.state.entities if item.active}
        while True:
            before_x = env.core.state.player.x
            action = int(agent.act(observation))
            observation, reward, terminated, truncated, info = env.step(action)
            active = {item.entity_id for item in env.core.state.entities if item.active}
            newly_collected = sorted(previous_active - active)
            collected.extend(newly_collected)
            previous_active = active
            trace.append(
                {
                    "step": len(trace) + 1,
                    "tick": info["tick"],
                    "action": action,
                    "before_x": before_x,
                    "x": env.core.state.player.x,
                    "y": env.core.state.player.y,
                    "collected": newly_collected,
                }
            )
            if terminated or truncated:
                break
        return {
            "outcome": info.get("outcome", "unknown"),
            "steps": len(trace),
            "coins_collected": len(collected),
            "coins_total": info["coins_total"],
            "collected_ids": collected,
            "trace": trace,
            "episode_return": info["episode_return"],
            "progress": info["progress"],
        }
    finally:
        env.close()


def diagnose_mixed_ground(
    suite_path: Path,
    output_dir: Path,
    *,
    route_seconds: float = 15.0,
    route_expansions: int = 25_000,
) -> dict:
    if output_dir.exists():
        raise FileExistsError("output directory exists; preserve previous diagnostics")
    suite = load_suite(suite_path)
    case = next(case for case in suite.cases if case.case_id == "mixed_ground")
    output_dir.mkdir(parents=True)
    rows = []
    for seed in suite.generation_seeds:
        request = case.expected_request(seed)
        generated = generate_level(request)
        folder = output_dir / request.level_id
        folder.mkdir()
        spec_path = folder / "level.json"
        spec_path.write_text(json.dumps(generated.level.model_dump(), indent=2) + "\n")
        factory = EnvironmentFactory(
            {
                "environment_id": "PlatformerState-v2",
                "level_spec": str(spec_path),
                "physics": generated.report["physics"],
                "action_repeat": 4,
                "episode_step_limit": 768,
            }
        )
        agent_results = {}
        for name, agent_type in {**AGENTS, "short_jump_probe": ShortJumpProbe}.items():
            result = run_agent(factory, agent_type, seed=suite.policy_seed)
            trace = result.pop("trace")
            (folder / f"{name}_trace.json").write_text(json.dumps(trace, indent=2) + "\n")
            result["missed_ids"] = sorted(
                item.entity_id for item in generated.level.collectibles
                if item.entity_id not in result["collected_ids"]
            )
            if name == "short_jump_probe" and result["outcome"] == "success" and not result["missed_ids"]:
                witness = collect_route(
                    factory, [step["action"] for step in trace],
                    seed=suite.policy_seed, min_coin_ratio=1.0,
                )
                replay = replay_route(spec_path, witness)
                if not replay["passed"]:
                    raise RuntimeError("short-jump witness failed independent replay")
                (folder / "short_jump_route.json").write_text(json.dumps(witness, indent=2) + "\n")
                result["route_verified"] = True
            agent_results[name] = result
        route = search_route(
            factory,
            seed=suite.policy_seed,
            max_steps=768,
            max_seconds=route_seconds,
            max_expansions=route_expansions,
            min_coin_ratio=1.0,
        )
        (folder / "route.json").write_text(json.dumps(route, indent=2) + "\n")
        rows.append(
            {
                "level_id": request.level_id,
                "generation_seed": seed,
                "geometry_sha256": geometry_hash(generated.level),
                "segments": generated.report["segments"],
                "coins": [
                    {"id": item.entity_id, "x": item.x, "y": item.y}
                    for item in generated.level.collectibles
                ],
                "agents": agent_results,
                "route": {
                    "status": route["status"],
                    "reason": route["reason"],
                    "expansions": route["expansions"],
                    "method": route.get("method"),
                },
            }
        )
    report = {
        "schema_version": 1,
        "purpose": "model_independent_mixed_ground_diagnosis",
        "source_suite": str(suite_path),
        "policy_seed": suite.policy_seed,
        "route_budget": {"seconds": route_seconds, "expansions": route_expansions},
        "historical_ppo_summary": "v9 report: 14/16 for each of three maps; checkpoint unavailable here",
        "maps": rows,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
