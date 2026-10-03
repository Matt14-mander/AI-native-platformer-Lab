"""Resolve explicit training/evaluation pools before allocating PPO resources."""

from __future__ import annotations

from copy import deepcopy
from math import isfinite
from typing import Any

from ai_platformer.envs.factory import EnvironmentFactory


def resolve_training_config(config: dict[str, Any]) -> tuple[dict, EnvironmentFactory, list[dict]]:
    config = deepcopy(config)
    if config.get("schema_version") not in (1, 2):
        raise ValueError("unsupported PPO config version")
    if config.get("environment_id") not in {
        "PlatformerState-v0",
        "PlatformerState-v1",
        "PlatformerState-v2",
    }:
        raise ValueError("unsupported environment_id")
    for key in ("seed", "total_timesteps", "n_envs", "torch_threads"):
        value = config.get(key, 1 if key == "torch_threads" else None)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < (0 if key == "seed" else 1)
        ):
            raise ValueError(f"invalid {key}")
        config[key] = value
    algorithm = config["algorithm"]
    for key in ("n_steps", "batch_size", "n_epochs"):
        if (
            not isinstance(algorithm[key], int)
            or isinstance(algorithm[key], bool)
            or algorithm[key] <= 0
        ):
            raise ValueError(f"invalid PPO {key}")
    for key in ("learning_rate", "clip_range", "gamma", "gae_lambda", "ent_coef", "vf_coef"):
        algorithm[key] = float(algorithm[key])
        if not isfinite(algorithm[key]):
            raise ValueError(f"non-finite PPO {key}")
    rollout = config["n_envs"] * algorithm["n_steps"]
    if rollout <= 1 or algorithm["batch_size"] <= 1 or rollout % algorithm["batch_size"]:
        raise ValueError("batch_size must divide n_envs * n_steps (both greater than one)")
    if float(algorithm["learning_rate"]) <= 0 or not 0 < float(algorithm["clip_range"]) < 1:
        raise ValueError("invalid learning rate or clip range")
    for key in ("gamma", "gae_lambda"):
        if not 0 < float(algorithm[key]) <= 1:
            raise ValueError(f"invalid PPO {key}")
    for key in ("ent_coef", "vf_coef"):
        if float(algorithm[key]) < 0:
            raise ValueError(f"invalid PPO {key}")
    for layers in config["network"].values():
        if not layers or any(not isinstance(item, int) or item <= 0 for item in layers):
            raise ValueError("network layers must be positive integers")
    evaluation = config["evaluation"]
    if "imitation" in config:
        imitation = config["imitation"]
        expected = {"rounds", "epochs", "batch_size", "learning_rate", "max_steps_per_episode"}
        if not isinstance(imitation, dict) or set(imitation) != expected:
            raise ValueError(
                "imitation requires explicit rounds/epochs/batch_size/learning_rate/max_steps_per_episode"
            )
        for key in expected - {"learning_rate"}:
            if (
                not isinstance(imitation[key], int)
                or isinstance(imitation[key], bool)
                or imitation[key] <= 0
            ):
                raise ValueError(f"invalid imitation {key}")
        if (
            isinstance(imitation["learning_rate"], bool)
            or not isinstance(imitation["learning_rate"], (int, float))
            or not isfinite(imitation["learning_rate"])
            or imitation["learning_rate"] <= 0
        ):
            raise ValueError("invalid imitation learning_rate")
    selection = evaluation.get("selection", "current_task")
    if selection not in {"current_task", "joint_v1"}:
        raise ValueError("unsupported checkpoint selection policy")
    evaluation.setdefault("every_timesteps", 10_000)
    config.setdefault("checkpoint_every_timesteps", 25_000)
    for value in (evaluation["every_timesteps"], config["checkpoint_every_timesteps"]):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("evaluation and checkpoint intervals must be positive integers")
    config.setdefault("train_seeds", [config["seed"] + rank for rank in range(config["n_envs"])])
    for seeds in (config["train_seeds"], evaluation["seeds"]):
        if (
            not seeds
            or len(set(seeds)) != len(seeds)
            or any(
                not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in seeds
            )
        ):
            raise ValueError("seed pools must contain distinct nonnegative integers")
    if set(config["train_seeds"]) & set(evaluation["seeds"]):
        raise ValueError("training and validation seed pools overlap")
    if (
        "environment_id" in config["environment"]
        and config["environment"]["environment_id"] != config["environment_id"]
    ):
        raise ValueError("top-level and environment environment_id disagree")
    config["environment"]["environment_id"] = config["environment_id"]
    factory = EnvironmentFactory(config["environment"])
    # Freeze effective values so a later gameplay.json edit does not alter this run's evaluation.
    protocol = factory.protocol([])
    config["environment"].update(
        {
            key: protocol[key]
            for key in ("physics", "reward", "sensor_range", "action_repeat", "episode_step_limit")
        }
    )
    config["environment"]["level_id"] = factory.level_id
    curriculum = config.get("curriculum", {})
    fraction = float(curriculum.get("replay_fraction", 0.2))
    if not 0 <= fraction < 1:
        raise ValueError("replay_fraction must be in [0, 1)")
    required = curriculum.get("required_evaluations", 2)
    if not isinstance(required, int) or required < 1:
        raise ValueError("required_evaluations must be positive")
    stages = []
    for spec in curriculum.get("stages", []):
        task = spec["task"]
        if task == "baseline":
            if len(curriculum["stages"]) != 1:
                raise ValueError("baseline cannot be combined with course stages")
            break  # Resolve saved baseline config using the new additional training budget.
        train = [factory.level_id] if task == "full" else factory.repository.split(task, "train")
        validation = (
            [factory.level_id] if task == "full" else factory.repository.split(task, "validation")
        )
        threshold = float(spec["success_threshold"])
        budget = spec["max_timesteps"]
        if not 0 < threshold <= 1 or not isinstance(budget, int) or budget <= 0:
            raise ValueError("stage threshold/budget is invalid")
        stages.append(
            {
                "task": task,
                "train_levels": train,
                "validation_levels": validation,
                "success_threshold": threshold,
                "max_timesteps": budget,
            }
        )
        if "max_success_steps" in spec:
            limit = spec["max_success_steps"]
            if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
                raise ValueError("max_success_steps must be a positive integer")
            stages[-1]["max_success_steps"] = limit
        if "sampling_weights" in spec:
            weights = spec["sampling_weights"]
            allowed_tasks = {stage["task"] for stage in stages}
            if (
                not isinstance(weights, dict)
                or not weights
                or set(weights) - allowed_tasks
                or task not in weights
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not isfinite(value)
                    or value <= 0
                    for value in weights.values()
                )
            ):
                raise ValueError(
                    "sampling_weights require positive weights for current/earlier tasks"
                )
            stages[-1]["sampling_weights"] = dict(weights)
            if not isfinite(sum(weights.values())):
                raise ValueError("sampling_weights sum must be finite")
    if not stages:
        stages = [
            {
                "task": "baseline",
                "train_levels": [factory.level_id],
                "validation_levels": [factory.level_id],
                "success_threshold": None,
                "max_timesteps": config["total_timesteps"],
            }
        ]
    if len({stage["task"] for stage in stages}) != len(stages):
        raise ValueError("curriculum stages must be distinct")
    if selection == "joint_v1" and any(stage["success_threshold"] is None for stage in stages):
        raise ValueError("joint_v1 selection requires explicit curriculum stages")
    all_ids = {
        item for stage in stages for item in stage["train_levels"] + stage["validation_levels"]
    }
    for level in all_ids:
        factory.repository.load(level)
    config["curriculum"] = {
        "replay_fraction": fraction,
        "required_evaluations": required,
        "stages": stages,
    }
    return config, factory, stages
