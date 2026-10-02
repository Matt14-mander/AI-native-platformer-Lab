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
    if config.get("environment_id") not in {"PlatformerState-v0", "PlatformerState-v1"}:
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
