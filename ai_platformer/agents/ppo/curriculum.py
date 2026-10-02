"""Episode-boundary curriculum sampling and serializable promotion decisions."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class CourseSampler(gym.Wrapper):
    def __init__(
        self, env, *, stages: list[dict], seeds: list[int], rng_seed: int, replay_fraction: float
    ) -> None:
        super().__init__(env)
        self.stages = stages
        self.seeds = seeds
        self.rng = np.random.default_rng(rng_seed)
        self.replay_fraction = replay_fraction
        self.stage_index = 0

    def reset(self, *, seed=None, options=None):
        # Explicit resets reproducibly initialize this worker; automatic resets advance its RNG.
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        index = self.stage_index
        if index > 0 and self.rng.random() < self.replay_fraction:
            index = int(self.rng.integers(index))
        levels = self.stages[index]["train_levels"]
        level_id = levels[int(self.rng.integers(len(levels)))]
        actual_seed = self.seeds[int(self.rng.integers(len(self.seeds)))]
        return self.env.reset(seed=actual_seed, options={"level_id": level_id})


class CurriculumState:
    def __init__(self, stages: list[dict], *, required_evaluations: int, saved: dict | None = None):
        self.stages = stages
        self.required = required_evaluations
        self.data = saved or {
            "stage_index": 0,
            "stage_start": 0,
            "consecutive_passes": 0,
            "status": "training",
            "history": [],
            "best": {},
        }
        if not 0 <= self.data["stage_index"] < len(stages):
            raise ValueError("invalid checkpoint curriculum stage")
        self.data["status"] = "training"

    @property
    def stage(self) -> dict:
        return self.stages[self.data["stage_index"]]

    def select_stage(self, task: str, timesteps: int) -> None:
        """Explicit experiment selection; never label skipped stages as mastered."""
        tasks = [stage["task"] for stage in self.stages]
        if task not in tasks:
            raise ValueError(f"stage is not configured: {task}")
        if task == self.stage["task"]:
            return
        self.data["history"].append(
            {
                "outcome": "manual_stage_change",
                "from": self.stage["task"],
                "to": task,
                "timesteps": timesteps,
                "reason": "explicit_start_stage",
            }
        )
        self.data["stage_index"] = tasks.index(task)
        self.data["stage_start"] = timesteps
        self.data["consecutive_passes"] = 0
        self.data["status"] = "training"

    def observe(self, report: dict, timesteps: int) -> bool:
        """Return True on promotion. Never consult final-test layouts."""
        threshold = self.stage["success_threshold"]
        if threshold is None:
            return False
        passed = all(
            report["by_level"][level]["success_rate"] >= threshold
            for level in self.stage["validation_levels"]
        )
        # Earlier tasks must still be mastered before promoting the next task.
        for stage in self.stages[: self.data["stage_index"]]:
            passed = passed and all(
                report["by_level"][level]["success_rate"] >= stage["success_threshold"]
                for level in stage["validation_levels"]
            )
        for stage in self.stages[: self.data["stage_index"] + 1]:
            limit = stage.get("max_success_steps")
            if limit is not None:
                # Gate each layout: averaging would hide a stalled successful episode.
                for level in stage["validation_levels"]:
                    successful = [
                        episode
                        for episode in report.get("episodes", [])
                        if episode["level_id"] == level and episode["outcome"] == "success"
                    ]
                    passed = (
                        passed
                        and bool(successful)
                        and all(episode["steps"] <= limit for episode in successful)
                    )
        self.data["consecutive_passes"] = self.data["consecutive_passes"] + 1 if passed else 0
        if self.data["consecutive_passes"] < self.required:
            return False
        self.data["history"].append(
            {
                "task": self.stage["task"],
                "outcome": "mastered",
                "start": self.data["stage_start"],
                "end": timesteps,
            }
        )
        if self.data["stage_index"] == len(self.stages) - 1:
            self.data["status"] = "completed"
        else:
            self.data["stage_index"] += 1
            self.data["stage_start"] = timesteps
            self.data["consecutive_passes"] = 0
        return True
