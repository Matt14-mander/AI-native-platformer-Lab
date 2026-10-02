"""Reproducible PPO with evaluation, curriculum, checkpoints, and resume."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3 import __version__ as sb3_version
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from ai_platformer.benchmark.scripted import evaluate_agent
from ai_platformer.core import Action
from ai_platformer.envs.factory import EnvironmentFactory, protocol_hash

from .configuration import resolve_training_config
from .curriculum import CourseSampler, CurriculumState


class PolicyAgent:
    def __init__(self, model: PPO):
        self.model = model

    def reset(self, *, seed: int) -> None:
        pass

    def act(self, observation: np.ndarray) -> int:
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action)


def evaluate_policy(
    model: PPO,
    *,
    seeds: list[int],
    action_repeat: int = 4,
    episode_step_limit: int | None = 1024,
    environment: dict | None = None,
    level_ids: list[str] | None = None,
    trace_path: Path | None = None,
) -> dict[str, Any]:
    config = dict(environment or {})
    config.setdefault("action_repeat", action_repeat)
    config.setdefault("episode_step_limit", episode_step_limit)
    return evaluate_agent(
        "ppo",
        lambda: PolicyAgent(model),
        seeds,
        factory=EnvironmentFactory(config),
        level_ids=level_ids,
        trace_path=trace_path,
    ).to_dict()


def write_json(path: Path, data: dict) -> None:
    """Keep reports readable even when a later evaluation/rollout fails."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def training_signature(config: dict, factory: EnvironmentFactory, stages: list[dict]) -> dict:
    levels = sorted(
        {level for stage in stages for level in stage["train_levels"] + stage["validation_levels"]}
    )
    # Runtime budgets/intervals may change on resume; protocols and selection rules may not.
    stage_contracts = [
        {key: value for key, value in stage.items() if key != "max_timesteps"} for stage in stages
    ]
    return {
        "protocol": factory.protocol(levels),
        "seed": config["seed"],
        "train_seeds": config["train_seeds"],
        "validation_seeds": config["evaluation"]["seeds"],
        "network": config["network"],
        "algorithm": config["algorithm"],
        "n_envs": config["n_envs"],
        "stages": stage_contracts,
        "replay_fraction": config["curriculum"]["replay_fraction"],
        "required_evaluations": config["curriculum"]["required_evaluations"],
        "action_ids": {action.name: int(action) for action in Action},
    }


def checkpoint_metadata(path: Path) -> dict:
    data = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if data.get("schema_version") != 2:
        raise ValueError("resume requires a v2 checkpoint sidecar, not an unversioned legacy model")
    if data["signature_hash"] != protocol_hash(data["signature"]):
        raise ValueError("checkpoint protocol metadata is inconsistent")
    if (
        "model_sha256" in data
        and data["model_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
    ):
        raise ValueError("checkpoint model does not match its sidecar")
    return data


class TrainingCallback(BaseCallback):
    def __init__(
        self, config, factory, samplers, output_dir, state, signature, initialization=None
    ):
        super().__init__()
        self.config = config
        self.factory = factory
        self.samplers = samplers
        self.output_dir = output_dir
        self.state = state
        self.signature = signature
        self.initialization = initialization
        self.last_eval = -1
        self.next_eval = 0
        self.next_checkpoint = 0
        self.evaluations = []

    def _on_training_start(self) -> None:
        self.next_eval = self.num_timesteps + self.config["evaluation"]["every_timesteps"]
        self.next_checkpoint = self.num_timesteps + self.config["checkpoint_every_timesteps"]
        for sampler in self.samplers:
            sampler.stage_index = self.state.data["stage_index"]

    def save(self, stem: Path) -> None:
        self.model.save(stem)
        write_json(
            stem.with_suffix(".json"),
            {
                "schema_version": 2,
                "saved_timesteps": int(self.model.num_timesteps),
                "model_sha256": hashlib.sha256(stem.with_suffix(".zip").read_bytes()).hexdigest(),
                "signature": self.signature,
                "signature_hash": protocol_hash(self.signature),
                "config": self.config,
                "curriculum_state": deepcopy(self.state.data),
                "initialization": self.initialization,
                "resume_semantics": "optimizer and timestep continuation; environment/RNG reset",
            },
        )

    def evaluate(self, *, allow_promotion: bool) -> dict:
        index = self.state.data["stage_index"]
        stage = self.state.stage
        # Diagnose the active task first while retaining all earlier regression tasks.
        levels = stage["validation_levels"] + [
            level for item in self.state.stages[:index] for level in item["validation_levels"]
        ]
        report = evaluate_policy(
            self.model,
            seeds=self.config["evaluation"]["seeds"],
            environment=self.config["environment"],
            level_ids=levels,
            trace_path=self.output_dir / "evaluation" / f"failure_{self.num_timesteps}.json",
        )
        current = [
            item for item in report["episodes"] if item["level_id"] in stage["validation_levels"]
        ]
        score = [
            sum(item["outcome"] == "success" for item in current) / len(current),
            sum(item["progress"] for item in current) / len(current),
        ]
        if "max_success_steps" in stage:
            successful_steps = [item["steps"] for item in current if item["outcome"] == "success"]
            score.append(-sum(successful_steps) / len(successful_steps) if successful_steps else 0)
        entry = {
            "timesteps": self.num_timesteps,
            "stage": stage["task"],
            "selection_score": score,
            "evaluation": report,
        }
        self.evaluations.append(entry)
        write_json(self.output_dir / "evaluation" / f"step_{self.num_timesteps}.json", entry)
        best = self.state.data["best"].get(stage["task"])
        if best is None or tuple(score) > tuple(best["score"]):
            stem = self.output_dir / "checkpoints" / f"best_{stage['task']}"
            self.state.data["best"][stage["task"]] = {
                "score": score,
                "timesteps": self.num_timesteps,
                "model": str(stem.with_suffix(".zip").resolve()),
            }
            self.save(stem)
        self.logger.record("validation/success_rate", score[0])
        self.logger.record("validation/mean_progress", score[1])
        self.logger.dump(self.num_timesteps)
        self.last_eval = self.num_timesteps
        if allow_promotion and self.state.observe(report, self.num_timesteps):
            for sampler in self.samplers:
                sampler.stage_index = self.state.data["stage_index"]
            self.save(self.output_dir / "checkpoints" / f"stage_{self.num_timesteps}")
        return report

    def _on_step(self) -> bool:
        stage = self.state.stage
        budget_expired = stage["success_threshold"] is not None and (
            self.num_timesteps - self.state.data["stage_start"] >= stage["max_timesteps"]
        )
        if self.num_timesteps >= self.next_eval or budget_expired:
            self.evaluate(allow_promotion=True)
            self.next_eval = self.num_timesteps + self.config["evaluation"]["every_timesteps"]
            if self.state.data["status"] == "completed":
                return False
            # Promotion starts a new budget, so recheck after the evaluation decision.
            stage = self.state.stage
            if (
                stage["success_threshold"] is not None
                and self.num_timesteps - self.state.data["stage_start"] >= stage["max_timesteps"]
            ):
                self.state.data["status"] = "budget_exhausted"
                self.save(self.output_dir / "checkpoints" / f"budget_{self.num_timesteps}")
                return False
        if self.num_timesteps >= self.next_checkpoint:
            self.save(self.output_dir / "checkpoints" / f"step_{self.num_timesteps}")
            self.next_checkpoint = self.num_timesteps + self.config["checkpoint_every_timesteps"]
        return True


def train_ppo(
    config: dict[str, Any],
    output_dir: Path,
    *,
    resume: Path | None = None,
    init_from: Path | None = None,
    start_stage: str | None = None,
) -> dict:
    """Train additional transitions; refuse output overwrites and incompatible resume."""
    config, factory, stages = resolve_training_config(config)
    signature = training_signature(config, factory, stages)
    saved = None
    initial_weights = None
    initialization = None
    if resume is not None and init_from is not None:
        raise ValueError("resume and init_from are mutually exclusive")
    if start_stage is not None and start_stage not in {stage["task"] for stage in stages}:
        raise ValueError(f"stage is not configured: {start_stage}")
    if resume is not None:
        resume = resume.with_suffix(".zip")
        saved = checkpoint_metadata(resume)
        if saved["signature_hash"] != protocol_hash(signature):
            raise ValueError(
                "checkpoint is incompatible with environment, content, seed pools or PPO config"
            )
        if saved["curriculum_state"]["status"] == "completed":
            raise ValueError("curriculum already completed; evaluate its checkpoints instead")
        initialization = saved.get("initialization")
    if init_from is not None:
        init_from = init_from.with_suffix(".zip")
        source = checkpoint_metadata(init_from)
        source_signature = source["signature"]
        content_keys = {"levels", "curriculum_generator_version", "curriculum_manifest_hash"}
        source_protocol = {
            key: value
            for key, value in source_signature["protocol"].items()
            if key not in content_keys
        }
        target_protocol = {
            key: value for key, value in signature["protocol"].items() if key not in content_keys
        }
        if (
            source_protocol != target_protocol
            or source_signature["network"] != signature["network"]
            or source_signature["action_ids"] != signature["action_ids"]
        ):
            raise ValueError("init_from is incompatible with observation/action/network protocol")
        source_model = PPO.load(init_from, device="cpu")
        if source_model.num_timesteps != source["saved_timesteps"]:
            raise ValueError("checkpoint timestep differs from its sidecar")
        initial_weights = deepcopy(source_model.policy.state_dict())
        del source_model
        initialization = {
            "source": str(init_from.resolve()),
            "source_sha256": hashlib.sha256(init_from.read_bytes()).hexdigest(),
            "source_timesteps": source["saved_timesteps"],
            "source_signature_hash": source["signature_hash"],
            "semantics": "actor and critic weights only; fresh optimizer, RNG, timesteps and curriculum",
        }
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("output directory is not empty; choose a new run directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("monitor", "checkpoints", "evaluation"):
        (output_dir / name).mkdir()
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "protocol.json", signature["protocol"])
    state = CurriculumState(
        stages,
        required_evaluations=config["curriculum"]["required_evaluations"],
        saved=deepcopy(saved["curriculum_state"]) if saved else None,
    )
    if start_stage is not None:
        state.select_stage(start_stage, saved["saved_timesteps"] if saved else 0)
    torch.set_num_threads(config["torch_threads"])
    set_random_seed(config["seed"])
    samplers = []

    def make_worker(rank):
        def make():
            stage = stages[state.data["stage_index"]]
            sampler = CourseSampler(
                factory.make(level_id=stage["train_levels"][0]),
                stages=stages,
                seeds=config["train_seeds"],
                rng_seed=config["seed"] + rank,
                replay_fraction=config["curriculum"]["replay_fraction"],
            )
            sampler.stage_index = state.data["stage_index"]
            samplers.append(sampler)
            return Monitor(
                sampler,
                filename=str(output_dir / "monitor" / f"env_{rank}"),
                info_keywords=("level_id", "seed", "progress", "outcome"),
            )

        return make

    env = DummyVecEnv([make_worker(rank) for rank in range(config["n_envs"])])
    try:
        if resume is not None:
            model = PPO.load(resume, env=env, device=config.get("device", "cpu"))
            if model.num_timesteps != saved["saved_timesteps"]:
                raise ValueError("checkpoint timestep differs from its sidecar")
        else:
            model = PPO(
                "MlpPolicy",
                env,
                **config["algorithm"],
                policy_kwargs={
                    "activation_fn": torch.nn.ReLU,
                    "net_arch": {
                        "pi": config["network"]["policy_layers"],
                        "vf": config["network"]["value_layers"],
                    },
                },
                seed=config["seed"],
                device=config.get("device", "cpu"),
                verbose=config.get("verbose", 1),
            )
            if initial_weights is not None:
                model.policy.load_state_dict(initial_weights, strict=True)
        initial_timesteps = int(model.num_timesteps)
        if state.stage["success_threshold"] is None:
            state.data["stage_start"] = initial_timesteps
        callback = TrainingCallback(
            config, factory, samplers, output_dir, state, signature, initialization
        )
        interrupted = False
        try:
            model.learn(
                total_timesteps=config["total_timesteps"],
                callback=callback,
                reset_num_timesteps=resume is None,
                progress_bar=False,
            )
        except KeyboardInterrupt:
            interrupted = True
            state.data["status"] = "interrupted"
        if state.data["status"] == "training":
            state.data["status"] = (
                "timesteps_limit" if state.stage["success_threshold"] else "finished"
            )
        # Evaluate latest parameters after the final optimization (callbacks run during collection).
        evaluation = callback.evaluate(allow_promotion=False)
        callback.save(output_dir / "model")
        try:
            revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[3],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            dirty = bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"],
                    cwd=Path(__file__).resolve().parents[3],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            )
        except (OSError, subprocess.CalledProcessError):
            revision, dirty = None, None
        report = {
            "schema_version": 2,
            "environment_id": factory.environment_id,
            "config": config,
            "signature_hash": protocol_hash(signature),
            "protocol": signature["protocol"],
            "action_ids": signature["action_ids"],
            "observation_size": signature["protocol"]["observation_size"],
            "runtime": {
                "python": platform.python_version(),
                "stable_baselines3": sb3_version,
                "torch": torch.__version__,
                "numpy": np.__version__,
                "gymnasium": gymnasium.__version__,
                "git_revision": revision,
                "git_dirty": dirty,
            },
            "initial_timesteps": initial_timesteps,
            "trained_timesteps": int(model.num_timesteps),
            "added_timesteps": int(model.num_timesteps) - initial_timesteps,
            "resumed_from": str(resume.resolve()) if resume else None,
            "initialization": initialization,
            "requested_start_stage": start_stage,
            "curriculum_state": state.data,
            "evaluation": evaluation,
            "evaluation_history": callback.evaluations,
            "artifacts": {
                "model": str((output_dir / "model.zip").resolve()),
                "monitor": str((output_dir / "monitor").resolve()),
                "checkpoints": str((output_dir / "checkpoints").resolve()),
            },
        }
        write_json(output_dir / "run.json", report)
        if interrupted:
            print("Training interrupted; final checkpoint and report saved.")
        return report
    finally:
        env.close()
