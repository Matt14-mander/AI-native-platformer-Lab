"""Export the supported state-vector PPO actor and fixed numerical references."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import FlattenExtractor

from ai_platformer.agents.ppo.training import checkpoint_metadata
from ai_platformer.core import Action
from ai_platformer.envs.factory import EnvironmentFactory, protocol_hash

from .package import file_hash, metadata_hash


def materialize_constant_aliases(graph: onnx.ModelProto) -> int:
    """PyTorch may emit Identity aliases for identical (e.g. zero) biases."""
    constants = {value.name: value for value in graph.graph.initializer}
    remaining = []
    count = 0
    for node in graph.graph.node:
        if (
            node.op_type == "Identity"
            and node.domain in {"", "ai.onnx"}
            and len(node.input) == len(node.output) == 1
            and node.input[0] in constants
        ):
            value = onnx.TensorProto()
            value.CopyFrom(constants[node.input[0]])
            value.name = node.output[0]
            graph.graph.initializer.append(value)
            constants[value.name] = value
            count += 1
        else:
            remaining.append(node)
    del graph.graph.node[:]
    graph.graph.node.extend(remaining)
    return count


def actor_module(policy) -> torch.nn.Module:
    if (
        not isinstance(policy.pi_features_extractor, FlattenExtractor)
        or len(policy.observation_space.shape) != 1
        or policy.action_space.n != len(Action)
    ):
        raise ValueError("export requires a flat state-vector, discrete-action MLP policy")
    # Flat Box observations already have shape [1,N]; FlattenExtractor is identity
    # here. Omit its ONNX Flatten node, which TinyInfer does not implement.
    actor = torch.nn.Sequential(policy.mlp_extractor.policy_net, policy.action_net).cpu().eval()
    if any(
        not isinstance(layer, (torch.nn.Sequential, torch.nn.Linear, torch.nn.ReLU))
        for layer in actor.modules()
    ):
        raise ValueError("export supports only Linear/ReLU actor layers")
    return actor


def collect_observations(
    model, factory, *, samples: int, seed: int
) -> tuple[np.ndarray, list[str], int]:
    rng = np.random.default_rng(seed)
    observations = []
    levels = []
    if "full" in factory.repository.manifest["splits"]:
        levels.append(factory.level_id)
    for task in ("flat", "obstacle", "gap", "mixed", "full"):
        if task not in factory.repository.manifest["splits"]:
            continue
        candidates = factory.repository.split(task, "validation")
        indices = sorted({0, len(candidates) // 2, len(candidates) - 1})
        levels.extend(candidates[index] for index in indices)
    for level_id in levels:
        env = factory.make(level_id=level_id, seed=seed)
        try:
            observation, _ = env.reset(seed=seed)
            done = False
            while not done:
                observations.append(observation.copy())
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, _ = env.step(int(action))
                done = terminated or truncated
        finally:
            env.close()
    real_count = min(len(observations), max(1, samples * 3 // 4))
    chosen = rng.choice(len(observations), real_count, replace=False)
    real = np.asarray(observations, dtype=np.float32)[chosen]
    random = rng.uniform(-1, 1, size=(samples - real_count, real.shape[1])).astype(np.float32)
    return np.concatenate((real, random)), levels, real_count


def export_actor(
    checkpoint: Path, output_dir: Path, *, samples: int = 1024, seed: int = 100
) -> dict:
    if samples < 2 or seed < 0:
        raise ValueError("samples must be >= 2 and seed must be nonnegative")
    checkpoint = Path(checkpoint).with_suffix(".zip")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("deployment output already exists; choose a new directory")
    metadata = checkpoint_metadata(checkpoint)
    factory = EnvironmentFactory(metadata["config"]["environment"])
    protocol = metadata["signature"]["protocol"]
    if metadata["signature"]["action_ids"] != {action.name: int(action) for action in Action}:
        raise ValueError("checkpoint action mapping differs from this project")
    if protocol_hash(factory.protocol(list(protocol["levels"]))) != protocol_hash(protocol):
        raise ValueError("checkpoint content/protocol has changed; restore its manifest/levels")
    model = PPO.load(checkpoint, device="cpu")
    torch.set_num_threads(1)
    if model.num_timesteps != metadata["saved_timesteps"]:
        raise ValueError("checkpoint timestep differs from its sidecar")
    env = factory.make()
    try:
        if (
            model.observation_space != env.observation_space
            or model.action_space != env.action_space
        ):
            raise ValueError("checkpoint observation/action spaces differ from environment")
    finally:
        env.close()
    actor = actor_module(model.policy)
    observations, levels, real_count = collect_observations(
        model, factory, samples=samples, seed=seed
    )
    with torch.no_grad():
        logits = np.concatenate(
            [actor(torch.from_numpy(row[None])).numpy() for row in observations]
        )
    actions, _ = model.predict(observations, deterministic=True)
    if not np.array_equal(actions, logits.argmax(axis=1)):
        raise ValueError("exported actor argmax differs from SB3 policy action selection")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir.parent, prefix=".actor-export-") as temporary:
        folder = Path(temporary) / "package"
        folder.mkdir()
        path = folder / "actor.onnx"
        torch.onnx.export(
            actor,
            torch.from_numpy(observations[:1]),
            str(path),
            opset_version=17,
            export_params=True,
            do_constant_folding=True,
            input_names=["observations"],
            output_names=["logits"],
        )
        graph = onnx.load(str(path))
        onnx.checker.check_model(graph)
        aliases = materialize_constant_aliases(graph)
        onnx.checker.check_model(graph)
        onnx.save(graph, str(path))
        kinds = [node.op_type for node in graph.graph.node]
        if set(kinds) - {"Gemm", "Relu"}:
            raise ValueError(f"unexpected exported actor operators: {kinds}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        reference = np.concatenate(
            [session.run(["logits"], {"observations": row[None]})[0] for row in observations]
        )
        np.testing.assert_allclose(reference, logits, atol=1e-5, rtol=1e-5)
        if not np.array_equal(reference.argmax(axis=1), actions):
            raise ValueError("ONNX Runtime actions differ from SB3 on validation observations")
        np.savez_compressed(
            folder / "validation.npz", observations=observations, logits=logits, actions=actions
        )
        deployment = {
            "schema_version": 1,
            "model_sha256": file_hash(path),
            "source": {
                "checkpoint": str(checkpoint.resolve()),
                "model_sha256": file_hash(checkpoint),
                "signature_hash": metadata["signature_hash"],
                "timesteps": metadata["saved_timesteps"],
            },
            "input": {
                "name": "observations",
                "shape": [1, observations.shape[1]],
                "dtype": "float32",
            },
            "output": {"name": "logits", "shape": [1, len(Action)], "dtype": "float32"},
            "action_ids": {action.name: int(action) for action in Action},
            "action_selection": "argmax_first",
            "environment": metadata["config"]["environment"],
            "protocol": protocol,
            "onnx": {"opset": 17, "operators": kinds, "constant_aliases_materialized": aliases},
            "versions": {
                "torch": torch.__version__,
                "onnx": onnx.__version__,
                "onnxruntime": ort.__version__,
            },
            "validation": {
                "file": "validation.npz",
                "sha256": file_hash(folder / "validation.npz"),
                "seed": seed,
                "samples": samples,
                "real_observations": real_count,
                "random_observations": samples - real_count,
                "levels": levels,
                "atol": 1e-5,
                "rtol": 1e-5,
                "onnxruntime_max_abs_error": float(np.max(np.abs(reference - logits))),
                "sb3_action_match_rate": 1.0,
                "onnxruntime_action_match_rate": 1.0,
            },
        }
        deployment["metadata_sha256"] = metadata_hash(deployment)
        (folder / "actor.json").write_text(
            json.dumps(deployment, indent=2) + "\n", encoding="utf-8"
        )
        folder.rename(output_dir)
    return deployment
