"""Load playback backends lazily and restore their verified environment protocol."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_platformer.envs.factory import EnvironmentFactory, protocol_hash


@dataclass
class LoadedBackend:
    policy: Any
    factory: EnvironmentFactory
    default_seed: int
    label: str
    model_path: Path

    def close(self) -> None:
        close = getattr(self.policy, "close", None)
        if close is not None:
            close()


def load_backend(
    backend: str,
    model_path: Path,
    *,
    library: Path | None = None,
    fuse_relu: bool = False,
) -> LoadedBackend:
    if backend == "sb3":
        if library is not None or fuse_relu:
            raise ValueError("--library/--fuse-relu apply only to the tinyinfer backend")
        from stable_baselines3 import PPO

        from ai_platformer.agents.ppo.training import checkpoint_metadata

        path = Path(model_path).with_suffix(".zip")
        metadata = checkpoint_metadata(path)
        factory = EnvironmentFactory(metadata["config"]["environment"])
        previous = metadata["signature"]["protocol"]
        seed = metadata["config"]["evaluation"]["seeds"][0]
    elif backend == "tinyinfer":
        if library is None:
            raise ValueError("tinyinfer requires --library pointing to the bridge dynamic library")
        from .package import load_actor_metadata

        path = Path(model_path)
        if path.suffix.lower() != ".onnx":
            raise ValueError("tinyinfer requires an exported actor.onnx deployment package")
        metadata = load_actor_metadata(path)
        factory = EnvironmentFactory(metadata["environment"])
        previous = metadata["protocol"]
        seed = metadata["validation"]["seed"]
    else:
        raise ValueError(f"unsupported backend: {backend}")
    if protocol_hash(factory.protocol(list(previous["levels"]))) != protocol_hash(previous):
        raise ValueError("model content/protocol has changed; restore its manifest/levels")
    if backend == "sb3":
        policy = PPO.load(path, device="cpu")
        if policy.num_timesteps != metadata["saved_timesteps"]:
            raise ValueError("checkpoint timestep differs from its sidecar")
        probe = factory.make(seed=seed)
        try:
            if (
                policy.observation_space != probe.observation_space
                or policy.action_space != probe.action_space
            ):
                raise ValueError("checkpoint observation/action spaces differ from environment")
        finally:
            probe.close()
        label = "SB3 PPO"
    else:
        from .tinyinfer import TinyInferPolicy

        policy = TinyInferPolicy(path, library, fuse_relu=fuse_relu)
        label = "TinyInfer CPU" + (" + ReLU fusion" if fuse_relu else "")
    return LoadedBackend(policy, factory, seed, label, path)
