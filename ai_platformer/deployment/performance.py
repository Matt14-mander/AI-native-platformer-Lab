"""Reproducible CPU deployment measurements, with explicit timing boundaries."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from .package import file_hash, load_actor_metadata, load_validation_data
from .tinyinfer import TinyInferPolicy


def statistics(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("timings must be nonempty, finite, nonnegative")
    return {
        "count": values.size,
        "p50_us": float(np.median(values)),
        "p95_us": float(np.percentile(values, 95)),
        "p99_us": float(np.percentile(values, 99)),
        "mean_us": float(values.mean()),
        "min_us": float(values.min()),
        "max_us": float(values.max()),
    }


def measure_calls(call, inputs, *, warmup: int, iterations: int) -> np.ndarray:
    for index in range(warmup):
        call(inputs[index % len(inputs)])
    timings = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        value = inputs[index % len(inputs)]
        begin = perf_counter_ns()
        call(value)
        timings[index] = (perf_counter_ns() - begin) / 1000
    return timings


def measure_frames(policies: dict, factory, levels: list[str], seed: int) -> tuple[dict, dict]:
    # Offscreen drawing is measured without FPS throttling, HUD, or terminal delay.
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    import pygame

    from ai_platformer.rendering.ppo_playback import create_renderer

    pygame.display.init()
    pygame.font.init()
    screen = pygame.display.set_mode((1000, 600))
    summaries = {name: [] for name in policies}
    raw = {
        name: {key: [] for key in ("predict_us", "environment_us", "render_us", "frame_us")}
        for name in policies
    }
    try:
        for episode, level_id in enumerate(levels):
            order = list(policies) if episode % 2 == 0 else list(reversed(policies))
            for name in order:
                policy = policies[name]
                env = factory.make(level_id=level_id, seed=seed)
                try:
                    observation, _ = env.reset(seed=seed)
                    # Renderer construction/asset loading is outside steady frames.
                    from types import SimpleNamespace

                    renderer = create_renderer(factory, SimpleNamespace(env=env), screen.get_size())
                    renderer.draw(screen, env.core.state)
                    trace = hashlib.sha256()
                    total_reward = 0.0
                    done = False
                    while not done:
                        begin = perf_counter_ns()
                        action, _ = policy.predict(observation, deterministic=True)
                        after_predict = perf_counter_ns()
                        action = int(np.asarray(action).item())
                        observation, reward, terminated, truncated, info = env.step(action)
                        after_env = perf_counter_ns()
                        renderer.draw(screen, env.core.state)
                        pygame.display.flip()
                        after_render = perf_counter_ns()
                        raw[name]["predict_us"].append((after_predict - begin) / 1000)
                        raw[name]["environment_us"].append((after_env - after_predict) / 1000)
                        raw[name]["render_us"].append((after_render - after_env) / 1000)
                        raw[name]["frame_us"].append((after_render - begin) / 1000)
                        # Hash outside timers: verify all world state, not just outcome.
                        trace.update(json.dumps(asdict(env.core.state), sort_keys=True).encode())
                        trace.update(np.asarray([action, reward], dtype=np.float64).tobytes())
                        total_reward += reward
                        done = terminated or truncated
                    summaries[name].append(
                        {
                            "level_id": level_id,
                            "seed": seed,
                            "outcome": info.get("outcome"),
                            "steps": info["episode_step"],
                            "ticks": info["tick"],
                            "return": total_reward,
                            "trace_sha256": trace.hexdigest(),
                        }
                    )
                finally:
                    env.close()
    finally:
        pygame.quit()
    first = next(iter(summaries.values()))
    if any(episodes != first for episodes in summaries.values()):
        raise ValueError(
            "deployment rollouts differ between backends; performance comparison rejected"
        )
    report = {
        name: {
            "episodes": summaries[name],
            "timings": {key: statistics(values) for key, values in times.items()},
        }
        for name, times in raw.items()
    }
    arrays = {
        f"{name}_frames_{key}": np.asarray(values)
        for name, times in raw.items()
        for key, values in times.items()
    }
    return report, arrays


def benchmark_deployment(
    model: Path,
    library: Path,
    *,
    checkpoint: Path | None = None,
    warmup: int = 200,
    iterations: int = 5000,
    rounds: int = 3,
    startup_repeats: int = 5,
    episodes: int = 3,
    seed: int | None = None,
) -> tuple[dict, dict]:
    if (
        not 0 <= warmup <= 1000000
        or not 1 <= iterations <= 1000000
        or not 1 <= rounds <= 100
        or not 1 <= startup_repeats <= 100
        or not 0 <= episodes <= 100
        or (seed is not None and seed < 0)
    ):
        raise ValueError("invalid benchmark counts or seed")
    model, library = Path(model), Path(library)
    metadata = load_actor_metadata(model)
    observations, reference_logits, reference_actions = load_validation_data(model, metadata)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(model.resolve()),
        "model_sha256": file_hash(model),
        "library": str(library.resolve()),
        "library_sha256": file_hash(library),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "tinyinfer_execution_threads": 1,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "parameters": {
            "warmup": warmup,
            "iterations": iterations,
            "rounds": rounds,
            "startup_repeats": startup_repeats,
            "episodes": episodes,
            "seed": metadata["validation"]["seed"] if seed is None else seed,
        },
        "validation_samples": len(observations),
        "timings": {},
        "correctness": {},
        "notes": [
            "All timings are microseconds. Input selection is outside Python timers.",
            "First session creation includes CDLL/model/plan; subsequent creates share OS caches. Not cold disk I/O.",
            "Native run excludes input binding/output extraction. Native call includes them and nested timers, excludes ctypes.",
            "Python predict includes validation, allocation, ctypes, output extraction and argmax.",
            "Correctness calls precede explicit warmup. Timers are not subtracted; median differences are not overhead decompositions.",
            "Offscreen frames include predict/environment/renderer+flip; no HUD, FPS wait, trace hashing or asset load.",
            "PyTorch actor timing uses prepared tensors, includes PyTorch dispatch, excludes input conversion/output extraction.",
        ],
    }
    build_path = library.parent / "bridge_build.json"
    if build_path.is_file():
        build = json.loads(build_path.read_text())
        if build.get("library_sha256") == report["library_sha256"]:
            report["bridge_build"] = build
    policies = {}
    arrays = {}
    baseline = None
    try:
        for name, fused in (("tinyinfer_original", False), ("tinyinfer_fused", True)):
            begin = perf_counter_ns()
            policy = TinyInferPolicy(model, library, fuse_relu=fused)
            creates = [(perf_counter_ns() - begin) / 1000]
            policies[name] = policy
            begin = perf_counter_ns()
            policy.predict(observations[0])
            first_call = (perf_counter_ns() - begin) / 1000
            for _ in range(startup_repeats - 1):
                begin = perf_counter_ns()
                temporary = TinyInferPolicy(model, library, fuse_relu=fused)
                creates.append((perf_counter_ns() - begin) / 1000)
                temporary.close()
            report["timings"][name] = {
                "first_create_us": creates[0],
                "first_predict_us": first_call,
                "session_create": statistics(creates),
                "rounds": [],
            }
            actual = np.stack([policy.logits(row) for row in observations])
            np.testing.assert_allclose(actual, reference_logits, atol=1e-5, rtol=1e-5)
            if not np.array_equal(actual.argmax(axis=1), reference_actions):
                raise ValueError(f"{name} actions differ from reference; benchmark rejected")
            report["correctness"][name] = {
                "action_match_rate": 1.0,
                "max_abs_error": float(np.abs(actual - reference_logits).max()),
            }
            arrays[f"{name}_create_us"] = np.asarray(creates)
        if checkpoint is not None:
            import torch

            from .backends import load_backend

            torch.set_num_threads(1)
            if (
                file_hash(Path(checkpoint).with_suffix(".zip"))
                != metadata["source"]["model_sha256"]
            ):
                raise ValueError("SB3 comparison checkpoint differs from exported actor source")
            begin = perf_counter_ns()
            baseline = load_backend("sb3", checkpoint)
            report["sb3_load_us"] = (perf_counter_ns() - begin) / 1000
            report["torch"] = torch.__version__
            report["torch_threads"] = torch.get_num_threads()
            report["torch_interop_threads"] = torch.get_num_interop_threads()
            policies["sb3"] = baseline.policy
            actor = torch.nn.Sequential(
                baseline.policy.policy.mlp_extractor.policy_net,
                baseline.policy.policy.action_net,
            ).eval()
            tensors = [torch.from_numpy(row[None]) for row in observations]
            with torch.inference_mode():
                actual = np.concatenate([actor(value).numpy() for value in tensors])
            np.testing.assert_allclose(actual, reference_logits, atol=1e-5, rtol=1e-5)
            actions = np.asarray(
                [int(baseline.policy.predict(row, deterministic=True)[0]) for row in observations]
            )
            if not np.array_equal(actions, reference_actions):
                raise ValueError("SB3 comparison actions differ from references")
            report["correctness"]["sb3"] = {"action_match_rate": 1.0}
            report["timings"]["sb3"] = {"rounds": []}
            report["timings"]["pytorch_actor"] = {"rounds": []}
        collected = {}
        for round_index in range(rounds):
            order = list(policies)
            if checkpoint is not None:
                order.append("pytorch_actor")
            if round_index % 2:
                order.reverse()
            report.setdefault("round_order", []).append(order)
            for name in order:
                metrics = {}
                if name.startswith("tinyinfer"):
                    native = policies[name].benchmark_native(
                        observations, warmup=warmup, iterations=iterations
                    )
                    metrics["native_run_us"], metrics["native_call_us"] = (
                        native["run_us"],
                        native["native_us"],
                    )
                    metrics["python_logits_us"] = measure_calls(
                        policies[name].logits, observations, warmup=warmup, iterations=iterations
                    )
                if name == "pytorch_actor":
                    with torch.inference_mode():
                        metrics["tensor_forward_us"] = measure_calls(
                            actor, tensors, warmup=warmup, iterations=iterations
                        )
                else:
                    metrics["predict_us"] = measure_calls(
                        lambda row, policy=policies[name]: policy.predict(row, deterministic=True),
                        observations,
                        warmup=warmup,
                        iterations=iterations,
                    )
                report["timings"][name]["rounds"].append(
                    {key: statistics(values) for key, values in metrics.items()}
                )
                for key, values in metrics.items():
                    collected.setdefault((name, key), []).append(values)
        for (name, key), values in collected.items():
            array = np.concatenate(values)
            report["timings"][name][key] = statistics(array)
            arrays[f"{name}_{key}"] = array
        report["timer_empty_call"] = statistics(
            measure_calls(lambda _: None, observations, warmup=0, iterations=iterations)
        )
        if checkpoint is not None:
            sb3 = report["timings"]["sb3"]["predict_us"]["p50_us"]
            report["sb3_predict_median_ratio"] = {
                name: sb3 / report["timings"][name]["predict_us"]["p50_us"]
                for name in ("tinyinfer_original", "tinyinfer_fused")
            }
        if episodes:
            from ai_platformer.envs.factory import EnvironmentFactory, protocol_hash

            factory = EnvironmentFactory(metadata["environment"])
            previous = metadata["protocol"]
            if protocol_hash(factory.protocol(list(previous["levels"]))) != protocol_hash(previous):
                raise ValueError("deployment environment content/protocol has changed")
            candidates = metadata["validation"]["levels"]
            indices = np.linspace(0, len(candidates) - 1, min(episodes, len(candidates)), dtype=int)
            selected = [candidates[index] for index in indices]
            levels = [selected[index % len(selected)] for index in range(episodes)]
            frames, frame_arrays = measure_frames(
                policies, factory, levels, report["parameters"]["seed"]
            )
            report["frames"] = frames
            report["trajectory_match"] = True
            arrays.update(frame_arrays)
        report["passed"] = True
        return report, arrays
    finally:
        for policy in policies.values():
            close = getattr(policy, "close", None)
            if close is not None:
                close()
        if baseline is not None:
            baseline.close()
