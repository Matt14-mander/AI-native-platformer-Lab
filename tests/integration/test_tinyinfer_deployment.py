"""Real ONNX export and native bridge numerical/lifecycle validation."""

import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
torch = pytest.importorskip("torch")
PPO = pytest.importorskip("stable_baselines3").PPO

from ai_platformer.core import Action
from ai_platformer.deployment.export import export_actor
from ai_platformer.deployment.package import file_hash, load_actor_metadata, metadata_hash
from ai_platformer.deployment.tinyinfer import TinyInferPolicy
from ai_platformer.envs.factory import EnvironmentFactory, protocol_hash
from scripts.validate_tinyinfer import validate


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    root = tmp_path_factory.mktemp("actor")
    environment = {
        "environment_id": "PlatformerState-v1",
        "curriculum_manifest": "config/curriculum_v4.json",
        "episode_step_limit": 16,
        "action_repeat": 4,
    }
    factory = EnvironmentFactory(environment)
    levels = factory.repository.split("gap", "validation")
    env = factory.make()
    model = PPO(
        "MlpPolicy",
        env,
        seed=42,
        n_steps=16,
        batch_size=16,
        policy_kwargs={
            "activation_fn": torch.nn.ReLU,
            "net_arch": {"pi": [64, 64], "vf": [64, 64]},
        },
        device="cpu",
    )
    checkpoint = root / "model.zip"
    model.save(checkpoint)
    env.close()
    signature = {
        "protocol": factory.protocol(levels),
        "action_ids": {action.name: int(action) for action in Action},
    }
    sidecar = {
        "schema_version": 2,
        "signature": signature,
        "signature_hash": protocol_hash(signature),
        "model_sha256": file_hash(checkpoint),
        "saved_timesteps": 0,
        "config": {"environment": environment, "evaluation": {"seeds": [123]}},
    }
    checkpoint.with_suffix(".json").write_text(json.dumps(sidecar))
    output = root / "deployment"
    export_actor(checkpoint, output, samples=128, seed=123)
    return checkpoint, output / "actor.onnx"


@pytest.fixture(scope="module")
def library():
    explicit = os.environ.get("PLATFORMER_TINYINFER_LIBRARY")
    if explicit:
        return Path(explicit)
    root = Path(__file__).resolve().parents[2]
    paths = [
        path
        for path in (root / "build/tinyinfer_bridge").rglob("*platformer_tinyinfer*")
        if path.suffix in {".dylib", ".so", ".dll"}
    ]
    if len(paths) != 1:
        pytest.skip("build the TinyInfer bridge or set PLATFORMER_TINYINFER_LIBRARY")
    return paths[0]


def test_export_package_has_only_actor_and_checked_references(exported):
    checkpoint, path = exported
    metadata = load_actor_metadata(path)
    assert metadata["onnx"]["operators"] == ["Gemm", "Relu", "Gemm", "Relu", "Gemm"]
    assert metadata["onnx"]["constant_aliases_materialized"] >= 1
    assert metadata["input"]["shape"] == [1, 15]
    assert metadata["output"]["shape"] == [1, 10]
    assert metadata["validation"]["onnxruntime_action_match_rate"] == 1
    assert metadata["validation"]["real_observations"] > 0
    factory = EnvironmentFactory(metadata["environment"])
    assert set(factory.repository.split("mixed", "validation")) & set(
        metadata["validation"]["levels"]
    )
    with pytest.raises(FileExistsError):
        export_actor(checkpoint, path.parent)


def test_episode_regression_compares_backends_even_when_policy_fails(exported, library):
    from scripts.regress_deployment import regress

    checkpoint, actor = exported
    report = regress(checkpoint, actor, library, seeds=[123])
    assert report["mismatch"] is None
    assert report["unchanged"]
    assert len(report["episodes"]) == len(report["levels"])
    assert all(ep["steps"] <= 16 for ep in report["episodes"])
    assert report["passed"] == all(ep["outcome"] == "success" for ep in report["episodes"])
    with pytest.raises(ValueError, match="distinct nonnegative"):
        regress(checkpoint, actor, library, seeds=[123, 123])


def test_v9_regression_covers_full_suites_without_generated_train_maps():
    from scripts.regress_deployment import regression_levels

    factory = EnvironmentFactory(
        {"curriculum_manifest": "config/curriculum_v9.json", "level_id": "level_1_main"}
    )
    levels = regression_levels(factory)
    assert len(levels) == len(set(levels)) == 47
    assert levels[0] == "level_1_main"
    for suite in ("validation", "test", "ood"):
        assert set(factory.repository.split("full", suite)) <= set(levels)
    assert not (set(factory.repository.split("full", "train")) - {"level_1_main"}) & set(levels)


def test_both_native_paths_match_pytorch_references(exported, library):
    _, path = exported
    report = validate(path, library)
    assert report["passed"]
    assert set(report["paths"]) == {"original", "fused_relu"}


def test_reused_contexts_copy_outputs_and_handle_invalid_inputs(exported, library):
    _, path = exported
    with np.load(path.parent / "validation.npz") as data:
        observations = data["observations"][:4].copy()
        expected = data["logits"][:4].copy()
    first = TinyInferPolicy(path, library)
    second = TinyInferPolicy(path, library, fuse_relu=True)
    try:
        saved = first.logits(observations[0])
        for index in (1, 3, 2, 0):
            np.testing.assert_allclose(
                first.logits(observations[index]), expected[index], atol=1e-5, rtol=1e-5
            )
            np.testing.assert_allclose(
                second.logits(observations[index]), expected[index], atol=1e-5, rtol=1e-5
            )
        np.testing.assert_array_equal(saved, first.logits(observations[0]))
        with pytest.raises(ValueError, match="float32"):
            first.logits(observations[0].astype(np.float64))
        with pytest.raises(ValueError, match="float32"):
            first.logits(observations[0][:-1])
        with pytest.raises(ValueError, match="finite"):
            first.logits(np.full(15, np.nan, dtype=np.float32))
        with pytest.raises(ValueError, match="deterministic"):
            first.predict(observations[0], deterministic=False)
        pointer = ctypes.POINTER(ctypes.c_float)
        output = np.empty(10, dtype=np.float32)
        status = first._library.pti_infer(
            first._handle,
            observations[0].ctypes.data_as(pointer),
            14,
            output.ctypes.data_as(pointer),
            10,
        )
        assert status == -1
        assert "counts" in first._error()
        # An ABI error must not poison the persistent execution context.
        np.testing.assert_array_equal(first.logits(observations[0]), saved)
        first.close()
        first.close()
        with pytest.raises(RuntimeError, match="closed"):
            first.logits(observations[0])
        np.testing.assert_allclose(
            second.logits(observations[1]), expected[1], atol=1e-5, rtol=1e-5
        )
    finally:
        first.close()
        second.close()


def test_corrupt_model_and_wrong_named_binding_are_rejected(exported, library, tmp_path):
    _, path = exported
    clone = tmp_path / "actor.onnx"
    clone.write_bytes(path.read_bytes() + b"corrupt")
    metadata = load_actor_metadata(path)
    clone.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="model does not match"):
        TinyInferPolicy(clone, library)
    clone.write_bytes(path.read_bytes())
    metadata["input"]["name"] = "missing_observations"
    metadata["metadata_sha256"] = metadata_hash(
        {key: value for key, value in metadata.items() if key != "metadata_sha256"}
    )
    clone.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError):
        TinyInferPolicy(clone, library)


def test_adapter_infers_without_importing_training_or_onnx_frameworks(exported, library):
    _, path = exported
    code = """
import sys
from pathlib import Path
import numpy as np
from ai_platformer.deployment.tinyinfer import TinyInferPolicy
with TinyInferPolicy(Path(sys.argv[1]), Path(sys.argv[2])) as policy:
    policy.predict(np.zeros(15, dtype=np.float32))
assert not any(name in sys.modules for name in ('torch', 'stable_baselines3', 'onnx', 'onnxruntime'))
"""
    subprocess.run([sys.executable, "-c", code, str(path), str(library)], check=True)


def test_native_benchmark_measures_run_and_io_and_preserves_context(exported, library):
    _, path = exported
    with np.load(path.parent / "validation.npz") as data:
        observations = data["observations"][:5].copy()
        expected = data["logits"][0].copy()
    with TinyInferPolicy(path, library) as policy:
        metrics = policy.benchmark_native(observations, warmup=4, iterations=16)
        assert metrics["run_us"].shape == (16,)
        assert np.isfinite(metrics["native_us"]).all()
        assert (metrics["run_us"] >= 0).all()
        assert (metrics["native_us"] >= metrics["run_us"]).all()
        np.testing.assert_allclose(policy.logits(observations[0]), expected, atol=1e-5, rtol=1e-5)
        with pytest.raises(ValueError):
            policy.benchmark_native(observations, warmup=0, iterations=0)
    with pytest.raises(RuntimeError, match="closed"):
        policy.benchmark_native(observations, warmup=0, iterations=1)


def test_tinyinfer_playback_runs_without_training_imports(exported, library, tmp_path):
    _, path = exported
    code = """
import sys, runpy
model, library, screenshot = sys.argv[1:]
sys.argv = ['play_ppo', '--backend', 'tinyinfer', '--model', model, '--library', library,
            '--fuse-relu', '--task', 'gap', '--headless', '--episodes', '1', '--screenshot', screenshot]
runpy.run_module('scripts.play_ppo', run_name='__main__')
assert not any(name in sys.modules for name in ('torch', 'stable_baselines3', 'onnx', 'onnxruntime'))
"""
    screenshot = tmp_path / "playback.png"
    subprocess.run(
        [sys.executable, "-c", code, str(path), str(library), str(screenshot)], check=True
    )
    assert screenshot.stat().st_size > 1000


def test_playback_rejects_changed_environment_and_missing_library(exported, library, tmp_path):
    from ai_platformer.deployment.backends import load_backend

    _, path = exported
    with pytest.raises(ValueError, match="requires --library"):
        load_backend("tinyinfer", path)
    clone = tmp_path / "actor.onnx"
    clone.write_bytes(path.read_bytes())
    metadata = load_actor_metadata(path)
    metadata["environment"]["action_repeat"] = 3
    metadata["metadata_sha256"] = metadata_hash(
        {key: value for key, value in metadata.items() if key != "metadata_sha256"}
    )
    clone.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="content/protocol"):
        load_backend("tinyinfer", clone, library=library)


def test_benchmark_validates_actions_traces_and_counts(exported, library):
    from ai_platformer.deployment.performance import benchmark_deployment

    checkpoint, path = exported
    report, samples = benchmark_deployment(
        path,
        library,
        checkpoint=checkpoint,
        warmup=2,
        iterations=8,
        rounds=2,
        startup_repeats=2,
        episodes=1,
    )
    assert report["passed"] and report["trajectory_match"]
    assert set(report["correctness"]) == {"tinyinfer_original", "tinyinfer_fused", "sb3"}
    for name in ("tinyinfer_original", "tinyinfer_fused", "sb3"):
        assert report["timings"][name]["predict_us"]["count"] == 16
        assert len(report["frames"][name]["episodes"]) == 1
        assert samples[f"{name}_predict_us"].shape == (16,)
    assert report["timings"]["pytorch_actor"]["tensor_forward_us"]["count"] == 16
    assert report["round_order"][0] == list(reversed(report["round_order"][1]))
