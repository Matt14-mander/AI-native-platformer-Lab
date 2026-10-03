"""Compare both TinyInfer graph paths with exported PyTorch reference logits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ai_platformer.deployment.package import file_hash, load_actor_metadata
from ai_platformer.deployment.tinyinfer import TinyInferPolicy


def validate(model: Path, library: Path) -> dict:
    metadata = load_actor_metadata(model)
    reference_path = model.parent / "validation.npz"
    if file_hash(reference_path) != metadata["validation"]["sha256"]:
        raise ValueError("validation references do not match deployment metadata")
    with np.load(reference_path, allow_pickle=False) as reference:
        observations = reference["observations"]
        logits = reference["logits"]
        actions = reference["actions"]
    if (
        observations.dtype != np.float32
        or observations.shape != (metadata["validation"]["samples"], metadata["input"]["shape"][1])
        or logits.shape != (len(observations), metadata["output"]["shape"][1])
        or actions.shape != (len(observations),)
        or not np.isfinite(observations).all()
        or not np.isfinite(logits).all()
    ):
        raise ValueError("invalid validation tensor shapes/types/values")
    sorted_logits = np.sort(logits, axis=1)
    margins = sorted_logits[:, -1] - sorted_logits[:, -2]
    report = {
        "model": str(model.resolve()),
        "model_sha256": file_hash(model),
        "library": str(library.resolve()),
        "library_sha256": file_hash(library),
        "samples": len(observations),
        "atol": 1e-5,
        "rtol": 1e-5,
        "minimum_action_margin": float(margins.min()),
        "paths": {},
        "note": "Fixed observations verify numerical/action parity; this is not a performance benchmark.",
    }
    for fused in (False, True):
        with TinyInferPolicy(model, library, fuse_relu=fused) as policy:
            actual = np.stack([policy.logits(row) for row in observations])
        differences = np.abs(actual - logits)
        numerical = np.isclose(actual, logits, atol=1e-5, rtol=1e-5)
        matches = actual.argmax(axis=1) == actions
        report["paths"]["fused_relu" if fused else "original"] = {
            "max_abs_error": float(differences.max()),
            "mean_abs_error": float(differences.mean()),
            "numerical_pass": bool(numerical.all()),
            "action_match_rate": float(matches.mean()),
            "action_mismatch_indices": np.flatnonzero(~matches).tolist(),
            "numerical_mismatch_samples": np.flatnonzero(~numerical.all(axis=1)).tolist(),
        }
    report["passed"] = all(
        path["numerical_pass"] and path["action_match_rate"] == 1.0
        for path in report["paths"].values()
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="actor.onnx with actor.json")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise FileExistsError("validation report already exists; choose a new path")
        report = validate(args.model, args.library)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"TinyInfer validation failed: {error}\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        parser.exit(1, "TinyInfer numerical/action validation did not pass\n")


if __name__ == "__main__":
    main()
