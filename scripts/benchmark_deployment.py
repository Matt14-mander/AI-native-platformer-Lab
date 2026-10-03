"""Measure TinyInfer deployment latency and optionally compare the source SB3 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ai_platformer.deployment.package import file_hash
from ai_platformer.deployment.performance import benchmark_deployment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, help="optional matching SB3 checkpoint for comparison"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--startup-repeats", type=int, default=5)
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="offscreen episodes per backend; 0 disables rendering",
    )
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    raw_path = args.output.with_suffix(".samples.npz")
    try:
        if args.output.suffix.lower() != ".json":
            raise ValueError("output must have a .json suffix")
        if args.output.exists() or raw_path.exists():
            raise FileExistsError("benchmark output already exists; choose a new path")
        report, arrays = benchmark_deployment(
            args.model,
            args.library,
            checkpoint=args.checkpoint,
            warmup=args.warmup,
            iterations=args.iterations,
            rounds=args.rounds,
            startup_repeats=args.startup_repeats,
            episodes=args.episodes,
            seed=args.seed,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(raw_path, **arrays)
        report["raw_samples"] = {"file": str(raw_path.resolve()), "sha256": file_hash(raw_path)}
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, RuntimeError, AssertionError, ImportError) as error:
        parser.exit(1, f"deployment benchmark failed: {error}\n")
    for name, metrics in report["timings"].items():
        key = "predict_us" if "predict_us" in metrics else "tensor_forward_us"
        summary = metrics[key]
        print(f"{name}: median={summary['p50_us']:.2f} us, P95={summary['p95_us']:.2f} us")
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
