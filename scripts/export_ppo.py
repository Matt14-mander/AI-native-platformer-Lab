"""Export a PPO actor to a checked, static FP32 ONNX deployment package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()
    try:
        from ai_platformer.deployment.export import export_actor

        report = export_actor(args.model, args.output_dir, samples=args.samples, seed=args.seed)
    except (OSError, ValueError, KeyError, AssertionError, ImportError) as error:
        parser.exit(1, f"actor export failed: {error}\n")
    print(
        json.dumps({"output_dir": str(args.output_dir.resolve()), **report["validation"]}, indent=2)
    )


if __name__ == "__main__":
    main()
