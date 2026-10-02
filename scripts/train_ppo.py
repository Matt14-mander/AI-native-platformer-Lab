"""Train or resume a reproducible PPO baseline/curriculum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_platformer.agents.ppo import train_ppo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/ppo_state_v0.json"),
    )
    parser.add_argument("--timesteps", type=int, help="override total training timesteps")
    parser.add_argument(
        "--resume", type=Path, help="v2 checkpoint to continue in a new run directory"
    )
    parser.add_argument("--eval-every", type=int, help="evaluation interval in transitions")
    parser.add_argument(
        "--start-stage",
        choices=("flat", "obstacle", "gap", "mixed", "full"),
        help="explicitly select a stage on resume; skipped stages remain unmastered",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/ppo_state_v0"),
    )
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if args.timesteps is not None:
        if args.timesteps <= 0:
            parser.error("--timesteps must be positive")
        config["total_timesteps"] = args.timesteps
    if args.eval_every is not None:
        if args.eval_every <= 0:
            parser.error("--eval-every must be positive")
        config["evaluation"]["every_timesteps"] = args.eval_every
    try:
        report = train_ppo(
            config, args.output_dir, resume=args.resume, start_stage=args.start_stage
        )
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps(report["evaluation"]["summary"], ensure_ascii=False, indent=2))
    print(f"model: {report['artifacts']['model']}")
    print(f"metadata: {(args.output_dir / 'run.json').resolve()}")
    print(f"status: {report['curriculum_state']['status']}")


if __name__ == "__main__":
    main()
