"""Find and independently replay a bounded route through a LevelSpec candidate."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ai_platformer.content.routes import search_route
from ai_platformer.envs.factory import EnvironmentFactory
from ai_platformer.settings import AI_GAMEPLAY_PATH, load_gameplay_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--settings", type=Path, default=AI_GAMEPLAY_PATH)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=768)
    parser.add_argument("--max-expansions", type=int, default=25000)
    parser.add_argument("--beam-width", type=int, default=24)
    parser.add_argument("--max-seconds", type=float, default=15.0)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--min-coin-ratio", type=float, default=0.0)
    parser.add_argument("--no-scripted-probes", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.exit(2, "output exists; choose a new path\n")
    try:
        factory = EnvironmentFactory(
            {
                "environment_id": "PlatformerState-v2",
                "level_spec": str(args.input),
                "episode_step_limit": args.max_steps,
                "action_repeat": args.action_repeat,
                "physics": asdict(load_gameplay_settings(args.settings).physics),
            }
        )
        report = search_route(
            factory,
            seed=args.seed,
            max_steps=args.max_steps,
            max_expansions=args.max_expansions,
            beam_width=args.beam_width,
            max_seconds=args.max_seconds,
            min_coin_ratio=args.min_coin_ratio,
            try_scripted=not args.no_scripted_probes,
        )
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        report = {"schema_version": 1, "status": "invalid", "reason": str(error), "witness": None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"route={report['status']}; reason={report['reason']}")
    if report["status"] != "verified":
        parser.exit(1)


if __name__ == "__main__":
    main()
