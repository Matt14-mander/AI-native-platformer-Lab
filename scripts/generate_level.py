"""Produce an atomic PCG candidate package, optionally with a verified route."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from pydantic import ValidationError

from ai_platformer.content.generation_request import load_generation_request
from ai_platformer.content.pcg import generate_level


def write(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def populate_package(result, folder, *, verify_route=False, max_seconds=15.0, max_expansions=25000):
    write(folder / "request.json", result.request.model_dump())
    write(folder / "level.json", result.level.model_dump())
    report = dict(result.report)
    if verify_route:
        from ai_platformer.content.routes import search_route
        from ai_platformer.envs.factory import EnvironmentFactory

        factory = EnvironmentFactory(
            {
                "environment_id": "PlatformerState-v2",
                "level_spec": str(folder / "level.json"),
                "physics": asdict(result.physics),
                "action_repeat": 4,
                "episode_step_limit": 768,
            }
        )
        route = search_route(
            factory,
            seed=100,
            max_seconds=max_seconds,
            max_expansions=max_expansions,
            min_coin_ratio=result.request.min_coin_ratio,
        )
        write(folder / "route.json", route)
        report["status"] = route["status"]
        report["note"] = (
            "Only verified means a collection-target route was independently replayed. No training split assignment."
        )
        report["route"] = {
            "file": "route.json",
            "status": route["status"],
            "reason": route["reason"],
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-route", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=15.0)
    parser.add_argument("--max-expansions", type=int, default=25000)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.exit(2, "output directory exists; choose a new path\n")
    try:
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.output_dir.parent, prefix=".pcg-") as temp:
            folder = Path(temp) / "package"
            folder.mkdir()
            try:
                result = generate_level(load_generation_request(args.request))
                report = populate_package(
                    result,
                    folder,
                    verify_route=args.verify_route,
                    max_seconds=args.max_seconds,
                    max_expansions=args.max_expansions,
                )
            except ValidationError as error:
                report = {
                    "schema_version": 1,
                    "status": "invalid",
                    "issues": error.errors(
                        include_input=False, include_context=False, include_url=False
                    ),
                }
            except (OSError, ValueError, RuntimeError, ImportError, RecursionError) as error:
                report = {"schema_version": 1, "status": "error", "reason": str(error)}
            write(folder / "report.json", report)
            folder.rename(args.output_dir)
    except OSError as error:
        parser.exit(2, f"cannot publish candidate package: {error}\n")
    print(f"generation={report['status']}; package={args.output_dir}")
    if report["status"] not in {"candidate", "verified"}:
        parser.exit(1)


if __name__ == "__main__":
    main()
