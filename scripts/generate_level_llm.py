"""Translate natural-language intent, then publish a validated PCG candidate."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from pydantic import ValidationError

from ai_platformer.agents.llm.generation import GenerationError, translate_intent
from ai_platformer.content.pcg import generate_level
from scripts.generate_level import populate_package, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--level-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--provider", choices=("gemini", "fixture"), default="gemini")
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL"))
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--verify-route", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--max-seconds", type=float, default=15)
    parser.add_argument("--max-expansions", type=int, default=25000)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.exit(2, "output directory exists; choose a new path\n")
    try:
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.output_dir.parent, prefix=".llm-") as temp:
            folder = Path(temp) / "package"
            folder.mkdir()
            write(
                folder / "intent.json",
                {
                    "schema_version": 1,
                    "prompt": args.prompt,
                    "provider": args.provider,
                    "model": args.model,
                    "level_id": args.level_id,
                    "seed": args.seed,
                },
            )
            try:
                request, record = translate_intent(
                    args.prompt,
                    level_id=args.level_id,
                    seed=args.seed,
                    provider=args.provider,
                    model=args.model,
                    fixture=args.fixture,
                    cache_dir=args.cache_dir,
                    max_attempts=args.max_attempts,
                    timeout=args.timeout,
                )
                write(folder / "llm.json", record)
                report = populate_package(
                    generate_level(request),
                    folder,
                    verify_route=args.verify_route,
                    max_seconds=args.max_seconds,
                    max_expansions=args.max_expansions,
                )
                report["llm"] = {
                    "file": "llm.json",
                    "provider": args.provider,
                    "cache_hit": record["cache_hit"],
                }
            except GenerationError as error:
                write(folder / "llm.json", {"attempts": error.attempts})
                report = {"schema_version": 1, "status": "error", "reason": str(error)}
            except ValidationError as error:
                report = {
                    "schema_version": 1,
                    "status": "invalid",
                    "issues": error.errors(
                        include_input=False, include_context=False, include_url=False
                    ),
                }
            except (OSError, ValueError, RuntimeError, ImportError, RecursionError):
                report = {
                    "schema_version": 1,
                    "status": "error",
                    "reason": "invalid configuration, input, or pipeline failure",
                }
            write(folder / "report.json", report)
            folder.rename(args.output_dir)
    except OSError:
        parser.exit(2, "cannot publish candidate package\n")
    print(f"generation={report['status']}; package={args.output_dir}")
    if report["status"] not in {"candidate", "verified"}:
        parser.exit(1)


if __name__ == "__main__":
    main()
