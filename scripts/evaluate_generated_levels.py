"""Evaluate a frozen prompt/seed suite through LLM, PCG, route search and PPO."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
from pathlib import Path

from ai_platformer.agents.llm.evaluation import run_suite
from ai_platformer.content.generation_suite import load_suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", type=Path, default=Path("game_content/generation_suites/small_v1.json")
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("runs/full_collection_v9/seed_20261004/model.zip")
    )
    parser.add_argument("--provider", choices=("gemini", "expected-fixture"), default="gemini")
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--tls-mode", choices=("default", "tls12"), default="tls12")
    parser.add_argument("--max-attempts", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--route-seconds", type=float, default=15)
    parser.add_argument("--route-expansions", type=int, default=25000)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, help="first N prompt/seed cases for a smoke run")
    parser.add_argument(
        "--plan", action="store_true", help="show budget without API or checkpoint loading"
    )
    args = parser.parse_args()
    if not 0 < args.timeout <= 60 or not 0 < args.route_seconds <= 300:
        parser.error("timeout must be in (0,60], route-seconds in (0,300]")
    if not 1 <= args.route_expansions <= 1_000_000 or (
        args.limit is not None and not 1 <= args.limit <= 64
    ):
        parser.error("route-expansions must be 1..1000000, limit 1..64")
    if not re.fullmatch(r"gemini-[A-Za-z0-9._-]{1,100}", args.model):
        parser.error("invalid Gemini model ID")
    try:
        suite = load_suite(args.suite)
    except (OSError, ValueError, RecursionError) as error:
        parser.error(str(error))
    planned = min(len(suite.cases) * len(suite.generation_seeds), args.limit or 64)
    if args.plan:
        print(
            json.dumps(
                {
                    "suite": suite.suite_id,
                    "provider": args.provider,
                    "prompts": len(suite.cases),
                    "generation_seeds": suite.generation_seeds,
                    "policy_seed": suite.policy_seed,
                    "planned_candidates": planned,
                    "maximum_api_attempts_this_invocation": planned * args.max_attempts
                    if args.provider == "gemini"
                    else 0,
                    "maximum_output_tokens_per_attempt": 4096,
                    "purpose": suite.purpose,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.output_dir is None:
        parser.error("--output-dir is required unless --plan is selected")
    if args.provider == "gemini" and not (
        os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    ):
        parser.error(
            "API Key missing in this terminal; export GEMINI_API_KEY (offline mode: --provider expected-fixture)"
        )
    from ai_platformer.deployment.backends import load_backend

    loaded = None
    try:
        loaded = load_backend("sb3", args.checkpoint)
        settings = {
            "provider": args.provider,
            "model": args.model if args.provider == "gemini" else "offline-expected-fixture",
            "tls_mode": args.tls_mode,
            "timeout": args.timeout,
            "max_attempts": args.max_attempts,
            "route_seconds": args.route_seconds,
            "route_expansions": args.route_expansions,
            "cache_dir": str((args.cache_dir or args.output_dir / "cache").resolve()),
        }
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        lock_path = args.output_dir.parent / f".{args.output_dir.name}.lock"
        with lock_path.open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                parser.error("another process is using this output directory")
            print(
                f"provider={args.provider}; candidates={planned}; maximum API attempts={planned * args.max_attempts if args.provider == 'gemini' else 0}",
                flush=True,
            )
            report = run_suite(
                suite,
                output=args.output_dir,
                loaded=loaded,
                settings=settings,
                resume=args.resume,
                limit=args.limit,
            )
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as error:
        parser.error(str(error))
    finally:
        if loaded is not None:
            loaded.close()
    if report["status"] != "complete":
        parser.exit(1)


if __name__ == "__main__":
    main()
