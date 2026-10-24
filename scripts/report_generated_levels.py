"""Consolidate existing generation batches and optional single-map PPO evidence."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from ai_platformer.agents.llm.evaluation import atomic_json, evaluate_ppo, sha256_file
from ai_platformer.agents.llm.reporting import build_report, load_batch, load_single, markdown
from ai_platformer.content.generation_request import load_generation_request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, nargs="+", default=[])
    parser.add_argument("--single-package", type=Path, nargs="+", default=[])
    parser.add_argument(
        "--single-checkpoint", type=Path, help="optional fresh CPU PPO inference on single packages"
    )
    parser.add_argument("--single-seed", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    inputs = [path.resolve() for path in args.batch_dir + args.single_package]
    if not 1 <= len(inputs) <= 32 or len(set(inputs)) != len(inputs):
        parser.error("provide 1..32 distinct input directories")
    if args.single_checkpoint and not args.single_package:
        parser.error("--single-checkpoint requires --single-package")
    if not 0 <= args.single_seed <= 2**32 - 1:
        parser.error("single-seed must be uint32")
    output = args.output_dir.resolve()
    if output.exists():
        parser.error("output exists; choose a new directory")
    if any(output.is_relative_to(path) for path in inputs):
        parser.error("report output must be outside the read-only source directories")
    loaded = None
    try:
        sources = [load_batch(path) for path in args.batch_dir]
        # Validate all source packages before any optional inference.
        singles = [load_single(path, policy_seed=args.single_seed) for path in args.single_package]
        if args.single_checkpoint:
            from ai_platformer.deployment.backends import load_backend

            loaded = load_backend("sb3", args.single_checkpoint)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output.parent, prefix=".unified-report-") as temp:
            folder = Path(temp) / "package"
            folder.mkdir()
            evaluations = {}
            for index, (path, source) in enumerate(zip(args.single_package, singles, strict=True)):
                if loaded and (path / "request.json").exists():
                    request = load_generation_request(path / "request.json")
                    factory = loaded.factory.for_level_spec(path / "level.json")
                    summary, trace = evaluate_ppo(
                        loaded,
                        factory,
                        seed=args.single_seed,
                        request_target=request.min_coin_ratio,
                        expected_target=request.min_coin_ratio,
                    )
                    filename = f"single_{index:02d}_ppo.json"
                    atomic_json(folder / filename, {**summary, **trace})
                    evaluations[filename] = sha256_file(folder / filename)
                    source = load_single(
                        path,
                        ppo={**summary, **trace},
                        checkpoint_sha256=sha256_file(loaded.model_path),
                        policy_seed=args.single_seed,
                    )
                    source["fresh_ppo_artifact"] = filename
                    source["checkpoint_path"] = str(loaded.model_path.resolve())
                sources.append(source)
            report = build_report(sources)
            report["output_artifact_sha256"] = evaluations
            atomic_json(folder / "report.json", report)
            (folder / "report.md").write_text(markdown(report), encoding="utf-8")
            folder.rename(output)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as error:
        parser.exit(2, f"cannot consolidate reports: {error}\n")
    finally:
        if loaded is not None:
            loaded.close()
    print(
        json.dumps(
            {
                "output": str(output),
                **report["evidence_counts"],
                "failure_events": len(report["failure_catalog"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
