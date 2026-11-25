"""Select on development, then open the frozen holdout once for final comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO

from ai_platformer.agents.ppo.pcg_experiment import evaluate_model, file_hash
from ai_platformer.benchmark.stratified import evaluate_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-run", type=Path, required=True)
    parser.add_argument("--ppo-run", type=Path, required=True)
    parser.add_argument("--sampling-run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("game_content/benchmark_suites/generalization_v2.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--open-holdout", action="store_true")
    args = parser.parse_args()
    if not args.open_holdout:
        parser.error("final evaluation requires --open-holdout")
    if args.output.exists():
        parser.error("output exists; preserve the first holdout result")
    base = json.loads((args.bc_run / "report.json").read_text(encoding="utf-8"))
    if base["model_sha256"] != file_hash(args.bc_run / "model.zip"):
        parser.error("BC checkpoint changed")
    candidates = []
    source_hashes = {}
    for directory in (args.ppo_run, args.sampling_run):
        report_path = directory / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["source_bc_sha256"] != base["model_sha256"]:
            parser.error("candidate source differs from BC checkpoint")
        if report["manifest_sha256"] != file_hash(args.manifest):
            parser.error("candidate manifest differs")
        if len(report["runs"]) != 2 * len(report["training_seeds"]):
            parser.error("candidate comparison is incomplete")
        source_hashes[str(report_path)] = file_hash(report_path)
        for row in report["runs"]:
            stem = (row.get("method") or row.get("mode")) + f"_seed_{row['training_seed']}"
            path = directory / f"{stem}.zip"
            if row["model_sha256"] != file_hash(path):
                parser.error("candidate checkpoint changed")
            dev = row["development"]
            score = (dev["targets_met"], dev["successes"], dev["mean_coin_ratio"])
            candidates.append({"path": str(path), "model_sha256": row["model_sha256"],
                               "method": row.get("method") or row.get("mode"),
                               "training_seed": row["training_seed"], "development_score": score})
    selected = max(candidates, key=lambda row: tuple(row["development_score"]))
    bc = PPO.load(args.bc_run / "model.zip", device="cpu")
    winner = PPO.load(selected["path"], device="cpu")
    bc_holdout = evaluate_model(bc, args.manifest, split="holdout", open_holdout=True)
    winner_holdout = evaluate_model(winner, args.manifest, split="holdout", open_holdout=True)
    scripted = evaluate_manifest(args.manifest, split="holdout")["by_agent"]["ground_coin_jump"]
    result = {"schema_version": 1, "selection_rule": "max (development targets, goals, mean coin ratio)",
              "selected": selected, "candidate_count": len(candidates),
              "bc_model_sha256": base["model_sha256"], "source_report_hashes": source_hashes,
              "manifest_sha256": file_hash(args.manifest),
              "holdout": {"bc": bc_holdout, "selected_ppo": winner_holdout,
                          "scripted_teacher": scripted},
              "scope": "One frozen 12-map holdout evaluation after development selection; no further tuning on holdout."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": selected, "bc_targets": bc_holdout["targets_met"],
                      "ppo_targets": winner_holdout["targets_met"],
                      "teacher_targets": scripted["overall"]["collection_targets_met"]}, indent=2))


if __name__ == "__main__":
    main()
