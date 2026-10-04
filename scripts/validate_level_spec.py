"""Validate LevelSpec structure and geometry, with machine-readable diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_platformer.content.level_spec import parse_level_json
from ai_platformer.content.level_validation import validate_level_spec
from ai_platformer.settings import AI_GAMEPLAY_PATH, load_gameplay_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--settings", type=Path, default=AI_GAMEPLAY_PATH)
    args = parser.parse_args()
    if args.output.exists():
        parser.exit(2, "output exists; choose a new path\n")
    try:
        with args.input.open("rb") as stream:
            raw = stream.read(8_000_001)
        if len(raw) > 8_000_000:
            raise ValueError("LevelSpec input exceeds byte limit")
        data = parse_level_json(raw)
        report = validate_level_spec(data, physics=load_gameplay_settings(args.settings).physics)
    except (OSError, ValueError, RecursionError) as error:
        report = {
            "schema_version": 1,
            "structure_valid": False,
            "static_valid": False,
            "reachability": "not_checked",
            "issues": [
                {"code": "input_error", "path": "/", "severity": "error", "message": str(error)}
            ],
        }
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    except OSError as error:
        parser.exit(2, f"cannot write report: {error}\n")
    print(
        f"structure={report['structure_valid']}; static={report['static_valid']}; reachability=not_checked"
    )
    if not report["static_valid"]:
        parser.exit(1)


if __name__ == "__main__":
    main()
