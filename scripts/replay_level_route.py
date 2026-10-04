"""Verify a route report against fresh simulation and the original environment."""

import argparse
import json
from pathlib import Path

from ai_platformer.content.level_spec import parse_level_json
from ai_platformer.content.routes import replay_route


def load_witness(path):
    with Path(path).open("rb") as stream:
        data = stream.read(8_000_001)
    if len(data) > 8_000_000:
        raise ValueError("route input exceeds byte limit")
    report = parse_level_json(data)
    if not isinstance(report, dict):
        raise TypeError("route report must be an object")
    return report.get("witness", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.exit(2, "output exists; choose a new path\n")
    try:
        report = replay_route(args.input, load_witness(args.route))
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        report = {"schema_version": 1, "passed": False, "error": str(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"replay={report['passed']}")
    if not report["passed"]:
        parser.exit(1)


if __name__ == "__main__":
    main()
