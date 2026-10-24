"""Validate a local telemetry session and inspect its attempt ledger (no profiling)."""

import argparse
import json
from collections import Counter
from pathlib import Path

from ai_platformer.adaptive.telemetry import read_session
from ai_platformer.adaptive.telemetry.storage import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="optional new JSON report outside the session")
    args = parser.parse_args()
    if args.output and (
        args.output.exists() or args.output.resolve().is_relative_to(args.input.resolve())
    ):
        parser.error("output must be a new file outside the source session")
    try:
        session = read_session(args.input)
        report = {
            key: value for key, value in session.items() if key not in {"events", "manifest"}
        }
        report.update(
            session_id=session["manifest"]["session_id"],
            actor_type=session["manifest"]["actor_type"],
            identity_scope=session["manifest"]["identity_scope"],
            event_counts=dict(Counter(event["type"] for event in session["events"])),
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(args.output, report)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"cannot inspect telemetry: {error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
