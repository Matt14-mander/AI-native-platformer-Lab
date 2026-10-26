"""Rebuild versioned behavior metrics and human profiles from local event sessions."""

import argparse
import json
import tempfile
from pathlib import Path

from ai_platformer.adaptive.reporting import build_report, markdown
from ai_platformer.adaptive.telemetry.storage import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True,
                        help="session directories, oldest to newest")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() or any(output.is_relative_to(path.resolve()) for path in args.input):
        parser.error("output must be a new directory outside source sessions")
    try:
        report = build_report(args.input, window=args.window)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output.parent, prefix=".play-report-") as temp:
            folder = Path(temp) / "report"
            folder.mkdir()
            atomic_json(folder / "report.json", report)
            (folder / "report.md").write_text(markdown(report), encoding="utf-8")
            folder.rename(output)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(2, f"cannot summarize play sessions: {error}\n")
    print(json.dumps({"output": str(output), "attempts": len(report["metrics"]["attempts"]),
                      "profiles": len(report["profiles"]),
                      "ready_profiles": sum(p["status"] == "ready" for p in report["profiles"])}))


if __name__ == "__main__":
    main()
