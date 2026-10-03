"""Quantify legacy and candidate gameplay physics before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_platformer.benchmark.gameplay import audit_settings, measure_jump
from ai_platformer.settings import AI_GAMEPLAY_PATH, load_gameplay_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, nargs="+", help="profiles to compare")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--plot", type=Path, help="optional jump trajectory PNG (requires matplotlib)"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = args.settings or [root / "config/gameplay.json", AI_GAMEPLAY_PATH]
    if args.output.exists():
        parser.error("output already exists; choose a new report path")
    if args.plot and (args.plot.exists() or args.plot.suffix.lower() != ".png"):
        parser.error("plot must be a new .png path")
    report = {
        "profiles": {path.stem: audit_settings(load_gameplay_settings(path)) for path in paths}
    }
    report["passed"] = all(
        all(profile["checks"].values()) for profile in report["profiles"].values()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.plot:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(9, 4.5))
        for path in paths:
            settings = load_gameplay_settings(path)
            fixed = report["profiles"][path.stem]["fixed_height"]
            for ticks in (1,) if fixed else (1, 300):
                jump = measure_jump(settings, ticks, include_trajectory=True)
                trace = jump["trajectory"]
                label = f"{path.stem}: " + (
                    "fixed tap/hold" if fixed else "tap" if ticks == 1 else "hold"
                )
                axis.plot(
                    [row[0] / settings.render_fps for row in trace],
                    [row[1] for row in trace],
                    label=label,
                    linewidth=2,
                )
        axis.set(
            xlabel="Time after jump (seconds)",
            ylabel="Height above takeoff (pixels)",
            title="Measured shared-core jump trajectories",
        )
        axis.grid(alpha=0.25)
        axis.legend()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        figure.tight_layout()
        figure.savefig(args.plot, dpi=150)
        plt.close(figure)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        parser.exit(1, "gameplay mechanics audit did not pass\n")


if __name__ == "__main__":
    main()
