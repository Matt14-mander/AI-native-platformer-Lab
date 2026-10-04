"""Export an existing course without modifying its source or gameplay hash."""

from __future__ import annotations

import argparse
from pathlib import Path

from ai_platformer.content.curriculum import TrainingLevelRepository
from ai_platformer.content.level_spec import from_level_definition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("config/curriculum_v9.json"))
    parser.add_argument("--level-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise FileExistsError("output exists; choose a new path")
        level = TrainingLevelRepository(args.manifest).load(args.level_id)
        spec = from_level_definition(level, metadata={"source": str(args.manifest)})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(spec.model_dump_json(indent=2) + "\n")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"LevelSpec export failed: {error}\n")
    print(args.output)


if __name__ == "__main__":
    main()
