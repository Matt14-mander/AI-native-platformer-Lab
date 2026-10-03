"""Write and validate the obstacle recovery curriculum."""

import json

from ai_platformer.content.curriculum import PROJECT_ROOT, TrainingLevelRepository
from ai_platformer.content.curriculum_v3 import build_manifest


def main():
    path = PROJECT_ROOT / "config/curriculum_v3.json"
    path.write_text(json.dumps(build_manifest(), indent=2) + "\n", encoding="utf-8")
    print(f"{len(TrainingLevelRepository(path).levels)} courses: {path}")


if __name__ == "__main__":
    main()
