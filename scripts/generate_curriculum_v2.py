"""Write and validate the reproducible curriculum v2 manifest."""

import json

from ai_platformer.content.curriculum import PROJECT_ROOT, TrainingLevelRepository
from ai_platformer.content.curriculum_v2 import build_manifest


def main() -> None:
    path = PROJECT_ROOT / "config/curriculum_v2.json"
    path.write_text(json.dumps(build_manifest(), indent=2) + "\n", encoding="utf-8")
    repository = TrainingLevelRepository(path)
    print(f"{len(repository.levels)} courses: {path}")


if __name__ == "__main__":
    main()
