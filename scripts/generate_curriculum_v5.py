"""Generate and validate mixed composition splits, preserving all v4 layouts."""

import json

from ai_platformer.content.curriculum import PROJECT_ROOT, TrainingLevelRepository
from ai_platformer.content.curriculum_v5 import PREFIX, build_manifest


def main():
    path = PROJECT_ROOT / "config/curriculum_v5.json"
    path.write_text(json.dumps(build_manifest(), indent=2) + "\n", encoding="utf-8")
    repo = TrainingLevelRepository(path)
    print(f"{len(repo.levels)} total courses: {path}")
    for suite in ("train", "validation", "test", "ood"):
        print(suite, sum(level.startswith(PREFIX) for level in repo.split("mixed", suite)))


if __name__ == "__main__":
    main()
