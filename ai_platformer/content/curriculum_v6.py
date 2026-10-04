"""Fresh mixed holdouts for the preregistered independent-training-seed cohort."""

from copy import deepcopy
from itertools import product

from ai_platformer.content.curriculum import TrainingLevelRepository
from ai_platformer.content.curriculum_v5 import composition_cell


def build_manifest() -> dict:
    manifest = deepcopy(TrainingLevelRepository("config/curriculum_v5.json").manifest)
    manifest["generator_version"] = 6
    manifest["distribution"]["fresh_mixed_v6"] = {
        "purpose": "New holdout geometry, created before independent cohort training. "
        "All v5 training and validation pools unchanged; old test/OOD now regression.",
        "test_spawn": [57, 153],
        "test_approach": [257, 437],
        "ood_height": 92,
        "ood_gap_width": 188,
        "ood_recovery": [420, 600, 800],
    }
    cells = sorted(
        {
            composition_cell(manifest["levels"][level])
            for level in manifest["splits"]["mixed"]["test"]
            if level.startswith("v5_mixed_")
        }
    )
    for suite, combinations, spawns, approaches in (
        ("test", cells, (57, 153), (257, 437)),
        ("ood", [(92, 188, r) for r in (420, 600, 800)], (61, 165), (255, 435)),
    ):
        for index, ((height, width, recovery), spawn, approach) in enumerate(
            product(combinations, spawns, approaches)
        ):
            obstacle = spawn + approach
            gap = obstacle + 64 + recovery
            level = f"v6_mixed_{suite}_{index:03d}"
            manifest["levels"][level] = {
                "task": "mixed",
                "width": gap + width + 500,
                "spawn_x": spawn,
                "obstacle_x": obstacle,
                "obstacle_height": height,
                "gap_x": gap,
                "gap_width": width,
            }
            manifest["splits"]["mixed"][suite].append(level)
    return manifest
