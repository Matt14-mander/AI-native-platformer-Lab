"""Mixed composition cells held out as whole groups, with legacy splits intact."""

from copy import deepcopy
from itertools import product

from ai_platformer.content.curriculum import TrainingLevelRepository

PREFIX = "v5_mixed_"


def composition_cell(spec: dict) -> tuple[int, int, int]:
    return (
        spec["obstacle_height"],
        spec["gap_width"],
        spec["gap_x"] - spec["obstacle_x"] - 64,
    )


def build_manifest() -> dict:
    manifest = deepcopy(TrainingLevelRepository("config/curriculum_v4.json").manifest)
    manifest["generator_version"] = 5
    manifest["distribution"]["mixed_composition_v5"] = {
        "obstacle_heights": [48, 64, 80],
        "gap_widths": [100, 130, 160],
        "recovery_distances": [400, 560, 760],
        "spawn_positions": [32, 128],
        "obstacle_approaches": [240, 420],
        "landing_distance_to_goal": 400,
        "cell_split": "(height_index + 2*width_index + 3*recovery_index) % 5: "
        "0 validation, 1 test, other train; all positions of each cell stay together",
        "ood": {"obstacle_height": 88, "gap_width": 180},
        "note": "Legacy assignments retained. New validation/test cells are absent from new "
        "training; individual height/width/recovery values are represented in training. "
        "One obstacle and one gap; recovery remains above the existing 360px guard.",
    }
    indices = dict.fromkeys(("train", "validation", "test", "ood"), 0)

    def append(suite, height, width, recovery):
        for spawn, approach in product((32, 128), (240, 420)):
            obstacle = spawn + approach
            gap = obstacle + 64 + recovery
            level_id = f"{PREFIX}{suite}_{indices[suite]:03d}"
            indices[suite] += 1
            manifest["levels"][level_id] = {
                "task": "mixed",
                "width": gap + width + 500,
                "spawn_x": spawn,
                "obstacle_x": obstacle,
                "obstacle_height": height,
                "gap_x": gap,
                "gap_width": width,
            }
            manifest["splits"]["mixed"][suite].append(level_id)

    for hi, wi, ri in product(range(3), repeat=3):
        partition = (hi + 2 * wi + 3 * ri) % 5
        suite = "validation" if partition == 0 else "test" if partition == 1 else "train"
        append(suite, (48, 64, 80)[hi], (100, 130, 160)[wi], (400, 560, 760)[ri])
    for recovery in (400, 560, 760):
        append("ood", 88, 180, recovery)
    return manifest
