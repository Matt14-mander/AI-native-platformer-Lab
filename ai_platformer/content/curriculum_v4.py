"""Bridge long gap approaches without training on reserved OOD geometry."""

from copy import deepcopy

from ai_platformer.content.curriculum import TrainingLevelRepository


def build_manifest() -> dict:
    manifest = deepcopy(TrainingLevelRepository("config/curriculum_v3.json").manifest)
    manifest["generator_version"] = 4
    manifest["distribution"]["long_gap_bridge"] = {
        "train_approach_distances": [680, 740],
        "validation_approach_distances": [687, 747],
        "test_approach_distances": [673, 733],
        "width": 1800,
        "ood_minimum_approach_distance": 780,
        "note": "All earlier split assignments retained; no OOD/test layouts enter training.",
    }
    for suite, offsets in {"train": (0, 0, 0), "validation": (3, 7, 4), "test": (5, -7, 8)}.items():
        index = 0
        for spawn in (32, 96, 160):
            for distance in (680, 740):
                for width in (100, 130, 160):
                    level_id = f"v4_gap_{suite}_{index:02d}"
                    manifest["levels"][level_id] = {
                        "task": "gap",
                        "width": 1800,
                        "spawn_x": spawn + offsets[0],
                        "gap_x": spawn + offsets[0] + distance + offsets[1],
                        "gap_width": width + offsets[2],
                    }
                    manifest["splits"]["gap"][suite].append(level_id)
                    index += 1
    return manifest
