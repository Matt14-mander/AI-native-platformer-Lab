"""Broaden obstacle coverage while retaining all existing held-out course assignments."""

from copy import deepcopy

from ai_platformer.content.curriculum import TrainingLevelRepository


def build_manifest() -> dict:
    manifest = deepcopy(TrainingLevelRepository("config/curriculum_v2.json").manifest)
    manifest["generator_version"] = 3
    manifest["distribution"]["obstacle_v3"] = {
        "spawn_strata": [32, 96, 160],
        "approach_strata": [260, 380, 500, 620],
        "heights": [48, 64, 80],
        "split_offsets_spawn_distance": {"train": [0, 0], "validation": [3, 7], "test": [5, -7]},
        "legacy": "all v1 obstacle split assignments retained; validation never added to training",
    }
    for suite, offsets in manifest["distribution"]["obstacle_v3"][
        "split_offsets_spawn_distance"
    ].items():
        index = 0
        for spawn in (32, 96, 160):
            for distance in (260, 380, 500, 620):
                for height in (48, 64, 80):
                    level_id = f"v3_obstacle_{suite}_{index:02d}"
                    manifest["levels"][level_id] = {
                        "task": "obstacle",
                        "width": 1500,
                        "spawn_x": spawn + offsets[0],
                        "obstacle_x": spawn + offsets[0] + distance + offsets[1],
                        "obstacle_height": height,
                    }
                    manifest["splits"]["obstacle"][suite].append(level_id)
                    index += 1
    return manifest
