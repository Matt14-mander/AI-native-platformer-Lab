"""Generate fixed, reviewable course splits without changing the v1 manifest."""

from __future__ import annotations

from ai_platformer.content.curriculum import TrainingLevelRepository


def build_manifest() -> dict:
    old = TrainingLevelRepository().manifest
    manifest = {
        "schema_version": 1,
        "generator_version": 2,
        "distribution": {
            "in_distribution": "matched strata; split-specific offsets prevent duplicate geometry",
            "gap_distance_strata": [280, 380, 500, 620],
            "mixed_distance_strata": [1000, 1120, 1240, 1360],
            "spawn_strata": [32, 96, 160],
            "gap_width_strata": [100, 130, 160],
            "split_offsets": {"train": [0, 0, 0], "validation": [3, 7, 4], "test": [5, -7, 8]},
            "ood": "longer approach distances and gap widths of 170; diagnostic only",
            "legacy_replay": "flat/obstacle splits preserve v1 geometry and reserved final tests",
        },
        "levels": {},
        "splits": {},
    }
    for task in ("flat", "obstacle"):
        manifest["splits"][task] = old["splits"][task]
        for ids in old["splits"][task].values():
            for level_id in ids:
                manifest["levels"][level_id] = old["levels"][level_id]
    for task in ("gap", "mixed"):
        groups = manifest["splits"][task] = {}
        distances = manifest["distribution"][f"{task}_distance_strata"]
        for suite, offsets in manifest["distribution"]["split_offsets"].items():
            groups[suite] = []
            for spawn in (32, 96, 160):
                for distance in distances:
                    for width in (100, 130, 160):
                        level_id = f"v2_{task}_{suite}_{len(groups[suite]):02d}"
                        actual_spawn = spawn + offsets[0]
                        spec = {
                            "task": task,
                            "width": 1500 if task == "gap" else 2100,
                            "spawn_x": actual_spawn,
                            "gap_x": actual_spawn + distance + offsets[1],
                            "gap_width": width + offsets[2],
                        }
                        if task == "mixed":
                            spec.update(obstacle_x=actual_spawn + 220, obstacle_height=48)
                        manifest["levels"][level_id] = spec
                        groups[suite].append(level_id)
        groups["ood"] = []
        for spawn in (48, 120, 192):
            for distance in [780, 840, 900, 960] if task == "gap" else [1500, 1560, 1620, 1680]:
                level_id = f"v2_{task}_ood_{len(groups['ood']):02d}"
                spec = {
                    "task": task,
                    "width": 1800 if task == "gap" else 2800,
                    "spawn_x": spawn,
                    "gap_x": spawn + distance,
                    "gap_width": 170,
                }
                if task == "mixed":
                    spec.update(obstacle_x=spawn + 220, obstacle_height=64)
                manifest["levels"][level_id] = spec
                groups["ood"].append(level_id)
    return manifest
