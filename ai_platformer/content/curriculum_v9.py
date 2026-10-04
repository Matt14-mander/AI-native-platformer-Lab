"""Playable legacy main area and fresh full-map geometry after v8 audit."""

from copy import deepcopy

from ai_platformer.content.curriculum import TrainingLevelRepository


def build_manifest() -> dict:
    manifest = deepcopy(TrainingLevelRepository("config/curriculum_v8.json").manifest)
    manifest["generator_version"] = 9
    manifest["levels"]["level_1_main"] = {"task": "full", "main_area_source": "level_1"}
    manifest["splits"]["full"]["train"].append("level_1_main")
    manifest["distribution"]["full_v9"] = {
        "main_area": "Explicit level_1_main filters 20 coins beyond the first area's goal; "
        "legacy level_1 unchanged. Five reachable intro coins, all required.",
        "fresh_counts": {"test": 4, "ood": 4},
        "fresh_scope": "New positions/spacing and test height-width combinations; OOD reuses "
        "already exposed bounded extremes, not new maximum height/width.",
    }
    for suite, offset in (("test", 210), ("ood", 240)):
        for index in range(4):
            level = f"v9_full_{suite}_{index:02d}"
            heights, widths = (
                ((60, 108, 156), (108, 152, 176))
                if suite == "test"
                else ((178, 136, 72), (200, 164, 128))
            )
            manifest["levels"][level] = {
                "task": "full",
                "spawn_x": 48 + 3 * (offset + index),
                "segments": [
                    {
                        "height": heights[(index + j) % 3],
                        "gap_width": widths[(index + j) % 3],
                        "approach": 660 + ((index + j) % 4) * 40,
                        "recovery": (1080 if suite == "ood" else 880) + ((index + j) % 3) * 80,
                    }
                    for j in (2, 0, 1)
                ],
            }
            manifest["splits"]["full"][suite].append(level)
    return manifest
