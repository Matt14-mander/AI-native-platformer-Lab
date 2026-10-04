"""Expanded full training orders and fresh geometry after the v7 failure audit."""

from copy import deepcopy
from itertools import permutations

from ai_platformer.content.curriculum import TrainingLevelRepository


def build_manifest() -> dict:
    manifest = deepcopy(TrainingLevelRepository("config/curriculum_v7.json").manifest)
    manifest["generator_version"] = 8
    manifest["distribution"]["full_v8"] = {
        "reason": "V7 seed 20261006 failed a reserved reversed-order map; v7 pools now regression.",
        "new_counts": {"train": 12, "validation": 6, "test": 4, "ood": 4},
        "test": "New interpolated height/width combinations and spawn/spacing, not new order cells.",
        "ood": "Height178 / gap200, recovery1040..1200 with new positions. Bounded physics family.",
    }
    orders = list(permutations(range(3)))
    for suite, count, offset in (
        ("train", 12, 80),
        ("validation", 6, 110),
        ("test", 4, 140),
        ("ood", 4, 170),
    ):
        for index in range(count):
            level_id = f"v8_full_{suite}_{index:02d}"
            heights, widths = (48, 96, 160), (96, 140, 172)
            if suite == "test":
                heights, widths = (56, 104, 152), (104, 148, 168)
            elif suite == "ood":
                heights, widths = (178, 136, 72), (200, 164, 128)
            order = orders[index % len(orders)]
            manifest["levels"][level_id] = {
                "task": "full",
                "spawn_x": 48 + (offset + index) * 3,
                "segments": [
                    {
                        "height": heights[j],
                        "gap_width": widths[j],
                        "approach": 620 + ((index + j) % 4) * 40,
                        "recovery": (1040 if suite == "ood" else 840) + ((index + j) % 3) * 80,
                    }
                    for j in order
                ],
            }
            manifest["splits"]["full"][suite].append(level_id)
    return manifest
