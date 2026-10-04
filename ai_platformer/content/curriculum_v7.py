"""Full-map pools frozen before multi-map training and holdout evaluation."""

from copy import deepcopy

from ai_platformer.content.curriculum import TrainingLevelRepository


def build_manifest() -> dict:
    manifest = deepcopy(TrainingLevelRepository("config/curriculum_v6.json").manifest)
    manifest["generator_version"] = 7
    manifest["distribution"]["full_v7"] = {
        "description": "Three consecutive obstacle-gap segments with ground and low-jump coins. "
        "Distinct geometry across splits. Legacy level_1 is separately retained as regression.",
        "counts": {"train": 12, "validation": 4, "test": 4, "ood": 4},
        "ood": {"height": 176, "gap_width": 196, "order": "reversed"},
    }
    manifest["splits"]["full"] = {}
    for suite, count, offset in (
        ("train", 12, 0),
        ("validation", 4, 20),
        ("test", 4, 40),
        ("ood", 4, 60),
    ):
        ids = []
        for index in range(count):
            level_id = f"v7_full_{suite}_{index:02d}"
            heights = (48, 96, 160) if suite != "ood" else (176, 128, 64)
            widths = (96, 140, 172) if suite != "ood" else (196, 160, 120)
            order = [(index + j) % 3 for j in range(3)]
            if suite in {"test", "ood"}:
                order.reverse()
            manifest["levels"][level_id] = {
                "task": "full",
                "spawn_x": 48 + (index + offset) * 3,
                "segments": [
                    {
                        "height": heights[j],
                        "gap_width": widths[j],
                        "approach": 600 + ((index + offset + j) % 4) * 40,
                        "recovery": 800 + ((index + j) % 3) * 80,
                    }
                    for j in order
                ],
            }
            ids.append(level_id)
        manifest["splits"]["full"][suite] = ids
    return manifest
