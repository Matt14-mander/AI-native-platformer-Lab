"""Long courses with repeated hazards and authored, reachable collectible routes."""

from ai_platformer.core.level import CollectibleSpawn, LevelDefinition, SolidRect


def make_full_course(level_id: str, spec: dict) -> LevelDefinition:
    floor = 540.0
    cursor = float(spec.get("spawn_x", 48))
    spawn = cursor
    solids, gaps, coins = [], [], []
    segments = spec["segments"]
    if len(segments) < 3:
        raise ValueError("full courses require at least three hazard segments")
    for index, segment in enumerate(segments):
        approach = float(segment["approach"])
        height = float(segment["height"])
        recovery = float(segment["recovery"])
        gap_width = float(segment["gap_width"])
        if approach < 540 or not 0 < height <= 178 or recovery < 420 or not 0 < gap_width <= 200:
            raise ValueError("invalid full course segment")
        obstacle = cursor + approach
        gap = obstacle + 64 + recovery
        solids.append(SolidRect(obstacle, floor - height, 64, height))
        gaps.append((gap, gap + gap_width))
        for number, (x, y) in enumerate(
            (
                (cursor + 180, 508),
                (cursor + 300, 430),
                (gap + gap_width + 220, 508),
                (gap + gap_width + 380, 430),
            )
        ):
            coins.append(CollectibleSpawn(f"coin-{index}-{number}", "coin", x, y))
        cursor = gap + gap_width + 600
    width = cursor + 160
    start = 0.0
    for left, right in gaps:
        solids.append(SolidRect(start, floor, left - start, 60))
        start = right
    solids.append(SolidRect(start, floor, width - start, 60))
    return LevelDefinition(
        level_id, width, 600, spawn, floor, width - 100, tuple(solids), tuple(coins)
    )
