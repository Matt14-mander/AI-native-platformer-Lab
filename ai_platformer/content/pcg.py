"""Deterministic generator dispatch with frozen versioned physics and RNG rules."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

from ai_platformer.core import CollectibleSpawn, LevelDefinition, PhysicsConfig, SolidRect

from .generation_request import GenerationRequest, request_hash
from .level_spec import LevelSpec, from_level_definition
from .level_validation import validate_level_spec

# Frozen profile: never read mutable gameplay settings during generation.
SEGMENTS_V1_PHYSICS = PhysicsConfig(
    player_width=24.0,
    player_height=32.0,
    walk_acceleration=0.3,
    run_acceleration=0.5,
    turn_acceleration=0.7,
    max_walk_speed=6.0,
    max_run_speed=10.0,
    jump_velocity=-13.8,
    rising_gravity=0.5,
    falling_gravity=1.8,
    max_fall_speed=11.0,
    max_episode_steps=36000,
)


class _Random:
    """SHA256 counter stream with rejection sampling, independent of Python RNG."""

    def __init__(self, seed):
        self.seed, self.counter = seed, 0

    def integer(self, lo, hi):
        span = hi - lo + 1
        limit = 2**256 - (2**256 % span)
        while True:
            raw = f"segments-v1:{self.seed}:{self.counter}".encode("ascii")
            self.counter += 1
            value = int.from_bytes(hashlib.sha256(raw).digest(), "big")
            if value < limit:
                return lo + value % span

    def shuffle(self, values):
        for index in range(len(values) - 1, 0, -1):
            other = self.integer(0, index)
            values[index], values[other] = values[other], values[index]


@dataclass(frozen=True)
class GenerationResult:
    request: GenerationRequest
    level: LevelSpec
    report: dict
    physics: PhysicsConfig


def _segments_v1(request):
    rng = _Random(request.seed)
    kinds = ["obstacle"] * request.obstacle_count + ["gap"] * request.gap_count
    rng.shuffle(kinds)
    if not kinds:
        kinds = ["flat"]
    cursor, floor = 48, 540
    obstacles, gaps, coins, segments = [], [], [], []
    for index, kind in enumerate(kinds):
        approach = rng.integer(540, 720)
        runout = rng.integer(600, 760)
        x = cursor + approach
        height = (
            rng.integer(request.obstacle_height.minimum, request.obstacle_height.maximum)
            if kind == "obstacle"
            else 0
        )
        width = (
            rng.integer(request.gap_width.minimum, request.gap_width.maximum)
            if kind == "gap"
            else 64
            if kind == "obstacle"
            else 0
        )
        end = x + width
        if kind == "obstacle":
            obstacles.append(SolidRect(x, floor - height, width, height))
        elif kind == "gap":
            gaps.append((x, end))
        if request.coin_layout != "none":
            for number, (coin_x, coin_y) in enumerate(
                ((cursor + 180, 508), (cursor + 300, 430), (end + 220, 508), (end + 380, 430))
            ):
                if request.coin_layout == "ground":
                    coin_y = 508
                coins.append(CollectibleSpawn(f"coin-{index}-{number}", "coin", coin_x, coin_y))
        segments.append(
            {
                "kind": kind,
                "x": x,
                "height": height,
                "width": width,
                "approach": approach,
                "runout": runout,
            }
        )
        cursor = end + runout
    world_width = cursor + 160
    solids, start = [], 0
    for left, right in gaps:
        solids.append(SolidRect(start, floor, left - start, 60))
        start = right
    solids.append(SolidRect(start, floor, world_width - start, 60))
    level = LevelDefinition(
        request.level_id,
        world_width,
        600,
        48,
        floor,
        world_width - 100,
        tuple(solids + obstacles),
        tuple(coins),
    )
    spec = from_level_definition(
        level,
        metadata={
            "theme": request.theme,
            "source": f"generation-request:{request_hash(request)}",
            "generator_version": request.generator_version,
            "seed": request.seed,
        },
    )
    validation = validate_level_spec(spec, physics=SEGMENTS_V1_PHYSICS)
    if not validation["static_valid"]:
        raise RuntimeError(f"generator produced invalid geometry: {validation['issues']}")
    return GenerationResult(
        request,
        spec,
        {
            "schema_version": 1,
            "status": "candidate",
            "generator_version": request.generator_version,
            "rng_version": "sha256-counter-rejection-v1",
            "request_sha256": request_hash(request),
            "physics_profile": request.physics_profile,
            "physics": asdict(SEGMENTS_V1_PHYSICS),
            "segments": segments,
            "counts": {"obstacles": len(obstacles), "gaps": len(gaps), "coins": len(coins)},
            "objective": {"min_coin_ratio": request.min_coin_ratio},
            "validation": validation,
            "note": "Static-valid candidate. Physical reachability and collection target need route replay.",
        },
        SEGMENTS_V1_PHYSICS,
    )


_GENERATORS = {"segments-v1": _segments_v1}


def available_generators() -> tuple[str, ...]:
    return tuple(sorted(_GENERATORS))


def generate_level(request: GenerationRequest | dict) -> GenerationResult:
    """Pure versioned dispatch; no files, environment, RNG globals or training mutation."""
    request = GenerationRequest.model_validate(
        request.model_dump() if isinstance(request, GenerationRequest) else request
    )
    return _GENERATORS[request.generator_version](request)
