"""Structural and static geometry diagnostics; never claims physical reachability."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite

from pydantic import ValidationError

from ai_platformer.core import PhysicsConfig

from .level_spec import LevelSpec, document_hash, gameplay_hash, geometry_hash


@dataclass(frozen=True)
class Issue:
    code: str
    path: str
    severity: str
    message: str


def _overlap(a, b) -> bool:
    return (
        a.x < b.x + b.width
        and a.x + a.width > b.x
        and a.y < b.y + b.height
        and a.y + a.height > b.y
    )


def validate_geometry(spec: LevelSpec, *, physics: PhysicsConfig | None = None) -> list[Issue]:
    physics = physics or PhysicsConfig()
    if not all(isfinite(v) for v in asdict(physics).values()):
        raise ValueError("physics values must be finite")
    if not all(v > 0 for v in (physics.player_width, physics.player_height)):
        raise ValueError("player dimensions must be positive and finite")
    issues = []

    def add(code, path, message, severity="error"):
        issues.append(Issue(code, path, severity, message))

    if not 0 <= spec.spawn.x <= spec.world.width - physics.player_width:
        add("spawn_bounds", "/spawn/x", "Player bounding box exceeds horizontal world bounds")
    if not physics.player_height <= spec.spawn.bottom <= spec.world.height:
        add("spawn_bounds", "/spawn/bottom", "Player bounding box exceeds vertical world bounds")
    if not spec.spawn.x < spec.goal.x <= spec.world.width - physics.player_width:
        add(
            "goal_bounds",
            "/goal/x",
            "Goal must be after spawn and reachable within player horizontal bounds",
        )
    from types import SimpleNamespace

    player = SimpleNamespace(
        x=spec.spawn.x,
        y=spec.spawn.bottom - physics.player_height,
        width=physics.player_width,
        height=physics.player_height,
    )
    ids = {}
    for group in ("solids", "collectibles", "blocks", "enemies", "powerups"):
        for index, item in enumerate(getattr(spec, group)):
            path = f"/{group}/{index}"
            if not (
                0 <= item.x
                and 0 <= item.y
                and item.x + item.width <= spec.world.width
                and item.y + item.height <= spec.world.height
            ):
                add("entity_bounds", path, "Rectangle exceeds world bounds")
            if group != "solids":
                if item.entity_id in ids:
                    add(
                        "duplicate_entity_id",
                        path + "/entity_id",
                        f"ID already used at {ids[item.entity_id]}",
                    )
                ids[item.entity_id] = path
            if group in {"solids", "blocks"} and _overlap(player, item):
                add("spawn_overlap", path, "Player spawn overlaps collision geometry")
            if group == "enemies":
                if (
                    not 0
                    <= item.patrol_left
                    <= item.x
                    <= item.patrol_right
                    <= spec.world.width - item.width
                ):
                    add("patrol_bounds", path, "Patrol range/spawn exceeds horizontal world bounds")
                if _overlap(player, item):
                    add("spawn_enemy_overlap", path, "Enemy overlaps player spawn")
            if group == "collectibles" and item.x >= spec.goal.x:
                add(
                    "collectible_after_goal",
                    path,
                    "Collectible starts beyond episode goal; inspect collection denominator",
                    "warning",
                )
    # Limit expensive optional diagnostics independently from schema size limits.
    pairs = len(spec.collectibles) * len(spec.solids)
    if pairs <= 100_000:
        for index, coin in enumerate(spec.collectibles):
            if any(
                s.x <= coin.x
                and s.y <= coin.y
                and s.x + s.width >= coin.x + coin.width
                and s.y + s.height >= coin.y + coin.height
                for s in spec.solids
            ):
                add(
                    "collectible_embedded",
                    f"/collectibles/{index}",
                    "Collectible is entirely inside immutable solid geometry",
                )
    else:
        add(
            "diagnostic_budget",
            "/collectibles",
            "Embedded collectible diagnostic skipped: pair budget exceeded",
            "error",
        )
    if spec.blocks or spec.enemies or spec.powerups:
        add(
            "interactive_reachability_unverified",
            "/",
            "Interactive entities are supported by core but reachability has not been checked",
            "warning",
        )
    return issues


def validate_level_spec(data: dict | LevelSpec, *, physics: PhysicsConfig | None = None) -> dict:
    try:
        spec = LevelSpec.model_validate(data.model_dump() if isinstance(data, LevelSpec) else data)
    except ValidationError as error:
        issues = [
            Issue(
                "schema_error",
                "/"
                + "/".join(
                    str(v).replace("~", "~0").replace("/", "~1")
                    for v in e["loc"]
                    if v not in ("constrained-float", "constrained-int")
                ),
                "error",
                e["msg"],
            )
            for e in error.errors(include_input=False, include_url=False)
        ]
        return {
            "schema_version": 1,
            "structure_valid": False,
            "static_valid": False,
            "reachability": "not_checked",
            "issues": [asdict(i) for i in issues],
        }
    issues = validate_geometry(spec, physics=physics)
    return {
        "schema_version": 1,
        "level_id": spec.level_id,
        "structure_valid": True,
        "static_valid": not any(i.severity == "error" for i in issues),
        "reachability": "not_checked",
        "document_sha256": document_hash(spec),
        "gameplay_sha256": gameplay_hash(spec)
        if not any(i.severity == "error" for i in issues)
        else None,
        "geometry_sha256": geometry_hash(spec)
        if not any(i.severity == "error" for i in issues)
        else None,
        "physics": asdict(physics or PhysicsConfig()),
        "issues": [asdict(i) for i in issues],
    }


def validate_dataset(
    groups: dict[str, list[LevelSpec]], *, physics: PhysicsConfig | None = None
) -> dict:
    """Check ID/content reuse across every explicitly supplied split."""
    seen_ids, seen_hashes, issues, reports = {}, {}, [], {}
    if not groups:
        issues.append(Issue("empty_dataset", "/", "error", "Dataset must declare nonempty splits"))
    for group, specs in groups.items():
        reports[group] = []
        if not specs:
            issues.append(Issue("empty_split", f"/{group}", "error", "Split must contain levels"))
        for index, spec in enumerate(specs):
            path = f"/{group}/{index}"
            report = validate_level_spec(spec, physics=physics)
            reports[group].append(report)
            if not report["structure_valid"]:
                continue
            level_id = report["level_id"]
            if level_id in seen_ids:
                issues.append(
                    Issue(
                        "duplicate_level_id",
                        path,
                        "error",
                        f"Level ID already used at {seen_ids[level_id]}",
                    )
                )
            seen_ids[level_id] = path
            digest = report["geometry_sha256"]
            if digest:
                if digest in seen_hashes:
                    issues.append(
                        Issue(
                            "duplicate_gameplay",
                            path,
                            "error",
                            f"Gameplay content already used at {seen_hashes[digest]}",
                        )
                    )
                seen_hashes[digest] = path
    return {
        "schema_version": 1,
        "static_valid": not issues
        and all(report["static_valid"] for group in reports.values() for report in group),
        "reachability": "not_checked",
        "groups": reports,
        "issues": [asdict(i) for i in issues],
    }
