"""Reserved pools and criteria, declared before observing policy holdout outcomes."""

from ai_platformer.agents.ppo.gates import assess_prerequisites


def validate_reserved_seeds(seeds: list[int], config: dict) -> None:
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds)
    ):
        raise ValueError("acceptance seeds must be distinct nonnegative integers")
    if set(seeds) & set(config["train_seeds"] + config["evaluation"]["seeds"]):
        raise ValueError("acceptance seeds overlap training/validation")


def reserved_pools(repository, stages: list[dict]) -> dict:
    seen = {
        level
        for stage in stages
        for key in ("train_levels", "validation_levels")
        for level in stage[key]
    }
    pools = {}
    reserved = set()
    for suite in ("test", "ood"):
        tasks = {}
        for stage in stages:
            levels = repository.manifest["splits"].get(stage["task"], {}).get(suite, [])
            if not levels:
                continue
            if len(set(levels)) != len(levels) or set(levels) & (seen | reserved):
                raise ValueError("reserved layout overlaps another pool or train/validation")
            tasks[stage["task"]] = list(levels)
            reserved.update(levels)
        if not tasks:
            raise ValueError(f"no reserved {suite} pool")
        pools[suite] = tasks
    return pools


def acceptance_gate(stages: list[dict], pools: dict[str, list[str]], evaluation: dict) -> dict:
    criteria = [
        {**stage, "validation_levels": pools[stage["task"]]}
        for stage in stages
        if stage["task"] in pools
    ]
    return assess_prerequisites(criteria, evaluation)
