"""Bounded prompt/seed evaluation; route feasibility and PPO ability stay separate."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

from ai_platformer.agents.llm.generation import (
    GenerationError,
    digest,
    read_json,
    translate_intent,
)
from ai_platformer.content.generation_suite import compare_expectations
from ai_platformer.content.level_spec import geometry_hash
from ai_platformer.content.pcg import SEGMENTS_V1_PHYSICS, generate_level

RUNNER_VERSION = "generation-evaluation-v1"


def sha256_file(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def component_fingerprints():
    root = Path(__file__).resolve().parents[2]
    files = [*root.joinpath("core").glob("*.py"), *root.joinpath("envs").glob("*.py")]
    files += [
        root / name
        for name in (
            "agents/llm/generation.py",
            "content/generation_suite.py",
            "content/generation_request.py",
            "content/level_spec.py",
            "content/level_repository.py",
            "content/level_validation.py",
            "content/pcg.py",
            "content/routes.py",
            "rendering/ppo_playback.py",
            "deployment/backends.py",
        )
    ]
    return {str(path.relative_to(root)): sha256_file(path) for path in sorted(files)}


def runtime_versions():
    result = {}
    for name in ("numpy", "pydantic", "stable-baselines3", "torch", "gymnasium"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=".writing-", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def evaluate_ppo(loaded, factory, *, seed, request_target, expected_target):
    from ai_platformer.rendering.ppo_playback import PlaybackSession

    session = PlaybackSession(loaded.policy, factory, level_id=factory.level_id, seed=seed)
    actions = []
    try:
        while not session.done:
            session.step()
            actions.append(int(session.action))
        summary = session.summary()
        ratio = (
            summary["coins_collected"] / summary["coins_total"] if summary["coins_total"] else None
        )

        def meets(target):
            return summary["outcome"] == "success" and (
                target == 0 or (ratio is not None and ratio + 1e-12 >= target)
            )

        return {
            "status": "ok",
            **summary,
            "collection_ratio": ratio,
            "meets_request_target": meets(request_target),
            "meets_expected_target": meets(expected_target),
        }, {
            "actions": actions,
            "final_state": asdict(session.env.core.state),
        }
    finally:
        session.env.close()


def candidate(case, seed, *, folder, suite, loaded, settings, run_id, transport=None):
    """A protocol-valid map is always evaluated by PPO, even if search is unknown."""
    from ai_platformer.content.routes import search_route
    from ai_platformer.envs.factory import protocol_hash

    expected = case.expected_request(seed)
    row = {
        "case_id": case.case_id,
        "level_id": expected.level_id,
        "generation_seed": seed,
        "policy_seed": suite.policy_seed,
        "run_sha256": run_id,
        "status": "error",
    }
    stage = "translation"
    started = perf_counter()
    atomic_json(
        folder / "intent.json",
        {
            "prompt": case.prompt,
            "expected": case.expected,
            "level_id": expected.level_id,
            "seed": seed,
        },
    )
    fixture = None
    if settings["provider"] == "expected-fixture":
        fixture = folder / "fixture.json"
        atomic_json(fixture, {"request": expected.model_dump(), "unsupported": None})
    try:
        begin = perf_counter()
        request, record = translate_intent(
            case.prompt,
            level_id=expected.level_id,
            seed=seed,
            provider="fixture" if fixture else "gemini",
            model=settings["model"],
            fixture=fixture,
            cache_dir=settings["cache_dir"],
            max_attempts=settings["max_attempts"],
            timeout=settings["timeout"],
            tls_mode=settings["tls_mode"],
            transport=transport,
        )
        atomic_json(folder / "llm.json", record)
        row["translation"] = {
            "status": "ok",
            "seconds": perf_counter() - begin,
            "cache_hit": record["cache_hit"],
            "attempt_count": max((a["attempt"] for a in record["attempts"]), default=0),
        }
        usages = {
            a["attempt"]: a["usage"] for a in record["attempts"] if isinstance(a.get("usage"), dict)
        }
        row["translation"]["reported_token_usage"] = (
            {
                key: sum(
                    usage.get(key, 0) for usage in usages.values() if type(usage.get(key, 0)) is int
                )
                for key in (
                    "promptTokenCount",
                    "candidatesTokenCount",
                    "totalTokenCount",
                    "thoughtsTokenCount",
                )
            }
            if usages
            else None
        )
        row["semantics"] = compare_expectations(request, case.expected)
        stage = "generation"
        generated = generate_level(request)
        atomic_json(folder / "request.json", request.model_dump())
        atomic_json(folder / "level.json", generated.level.model_dump())
        atomic_json(folder / "generation.json", generated.report)
        row["generation"] = {
            "status": "valid",
            "geometry_sha256": geometry_hash(generated.level),
            "request_sha256": record["request_sha256"],
        }
        factory = loaded.factory.for_level_spec(folder / "level.json")
        row["candidate_protocol_sha256"] = protocol_hash(factory.protocol([factory.level_id]))
        # Same physics, action repeat and decision budget for search and PPO.
        begin = perf_counter()
        try:
            route = search_route(
                factory,
                seed=suite.policy_seed,
                max_steps=factory.step_limit,
                max_seconds=settings["route_seconds"],
                max_expansions=settings["route_expansions"],
                min_coin_ratio=request.min_coin_ratio,
            )
            atomic_json(folder / "route.json", route)
            row["route"] = {"status": route["status"], "reason": route["reason"]}
        except (OSError, ValueError, RuntimeError, ImportError):
            row["route"] = {"status": "error", "reason": "route_verification_failed"}
        row["route"]["seconds"] = perf_counter() - begin
        begin = perf_counter()
        try:
            summary, trace = evaluate_ppo(
                loaded,
                factory,
                seed=suite.policy_seed,
                request_target=request.min_coin_ratio,
                expected_target=expected.min_coin_ratio,
            )
            atomic_json(folder / "ppo.json", {**summary, **trace})
            row["ppo"] = summary
        except (OSError, ValueError, RuntimeError, ImportError):
            row["ppo"] = {"status": "error", "reason": "policy_evaluation_failed"}
        row["ppo"]["seconds"] = perf_counter() - begin
        row["status"] = "completed"
    except GenerationError as error:
        atomic_json(folder / "llm.json", {"attempts": error.attempts})
        row["translation"] = {
            "status": "error",
            "seconds": perf_counter() - started,
            "attempt_count": max((a["attempt"] for a in error.attempts), default=0),
        }
        row["failure"] = {
            "stage": stage,
            "reason": str(error),
            "http_statuses": [a["http_status"] for a in error.attempts if "http_status" in a],
        }
    except (OSError, ValueError, RuntimeError, ImportError, RecursionError):
        row["failure"] = {"stage": stage, "reason": "candidate_pipeline_failed"}
    row["seconds"] = perf_counter() - started
    row["artifact_sha256"] = {p.name: sha256_file(p) for p in folder.glob("*.json")}
    atomic_json(folder / "result.json", row)
    return row


def summarize(rows, *, planned, provider):
    def count(stage, key, value):
        return sum(row.get(stage, {}).get(key) == value for row in rows)

    valid = count("generation", "status", "valid")
    evaluated = count("ppo", "status", "ok")
    ratios = [
        row["ppo"]["collection_ratio"]
        for row in rows
        if row.get("ppo", {}).get("collection_ratio") is not None
    ]
    hashes = {}
    for row in rows:
        value = row.get("generation", {}).get("geometry_sha256")
        if value:
            hashes.setdefault(value, []).append(row["level_id"])
    matched = count("semantics", "passed", True)
    route_verified = count("route", "status", "verified")
    succeeded = count("ppo", "outcome", "success")
    met = count("ppo", "meets_expected_target", True)
    all_passed = sum(
        row.get("semantics", {}).get("passed", False)
        and row.get("route", {}).get("status") == "verified"
        and row.get("ppo", {}).get("meets_expected_target", False)
        for row in rows
    )
    live = [
        row["translation"]
        for row in rows
        if row.get("translation", {}).get("status") == "ok"
        and not row["translation"]["cache_hit"]
        and provider == "gemini"
    ]
    return {
        "planned": planned,
        "processed": len(rows),
        "remaining": planned - len(rows),
        "translation_valid": count("translation", "status", "ok"),
        "static_valid": valid,
        "request_expectation_matches": matched,
        "route_verified": route_verified,
        "route_unknown": count("route", "status", "unknown"),
        "ppo_evaluated": evaluated,
        "ppo_successes": succeeded,
        "ppo_expected_target_met": met,
        "all_stage_passes": all_passed,
        "failure_categories": {
            "translation_error": count("translation", "status", "error"),
            "semantic_mismatch": count("semantics", "passed", False),
            "route_unknown": count("route", "status", "unknown"),
            "route_error": count("route", "status", "error"),
            "ppo_error": count("ppo", "status", "error"),
            "ppo_non_success": sum(
                r.get("ppo", {}).get("status") == "ok" and r["ppo"].get("outcome") != "success"
                for r in rows
            ),
            "ppo_success_collection_miss": sum(
                r.get("ppo", {}).get("outcome") == "success"
                and not r["ppo"].get("meets_expected_target", False)
                for r in rows
            ),
        },
        "rates_over_planned": {
            "static_valid": valid / planned,
            "route_verified": route_verified / planned,
            "ppo_success": succeeded / planned,
            "all_stage_pass": all_passed / planned,
        },
        "ppo_success_rate_over_evaluated": succeeded / evaluated if evaluated else None,
        "ppo_target_rate_over_evaluated": met / evaluated if evaluated else None,
        "mean_collection_ratio_nonempty_maps": sum(ratios) / len(ratios) if ratios else None,
        "maps_with_collection_denominator": len(ratios),
        "llm_semantic_match_rate_over_planned": matched / planned if provider == "gemini" else None,
        "mean_uncached_valid_translation_seconds": sum(r["seconds"] for r in live) / len(live)
        if live
        else None,
        "uncached_valid_translation_samples": len(live),
        "reported_uncached_token_usage": {
            key: sum(
                r.get("reported_token_usage", {}).get(key, 0)
                for r in live
                if r.get("reported_token_usage")
            )
            for key in (
                "promptTokenCount",
                "candidatesTokenCount",
                "totalTokenCount",
                "thoughtsTokenCount",
            )
        },
        "billed_cost_usd": None,
        "unique_geometry_count": len(hashes),
        "duplicate_geometry_groups": [ids for ids in hashes.values() if len(ids) > 1],
        "failures": [{"level_id": r["level_id"], **r["failure"]} for r in rows if "failure" in r],
    }


def write_report(output, *, suite, settings, run_id, selected, rows):
    summary = summarize(rows, planned=len(selected), provider=settings["provider"])
    groups = {
        case.case_id: summarize(
            [r for r in rows if r["case_id"] == case.case_id],
            planned=sum(c.case_id == case.case_id for c, _ in selected),
            provider=settings["provider"],
        )
        for case in suite.cases
        if any(c.case_id == case.case_id for c, _ in selected)
    }
    report = {
        "schema_version": 1,
        "run_sha256": run_id,
        "suite_id": suite.suite_id,
        "status": "complete" if len(rows) == len(selected) else "incomplete",
        "mode": settings["provider"],
        "summary": summary,
        "by_prompt": groups,
        "cases": rows,
        "scope": "Exploratory static segments-v1 evaluation; no training split assignment. Fixture mode is not LLM evaluation.",
        "notes": [
            "Rates over planned include failed and unprocessed candidates in the denominator.",
            "Route verification and PPO ability are independent; PPO runs on every statically valid candidate.",
            "Token usage covers reported uncached successful translation attempts; not a billing estimate.",
            "One fixed deterministic policy seed per map; map seeds do not constitute independent policy trials.",
        ],
    }
    atomic_json(output / "report.json", report)
    lines = [
        f"# {suite.suite_id}",
        "",
        f"Mode: {settings['provider']}; status: {report['status']}",
        "",
        "| Prompt | Processed/planned | Static valid | Route verified | PPO success | PPO target met |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, group in groups.items():
        lines.append(
            f"| {name} | {group['processed']}/{group['planned']} | {group['static_valid']} | {group['route_verified']} | {group['ppo_successes']} | {group['ppo_expected_target_met']} |"
        )
    lines.extend(
        [
            "",
            report["scope"],
            "",
            "Costs are not estimated; failed-call token usage may be unavailable.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def run_suite(suite, *, output, loaded, settings, resume=False, limit=None, transport=None):
    if asdict(loaded.factory.settings.physics) != asdict(SEGMENTS_V1_PHYSICS):
        raise ValueError(
            "checkpoint physics differ from segments-v1; choose a compatible checkpoint"
        )
    if (
        not isinstance(loaded.factory.step_limit, int)
        or not 1 <= loaded.factory.step_limit <= 10000
    ):
        raise ValueError("checkpoint must have a bounded decision limit")
    selected = [(case, seed) for case in suite.cases for seed in suite.generation_seeds]
    selected = selected[:limit] if limit is not None else selected
    output = Path(output)
    identity = {
        "runner_version": RUNNER_VERSION,
        "runner_sha256": sha256_file(__file__),
        "component_sha256": component_fingerprints(),
        "runtime_versions": runtime_versions(),
        "suite": suite.model_dump(),
        "settings": settings,
        "limit": limit,
        "checkpoint_sha256": sha256_file(loaded.model_path),
        "checkpoint_sidecar_sha256": sha256_file(loaded.model_path.with_suffix(".json")),
    }
    run_id = digest(identity)
    if output.exists():
        if not resume:
            raise ValueError("output exists; use a new directory or --resume")
        if read_json(output / "run.json")["identity"] != identity:
            raise ValueError(
                "resume identity mismatch (suite, checkpoint, code or settings changed)"
            )
    else:
        if resume:
            raise ValueError("resume output does not exist")
        output.mkdir(parents=True)
        atomic_json(output / "run.json", {"identity": identity, "run_sha256": run_id})
        atomic_json(output / "suite.json", suite.model_dump())
    cases_dir = output / "cases"
    cases_dir.mkdir(exist_ok=True)
    rows = []
    previous = read_json(output / "report.json") if (output / "report.json").exists() else {}
    previous_rows = {row["level_id"]: row for row in previous.get("cases", [])}
    for case, seed in selected:
        package = cases_dir / case.expected_request(seed).level_id
        if not package.exists():
            continue
        row = read_json(package / "result.json")
        if row.get("run_sha256") != run_id or row.get("level_id") != package.name:
            raise ValueError("candidate resume identity mismatch")
        if package.name in previous_rows and previous_rows[package.name] != row:
            raise ValueError("candidate result changed; refusing resume")
        for name, expected_hash in row["artifact_sha256"].items():
            if (
                Path(name).name != name
                or not name.endswith(".json")
                or sha256_file(package / name) != expected_hash
            ):
                raise ValueError("candidate artifact changed; refusing resume")
        rows.append(row)
    done = {r["level_id"] for r in rows}
    report = write_report(
        output, suite=suite, settings=settings, run_id=run_id, selected=selected, rows=rows
    )
    consecutive_translation_errors = 0
    for case, seed in selected:
        level_id = case.expected_request(seed).level_id
        if level_id in done:
            continue
        with tempfile.TemporaryDirectory(dir=cases_dir, prefix=".candidate-") as temp:
            folder = Path(temp) / "package"
            folder.mkdir()
            row = candidate(
                case,
                seed,
                folder=folder,
                suite=suite,
                loaded=loaded,
                settings=settings,
                run_id=run_id,
                transport=transport,
            )
            folder.rename(cases_dir / level_id)
        rows.append(row)
        report = write_report(
            output, suite=suite, settings=settings, run_id=run_id, selected=selected, rows=rows
        )
        print(
            f"[{len(rows)}/{len(selected)}] {level_id}: request={row.get('translation', {}).get('status')} route={row.get('route', {}).get('status')} ppo={row.get('ppo', {}).get('outcome')} target={row.get('ppo', {}).get('meets_expected_target')}",
            flush=True,
        )
        if row.get("failure", {}).get("stage") == "translation":
            consecutive_translation_errors += 1
        else:
            consecutive_translation_errors = 0
        permanent = set(row.get("failure", {}).get("http_statuses", [])) & {400, 401, 402, 403, 404}
        if permanent or consecutive_translation_errors >= 3:
            print(
                "Stopped after permanent API error or three consecutive translation failures; inspect report before --resume.",
                flush=True,
            )
            break
    return report
