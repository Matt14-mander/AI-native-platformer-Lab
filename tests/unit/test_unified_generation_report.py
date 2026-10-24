"""Unified metrics preserve denominators, provenance and offline/live separation."""

import copy
import json
import subprocess
import sys

import pytest

from ai_platformer.agents.llm.evaluation import atomic_json, sha256_file
from ai_platformer.agents.llm.generation import digest
from ai_platformer.agents.llm.reporting import (
    api_metrics,
    build_report,
    cohort_summary,
    distribution,
    load_batch,
    load_single,
    markdown,
)
from ai_platformer.content.generation_request import request_hash
from ai_platformer.content.generation_suite import GenerationCase, GenerationSuite
from ai_platformer.content.level_spec import geometry_hash
from ai_platformer.content.pcg import generate_level


def make_batch(root, *, mode="expected-fixture", processed=2):
    case = GenerationCase(
        case_id="flat",
        prompt="平地，地面松果，全部收集",
        expected={
            "obstacle_count": 0,
            "gap_count": 0,
            "coin_layout": "ground",
            "min_coin_ratio": 1.0,
        },
    )
    suite = GenerationSuite(
        schema_version=1,
        suite_id="unit_suite",
        purpose="exploratory_evaluation_only",
        generation_seeds=[1, 2],
        cases=[case],
    )
    identity = {
        "suite": suite.model_dump(),
        "settings": {"provider": mode, "model": "gemini-test", "tls_mode": "tls12"},
        "checkpoint_sha256": "f" * 64,
        "limit": None,
    }
    run_id = digest(identity)
    atomic_json(root / "run.json", {"identity": identity, "run_sha256": run_id})
    atomic_json(root / "suite.json", suite.model_dump())
    rows = []
    for seed in [1, 2][:processed]:
        request = case.expected_request(seed)
        package = root / "cases" / request.level_id
        generated = generate_level(request)
        coins = len(generated.level.collectibles)
        ppo = {
            "status": "ok",
            "outcome": "success",
            "steps": 2,
            "ticks": 8,
            "coins_collected": coins if seed == 1 else coins - 1,
            "coins_total": coins,
            "collection_ratio": 1.0 if seed == 1 else (coins - 1) / coins,
            "meets_expected_target": seed == 1,
        }
        atomic_json(package / "request.json", request.model_dump())
        atomic_json(package / "level.json", generated.level.model_dump())
        atomic_json(
            package / "ppo.json", {**ppo, "actions": [1, 1], "final_state": {"player": {"x": 20}}}
        )
        atomic_json(
            package / "llm.json",
            {
                "attempts": [
                    {"attempt": 1, "output": request.model_dump(), "usage": {"totalTokenCount": 10}}
                ]
            },
        )
        row = {
            "level_id": request.level_id,
            "case_id": case.case_id,
            "generation_seed": seed,
            "run_sha256": run_id,
            "status": "completed",
            "policy_seed": 100,
            "semantics": {"passed": True, "mismatches": []},
            "generation": {
                "status": "valid",
                "geometry_sha256": geometry_hash(generated.level),
                "request_sha256": request_hash(request),
            },
            "translation": {"status": "ok", "seconds": seed, "cache_hit": False},
            "route": {
                "status": "verified" if seed == 1 else "unknown",
                "reason": "budget_exhausted",
            },
            "ppo": ppo,
            "artifact_sha256": {p.name: sha256_file(p) for p in package.glob("*.json")},
        }
        atomic_json(package / "result.json", row)
        rows.append(row)
    atomic_json(
        root / "report.json",
        {
            "run_sha256": run_id,
            "suite_id": suite.suite_id,
            "mode": mode,
            "cases": rows,
            "summary": {"ppo_successes": 999},
        },
    )
    return root


def test_metrics_recomputed_from_cases_and_failures_not_pooled(tmp_path):
    source = load_batch(make_batch(tmp_path / "offline"))
    report = build_report([source])
    summary = report["sources"][0]["summary"]
    assert summary["ppo_success_over_evaluated"]["rate"] == 1
    assert summary["ppo_target_over_evaluated"]["rate"] == 0.5
    assert summary["llm_semantic_matches_over_planned"] is None
    assert summary["api"]["recorded_uncached_attempts"] == 0
    assert len(report["failure_catalog"]) == 2  # one map, two overlapping failure categories
    assert "attempt_records" in source["cases"][0]  # input is not mutated
    assert "attempt_records" not in report["sources"][0]["cases"][0]
    assert "summary" not in report
    assert "不合并为总成功率" in markdown(report)
    assert "| 生成seed |" in markdown(report)


def test_pending_cases_stay_in_planned_denominators(tmp_path):
    source = load_batch(make_batch(tmp_path / "partial", mode="gemini", processed=1))
    assert source["summary"]["valid_requests_over_planned"]["rate"] == 0.5
    assert source["summary"]["ppo_success_over_evaluated"]["rate"] == 1
    assert source["summary"]["pending"] == 1
    assert len(source["pending_level_ids"]) == 1
    assert source["summary"]["api"]["reported_tokens"]["totalTokenCount"] == 10


def test_attempt_dedup_cache_exclusion_and_missing_telemetry():
    records = [
        {"attempt": 1, "output": {}, "usage": {"totalTokenCount": 10}},
        {"attempt": 1, "error": "invalid structured response"},
        {"attempt": 2, "http_status": 503},
        {"attempt": 3, "error": "invalid structured response"},
    ]
    cases = [
        {"translation": {"status": "error", "seconds": 4}, "attempt_records": records},
        {
            "translation": {"status": "ok", "cache_hit": True, "seconds": 0.1},
            "attempt_records": records,
        },
    ]
    api = api_metrics(cases, "gemini")
    assert api["recorded_uncached_attempts"] == 3
    assert api["observed_json_outputs"] == 1
    assert api["attempts_with_unknown_response_outcome"] == 1
    assert api["reported_tokens"]["totalTokenCount"] == 10
    assert api["reported_tokens"]["thoughtsTokenCount"] is None
    assert api["translation_latency_seconds"]["samples"] == 1
    assert api["valid_translation_latency_seconds"]["samples"] == 0
    assert api["billed_cost_usd"] is None


def test_zero_coin_maps_do_not_inflate_collection_average():
    cases = [
        {
            "ppo": {
                "status": "ok",
                "outcome": "success",
                "coins_collected": 0,
                "coins_total": 0,
                "meets_expected_target": True,
            }
        }
    ]
    result = cohort_summary(cases, 1, "expected-fixture", expectations_planned=0)
    assert result["collection_ratio_nonempty_maps"]["mean"] is None
    assert result["weighted_coin_collection"]["rate"] is None
    assert result["ppo_target_over_evaluated"]["rate"] == 1


def test_percentiles_and_unknown_values():
    result = distribution([1, 2, 3, 4, float("nan"), True, -1, None])
    assert result["samples"] == 4
    assert result["p50"] == 2.5
    assert result["p95"] == pytest.approx(3.85)
    assert distribution([])["mean"] is None


def test_tampered_artifact_and_duplicate_source_rejected(tmp_path):
    folder = make_batch(tmp_path / "batch")
    source = load_batch(folder)
    with pytest.raises(ValueError, match="duplicate input"):
        build_report([source, copy.deepcopy(source)])
    next((folder / "cases").glob("*/ppo.json")).write_text("{}")
    with pytest.raises(ValueError, match="artifact changed"):
        load_batch(folder)


def test_repeated_runs_have_geometry_duplicates_without_merged_rates(tmp_path):
    first = load_batch(make_batch(tmp_path / "first"))
    second = load_batch(make_batch(tmp_path / "second"))
    report = build_report([first, second])
    assert report["evidence_counts"]["offline_batches"] == 2
    assert len(report["sources"]) == 2
    assert len(report["geometry_audit"]["duplicate_groups"]) == 2


def test_single_package_does_not_invent_semantic_or_ppo_measurements(tmp_path):
    case = GenerationCase(
        case_id="single", prompt="test", expected={"obstacle_count": 0, "gap_count": 0}
    )
    request = case.expected_request(1)
    root = tmp_path / "single"
    atomic_json(root / "request.json", request.model_dump())
    atomic_json(root / "level.json", generate_level(request).level.model_dump())
    atomic_json(root / "report.json", {"status": "candidate"})
    atomic_json(
        root / "intent.json", {"provider": "gemini", "level_id": request.level_id, "seed": 1}
    )
    atomic_json(
        root / "llm.json",
        {
            "identity": {"provider": "gemini", "model": "gemini-test"},
            "request_sha256": request_hash(request),
            "attempts": [],
        },
    )
    source = load_single(root)
    assert source["summary"]["llm_semantic_matches_over_planned"] is None
    assert source["summary"]["ppo_evaluated"] == 0
    assert source["summary"]["api"]["translation_latency_seconds"]["mean"] is None
    assert source["checkpoint_sha256"] is None


def test_cli_publishes_readonly_report_and_refuses_overwrite(tmp_path):
    folder = make_batch(tmp_path / "batch")
    before = {str(p): sha256_file(p) for p in folder.rglob("*.json")}
    output = tmp_path / "unified"
    command = [
        sys.executable,
        "-m",
        "scripts.report_generated_levels",
        "--batch-dir",
        str(folder),
        "--output-dir",
        str(output),
    ]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    assert process.returncode == 0, process.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["evidence_counts"]["offline_batches"] == 1
    assert (output / "report.md").exists()
    assert {str(p): sha256_file(p) for p in folder.rglob("*.json")} == before
    assert subprocess.run(command, capture_output=True, check=False).returncode == 2
