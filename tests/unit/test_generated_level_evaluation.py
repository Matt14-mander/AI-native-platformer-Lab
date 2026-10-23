import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from ai_platformer.agents.llm import evaluation
from ai_platformer.content.generation_suite import load_suite
from ai_platformer.content.pcg import SEGMENTS_V1_PHYSICS
from ai_platformer.envs.factory import EnvironmentFactory

SUITE = "game_content/generation_suites/small_v1.json"


def settings(tmp_path):
    return {
        "provider": "expected-fixture",
        "model": "offline-expected-fixture",
        "tls_mode": "tls12",
        "cache_dir": str(tmp_path / "cache"),
        "max_attempts": 1,
        "timeout": 1,
        "route_seconds": 1,
        "route_expansions": 100,
    }


def backend(tmp_path):
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"test checkpoint")
    checkpoint.with_suffix(".json").write_text("{}")
    return SimpleNamespace(
        model_path=checkpoint,
        factory=EnvironmentFactory(
            {
                "environment_id": "PlatformerState-v2",
                "physics": asdict(SEGMENTS_V1_PHYSICS),
                "episode_step_limit": 1024,
            }
        ),
    )


def test_summary_has_explicit_denominators_and_no_fake_llm_score():
    rows = [
        {
            "level_id": "a",
            "generation": {"status": "valid", "geometry_sha256": "same"},
            "translation": {"status": "ok", "cache_hit": False, "seconds": 2},
            "semantics": {"passed": True},
            "route": {"status": "unknown"},
            "ppo": {
                "status": "ok",
                "outcome": "success",
                "collection_ratio": 0.5,
                "meets_expected_target": False,
            },
        },
        {
            "level_id": "b",
            "generation": {"status": "valid", "geometry_sha256": "same"},
            "ppo": {
                "status": "ok",
                "outcome": "success",
                "collection_ratio": None,
                "meets_expected_target": True,
            },
        },
    ]
    report = evaluation.summarize(rows, planned=4, provider="expected-fixture")
    assert report["rates_over_planned"]["ppo_success"] == 0.5
    assert report["ppo_success_rate_over_evaluated"] == 1
    assert report["mean_collection_ratio_nonempty_maps"] == 0.5
    assert report["maps_with_collection_denominator"] == 1
    assert report["llm_semantic_match_rate_over_planned"] is None
    assert report["mean_uncached_valid_translation_seconds"] is None
    assert report["billed_cost_usd"] is None
    assert report["duplicate_geometry_groups"] == [["a", "b"]]
    assert report["remaining"] == 2


def test_unknown_search_does_not_skip_ppo(tmp_path, monkeypatch):
    suite = load_suite(SUITE)
    folder = tmp_path / "package"
    folder.mkdir()
    called = []
    monkeypatch.setattr(
        "ai_platformer.content.routes.search_route",
        lambda factory, **kwargs: {
            "status": "unknown",
            "reason": "budget_exhausted",
            "witness": None,
        },
    )

    def evaluate(loaded, factory, **kwargs):
        called.append((factory.step_limit, kwargs))
        return {"status": "ok", "outcome": "success", "meets_expected_target": True}, {
            "actions": []
        }

    monkeypatch.setattr(evaluation, "evaluate_ppo", evaluate)
    row = evaluation.candidate(
        suite.cases[0],
        42,
        folder=folder,
        suite=suite,
        loaded=backend(tmp_path),
        settings=settings(tmp_path),
        run_id="test",
    )
    assert row["status"] == "completed"
    assert row["route"]["status"] == "unknown"
    assert row["ppo"]["outcome"] == "success"
    assert called[0][0] == 1024


def fake_candidate(case, seed, *, folder, run_id, **kwargs):
    evaluation.atomic_json(folder / "probe.json", {"test": True})
    row = {
        "case_id": case.case_id,
        "level_id": case.expected_request(seed).level_id,
        "generation_seed": seed,
        "run_sha256": run_id,
        "status": "completed",
        "translation": {"status": "ok", "cache_hit": False},
        "artifact_sha256": {"probe.json": evaluation.sha256_file(folder / "probe.json")},
    }
    evaluation.atomic_json(folder / "result.json", row)
    return row


def test_resume_reuses_completed_cases_and_rejects_changed_inputs_or_artifacts(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(evaluation, "candidate", fake_candidate)
    suite, loaded, config = load_suite(SUITE), backend(tmp_path), settings(tmp_path)
    output = tmp_path / "output"
    first = evaluation.run_suite(suite, output=output, loaded=loaded, settings=config, limit=2)
    assert first["status"] == "complete"
    monkeypatch.setattr(
        evaluation, "candidate", lambda *a, **k: pytest.fail("completed case reran")
    )
    resumed = evaluation.run_suite(
        suite, output=output, loaded=loaded, settings=config, limit=2, resume=True
    )
    assert resumed["summary"] == first["summary"]
    with pytest.raises(ValueError, match="identity mismatch"):
        evaluation.run_suite(
            suite,
            output=output,
            loaded=loaded,
            settings={**config, "timeout": 2},
            limit=2,
            resume=True,
        )
    next((output / "cases").glob("*/probe.json")).write_text("{}")
    with pytest.raises(ValueError, match="artifact changed"):
        evaluation.run_suite(
            suite, output=output, loaded=loaded, settings=config, limit=2, resume=True
        )


def test_three_translation_failures_stop_batch_and_preserve_pending_candidates(
    tmp_path, monkeypatch
):
    def fail(case, seed, *, folder, run_id, **kwargs):
        row = fake_candidate(case, seed, folder=folder, run_id=run_id)
        row.update(
            status="error",
            failure={"stage": "translation", "http_statuses": [503]},
            translation={"status": "error"},
        )
        evaluation.atomic_json(folder / "result.json", row)
        return row

    monkeypatch.setattr(evaluation, "candidate", fail)
    output = tmp_path / "output"
    report = evaluation.run_suite(
        load_suite(SUITE), output=output, loaded=backend(tmp_path), settings=settings(tmp_path)
    )
    assert report["status"] == "incomplete"
    assert report["summary"]["processed"] == 3
    assert report["summary"]["remaining"] == 15
    assert json.loads((output / "report.json").read_text())["summary"]["processed"] == 3


def test_semantic_mismatch_is_retained_without_rewriting_model_design(tmp_path, monkeypatch):
    suite = load_suite(SUITE)
    case = suite.cases[0]
    request = case.expected_request(42).model_dump()
    request["obstacle_count"] = 1
    folder = tmp_path / "package"
    folder.mkdir()
    monkeypatch.setattr(
        "ai_platformer.content.routes.search_route",
        lambda *a, **k: {"status": "unknown", "reason": "budget_exhausted", "witness": None},
    )
    monkeypatch.setattr(
        evaluation,
        "evaluate_ppo",
        lambda *a, **k: (
            {"status": "ok", "outcome": "success", "meets_expected_target": False},
            {"actions": []},
        ),
    )

    def transport(payload, **kwargs):
        return {
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {
                        "parts": [{"text": json.dumps({"request": request, "unsupported": None})}]
                    },
                }
            ]
        }

    config = {**settings(tmp_path), "provider": "gemini", "model": "gemini-test"}
    row = evaluation.candidate(
        case,
        42,
        folder=folder,
        suite=suite,
        loaded=backend(tmp_path),
        settings=config,
        run_id="test",
        transport=transport,
    )
    assert row["status"] == "completed"
    assert not row["semantics"]["passed"]
    assert row["semantics"]["mismatches"][0]["field"] == "obstacle_count"
    assert json.loads((folder / "request.json").read_text())["obstacle_count"] == 1
    assert (folder / "ppo.json").exists()
