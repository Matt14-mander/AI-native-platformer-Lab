"""Candidate package relocation, target enforcement and route-status boundaries."""

import json
import subprocess
import sys

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("gymnasium")

from ai_platformer.content.routes import replay_route


def test_generated_package_verifies_and_replays_after_atomic_publish(tmp_path):
    output = tmp_path / "package"
    command = [
        sys.executable,
        "-m",
        "scripts.generate_level",
        "--request",
        "game_content/generation_requests/mixed_v1.json",
        "--output-dir",
        str(output),
        "--verify-route",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "report.json").read_text())
    route = json.loads((output / "route.json").read_text())
    assert report["status"] == "verified" and route["witness"]["min_coin_ratio"] == 0.75
    assert route["witness"]["coins_collected"] / route["witness"]["coins_total"] >= 0.75
    relocated = tmp_path / "relocated"
    output.rename(relocated)
    assert replay_route(relocated / "level.json", route["witness"])["passed"]


def test_unknown_candidate_is_retained_but_not_marked_verified(tmp_path):
    output = tmp_path / "unknown"
    command = [
        sys.executable,
        "-m",
        "scripts.generate_level",
        "--request",
        "game_content/generation_requests/mixed_v1.json",
        "--output-dir",
        str(output),
        "--verify-route",
        "--max-expansions",
        "1",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert (output / "level.json").exists()
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "unknown"
    route = json.loads((output / "route.json").read_text())
    assert route["witness"] is None and route["reason"] == "budget_exhausted"
