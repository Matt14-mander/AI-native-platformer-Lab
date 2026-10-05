"""Offline fixture pipeline exercises production packaging and route verification."""

import json
import subprocess
import sys

from ai_platformer.content.routes import replay_route


def test_llm_package_cache_and_replay(tmp_path):
    output = tmp_path / "first"
    command = [
        sys.executable,
        "-m",
        "scripts.generate_level_llm",
        "--provider",
        "fixture",
        "--fixture",
        "game_content/llm_fixtures/mixed_v1.json",
        "--prompt",
        "三个障碍两个坑，收集75%松果",
        "--level-id",
        "pcg_mixed_v1",
        "--seed",
        "20261005",
        "--cache-dir",
        str(tmp_path / "cache"),
        "--verify-route",
        "--output-dir",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "verified"
    assert not report["llm"]["cache_hit"]
    witness = json.loads((output / "route.json").read_text())["witness"]
    assert replay_route(output / "level.json", witness)["passed"]
    command[-1] = str(tmp_path / "second")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "second" / "report.json").read_text())["llm"]["cache_hit"]
    assert (output / "level.json").read_bytes() == (tmp_path / "second" / "level.json").read_bytes()
    assert subprocess.run(command, capture_output=True, check=False).returncode == 2


def test_llm_failure_preserves_report_without_geometry(tmp_path):
    fixture = tmp_path / "unsupported.json"
    fixture.write_text(json.dumps({"request": None, "unsupported": "moving platforms"}))
    output = tmp_path / "failure"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.generate_level_llm",
            "--provider",
            "fixture",
            "--fixture",
            str(fixture),
            "--prompt",
            "移动平台",
            "--level-id",
            "test",
            "--seed",
            "1",
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    assert not (output / "level.json").exists()
    assert json.loads((output / "report.json").read_text())["status"] == "error"
