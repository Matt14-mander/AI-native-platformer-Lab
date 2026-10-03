"""Shared per-layout readiness gates and balanced multi-task checkpoint ranking."""

from __future__ import annotations


def assess_prerequisites(stages: list[dict], report: dict) -> dict:
    tasks = {}
    for stage in stages:
        failures = {}
        for level in stage["validation_levels"]:
            result = report.get("by_level", {}).get(level)
            reasons = []
            if result is None or result["success_rate"] < stage["success_threshold"]:
                reasons.append("success_rate")
            limit = stage.get("max_success_steps")
            if limit is not None:
                successful = [
                    episode
                    for episode in report.get("episodes", [])
                    if episode["level_id"] == level and episode["outcome"] == "success"
                ]
                if not successful or any(episode["steps"] > limit for episode in successful):
                    reasons.append("success_steps")
            if reasons:
                failures[level] = reasons
        count = len(stage["validation_levels"])
        tasks[stage["task"]] = {
            "passed": not failures,
            "layouts": count,
            "qualified_layouts": count - len(failures),
            "qualified_fraction": (count - len(failures)) / count,
            "failures": failures,
            "success_threshold": stage["success_threshold"],
            "max_success_steps": stage.get("max_success_steps"),
        }
    return {
        "passed": bool(tasks) and all(task["passed"] for task in tasks.values()),
        "tasks": tasks,
    }


def joint_selection_score(stages: list[dict], report: dict) -> list[float]:
    gate = assess_prerequisites(stages, report)
    fractions = [task["qualified_fraction"] for task in gate["tasks"].values()]
    levels = {level for stage in stages for level in stage["validation_levels"]}
    successes = [
        item["steps"]
        for item in report.get("episodes", [])
        if item["level_id"] in levels and item["outcome"] == "success"
    ]
    progress = [report["by_level"].get(level, {}).get("mean_progress", 0) for level in levels]
    return [
        float(gate["passed"]),
        min(fractions),
        sum(fractions) / len(fractions),
        sum(progress) / len(progress),
        -sum(successes) / len(successes) if successes else 0,
    ]
