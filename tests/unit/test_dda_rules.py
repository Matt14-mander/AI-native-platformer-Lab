import copy
import json
import subprocess
import sys

import pytest

from ai_platformer.adaptive.dda import DDAContext, profile_reference, validate_proposal
from ai_platformer.adaptive.profile import build_profiles
from ai_platformer.adaptive.rules import suggest
from ai_platformer.content.generation_request import GenerationRequest, request_hash


def fixture(outcomes=("success",) * 6, *, permissions=None, progress=1.0):
    rows = []
    for i, outcome in enumerate(outcomes):
        rows.append({
            "actor_type": "human", "identity": {"scope": "player", "id": "rules-test"},
            "environment_sha256": "e" * 64, "profile_eligible": True,
            "profile_exclusion_reasons": [], "attempt_id": f"a-{i}", "session_id": "s-1",
            "level_sha256": ("a" if i % 2 else "b") * 64,
            "protocol_sha256": "c" * 64, "level_id": f"map-{i % 2}",
            "valid_terminal": True, "reason": outcome, "retry_after_failure": False,
            "coins_total": 0, "coins_collected": 0, "motion": None, "ticks": 60,
            "simulation_seconds": 1.0, "max_progress": progress,
            "net_horizontal_speed_px_s": 1.0, "jump_requests": 0,
            "accepted_jumps": 0, "landings": 0, "jump_hold_completed_ticks": [],
            "jump_hold_censored_ticks": None, "death_location": None, "interactions": {},
        })
    profile = build_profiles({"metrics_version": 1, "attempts": rows})[0]
    request = GenerationRequest(schema_version=1, level_id="next-map", seed=42)
    context = DDAContext(
        schema_version=1, kind="dda_context_v1", context_id="rules-context",
        profile=profile_reference(profile), target_environment_sha256="e" * 64,
        base_request=request, base_request_sha256=request_hash(request),
        permissions=permissions if permissions is not None else {
            "obstacle_height": {"max_endpoint_delta_px": 16},
            "gap_width": {"max_endpoint_delta_px": 4}},
    )
    return context, profile


def rebind(context, profile):
    return DDAContext.model_validate({**context.model_dump(), "profile": profile_reference(profile)})


@pytest.mark.parametrize("outcomes,progress,action,reason", [
    (("death",) * 6, 0.3, "ease", "performance_decline"),
    (("success",) * 6, 1.0, "challenge", "performance_stable"),
    (("success", "death") * 3, 0.8, "hold", "no_change_needed"),
    (("success",) * 4, 1.0, "hold", "insufficient_data"),
    (("success",) * 6, 0.8, "hold", "no_change_needed"),
])
def test_policy_directions_and_map_disagreement(outcomes, progress, action, reason):
    context, profile = fixture(outcomes, progress=progress)
    original = copy.deepcopy((context.model_dump(), profile))
    proposal = suggest(context, profile)
    assert proposal.action == action and proposal.reason.code == reason
    assert proposal == suggest(context, profile)
    assert validate_proposal(context, proposal, profile)["contract_valid"]
    assert (context.model_dump(), profile) == original
    assert proposal.candidate_request.gap_width == context.base_request.gap_width
    if action == "ease":
        assert proposal.candidate_request.obstacle_height.maximum == 88
    elif action == "challenge":
        assert proposal.candidate_request.obstacle_height.minimum == 56


def test_permissions_and_single_parameter_limit():
    context, profile = fixture(permissions={})
    assert suggest(context, profile).reason.code == "constraints_locked"
    context, profile = fixture(permissions={"gap_width": {"max_endpoint_delta_px": 4}})
    proposal = suggest(context, profile)
    assert proposal.candidate_request.gap_width.minimum == 100
    assert proposal.candidate_request.obstacle_height == context.base_request.obstacle_height


@pytest.mark.parametrize("wins,progress,expected", [
    (2, 0.5, "ease"), (3, 0.8, "hold"), (4, 0.9, "challenge"), (4, 0.899, "hold"),
])
def test_exact_rate_and_progress_boundaries(wins, progress, expected):
    # Five samples per map; both maps have the same rate.
    outcomes = tuple(value for i in range(5) for value in
                     (("success",) * 2 if i < wins else ("death",) * 2))
    context, profile = fixture(outcomes, progress=progress)
    assert suggest(context, profile).action == expected


@pytest.mark.parametrize("missing", ["samples", "measurement", "dimension"])
def test_missing_evidence_holds(missing):
    context, profile = fixture(("success",) * 5 if missing == "samples" else ("success",) * 6)
    if missing == "samples":
        # Preserve global readiness while creating a single-sample protocol group.
        group = profile["by_map"][0]
        isolated = copy.deepcopy(group)
        isolated["attempt_ids"] = [group["attempt_ids"].pop()]
        isolated["protocol_sha256"] = "d" * 64
        for g in (group, isolated):
            n = len(g["attempt_ids"])
            for key in ("attempt_count", "valid_terminal_attempts", "successes"):
                g["observed"][key] = n
        profile["by_map"].append(isolated)
    elif missing == "measurement":
        profile["by_map"][0]["observed"]["mean_max_progress"] = None
    else:
        profile["dimensions"]["completion_stability"]["status"] = "unknown"
    assert suggest(rebind(context, profile), profile).reason.code == "missing_dimension_evidence"


def test_stale_snapshot_and_inconsistent_metrics_are_rejected():
    context, profile = fixture()
    profile["by_map"][0]["observed"]["successes"] = 0
    with pytest.raises(ValueError, match="snapshot"):
        suggest(context, profile)
    with pytest.raises(ValueError, match="counts"):
        suggest(rebind(context, profile), profile)


def test_exhausted_range_holds():
    context, profile = fixture(permissions={"obstacle_height": {"max_endpoint_delta_px": 8}})
    request = context.base_request.model_dump()
    request["obstacle_height"] = {"minimum": 48, "maximum": 48}
    request = GenerationRequest.model_validate(request)
    context = DDAContext.model_validate({**context.model_dump(), "base_request": request,
                                        "base_request_sha256": request_hash(request)})
    assert suggest(context, profile).action == "hold"


def test_cli_output_passes_existing_validator_and_does_not_overwrite(tmp_path):
    context, profile = fixture()
    for name, data in (("context", context.model_dump()), ("profile", profile)):
        (tmp_path / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")
    command = [sys.executable, "-m", "scripts.suggest_dda", "--context", str(tmp_path / "context.json"),
               "--profile", str(tmp_path / "profile.json"), "--output", str(tmp_path / "proposal.json")]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    data = (tmp_path / "proposal.json").read_bytes()
    assert validate_proposal(context, json.loads(data), profile)["action"] == "challenge"
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert (tmp_path / "proposal.json").read_bytes() == data
