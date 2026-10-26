"""DDA authority, immutable request pins, hash binding and offline validation."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

from ai_platformer.adaptive.dda import (
    DDAContext,
    DDAProposal,
    context_hash,
    hold_proposal,
    profile_reference,
    validate_proposal,
)
from ai_platformer.adaptive.profile import build_profiles
from ai_platformer.content.generation_request import GenerationRequest, request_hash


def synthetic_profile(*, ready=True):
    rows = []
    for index in range(5 if ready else 1):
        rows.append({
            "actor_type": "human", "identity": {"scope": "player", "id": "protocol_fixture"},
            "environment_sha256": "e" * 64, "profile_eligible": ready,
            "profile_exclusion_reasons": [] if ready else ["nonterminal_attempt"],
            "attempt_id": f"attempt-{index}", "session_id": "fixture-session",
            "level_sha256": ("a" if index < 3 else "b") * 64,
            "protocol_sha256": ("c" if index < 3 else "d") * 64, "level_id": f"map-{index%2}",
            "valid_terminal": ready, "reason": "success" if ready else "quit",
            "retry_after_failure": False, "coins_total": 1, "coins_collected": 1,
            "motion": None, "ticks": 60, "simulation_seconds": 1.0, "tick_hz": 60,
            "max_progress": 1.0, "net_horizontal_speed_px_s": 10.0,
            "jump_requests": 0, "accepted_jumps": 0, "landings": 0,
            "jump_hold_completed_ticks": [], "jump_hold_censored_ticks": None,
            "death_location": None, "interactions": {},
        })
    return build_profiles({"metrics_version": 1, "attempts": rows})[0]


def fixture_context(*, ready=True, unlocked=True):
    profile = synthetic_profile(ready=ready)
    request = GenerationRequest(schema_version=1, level_id="dda_next_001", seed=20261026)
    context = DDAContext(
        schema_version=1, kind="dda_context_v1", context_id="fixture-context",
        profile=profile_reference(profile), target_environment_sha256=profile["environment_sha256"],
        base_request=request, base_request_sha256=request_hash(request),
        permissions=({"obstacle_height": {"max_endpoint_delta_px": 16},
                      "gap_width": {"max_endpoint_delta_px": 8}} if unlocked else {}),
    )
    return context, profile


def proposal(context, action="ease", **updates):
    data = context.base_request.model_dump()
    data.update(updates or {"obstacle_height": {"minimum": 48, "maximum": 80}})
    candidate = GenerationRequest.model_validate(data)
    return DDAProposal(
        schema_version=1, kind="dda_proposal_v1", decision_id="fixture-decision",
        context_sha256=context_hash(context), profile_sha256=context.profile.profile_sha256,
        base_request_sha256=context.base_request_sha256, policy_version="fixture-policy-v1",
        action=action, reason={"code": "manual_review", "summary": "Synthetic protocol test only.",
                               "evidence": [context.profile.evidence[0].model_dump()]},
        candidate_request=candidate, candidate_request_sha256=request_hash(candidate),
    )


def test_valid_suggestions_are_deterministic_and_do_not_mutate_inputs():
    context, profile = fixture_context()
    original = copy.deepcopy((context.model_dump(), profile))
    candidate = proposal(context)
    result = validate_proposal(context, candidate, profile)
    assert result == validate_proposal(context, candidate, profile)
    assert result["contract_valid"] and result["application"] == "not_applied"
    assert result["generation"] == result["ppo_evaluation"] == "not_run"
    assert result["changes"] == [{"parameter": "obstacle_height", "before": {"minimum": 48, "maximum": 96},
                                  "after": {"minimum": 48, "maximum": 80},
                                  "endpoint_delta_px": {"minimum": 0, "maximum": -16}}]
    assert (context.model_dump(), profile) == original
    harder = proposal(context, "challenge", obstacle_height={"minimum": 64, "maximum": 96})
    assert validate_proposal(context, harder, profile)["action"] == "challenge"


@pytest.mark.parametrize("ready,unlocked,reason", [
    (False, True, "insufficient_data"), (False, False, "insufficient_data"),
    (True, False, "constraints_locked"), (True, True, "no_change_needed"),
])
def test_hold_fallback_enforces_data_and_permission_gates(ready, unlocked, reason):
    context, profile = fixture_context(ready=ready, unlocked=unlocked)
    candidate = hold_proposal(context, decision_id="hold-1")
    assert candidate.reason.code == reason
    assert validate_proposal(context, candidate, profile)["changes"] == []


@pytest.mark.parametrize("updates", [
    {"level_id": "another"}, {"seed": 1}, {"obstacle_count": 2}, {"gap_count": 1},
    {"min_coin_ratio": 0.5}, {"coin_layout": "ground"},
    {"obstacle_height": {"minimum": 40, "maximum": 96}},
    {"obstacle_height": {"minimum": 48, "maximum": 112}},
    {"obstacle_height": {"minimum": 48, "maximum": 64}},
    {"gap_width": {"minimum": 96, "maximum": 136}},
    {"obstacle_height": {"minimum": 64, "maximum": 80}},
])
def test_request_pins_envelope_and_endpoint_limits_cannot_be_bypassed(updates):
    context, profile = fixture_context()
    with pytest.raises(ValueError):
        validate_proposal(context, proposal(context, **updates), profile)


@pytest.mark.parametrize("change", ["profile", "permission", "base", "environment", "evidence", "version"])
def test_cross_document_binding_and_versions_reject_stale_or_forged_inputs(change):
    context, profile = fixture_context()
    candidate = proposal(context)
    if change == "profile":
        profile["lifetime"]["retry_starts_after_failure"] += 1
    elif change == "permission":
        context = DDAContext.model_validate({**context.model_dump(), "permissions": {}})
    elif change == "base":
        data = candidate.model_dump(); data["base_request_sha256"] = "0"*64
        candidate = DDAProposal.model_validate(data)
    elif change == "environment":
        data = context.model_dump(); data["target_environment_sha256"] = "0"*64
        with pytest.raises(ValueError):
            DDAContext.model_validate(data)
        return
    elif change == "version":
        data = candidate.model_dump(); data["schema_version"] = True
        with pytest.raises(ValueError):
            DDAProposal.model_validate(data)
        return
    else:
        data = candidate.model_dump(); data["reason"]["evidence"][0]["attempt_id"] = "foreign"
        candidate = DDAProposal.model_validate(data)
    with pytest.raises(ValueError):
        validate_proposal(context, candidate, profile)


@pytest.mark.parametrize("updates", [
    {"mode": "auto_apply"}, {"boundary": "during_episode"}, {"schema_version": 2},
    {"permissions": {"min_coin_ratio": {"max_endpoint_delta_px": 16}}},
    {"permissions": {"obstacle_height": {"max_endpoint_delta_px": True}}},
    {"permissions": {"obstacle_height": {"max_endpoint_delta_px": 33}}},
    {"run_command": "anything"},
])
def test_protocol_rejects_runtime_mutation_and_unlisted_authority(updates):
    context, _ = fixture_context()
    with pytest.raises(ValueError):
        DDAContext.model_validate({**context.model_dump(), **updates})


def test_insufficient_data_and_locked_permissions_cannot_adjust():
    context, profile = fixture_context(ready=False)
    data = hold_proposal(context, decision_id="bad").model_dump()
    data["action"] = "ease"
    with pytest.raises(ValueError, match="insufficient"):
        validate_proposal(context, data, profile)
    context, profile = fixture_context(unlocked=False)
    with pytest.raises(ValueError, match="locked"):
        validate_proposal(context, proposal(context), profile)


def test_profile_reference_checks_counts_identity_and_map_evidence():
    profile = synthetic_profile()
    for mutate in (
        lambda p: p["readiness"].update(valid_attempts=50),
        lambda p: p["readiness"].update(independent_map_count=50),
        lambda p: p["identity"].update(id="another"),
        lambda p: p["by_map"][0]["attempt_ids"].append("missing"),
        lambda p: p.update(status="insufficient_data"),
    ):
        broken = copy.deepcopy(profile); mutate(broken)
        with pytest.raises(ValueError):
            profile_reference(broken)


def test_authorized_gap_change_and_multiple_geometry_changes_are_valid():
    context, profile = fixture_context()
    candidate = proposal(context, gap_width={"minimum": 96, "maximum": 144},
                         obstacle_height={"minimum": 48, "maximum": 80})
    report = validate_proposal(context, candidate, profile)
    assert {change["parameter"] for change in report["changes"]} == {"gap_width", "obstacle_height"}


@pytest.mark.parametrize("field,value", [
    ("application", "applied"), ("validation_status", "verified"),
    ("fallback", "challenge"), ("schema_version", True),
])
def test_proposal_cannot_claim_execution_or_validation(field, value):
    context, _ = fixture_context()
    data = proposal(context).model_dump(); data[field] = value
    with pytest.raises(ValueError):
        DDAProposal.model_validate(data)


def test_adjustments_need_evidence_real_changes_and_direction_consistency():
    context, profile = fixture_context()
    data = proposal(context).model_dump(); data["reason"]["evidence"] = []
    with pytest.raises(ValueError, match="cite"):
        validate_proposal(context, data, profile)
    data = proposal(context).model_dump()
    data["candidate_request"] = context.base_request.model_dump()
    data["candidate_request_sha256"] = context.base_request_sha256
    with pytest.raises(ValueError, match="real change"):
        validate_proposal(context, data, profile)
    with pytest.raises(ValueError, match="cannot decrease"):
        validate_proposal(context, proposal(context, "challenge"), profile)


def test_no_authority_for_absent_hazards_and_profile_versions_are_strict():
    context, profile = fixture_context()
    data = context.model_dump(); data["base_request"]["gap_count"] = 0
    data["base_request_sha256"] = request_hash(GenerationRequest.model_validate(data["base_request"]))
    with pytest.raises(ValueError, match="without"):
        DDAContext.model_validate(data)
    profile["profile_version"] = True
    with pytest.raises(ValueError, match="versions"):
        profile_reference(profile)


@pytest.mark.parametrize("model,filename", [
    (DDAContext, "dda_context_v1.schema.json"), (DDAProposal, "dda_proposal_v1.schema.json"),
])
def test_checked_in_json_schemas_match_models(model, filename):
    schema = model.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    assert json.loads((Path("schemas") / filename).read_text()) == schema


def test_validation_cli_accepts_hold_and_rejects_bad_candidate(tmp_path):
    context, profile = fixture_context(ready=False)
    candidate = hold_proposal(context, decision_id="fixture-hold")
    paths = {}
    for name, value in (("context", context.model_dump()), ("profile", profile),
                        ("proposal", candidate.model_dump())):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(value), encoding="utf-8")
    command = [sys.executable, "-m", "scripts.validate_dda_proposal", "--context", str(paths["context"]),
               "--profile", str(paths["profile"]), "--proposal", str(paths["proposal"])]
    for valid in (True, False):
        if not valid:
            bad = candidate.model_dump(); bad["candidate_request"]["seed"] += 1
            paths["proposal"].write_text(json.dumps(bad))
        output = tmp_path / f"result_{valid}.json"
        result = subprocess.run([*command, "--output", str(output)], capture_output=True, text=True, check=False)
        assert result.returncode == (0 if valid else 1), result.stderr
        report = json.loads(output.read_text())
        assert report["contract_valid"] is valid
        assert report["application"] == "not_applied"
    subprocess.run([sys.executable, "-c", ("import sys; import scripts.validate_dda_proposal; "
                    "assert not any(k in sys.modules for k in ('gymnasium','pygame','torch'))")], check=True)
