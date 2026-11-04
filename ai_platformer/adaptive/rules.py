"""Deterministic, conservative DDA suggestions from an exact profile snapshot."""

from math import isclose, isfinite

from .dda import (DDAContext, DDAProposal, DecisionReason, context_hash,
                  profile_reference, validate_proposal)
from .metrics import digest
from ai_platformer.content.generation_request import GenerationRequest, request_hash

POLICY_VERSION = "rules-v1"
MIN_GROUP_ATTEMPTS = 2
EASE_SUCCESS_RATE = 0.4
CHALLENGE_SUCCESS_RATE = 0.8
CHALLENGE_PROGRESS = 0.9
STEP_PX = 8


def _observations(profile):
    """Reject contradictory facts; missing measurements produce a hold."""
    observations = []
    for group in profile["by_map"]:
        facts = group["observed"]
        count = len(group["attempt_ids"])
        fields = ("attempt_count", "valid_terminal_attempts", "successes", "deaths", "timeouts")
        if any(type(facts.get(key)) is not int or facts[key] < 0 for key in fields):
            raise ValueError("invalid per-map outcome counts")
        if (facts["attempt_count"] != count or facts["valid_terminal_attempts"] != count
                or sum(facts[k] for k in ("successes", "deaths", "timeouts")) != count):
            raise ValueError("per-map outcome counts disagree with evidence")
        rate, progress = facts.get("success_rate"), facts.get("mean_max_progress")
        for value in (rate, progress):
            if value is not None and (type(value) not in (float, int)
                                      or not isfinite(value) or not 0 <= value <= 1):
                raise ValueError("invalid per-map measurement")
        if rate is not None and (not count or not isclose(rate, facts["successes"] / count)):
            raise ValueError("success rate disagrees with outcome counts")
        observations.append((count, rate, progress))
    return observations


def suggest(context, profile, *, decision_id=None):
    """Return a contract-checked suggestion, without generating or applying content.

    Every map/protocol group must have two samples and agree on direction. This
    avoids letting repeated easy-map wins hide poor results on another map.
    Geometry changes are experiments, not diagnoses of a particular jump error.
    """
    context = DDAContext.model_validate(
        context.model_dump() if isinstance(context, DDAContext) else context)
    if context.profile != profile_reference(profile):
        raise ValueError("profile snapshot does not match context")
    observations = _observations(profile)
    action, code, summary = "hold", "no_change_needed", "各地图表现未一致达到调整阈值，保持当前请求。"
    candidate = context.base_request
    evidence = []
    if context.profile.status != "ready":
        code, summary = "insufficient_data", "至少需要 5 次有效尝试和 2 张不同地图。"
    elif not any(context.permissions.model_dump().values()):
        code, summary = "constraints_locked", "没有获准调整的参数，保持当前请求。"
    elif (any(profile["dimensions"].get(name, {}).get("status") != "observed"
              for name in ("completion_stability", "progress_efficiency"))
          or not observations
          or any(n < MIN_GROUP_ATTEMPTS or rate is None or progress is None
                 for n, rate, progress in observations)):
        code, summary = "missing_dimension_evidence", "每个地图与协议分组需至少 2 次尝试及通关率、进度记录。"
    else:
        if all(rate <= EASE_SUCCESS_RATE for _, rate, _ in observations):
            action, code = "ease", "performance_decline"
            summary = "各地图分组通关率均不超过 40%，建议小幅降低几何范围上限；不推断具体失败原因。"
        elif all(rate >= CHALLENGE_SUCCESS_RATE and progress >= CHALLENGE_PROGRESS
                 for _, rate, progress in observations):
            action, code = "challenge", "performance_stable"
            summary = "各地图分组通关率均至少 80%，平均最大进度至少 90%，建议小幅提高几何范围下限。"
        if action != "hold":
            # Change one authorized parameter per suggestion in a frozen order.
            data = context.base_request.model_dump()
            for name in ("obstacle_height", "gap_width"):
                permission = getattr(context.permissions, name)
                bounds = data[name]
                if permission is None or bounds["minimum"] == bounds["maximum"]:
                    continue
                step = min(STEP_PX, permission.max_endpoint_delta_px,
                           bounds["maximum"] - bounds["minimum"])
                if action == "ease":
                    bounds["maximum"] -= step
                else:
                    bounds["minimum"] += step
                candidate = GenerationRequest.model_validate(data)
                evidence = context.profile.evidence
                break
            else:
                action, code, summary = "hold", "no_change_needed", "获准参数已无可收窄范围，保持当前请求。"
    proposal = DDAProposal(
        schema_version=1, kind="dda_proposal_v1",
        decision_id=decision_id or "rules-" + digest({"context": context_hash(context),
                                                    "policy": POLICY_VERSION})[:24],
        context_sha256=context_hash(context), profile_sha256=context.profile.profile_sha256,
        base_request_sha256=context.base_request_sha256, policy_version=POLICY_VERSION,
        action=action, reason=DecisionReason(code=code, summary=summary, evidence=evidence),
        candidate_request=candidate, candidate_request_sha256=request_hash(candidate),
    )
    validate_proposal(context, proposal, profile)
    return proposal
