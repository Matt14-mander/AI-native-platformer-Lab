"""DDA protocol v1: bounded suggestions for a future request, never game mutation."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from ai_platformer.content.generation_request import GenerationRequest, request_hash
from ai_platformer.content.level_spec import Identifier, SpecModel

from .metrics import digest

SHA256 = Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
Count = Annotated[int, Field(strict=True, ge=0, le=100)]


class ProtocolModel(SpecModel):
    schema_version: Literal[1]

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_version(cls, value):
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value


class PlayerIdentity(SpecModel):
    scope: Literal["player", "session_only"]
    id: Identifier


class EvidenceRef(SpecModel):
    session_id: Identifier
    attempt_id: Identifier


class ProfileRef(SpecModel):
    profile_version: Literal[1]
    metrics_version: Literal[1]
    profile_id: SHA256
    profile_sha256: SHA256
    environment_sha256: SHA256
    identity: PlayerIdentity
    status: Literal["ready", "insufficient_data"]
    valid_attempts: Count
    map_content_sha256: Annotated[list[SHA256], Field(max_length=100)]
    evidence: Annotated[list[EvidenceRef], Field(max_length=100)]

    @field_validator("profile_version", "metrics_version", mode="before")
    @classmethod
    def integer_versions(cls, value):
        if type(value) is not int:
            raise ValueError("profile/metrics versions must be integers")
        return value

    @model_validator(mode="after")
    def consistent(self):
        if self.valid_attempts != len(self.evidence):
            raise ValueError("profile count disagrees with evidence")
        if len({ref.attempt_id for ref in self.evidence}) != len(self.evidence):
            raise ValueError("duplicate profile evidence")
        if len(set(self.map_content_sha256)) != len(self.map_content_sha256):
            raise ValueError("duplicate map content hashes")
        if len(self.map_content_sha256) > self.valid_attempts:
            raise ValueError("more maps than valid attempts")
        ready = self.valid_attempts >= 5 and len(self.map_content_sha256) >= 2
        if (self.status == "ready") != ready:
            raise ValueError("profile readiness disagrees with v1 sample gate")
        expected = digest({"identity": self.identity.model_dump(),
                           "environment": self.environment_sha256})
        if self.profile_id != expected:
            raise ValueError("profile identity/environment hash mismatch")
        if self.identity.scope == "session_only" and any(
            ref.session_id != self.identity.id for ref in self.evidence
        ):
            raise ValueError("anonymous evidence crosses sessions")
        return self


class ParameterPermission(SpecModel):
    max_endpoint_delta_px: Annotated[int, Field(strict=True, ge=1, le=32)]


class AdjustmentPermissions(SpecModel):
    # No permission is inferred from a base request's broad generation range.
    obstacle_height: ParameterPermission | None = None
    gap_width: ParameterPermission | None = None


class DDAContext(ProtocolModel):
    kind: Literal["dda_context_v1"]
    context_id: Identifier
    mode: Literal["suggestion_only"] = "suggestion_only"
    boundary: Literal["before_next_level_generation"] = "before_next_level_generation"
    profile: ProfileRef
    target_environment_sha256: SHA256
    base_request: GenerationRequest
    base_request_sha256: SHA256
    permissions: AdjustmentPermissions = Field(default_factory=AdjustmentPermissions)

    @model_validator(mode="after")
    def consistent(self):
        if request_hash(self.base_request) != self.base_request_sha256:
            raise ValueError("base request hash mismatch")
        if self.target_environment_sha256 != self.profile.environment_sha256:
            raise ValueError("target environment differs from profile")
        for parameter, count in (("obstacle_height", self.base_request.obstacle_count),
                                 ("gap_width", self.base_request.gap_count)):
            if getattr(self.permissions, parameter) is not None and count == 0:
                raise ValueError(f"cannot authorize {parameter} without the corresponding hazard")
        return self


class DecisionReason(SpecModel):
    code: Literal[
        "insufficient_data", "constraints_locked", "missing_dimension_evidence",
        "performance_decline", "performance_stable", "no_change_needed", "manual_review",
    ]
    summary: Annotated[str, Field(strict=True, min_length=1, max_length=1000)]
    evidence: Annotated[list[EvidenceRef], Field(max_length=100)] = Field(default_factory=list)


class DDAProposal(ProtocolModel):
    kind: Literal["dda_proposal_v1"]
    decision_id: Identifier
    context_sha256: SHA256
    profile_sha256: SHA256
    base_request_sha256: SHA256
    policy_version: Identifier
    action: Literal["hold", "ease", "challenge"]
    reason: DecisionReason
    candidate_request: GenerationRequest
    candidate_request_sha256: SHA256
    application: Literal["not_applied"] = "not_applied"
    validation_status: Literal["pending"] = "pending"
    fallback: Literal["hold"] = "hold"

    @model_validator(mode="after")
    def candidate_hash(self):
        if request_hash(self.candidate_request) != self.candidate_request_sha256:
            raise ValueError("candidate request hash mismatch")
        refs = [(ref.session_id, ref.attempt_id) for ref in self.reason.evidence]
        if len(refs) != len(set(refs)):
            raise ValueError("duplicate decision evidence")
        return self


def context_hash(context: DDAContext) -> str:
    return digest(context.model_dump())


def profile_reference(profile: dict) -> ProfileRef:
    """Bind the exact v1 snapshot; this is integrity checking, not authentication."""
    if type(profile.get("schema_version")) is not int or profile["schema_version"] != 1:
        raise ValueError("unsupported profile schema")
    evidence = profile["evidence"]
    groups = profile["by_map"]
    ids = [ref["attempt_id"] for ref in evidence]
    grouped_ids = [attempt_id for group in groups for attempt_id in group["attempt_ids"]]
    if sorted(ids) != sorted(grouped_ids) or len(set(grouped_ids)) != len(grouped_ids):
        raise ValueError("profile map groups disagree with evidence")
    maps = sorted({group["level_sha256"] for group in groups})
    readiness = profile["readiness"]
    if type(readiness["independent_map_count"]) is not int or readiness["independent_map_count"] != len(maps):
        raise ValueError("profile map count mismatch")
    if readiness["minimum_attempts"] != 5 or readiness["minimum_maps"] != 2:
        raise ValueError("unsupported readiness thresholds")
    return ProfileRef.model_validate({
        "profile_version": profile["profile_version"], "metrics_version": profile["metrics_version"],
        "profile_id": profile["profile_id"], "profile_sha256": digest(profile),
        "environment_sha256": profile["environment_sha256"], "identity": profile["identity"],
        "status": profile["status"], "valid_attempts": readiness["valid_attempts"],
        "map_content_sha256": maps, "evidence": evidence,
    })


def validate_proposal(context, proposal, profile):
    """Revalidate objects and all cross-document constraints before accepting a suggestion."""
    context = DDAContext.model_validate(context.model_dump() if isinstance(context, DDAContext) else context)
    proposal = DDAProposal.model_validate(proposal.model_dump() if isinstance(proposal, DDAProposal) else proposal)
    if context.profile != profile_reference(profile):
        raise ValueError("profile snapshot does not match context")
    if proposal.context_sha256 != context_hash(context):
        raise ValueError("proposal belongs to another context or permission set")
    if proposal.profile_sha256 != context.profile.profile_sha256:
        raise ValueError("proposal profile hash mismatch")
    if proposal.base_request_sha256 != context.base_request_sha256:
        raise ValueError("proposal base request hash mismatch")
    available = {(ref.session_id, ref.attempt_id) for ref in context.profile.evidence}
    if any((ref.session_id, ref.attempt_id) not in available for ref in proposal.reason.evidence):
        raise ValueError("decision cites evidence outside the profile window")
    if context.profile.status == "insufficient_data":
        if proposal.action != "hold" or proposal.reason.code != "insufficient_data":
            raise ValueError("insufficient data requires hold/insufficient_data")
    elif proposal.reason.code == "insufficient_data":
        raise ValueError("ready profile cannot claim insufficient sample count")
    if (
        not any(context.permissions.model_dump().values()) and context.profile.status == "ready"
        and (proposal.action != "hold" or proposal.reason.code != "constraints_locked")
    ):
        raise ValueError("locked permissions require hold/constraints_locked")
    before, after = context.base_request.model_dump(), proposal.candidate_request.model_dump()
    changes = []
    for key in before:
        if before[key] == after[key]:
            continue
        if key not in {"obstacle_height", "gap_width"}:
            raise ValueError(f"immutable generation field changed: {key}")
        permission = getattr(context.permissions, key)
        if permission is None:
            raise ValueError(f"parameter is locked: {key}")
        original, candidate = before[key], after[key]
        if not original["minimum"] <= candidate["minimum"] <= candidate["maximum"] <= original["maximum"]:
            raise ValueError(f"candidate expands user geometry envelope: {key}")
        delta = {endpoint: candidate[endpoint] - original[endpoint] for endpoint in ("minimum", "maximum")}
        if any(abs(value) > permission.max_endpoint_delta_px for value in delta.values()):
            raise ValueError(f"endpoint change exceeds authorized step: {key}")
        if proposal.action == "ease" and any(value > 0 for value in delta.values()):
            raise ValueError("ease cannot increase a geometry endpoint")
        if proposal.action == "challenge" and any(value < 0 for value in delta.values()):
            raise ValueError("challenge cannot decrease a geometry endpoint")
        changes.append({"parameter": key, "before": original, "after": candidate,
                        "endpoint_delta_px": delta})
    if (proposal.action == "hold") != (len(changes) == 0):
        raise ValueError("hold must preserve the request; ease/challenge require a real change")
    if proposal.action != "hold" and not proposal.reason.evidence:
        raise ValueError("adjustment must cite profile evidence")
    return {
        "schema_version": 1, "kind": "dda_contract_validation_v1", "contract_valid": True,
        "context_sha256": context_hash(context), "proposal_sha256": digest(proposal.model_dump()),
        "profile_sha256": context.profile.profile_sha256,
        "base_request_sha256": context.base_request_sha256,
        "candidate_request_sha256": proposal.candidate_request_sha256,
        "action": proposal.action, "changes": changes, "application": "not_applied",
        "generation": "not_run", "static_validation": "not_run",
        "route_verification": "not_run", "ppo_evaluation": "not_run",
    }


def hold_proposal(context: DDAContext, *, decision_id: str) -> DDAProposal:
    """A deterministic protocol fallback, not a behavior-based DDA policy."""
    reason = "insufficient_data" if context.profile.status == "insufficient_data" else (
        "constraints_locked" if not any(context.permissions.model_dump().values()) else "no_change_needed"
    )
    return DDAProposal(
        schema_version=1, kind="dda_proposal_v1", decision_id=decision_id,
        context_sha256=context_hash(context), profile_sha256=context.profile.profile_sha256,
        base_request_sha256=context.base_request_sha256, policy_version="protocol-hold-v1",
        action="hold", reason=DecisionReason(code=reason, summary="Keep the caller's request unchanged."),
        candidate_request=context.base_request, candidate_request_sha256=context.base_request_sha256,
    )
