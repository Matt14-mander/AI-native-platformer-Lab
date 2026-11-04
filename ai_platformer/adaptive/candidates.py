"""Player-physics candidate preparation and explicit, transactional selection."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from uuid import uuid4
from concurrent.futures import CancelledError

from ai_platformer.content.play_session import PlayAdmissionError, PlayEntry, PreparedPlayLevel, PlayRepository
from ai_platformer.content.generation_request import GenerationRequest, request_hash
from ai_platformer.content.pcg import generate_player_level
from ai_platformer.content.level_spec import document_hash, load_level_spec
from ai_platformer.content.routes import search_route
from ai_platformer.envs.factory import EnvironmentFactory
from .dda import DDAContext, profile_reference, validate_proposal
from .human_play import HumanPlayProtocol
from .metrics import digest
from .progress import atomic_write
from .rules import suggest
from .candidate_catalog import read_catalog


@dataclass(frozen=True)
class CandidateOffer:
    origin: PreparedPlayLevel
    message: str
    prepared: PreparedPlayLevel | None = None
    entry: PlayEntry | None = None
    directory: Path | None = None
    document_sha256: str | None = None


def prepare_offer(session, *, cancelled=lambda: False):
    """No selection changes until accept_offer; failures leave current play intact."""
    origin = session.prepared
    records = session.records
    if not records:
        return CandidateOffer(origin, "请开启游玩记录后查看建议。")
    directory, report = None, None
    def check_cancelled():
        if cancelled():
            raise CancelledError()
    try:
        check_cancelled()
        profile = records.profile_snapshot(origin)
        check_cancelled()
        if profile is None:
            return CandidateOffer(origin, "有效记录不足，先在不同关卡完成几次游玩。")
        seed = int(digest({"profile": profile, "seed": origin.settings.seed})[:8], 16)
        # A new practice course, not a claim to reproduce an arbitrary source map.
        base = dict(origin.design_request or {
            "schema_version": 1, "obstacle_count": 1, "gap_count": 1,
            "obstacle_height": {"minimum": 48, "maximum": 80},
            "gap_width": {"minimum": 96, "maximum": 128},
            "coin_layout": "ground", "min_coin_ratio": 0.0,
        })
        base.update(level_id=f"practice-{seed:08x}", seed=seed)
        request = GenerationRequest.model_validate(base)
        environment = digest({"environment": HumanPlayProtocol(origin).environment(), "tick_hz": 60})
        context = DDAContext(
            schema_version=1, kind="dda_context_v1", context_id=f"player-{seed:08x}",
            profile=profile_reference(profile), target_environment_sha256=environment,
            base_request=request, base_request_sha256=request_hash(request),
            permissions={"obstacle_height": {"max_endpoint_delta_px": 8},
                         "gap_width": {"max_endpoint_delta_px": 8}},
        )
        proposal = suggest(context, profile)
        report = {"schema_version": 1, "profile": profile, "context": context.model_dump(),
                  "proposal": proposal.model_dump(), "contract": validate_proposal(context, proposal, profile),
                  "player_physics": asdict(origin.settings.physics), "tick_hz": 60,
                  "application": "not_applied", "status": "hold"}
        directory = records.directory / "candidates" / uuid4().hex
        directory.mkdir(parents=True)
        if proposal.action == "hold":
            atomic_write(directory / "report.json", report)
            messages = {"insufficient_data": "先完成至少 5 次尝试，覆盖 2 张不同地图。",
                        "missing_dimension_evidence": "每张地图还需要更多完整游玩记录。"}
            return CandidateOffer(origin, messages.get(proposal.reason.code, "建议保持当前难度，可继续原关卡。"))
        check_cancelled()
        atomic_write(directory / "report.json", {**report, "status": "validating"})
        result = generate_player_level(proposal.candidate_request, origin.settings.physics)
        path = directory / "level.json"
        atomic_write(path, result.level.model_dump())
        atomic_write(directory / "request.json", proposal.candidate_request.model_dump())
        factory = EnvironmentFactory({
            "environment_id": "PlatformerState-v2", "level_spec": str(path),
            "physics": asdict(origin.settings.physics), "action_repeat": 1,
            "episode_step_limit": min(2400, origin.settings.physics.max_episode_steps),
        })
        route = search_route(factory, seed=origin.settings.seed, min_coin_ratio=request.min_coin_ratio,
                             max_steps=2400, max_expansions=6000, beam_width=12, max_seconds=2.0,
                             cancelled=cancelled)
        check_cancelled()
        report.update(generation=result.report, route=route,
                      document_sha256=document_hash(result.level), status=route["status"])
        atomic_write(directory / "report.json", report)
        if route["status"] != "verified":
            return CandidateOffer(origin, "暂未验证出通关路线，保留原关卡，可重试或返回菜单。")
        definition = factory.repository.load(factory.level_id)
        prepared = PreparedPlayLevel(
            replace(origin.settings, level_id=factory.level_id), PlayRepository(definition),
            factory.repository.solids_by_kind(factory.level_id), "player_dda_verified", route["witness"],
            proposal.candidate_request.model_dump(),
        )
        message = ("依据近期通关表现，建议尝试更轻松的新练习关。" if proposal.action == "ease"
                   else "依据近期通关表现，建议尝试稍有挑战的新练习关。")
        return CandidateOffer(origin, message, prepared, PlayEntry(
                              "建议练习关", path=path, design_request=proposal.candidate_request.model_dump()),
                              directory, document_hash(result.level))
    except CancelledError:
        if directory is not None and report is not None:
            try:
                atomic_write(directory / "report.json", {**report, "status": "cancelled"})
            except (OSError, ValueError):
                pass
        raise
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, ImportError, RecursionError) as error:
        if directory is not None and report is not None:
            try:
                atomic_write(directory / "report.json", {
                    **report, "status": "failed", "error_type": type(error).__name__})
            except (OSError, ValueError):
                pass
        raise PlayAdmissionError("建议生成或验证失败，原关卡已保留。") from error


def accept_offer(session, offer):
    """Persist the user's choice before changing the active catalog or map."""
    if (offer is None or offer.prepared is None or session.prepared is not offer.origin
            or not session.records or session.records.store.disabled):
        raise PlayAdmissionError("建议已失效，请重新查看。")
    try:
        if document_hash(load_level_spec(offer.entry.path)) != offer.document_sha256:
            raise ValueError("candidate changed after verification")
        catalog = read_catalog(session.records.directory)
        if len(catalog) >= 128:
            raise ValueError("candidate catalog is full")
        name = offer.directory.name
        if name not in catalog:
            catalog.append(name)
        atomic_write(session.records.directory / "candidate_catalog.json", catalog)
        # The catalog is the durable selection record; proposal remains immutable.
        session.entries += (offer.entry,)
        session.prepared, session.index = offer.prepared, len(session.entries) - 1
    except (OSError, ValueError, TypeError, RecursionError) as error:
        raise PlayAdmissionError("候选关卡无法保存或已改变，原关卡已保留。") from error
