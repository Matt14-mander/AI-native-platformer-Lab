"""Real player-physics route verification and result-screen opt-in lifecycle."""
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pygame
import pytest

from ai_platformer.adaptive import candidates
from ai_platformer.adaptive.human_play import HumanPlayProtocol, HumanPlayRecords
from ai_platformer.adaptive.metrics import digest
from ai_platformer.content.play_session import PlayAdmissionError, default_play_session
from ai_platformer.content.routes import replay_route
from ai_platformer.content.pcg import generate_level, generate_player_level
from ai_platformer.content.generation_request import GenerationRequest
from ai_platformer.settings import load_gameplay_settings
from tests.unit.test_dda_rules import fixture
from tests.integration.test_human_play_records import app, tap, screen


def ready_profile(session):
    _, profile = fixture(("death",) * 6, progress=0.3)
    profile["identity"] = {"scope": "player", "id": session.records.store.data["player_id"]}
    environment = digest({"environment": HumanPlayProtocol(session.prepared).environment(), "tick_hz": 60})
    profile["environment_sha256"] = environment
    profile["profile_id"] = digest({"identity": profile["identity"], "environment": environment})
    return profile


def result_screen(game):
    game.state = game.state_dict["game_over"]
    game.state.enter()


def mock_profile_source(monkeypatch, records, profile):
    source = SimpleNamespace(directory=records.directory, profile_snapshot=lambda prepared: profile)
    monkeypatch.setattr(records, "freeze_profile_source", lambda prepared: source)


def request_offer(game):
    tap(game, pygame.K_d)
    job = game.state.suggestion_job
    if job:
        try:
            job.future.result(timeout=20)
        except PlayAdmissionError:
            pass
        game.advance(1 / 60)


def test_player_generator_preserves_legacy_contract_and_uses_player_physics():
    settings = load_gameplay_settings(Path("config/gameplay_player.json"))
    request = GenerationRequest(schema_version=1, level_id="player-generation", seed=17)
    before = generate_level(request)
    player = generate_player_level(request, settings.physics)
    assert player.physics == settings.physics
    assert player.report["physics"] == asdict(settings.physics)
    assert player.report["physics_profile"] == "human-custom-v1"
    assert player.level == before.level  # Geometry/RNG remains versioned independently.
    assert generate_level(request) == before


def test_real_candidate_replay_then_explicit_accept_and_restart_catalog(screen, tmp_path, monkeypatch):
    session, records, game = app(tmp_path)
    try:
        original = session.prepared
        records.store.data["player_id"] = "candidate-test-player"
        profile = ready_profile(session)
        mock_profile_source(monkeypatch, records, profile)
        result_screen(game)
        request_offer(game)
        offer = game.state.offer
        assert offer is not None and offer.prepared is not None, game.state.suggestion_message
        assert session.prepared is original and not records.active
        assert offer.prepared.settings.physics == original.settings.physics
        assert offer.prepared.route["environment"]["physics"] == asdict(original.settings.physics)
        assert offer.prepared.route["environment"]["action_repeat"] == 1
        assert replay_route(offer.entry.path, offer.prepared.route)["passed"]
        report = json.loads((offer.directory / "report.json").read_text(encoding="utf-8"))
        assert report["route"]["replay_passed"] and report["application"] == "not_applied"
        # A changed file, stale screen or failed durable selection cannot switch maps.
        raw = offer.entry.path.read_bytes()
        modified = json.loads(raw)
        modified["level_id"] = "changed-candidate"
        offer.entry.path.write_text(json.dumps(modified), encoding="utf-8")
        with pytest.raises(PlayAdmissionError):
            candidates.accept_offer(session, offer)
        offer.entry.path.write_bytes(raw)
        session.prepared = replace(original)
        with pytest.raises(PlayAdmissionError):
            candidates.accept_offer(session, offer)
        session.prepared = original
        with monkeypatch.context() as patch:
            def fail_write(*args):
                raise OSError("disk full")
            patch.setattr(candidates, "atomic_write", fail_write)
            with pytest.raises(PlayAdmissionError):
                candidates.accept_offer(session, offer)
        assert session.prepared is original and len(session.entries) == 2
        tap(game, pygame.K_d)
        assert game.state.offer is offer  # Repeated D does not generate again.
        tap(game, pygame.K_a)
        assert game.state is game.state_dict["level"] and records.active
        assert session.prepared is offer.prepared
        assert records.state.level_id == offer.prepared.settings.level_id
        # Retry uses the admitted immutable map.
        tap(game, pygame.K_r)
        assert session.prepared is offer.prepared
        records.close()
        restored = default_play_session(original.settings)
        restored.records = HumanPlayRecords(tmp_path)
        try:
            restored.records.restore_candidates(restored)
            assert restored.entries[-1].path == offer.entry.path
            assert restored.entries[-1].design_request == offer.prepared.design_request
            restored.select(len(restored.entries) - 1)
            assert restored.prepared.admission == "route_verified"
            assert restored.prepared.settings.physics == original.settings.physics
            assert restored.prepared.design_request == offer.prepared.design_request
        finally:
            restored.records.close()
    finally:
        records.close()


@pytest.mark.parametrize("failure", ["budget", "exception", "storage"])
def test_failed_validation_preserves_current_run(screen, tmp_path, monkeypatch, failure):
    session, records, game = app(tmp_path)
    try:
        original = session.prepared
        mock_profile_source(monkeypatch, records, ready_profile(session))
        if failure == "budget":
            monkeypatch.setattr(candidates, "search_route", lambda *a, **kw: {"status": "unverified"})
        else:
            def fail(*args, **kwargs):
                raise OSError("disk unavailable") if failure == "storage" else RuntimeError("route failed")
            monkeypatch.setattr(candidates, "atomic_write" if failure == "storage" else "search_route", fail)
        result_screen(game)
        request_offer(game)
        tap(game, pygame.K_a)
        assert session.prepared is original and len(session.entries) == 2
        assert not records.active and game.state is game.state_dict["game_over"]
        tap(game, pygame.K_r)
        assert records.active and game.state is game.state_dict["level"]
        assert session.prepared is original
    finally:
        records.close()


def test_profile_snapshot_seals_sessions_and_restores_explicit_order(screen, tmp_path):
    settings = load_gameplay_settings(Path("config/gameplay_player.json"))
    settings = replace(settings, physics=replace(settings.physics, max_episode_steps=4))
    session, records, game = app(tmp_path, settings)
    try:
        tap(game, pygame.K_RETURN)
        for _ in range(4):
            game.advance(1 / 60)
        profile = records.profile_snapshot(session.prepared)
        assert profile["readiness"]["valid_attempts"] == 1
        assert not records.active and records.recorder is None
        game.state.enter()
        for _ in range(4):
            game.advance(1 / 60)
        profile = records.profile_snapshot(session.prepared)
        assert profile["readiness"]["valid_attempts"] == 2
        order = list(records.session_order)
        assert len(order) == 2
        records.close()
        reopened = HumanPlayRecords(tmp_path)
        try:
            assert reopened.session_order == order
            assert reopened.profile_snapshot(session.prepared)["evidence"] == profile["evidence"]
        finally:
            reopened.close()
    finally:
        records.close()


def test_headless_and_no_save_never_create_candidate(screen, tmp_path):
    settings = load_gameplay_settings(Path("config/gameplay_player.json"))
    settings = replace(settings, physics=replace(settings.physics, max_episode_steps=4))
    session, records, game = app(tmp_path, settings)
    try:
        records.headless = True
        tap(game, pygame.K_RETURN)
        for _ in range(4):
            game.advance(1 / 60)
        result_screen(game)
        request_offer(game)
        assert game.state.offer.prepared is None
        session.records = None
        game.state.enter()
        request_offer(game)
        assert game.state.offer.prepared is None
        assert not (tmp_path / "candidates").exists()
    finally:
        records.close()


def test_foreign_physics_profile_is_rejected_before_generation(screen, tmp_path, monkeypatch):
    session, records, game = app(tmp_path)
    try:
        profile = ready_profile(session)
        profile["environment_sha256"] = "f" * 64
        profile["profile_id"] = digest({"identity": profile["identity"], "environment": "f" * 64})
        monkeypatch.setattr(records, "profile_snapshot", lambda prepared: profile)
        original = session.prepared
        with pytest.raises(PlayAdmissionError):
            candidates.prepare_offer(session)
        assert session.prepared is original and not (tmp_path / "candidates").exists()
    finally:
        records.close()


def test_corrupt_order_disables_suggestions_without_overwriting_history(tmp_path):
    path = tmp_path / "session_order.json"
    path.write_text('["../foreign"]', encoding="utf-8")
    records = HumanPlayRecords(tmp_path)
    try:
        assert not records.order_valid
        with pytest.raises(ValueError):
            records.profile_snapshot(None)
        assert path.read_text(encoding="utf-8") == '["../foreign"]'
    finally:
        records.close()
