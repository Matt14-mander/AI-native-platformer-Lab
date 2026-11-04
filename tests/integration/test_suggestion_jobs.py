"""Controlled worker completion tests, without sleep-based race assumptions."""
from concurrent.futures import CancelledError
from threading import Event

import pygame
import pytest

from ai_platformer.adaptive import candidates
from ai_platformer.adaptive.candidates import CandidateOffer
from ai_platformer.content.routes import search_route, collect_route, replay_route
from ai_platformer.envs.factory import EnvironmentFactory
from tests.integration.test_human_play_records import app, tap, screen
from tests.integration.test_player_candidates import result_screen


@pytest.mark.parametrize("key", [pygame.K_r, pygame.K_m])
def test_leave_cancels_job_without_blocking_or_mutating_next_state(screen, tmp_path, monkeypatch, key):
    entered, release = Event(), Event()
    calls = []
    def slow(snapshot, *, cancelled):
        calls.append(snapshot)
        entered.set()
        assert release.wait(10)
        return CandidateOffer(snapshot.prepared, "late result")
    monkeypatch.setattr(candidates, "prepare_offer", slow)
    session, records, game = app(tmp_path)
    try:
        original = session.prepared
        result_screen(game)
        old_screen = game.state
        tap(game, pygame.K_d)
        job = old_screen.suggestion_job
        assert entered.wait(5)
        tap(game, pygame.K_d)
        assert old_screen.suggestion_job is job and len(calls) == 1
        assert not hasattr(calls[0].records, "recorder")
        assert not hasattr(calls[0].records, "store")
        tap(game, key)
        assert game.state is game.state_dict["level" if key == pygame.K_r else "main_menu"]
        assert job.cancelled.is_set() and not job.future.done()
        if key == pygame.K_m:
            result_screen(game)
            tap(game, pygame.K_d)
            assert game.state.suggestion_job is None and len(calls) == 1
            assert "上次检查" in game.state.suggestion_message
        release.set()
        with pytest.raises(CancelledError):
            job.future.result(timeout=5)
        game.advance(1 / 60)
        assert old_screen.offer is None and old_screen.suggestion_job is None
        assert session.prepared is original and len(session.entries) == 2
    finally:
        release.set()
        if getattr(session, "suggestion_job", None):
            session.suggestion_job.thread.join(5)
        records.close()


def test_completion_frame_cannot_accept_early_a_press(screen, tmp_path, monkeypatch):
    release = Event()
    def slow(snapshot, *, cancelled):
        assert release.wait(10)
        return CandidateOffer(snapshot.prepared, "ready", prepared=snapshot.prepared)
    monkeypatch.setattr(candidates, "prepare_offer", slow)
    accepted = []
    monkeypatch.setattr(candidates, "accept_offer", lambda *args: accepted.append(args))
    session, records, game = app(tmp_path)
    try:
        result_screen(game)
        tap(game, pygame.K_d)
        job = game.state.suggestion_job
        tap(game, pygame.K_a)
        assert not accepted
        release.set()
        job.future.result(timeout=5)
        tap(game, pygame.K_a)  # This frame first displays the result.
        assert not accepted and game.state is game.state_dict["game_over"]
        tap(game, pygame.K_a)  # A fresh press after displaying is required.
        assert len(accepted) == 1
    finally:
        release.set()
        records.close()


def test_background_exception_is_shown_and_original_retry_still_works(screen, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("worker failure")
    monkeypatch.setattr(candidates, "prepare_offer", fail)
    session, records, game = app(tmp_path)
    try:
        result_screen(game)
        tap(game, pygame.K_d)
        job = game.state.suggestion_job
        with pytest.raises(RuntimeError):
            job.future.result(timeout=5)
        game.advance(1 / 60)
        assert "暂不可用" in game.state.suggestion_message
        assert game.state.offer is None
        tap(game, pygame.K_r)
        assert game.state is game.state_dict["level"] and records.active
    finally:
        records.close()


def test_search_and_witness_collection_are_cooperatively_cancellable():
    factory = EnvironmentFactory({"environment_id": "PlatformerState-v2",
                                  "level_spec": "game_content/level_specs/flat_v1.json"})
    with pytest.raises(CancelledError):
        search_route(factory, cancelled=lambda: True)
    with pytest.raises(CancelledError):
        collect_route(factory, [0], seed=0, min_coin_ratio=0, cancelled=lambda: True)
    report = search_route(factory)
    assert report["status"] == "verified"
    with pytest.raises(CancelledError):
        replay_route(factory.repository.path, report["witness"], cancelled=lambda: True)


def test_window_exit_cancels_without_waiting_for_worker_or_leaking_save_lock(screen, tmp_path, monkeypatch):
    from source.main import main
    from source.tools import Game
    from ai_platformer.adaptive.progress import ProgressStore

    entered, release = Event(), Event()
    captured = []
    def slow(snapshot, *, cancelled):
        entered.set()
        assert release.wait(10)
        return CandidateOffer(snapshot.prepared, "late")
    def run(game):
        result_screen(game)
        tap(game, pygame.K_d)
        captured.append(game.state.suggestion_job)
        assert entered.wait(5)
        # Returning from run is the normal main-level window exit path.
    monkeypatch.setattr(candidates, "prepare_offer", slow)
    monkeypatch.setattr(Game, "run", run)
    monkeypatch.setattr("sys.argv", ["main", "--save-dir", str(tmp_path)])
    try:
        main()
        assert captured[0].cancelled.is_set() and not captured[0].future.done()
        reopened = ProgressStore(tmp_path)
        assert not reopened.disabled
        reopened.close()
    finally:
        release.set()
        if captured:
            captured[0].thread.join(5)


@pytest.mark.parametrize("valid", [True, False])
def test_next_level_cancels_before_admission_and_discards_late_result(screen, tmp_path, monkeypatch, valid):
    from ai_platformer.content.play_session import PlayAdmissionError
    release = Event()
    def slow(snapshot, *, cancelled):
        assert release.wait(10)
        return CandidateOffer(snapshot.prepared, "late", prepared=snapshot.prepared)
    monkeypatch.setattr(candidates, "prepare_offer", slow)
    session, records, game = app(tmp_path)
    try:
        original = session.prepared
        game.state = game.state_dict["level_complete"]
        game.state.enter()
        result = game.state
        tap(game, pygame.K_d)
        job = result.suggestion_job
        next_level = session.next_level
        def admission():
            assert job.cancelled.is_set()
            if not valid:
                raise PlayAdmissionError("test admission failure")
            next_level()
        monkeypatch.setattr(session, "next_level", admission)
        tap(game, pygame.K_n)
        if valid:
            assert game.state is game.state_dict["level"]
            assert session.index == 1 and session.prepared is not original
            assert game.state.state.tick == 0
        else:
            assert game.state is result and session.prepared is original
            assert result.error == "test admission failure" and not records.active
            assert "正在后台" not in result.suggestion_message
        release.set()
        with pytest.raises(CancelledError):
            job.future.result(timeout=5)
        game.advance(1 / 60)
        assert result.offer is None
        assert session.index == (1 if valid else 0)
    finally:
        release.set()
        if getattr(session, "suggestion_job", None):
            session.suggestion_job.thread.join(5)
        records.close()


def test_worker_start_failure_does_not_break_retry(screen, tmp_path, monkeypatch):
    from ai_platformer.adaptive import suggestion_job
    def fail_start(thread):
        raise RuntimeError("cannot start thread")
    monkeypatch.setattr(suggestion_job.Thread, "start", fail_start)
    session, records, game = app(tmp_path)
    try:
        result_screen(game)
        tap(game, pygame.K_d)
        assert game.state.suggestion_job is None
        assert "暂不可用" in game.state.suggestion_message
        tap(game, pygame.K_r)
        assert records.active and game.state is game.state_dict["level"]
    finally:
        records.close()
