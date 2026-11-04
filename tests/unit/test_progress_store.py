import json
from types import SimpleNamespace

import pytest

from ai_platformer.adaptive.progress import ProgressStore


KEY = "a" * 64


def state(score=100, coins=2, tick=120):
    return SimpleNamespace(tick=tick, metadata={"score": score, "coins_collected": coins})


def test_persists_best_values_and_does_not_count_interruptions_as_completion(tmp_path):
    store = ProgressStore(tmp_path)
    identity = store.data["player_id"]
    store.begin(KEY, "flat", "local-flat")
    store.finish(KEY, state(), "success")
    store.begin(KEY, "flat", "local-flat")
    store.finish(KEY, state(10, 1, 200), "success")
    store.begin(KEY, "flat", "local-flat")
    store.finish(KEY, state(9999, 99, 1), "quit")
    store.close()
    loaded = ProgressStore(tmp_path)
    assert loaded.data["player_id"] == identity
    assert loaded.data["last_entry"] == "local-flat"
    assert loaded.data["levels"][KEY] == {
        "level_id": "flat", "attempts": 3, "completed": True,
        "best_score": 100, "best_coins": 2, "fastest_ticks": 120,
    }
    loaded.close()


def test_recovers_last_good_backup_and_preserves_corrupt_primary(tmp_path):
    store = ProgressStore(tmp_path)
    store.begin(KEY, "flat", "local-flat")
    store.finish(KEY, state(), "success")
    identity = store.data["player_id"]
    store.close()
    (tmp_path / "progress.json").write_text("truncated", encoding="utf-8")
    recovered = ProgressStore(tmp_path)
    assert recovered.data["player_id"] == identity
    assert recovered.data["levels"][KEY]["attempts"] == 1
    assert "备份" in recovered.warning
    assert list(tmp_path.glob("progress.corrupt-*.json"))[0].read_text() == "truncated"
    recovered.save()
    recovered.close()
    assert json.loads((tmp_path / "progress.json").read_text())["schema_version"] == 1


def test_unknown_version_is_never_overwritten(tmp_path):
    path = tmp_path / "progress.json"
    path.write_text('{"schema_version": 999}')
    store = ProgressStore(tmp_path)
    store.begin(KEY, "flat", "flat")
    assert store.disabled and "不兼容" in store.warning
    assert path.read_text() == '{"schema_version": 999}'
    store.close()


def test_write_failure_leaves_primary_intact_and_game_progress_in_memory(tmp_path, monkeypatch):
    from ai_platformer.adaptive import progress

    store = ProgressStore(tmp_path)
    store.begin(KEY, "flat", "flat")
    original = store.path.read_bytes()
    monkeypatch.setattr(progress.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("disk full")))
    store.finish(KEY, state(), "success")
    assert store.disabled and store.data["levels"][KEY]["completed"]
    assert store.path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))
    store.close()


def test_second_writer_fails_open_without_overwriting_first(tmp_path):
    first = ProgressStore(tmp_path)
    first.begin(KEY, "flat", "flat")
    second = ProgressStore(tmp_path)
    assert second.disabled
    second.begin(KEY, "flat", "other")
    assert json.loads(first.path.read_text())["last_entry"] == "flat"
    second.close()
    first.close()


def test_corrupt_without_backup_starts_clean_and_retains_original(tmp_path):
    (tmp_path / "progress.json").write_text("[]")
    store = ProgressStore(tmp_path)
    assert store.data["levels"] == {}
    assert list(tmp_path.glob("progress.corrupt-*.json"))
    store.close()
