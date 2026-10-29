"""Versioned local progress with atomic replacement and last-good recovery."""

import json
import os
from pathlib import Path
import tempfile
from uuid import uuid4


class UnsupportedProgressVersion(ValueError):
    pass


def unique_fields(pairs):
    data = {}
    for key, value in pairs:
        if key in data:
            raise ValueError("duplicate field")
        data[key] = value
    return data


def atomic_write(path, value):
    path = Path(path)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name, suffix=".tmp", delete=False) as stream:
            name = stream.name
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if name and Path(name).exists():
            Path(name).unlink()


def validate_progress(data):
    if not isinstance(data, dict):
        raise ValueError("invalid progress document")
    if type(data.get("schema_version")) is not int:
        raise ValueError("invalid progress version")
    if data["schema_version"] != 1:
        raise UnsupportedProgressVersion("unsupported progress version")
    if set(data) != {"schema_version", "player_id", "last_entry", "levels"}:
        raise ValueError("invalid progress fields")
    if not isinstance(data["player_id"], str) or not data["player_id"].strip():
        raise ValueError("invalid player identity")
    if data["last_entry"] is not None and not isinstance(data["last_entry"], str):
        raise ValueError("invalid last entry")
    if not isinstance(data["levels"], dict):
        raise ValueError("invalid levels")
    for key, row in data["levels"].items():
        if not isinstance(key, str) or len(key) != 64 or not isinstance(row, dict):
            raise ValueError("invalid level record")
        if set(row) != {"level_id", "attempts", "completed", "best_score", "best_coins", "fastest_ticks"}:
            raise ValueError("invalid record fields")
        if not isinstance(row["level_id"], str) or type(row["completed"]) is not bool:
            raise ValueError("invalid completion")
        for field in ("attempts", "best_score", "best_coins"):
            if type(row[field]) is not int or row[field] < 0:
                raise ValueError("invalid counters")
        if row["fastest_ticks"] is not None and (
            type(row["fastest_ticks"]) is not int or row["fastest_ticks"] < 0
        ):
            raise ValueError("invalid time")
        if row["completed"] != (row["fastest_ticks"] is not None):
            raise ValueError("inconsistent completion record")
    return data


class ProgressStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.path = self.directory / "progress.json"
        self.backup = self.directory / "progress.backup.json"
        self.warning = ""
        self.disabled = False
        self.lock = None
        self.data = {"schema_version": 1, "player_id": uuid4().hex,
                     "last_entry": None, "levels": {}}
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.lock = (self.directory / "progress.lock").open("a+b")
            if self.lock.tell() == 0:
                self.lock.write(b"0")
                self.lock.flush()
            self.lock.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._load()
        except OSError:
            self.disabled = True
            self.warning = "存档不可写或已被另一窗口使用，本次仅保留内存进度。"
            self.close()

    def _read(self, path):
        if path.stat().st_size > 4_000_000:
            raise ValueError("progress too large")
        return validate_progress(json.loads(path.read_text(encoding="utf-8"),
                                            object_pairs_hook=unique_fields))

    def _load(self):
        if self.path.exists():
            try:
                self.data = self._read(self.path)
                return
            except UnsupportedProgressVersion:
                self.disabled = True
                self.warning = "存档版本不兼容，已保留原文件，本次不写入进度。"
                return
            except (ValueError, UnicodeError):
                # Preserve the original before recovering; never discard user data.
                self.path.replace(self.directory / f"progress.corrupt-{uuid4().hex}.json")
                self.warning = "存档损坏，已保留原文件并尝试恢复备份。"
        if self.backup.exists():
            try:
                self.data = self._read(self.backup)
                self.warning = "已从备份恢复进度。"
            except UnsupportedProgressVersion:
                self.disabled = True
                self.warning = "备份版本不兼容，已保留原文件，本次不写入进度。"
            except (ValueError, UnicodeError):
                self.warning = "存档及备份无法读取，本次从新进度开始。"

    def save(self):
        if self.disabled:
            return
        try:
            validate_progress(self.data)
            if self.path.exists():
                atomic_write(self.backup, self._read(self.path))
            atomic_write(self.path, self.data)
        except (OSError, ValueError, UnicodeError):
            self.disabled = True
            self.warning = "保存失败，本次进度暂存于内存，游戏可继续。"

    def begin(self, key, level_id, entry_key):
        row = self.data["levels"].setdefault(key, {
            "level_id": level_id, "attempts": 0, "completed": False,
            "best_score": 0, "best_coins": 0, "fastest_ticks": None,
        })
        row["attempts"] += 1
        self.data["last_entry"] = entry_key
        self.save()

    def finish(self, key, state, outcome):
        if outcome not in {"success", "death", "time_limit"}:
            return
        row = self.data["levels"][key]
        row["best_score"] = max(row["best_score"], int(state.metadata.get("score", 0)))
        coins = int(state.metadata.get("coins_collected", 0) + state.metadata.get("box_coins", 0))
        row["best_coins"] = max(row["best_coins"], coins)
        if outcome == "success":
            row["completed"] = True
            row["fastest_ticks"] = min(row["fastest_ticks"] or state.tick, state.tick)
        self.save()

    def close(self):
        if self.lock:
            try:
                self.lock.close()
            except OSError:
                self.warning = "关闭存档文件失败，请检查保存目录。"
            self.lock = None
