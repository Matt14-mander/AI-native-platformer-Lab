"""Bounded deterministic checkpoint journals, sharing the progress store lock."""

from dataclasses import asdict
import json

from ai_platformer.core import BasicPlatformerCore
from .metrics import digest
from .progress import atomic_write, unique_fields


class CheckpointSave:
    def __init__(self, store):
        self.store = store
        self.path = store.directory / "checkpoint.json"
        self.data = None
        self.disabled = store.disabled
        try:
            if self.path.exists():
                if self.path.stat().st_size > 2_000_000:
                    raise ValueError("oversized checkpoint")
                data = json.loads(self.path.read_text(encoding="utf-8"), object_pairs_hook=unique_fields)
                if data is not None:
                    if not isinstance(data, dict) or set(data) != {
                        "version", "key", "entry", "seed", "actions", "state_hash"
                    } or type(data["version"]) is not int or data["version"] != 1:
                        raise ValueError("unsupported checkpoint")
                    if (any(not isinstance(data[k], str) for k in ("key", "entry", "state_hash"))
                            or type(data["seed"]) is not int
                            or not isinstance(data["actions"], list)
                            or not 0 < len(data["actions"]) <= 36_000):
                        raise ValueError("invalid checkpoint")
                    for row in data["actions"]:
                        if (not isinstance(row, list) or len(row) != 2
                                or type(row[0]) is not int or not 0 <= row[0] <= 9
                                or type(row[1]) is not bool):
                            raise ValueError("invalid action")
                self.data = data
        except (OSError, ValueError, RecursionError):
            self.disabled = True
            store.warning = "检查点存档无法读取，已保留原文件；可重新开始关卡。"

    def write(self, data):
        self.data = data
        if self.disabled or self.store.disabled:
            return
        try:
            atomic_write(self.path, data)
        except (OSError, ValueError):
            self.disabled = True
            self.store.warning = "检查点保存失败，本次仍可继续游玩。"

    def restore(self, prepared, entry):
        from .human_play import entry_key, progress_key
        data = self.data
        if (not data or data["key"] != progress_key(prepared)
                or data["entry"] != entry_key(entry) or data["seed"] != prepared.settings.seed):
            raise ValueError("关卡或物理配置已变化，无法继续旧检查点；请重新开始。")
        core = BasicPlatformerCore(prepared.repository.load, config=prepared.settings.physics,
                                   checkpoints=True)
        core.reset(seed=data["seed"], level_id=prepared.settings.level_id)
        if len(data["actions"]) >= prepared.settings.physics.max_episode_steps:
            raise ValueError("检查点已超时，请重新开始。")
        for action, jump in data["actions"]:
            result = core.step(action, jump_pressed=jump)
            if result.terminated or result.truncated:
                raise ValueError("检查点已结束，请重新开始。")
        if (not any(name == "checkpoint_activated" for name, _ in result.info.get("events", ()))
                or digest(asdict(core.state)) != data["state_hash"]):
            raise ValueError("检查点验证失败，请重新开始。")
        return tuple((a, j) for a, j in data["actions"])

    def capture(self, prepared, entry, actions, state):
        from .human_play import entry_key, progress_key
        if len(actions) > 36_000:
            return
        self.write({"version": 1, "key": progress_key(prepared), "entry": entry_key(entry),
                    "seed": prepared.settings.seed, "actions": [list(row) for row in actions],
                    "state_hash": digest(asdict(state))})
