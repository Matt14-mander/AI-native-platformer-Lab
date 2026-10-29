"""Local human-play catalog; admission never changes training protocols."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path

from ai_platformer.settings import GameplaySettings
from ai_platformer.core.level import LevelDefinition
from .legacy import LegacyLevelRepository


class PlayAdmissionError(ValueError):
    """A candidate cannot be started; keep the current selection intact."""


@dataclass(frozen=True)
class PlayEntry:
    title: str
    level_id: str | None = None
    path: Path | None = None


@dataclass(frozen=True)
class PlayRepository:
    """A frozen map prevents disk edits from changing an active run or retry."""

    definition: LevelDefinition

    def load(self, level_id):
        if level_id != self.definition.level_id:
            raise KeyError(level_id)
        return self.definition


@dataclass(frozen=True)
class PreparedPlayLevel:
    settings: GameplaySettings
    repository: PlayRepository
    solids: dict
    admission: str
    route: dict | None = None


class PlaySession:
    """Keep an admitted in-memory map stable across retries and result screens."""

    def __init__(self, settings: GameplaySettings, entries: tuple[PlayEntry, ...]):
        if not entries:
            raise ValueError("play catalog must not be empty")
        self.settings = settings
        self.entries = entries
        self.index = 0
        self.prepared = self._prepare(entries[0])

    @property
    def has_next(self):
        return self.index + 1 < len(self.entries)

    @property
    def title(self):
        return self.entries[self.index].title

    def select(self, index: int):
        if not 0 <= index < len(self.entries):
            raise PlayAdmissionError("没有更多关卡了。")
        if index == self.index:
            return
        prepared = self._prepare(self.entries[index])
        self.prepared, self.index = prepared, index

    def next_level(self):
        self.select(self.index + 1)

    def _prepare(self, entry):
        try:
            if entry.path is None:
                repository = LegacyLevelRepository()
                level_id = entry.level_id or self.settings.level_id
                definition = repository.load(level_id)
                return PreparedPlayLevel(
                    replace(self.settings, level_id=level_id), PlayRepository(definition),
                    repository.load_solids_by_group(level_id), "legacy_compatibility",
                )

            # Lazy optional imports keep legacy-only play usable without Gymnasium.
            from ai_platformer.envs.factory import EnvironmentFactory
            from .routes import search_route

            factory = EnvironmentFactory({
                "environment_id": "PlatformerState-v2",
                "level_spec": str(entry.path),
                "physics": asdict(self.settings.physics),
                "action_repeat": 1,
                "episode_step_limit": min(2400, self.settings.physics.max_episode_steps),
            })
            definition = factory.repository.load(factory.level_id)
            if definition.height != 600:
                raise PlayAdmissionError("当前画面仅支持高度为 600 的关卡。")
            report = search_route(
                factory, seed=self.settings.seed, max_steps=2400,
                max_expansions=6000, beam_width=12, max_seconds=2.0,
            )
            if report["status"] != "verified":
                raise PlayAdmissionError("暂未找到可验证路线，请换一关。")
            return PreparedPlayLevel(
                replace(self.settings, level_id=factory.level_id), PlayRepository(definition),
                factory.repository.solids_by_kind(factory.level_id), "route_verified",
                report["witness"],
            )
        except PlayAdmissionError:
            raise
        except ImportError as error:
            raise PlayAdmissionError("请先安装 content、benchmark 扩展依赖。") from error
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
            raise PlayAdmissionError("关卡读取或验证失败，请检查文件。") from error


def default_play_session(settings, extra_specs=()):
    root = Path(__file__).resolve().parents[2]
    paths = [root / "game_content/level_specs/flat_v1.json", *extra_specs]
    entries = [PlayEntry("湖岸旅程", level_id=settings.level_id)]
    seen = set()
    for path in paths:
        path = Path(path).resolve()
        if path in seen:
            continue
        seen.add(path)
        title = "湖岸练习场" if len(entries) == 1 else f"本地：{path.stem[:12]}"
        entries.append(PlayEntry(title, path=path))
    return PlaySession(settings, tuple(entries))
