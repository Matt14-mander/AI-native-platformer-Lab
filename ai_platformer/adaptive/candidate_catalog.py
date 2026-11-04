"""Lightweight persisted catalog; optional content dependencies stay lazy."""

import json
import re
from ai_platformer.content.play_session import PlayEntry


def read_catalog(directory):
    path = directory / "candidate_catalog.json"
    if not path.exists():
        return []
    if path.stat().st_size > 100_000:
        raise ValueError("candidate catalog too large")
    names = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(names, list) or len(names) > 128
            or any(not isinstance(n, str) or not re.fullmatch(r"[a-f0-9]{32}", n) for n in names)
            or len(set(names)) != len(names)):
        raise ValueError("invalid candidate catalog")
    return names


def restore_catalog(session):
    """Restore selected entries only; PlaySession revalidates before play."""
    try:
        for name in read_catalog(session.records.directory):
            path = session.records.directory / "candidates" / name / "level.json"
            if path.is_file() and all(e.path != path for e in session.entries):
                request_path = path.with_name("request.json")
                if request_path.stat().st_size > 100_000:
                    raise ValueError("candidate request too large")
                request = json.loads(request_path.read_text(encoding="utf-8"))
                session.entries += (PlayEntry("建议练习关", path=path, design_request=request),)
    except (OSError, ValueError, TypeError, RecursionError):
        session.records.warning = "建议关卡目录无法读取，原有关卡仍可游玩。"
