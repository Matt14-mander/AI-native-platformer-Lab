"""Deployment package integrity checks, independent of training frameworks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ai_platformer.core import Action


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metadata_hash(data: dict) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def load_actor_metadata(model_path: Path) -> dict:
    model_path = Path(model_path)
    data = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("unsupported actor deployment schema")
    payload = {key: value for key, value in data.items() if key != "metadata_sha256"}
    if metadata_hash(payload) != data.get("metadata_sha256"):
        raise ValueError("actor deployment metadata hash mismatch")
    if file_hash(model_path) != data["model_sha256"]:
        raise ValueError("ONNX model does not match deployment metadata")
    if data["action_ids"] != {action.name: int(action) for action in Action}:
        raise ValueError("deployment action mapping differs from this project")
    protocol = data["protocol"]
    if (
        data["input"]["shape"] != [1, protocol["observation_size"]]
        or data["output"]["shape"] != [1, len(Action)]
        or data["input"]["dtype"] != "float32"
        or data["output"]["dtype"] != "float32"
        or data["action_selection"] != "argmax_first"
    ):
        raise ValueError("unsupported actor tensor/action contract")
    return data
