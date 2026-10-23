"""Gemini translates intent to bounded requests; PCG alone creates geometry."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import socket
import ssl
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from pydantic import ValidationError

from ai_platformer.content.generation_request import GenerationRequest, request_hash
from ai_platformer.content.level_spec import parse_level_json

PROMPT_VERSION = "generation-intent-v1"
SYSTEM_PROMPT = """Translate the user's platformer design into GenerationRequest v1 JSON.
Only segments-v1, gameplay-ai-v2 and lake are supported. No enemies, moving platforms,
interactive blocks, assets or arbitrary geometry. Do not claim these are supported.
Use the supplied level_id and seed exactly. Emit all fields in the schema.
At most 12 hazards total; each count is 0..8. Range minimum <= maximum.
coin_layout none requires min_coin_ratio 0. Heights 16..160; gaps 64..176 pixels.
Defaults: 3 obstacles, 2 gaps, heights 48..96, gaps 96..152, low_jump coins,
minimum collection ratio .75. If requirements cannot be expressed, set unsupported
with a concise explanation and request null. Otherwise set unsupported null and
request to the design. Never emit code or instructions to execute."""
MAX_BYTES = 1_000_000


def safe_provider_text(value):
    """Retain a bounded diagnostic, never configured credentials or query keys."""
    if not isinstance(value, str):
        return None
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        secret = os.environ.get(name)
        if secret:
            value = value.replace(secret, "[REDACTED]")
    value = re.sub(r"AIza[A-Za-z0-9_-]+", "[REDACTED]", value)
    value = re.sub(
        r"(?i)((?:x-goog-api-key|api[_-]?key|key)\s*[=:]\s*)[^\s&\"'<>]+",
        r"\1[REDACTED]",
        value,
    )
    value = re.sub(r"(?i)(Bearer\s+)\S+", r"\1[REDACTED]", value)
    return " ".join(value.split())[:1000]


def http_error_details(error):
    """Read only the Google JSON error envelope; discard headers and HTML bodies."""
    try:
        raw = error.read(16385)
        if len(raw) > 16384:
            return {}
        document = parse_level_json(raw)
        details = document.get("error", {}) if isinstance(document, dict) else {}
        if not isinstance(details, dict):
            return {}
        result = {}
        for source, target in (("status", "provider_status"), ("message", "provider_message")):
            text = safe_provider_text(details.get(source))
            if text:
                result[target] = text
        return result
    except (OSError, ValueError, TypeError, RecursionError):
        return {}
    finally:
        error.close()


def connection_error_details(error):
    reason = error.reason if isinstance(error, urllib.error.URLError) else error
    category = "connection"
    if isinstance(reason, TimeoutError):
        category = "tls_handshake_timeout" if "handshake" in str(reason).lower() else "timeout"
    elif isinstance(reason, socket.gaierror):
        category = "dns"
    elif isinstance(reason, ssl.SSLCertVerificationError):
        category = "tls_certificate"
    elif isinstance(reason, ssl.SSLError):
        category = "tls"
    return {"connection_category": category, "exception_type": type(reason).__name__}


class GenerationError(RuntimeError):
    def __init__(self, reason, attempts=()):
        super().__init__(reason)
        self.attempts = list(attempts)


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read_json(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("JSON exceeds 1 MB limit")
    return parse_level_json(raw)


def output_schema():
    schema = GenerationRequest.model_json_schema()
    defs = schema.pop("$defs", {})

    def inline(node):
        if isinstance(node, list):
            return [inline(item) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return inline(defs[node["$ref"].split("/")[-1]])
        result = {k: inline(v) for k, v in node.items() if k not in {"default", "title", "const"}}
        if "const" in node:
            result["enum"] = [node["const"]]
        return result

    request = inline(schema)
    request["required"] = list(request["properties"])
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "request": {"anyOf": [request, {"type": "null"}]},
            "unsupported": {"anyOf": [{"type": "string", "maxLength": 1000}, {"type": "null"}]},
        },
        "required": ["request", "unsupported"],
    }


def validate_output(data, *, level_id, seed):
    if not isinstance(data, dict) or set(data) != {"request", "unsupported"}:
        raise ValueError("expected request and unsupported fields")
    if data["unsupported"] is not None:
        if data["request"] is not None or not isinstance(data["unsupported"], str):
            raise ValueError("invalid unsupported response")
        raise GenerationError("unsupported design: " + data["unsupported"][:1000])
    request = GenerationRequest.model_validate(data["request"])
    if request.level_id != level_id or request.seed != seed:
        raise ValueError("model changed caller-owned level_id or seed")
    return request


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Keep credentials on the fixed official host.
        return None


def https_handlers(tls_mode="default"):
    if tls_mode == "default":
        return [NoRedirect()]
    if tls_mode != "tls12":
        raise ValueError("tls_mode must be default or tls12")
    context = ssl.create_default_context()
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.maximum_version = ssl.TLSVersion.TLSv1_2
    context.set_alpn_protocols(["http/1.1"])
    return [NoRedirect(), urllib.request.HTTPSHandler(context=context)]


def gemini_call(payload, *, model, timeout, tls_mode="default"):
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise GenerationError("GEMINI_API_KEY or GOOGLE_API_KEY is required")
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=canonical(payload).encode(),
        method="POST",
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
    )
    with urllib.request.build_opener(*https_handlers(tls_mode)).open(
        request, timeout=timeout
    ) as response:
        raw = response.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("provider response exceeds 1 MB limit")
    return parse_level_json(raw)


def translate_intent(
    prompt,
    *,
    level_id,
    seed,
    provider="gemini",
    model=None,
    fixture=None,
    cache_dir=None,
    max_attempts=3,
    timeout=30.0,
    transport=None,
    tls_mode="default",
):
    # Validate caller fields before any external request.
    GenerationRequest(schema_version=1, level_id=level_id, seed=seed)
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt.encode()) > 16000:
        raise ValueError("prompt must contain 1..16000 UTF-8 bytes")
    if type(max_attempts) is not int or not 1 <= max_attempts <= 3:
        raise ValueError("max_attempts must be 1..3")
    if not math.isfinite(timeout) or not 0 < timeout <= 60:
        raise ValueError("timeout must be in (0,60]")
    if provider not in {"gemini", "fixture"}:
        raise ValueError("unknown LLM provider")
    if tls_mode not in {"default", "tls12"}:
        raise ValueError("tls_mode must be default or tls12")
    if provider == "gemini":
        model = model or os.environ.get("GEMINI_MODEL")
        if not model or not re.fullmatch(r"gemini-[A-Za-z0-9._-]{1,100}", model):
            raise GenerationError("set --model or GEMINI_MODEL to a Gemini model ID")
        if fixture is not None:
            raise ValueError("fixture is only available with provider=fixture")
        fixture_data = None
    else:
        if fixture is None:
            raise ValueError("fixture provider requires a JSON response file")
        fixture_data = read_json(fixture)
        model = "offline-fixture"
    identity = {
        "schema_version": 1,
        "prompt_version": PROMPT_VERSION,
        "system_sha256": digest(SYSTEM_PROMPT),
        "schema_sha256": digest(output_schema()),
        "prompt": prompt,
        "level_id": level_id,
        "seed": seed,
        "provider": provider,
        "model": model,
        "fixture_sha256": digest(fixture_data) if fixture_data is not None else None,
    }
    if tls_mode != "default":
        identity["tls_mode"] = tls_mode
    cache_key = digest(identity)
    cache_path = Path(cache_dir) / f"{cache_key}.json" if cache_dir is not None else None
    if cache_path is not None and cache_path.exists():
        try:
            cached = read_json(cache_path)
            if not isinstance(cached, dict) or cached.get("identity") != identity:
                raise GenerationError("cache identity mismatch")
            request = validate_output(cached["output"], level_id=level_id, seed=seed)
            if cached.get("request_sha256") != request_hash(request):
                raise GenerationError("cache request hash mismatch")
            return request, {**cached, "cache_hit": True, "cache_key": cache_key}
        except (OSError, ValueError, KeyError, TypeError, RecursionError):
            raise GenerationError("invalid cached response") from None
    attempts = []
    feedback = ""
    for index in range(max_attempts):
        try:
            if provider == "fixture":
                output = fixture_data
                usage = None
            else:
                payload = {
                    "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "text": canonical(
                                        {
                                            "intent": prompt,
                                            "level_id": level_id,
                                            "seed": seed,
                                            "validation_feedback": feedback,
                                        }
                                    )
                                }
                            ],
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0,
                        "maxOutputTokens": 4096,
                        "responseMimeType": "application/json",
                        "responseJsonSchema": output_schema(),
                    },
                }
                transport_options = {"model": model, "timeout": timeout}
                if tls_mode != "default":
                    transport_options["tls_mode"] = tls_mode
                response = (transport or gemini_call)(payload, **transport_options)
                candidates = response.get("candidates", [])
                if not candidates or candidates[0].get("finishReason") != "STOP":
                    raise GenerationError("provider blocked or incomplete response", attempts)
                raw = "".join(
                    p.get("text", "")
                    for p in candidates[0]["content"]["parts"]
                    if not p.get("thought", False)
                )
                output = parse_level_json(raw.encode())
                usage = response.get("usageMetadata")
            attempts.append({"attempt": index + 1, "output": output, "usage": usage})
            request = validate_output(output, level_id=level_id, seed=seed)
            record = {
                "identity": identity,
                "output": output,
                "attempts": attempts,
                "request_sha256": request_hash(request),
                "cache_key": cache_key,
                "cache_hit": False,
            }
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    mode="w", dir=cache_path.parent, encoding="utf-8", delete=False
                ) as stream:
                    temp = Path(stream.name)
                    stream.write(canonical(record))
                try:
                    temp.replace(cache_path)
                finally:
                    temp.unlink(missing_ok=True)
            return request, record
        except GenerationError as error:
            raise GenerationError(str(error), attempts) from None
        except urllib.error.HTTPError as error:
            reason = f"provider HTTP {error.code}"
            attempts.append(
                {
                    "attempt": index + 1,
                    "error": reason,
                    "http_status": error.code,
                    **http_error_details(error),
                }
            )
            if error.code != 429 and not 500 <= error.code < 600:
                raise GenerationError(reason, attempts) from None
            feedback = reason
            if index + 1 < max_attempts:
                time.sleep(min(2**index, 4))
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            feedback = "provider connection failed"
            attempts.append(
                {
                    "attempt": index + 1,
                    "error": feedback,
                    "timeout_seconds": timeout,
                    **connection_error_details(error),
                }
            )
        except (
            ValidationError,
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            RecursionError,
        ) as error:
            feedback = (
                canonical(
                    error.errors(include_input=False, include_context=False, include_url=False)
                )
                if isinstance(error, ValidationError)
                else "invalid structured response"
            )
            attempts.append({"attempt": index + 1, "error": feedback})
        if provider == "fixture":
            break
    raise GenerationError("request translation failed after bounded attempts", attempts)
