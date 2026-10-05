import copy
import json
import urllib.error
from pathlib import Path

import pytest

from ai_platformer.agents.llm.generation import (
    GenerationError,
    output_schema,
    read_json,
    translate_intent,
)

FIXTURE = Path("game_content/llm_fixtures/mixed_v1.json")


def response(output):
    return {
        "candidates": [
            {"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(output)}]}}
        ],
        "usageMetadata": {"totalTokenCount": 12},
    }


def call(**kwargs):
    return translate_intent(
        "三个障碍两个坑，至少收集75%松果",
        level_id="pcg_mixed_v1",
        seed=20261005,
        model="gemini-test",
        **kwargs,
    )


def test_structured_payload_and_cache(tmp_path):
    seen = []

    def transport(payload, **kwargs):
        seen.append(payload)
        assert kwargs == {"model": "gemini-test", "timeout": 30.0}
        return response(read_json(FIXTURE))

    request, record = call(transport=transport, cache_dir=tmp_path)
    assert request.obstacle_count == 3
    assert len(seen) == 1
    assert seen[0]["generationConfig"]["responseJsonSchema"] == output_schema()
    assert "三个障碍" in seen[0]["contents"][0]["parts"][0]["text"]
    assert not record["cache_hit"]
    assert call(transport=transport, cache_dir=tmp_path)[1]["cache_hit"]
    assert len(seen) == 1
    cache = next(tmp_path.glob("*.json"))
    data = read_json(cache)
    data["output"]["request"]["gap_count"] = 1
    cache.write_text(json.dumps(data))
    with pytest.raises(GenerationError, match="hash mismatch"):
        call(transport=transport, cache_dir=tmp_path)


def test_invalid_retry_and_caller_pins():
    good = read_json(FIXTURE)
    bad = copy.deepcopy(good)
    bad["request"]["seed"] = 0
    outputs = iter([bad, good])
    seen = []

    def transport(payload, **kwargs):
        seen.append(payload)
        return response(next(outputs))

    request, record = call(transport=transport)
    assert request.seed == 20261005
    assert len(record["attempts"]) == 3
    assert "invalid structured response" in seen[1]["contents"][0]["parts"][0]["text"]


def test_http_non_retry_sanitized():
    calls = []

    def transport(*args, **kwargs):
        calls.append(1)
        raise urllib.error.HTTPError("https://secret", 403, "SECRET", None, None)

    with pytest.raises(GenerationError, match="^provider HTTP 403$") as exc:
        call(transport=transport)
    assert len(calls) == 1
    assert "SECRET" not in str(exc.value.attempts)


def test_retry_budget_and_refusal():
    count = []

    def invalid(*args, **kwargs):
        count.append(1)
        return response({"bad": True})

    with pytest.raises(GenerationError, match="bounded attempts"):
        call(transport=invalid, max_attempts=2)
    assert len(count) == 2
    with pytest.raises(GenerationError, match="unsupported design"):
        call(
            transport=lambda *a, **k: response({"request": None, "unsupported": "moving platforms"})
        )


def test_fixture_is_explicit_and_no_key_needed(tmp_path):
    request, record = call(provider="fixture", fixture=FIXTURE, cache_dir=tmp_path)
    assert request.gap_count == 2
    assert record["identity"]["model"] == "offline-fixture"
    assert record["identity"]["fixture_sha256"]


@pytest.mark.parametrize(
    "options",
    [{"max_attempts": 4}, {"timeout": float("nan")}, {"provider": "unknown"}, {"fixture": FIXTURE}],
)
def test_bad_configuration(options):
    with pytest.raises(ValueError):
        call(**options)


def test_missing_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(GenerationError, match="API_KEY"):
        call()


def test_incomplete_response():
    with pytest.raises(GenerationError, match="incomplete"):
        call(transport=lambda *a, **k: {"candidates": [{"finishReason": "MAX_TOKENS"}]})


def test_real_transport_contract(monkeypatch):
    from ai_platformer.agents.llm.generation import NoRedirect, gemini_call

    monkeypatch.setenv("GEMINI_API_KEY", "test-secret")
    seen = []

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, size):
            assert size == 1_000_001
            return b'{"candidates": []}'

    class Opener:
        def open(self, request, timeout):
            seen.append(request)
            assert request.get_header("X-goog-api-key") == "test-secret"
            assert (
                request.full_url
                == "https://generativelanguage.googleapis.com/v1beta/models/gemini-test:generateContent"
            )
            assert "test-secret" not in request.full_url
            assert json.loads(request.data) == {"hello": "world"}
            assert timeout == 10
            return Reply()

    monkeypatch.setattr("urllib.request.build_opener", lambda handler: Opener())
    assert gemini_call({"hello": "world"}, model="gemini-test", timeout=10) == {"candidates": []}
    assert len(seen) == 1
    assert NoRedirect().redirect_request(None, None, 302, None, None, "https://other") is None


def test_corrupted_cache_is_rejected(tmp_path):
    call(provider="fixture", fixture=FIXTURE, cache_dir=tmp_path)
    cache = next(tmp_path.glob("*.json"))
    record = read_json(cache)
    del record["output"]
    cache.write_text(json.dumps(record))
    with pytest.raises(GenerationError, match="invalid cached response"):
        call(provider="fixture", fixture=FIXTURE, cache_dir=tmp_path)


def test_malformed_provider_envelope_is_bounded():
    with pytest.raises(GenerationError, match="bounded attempts"):
        call(transport=lambda *a, **k: [], max_attempts=1)
