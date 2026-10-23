"""Probe TLS, authenticated listing and minimal generation without exposing keys."""

from __future__ import annotations

import argparse
import json
import os
import socket
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

from ai_platformer.agents.llm.generation import (
    MAX_BYTES,
    connection_error_details,
    gemini_call,
    http_error_details,
    https_handlers,
)
from ai_platformer.content.level_spec import parse_level_json

HOST = "generativelanguage.googleapis.com"


def probe(operation):
    started = time.monotonic()
    try:
        result = {"status": "ok", **operation()}
    except urllib.error.HTTPError as error:
        result = {"status": "error", "http_status": error.code, **http_error_details(error)}
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        result = {"status": "error", **connection_error_details(error)}
    except (ValueError, TypeError, KeyError):
        result = {"status": "error", "reason": "invalid API response"}
    return {**result, "seconds": round(time.monotonic() - started, 3)}


def direct_tls(timeout, tls_mode="default"):
    context = ssl.create_default_context()
    if tls_mode == "tls12":
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.maximum_version = ssl.TLSVersion.TLSv1_2
    with (
        socket.create_connection((HOST, 443), timeout=timeout) as connection,
        context.wrap_socket(connection, server_hostname=HOST) as secured,
    ):
        issuer = secured.getpeercert().get("issuer", ())
        common_names = [value for group in issuer for key, value in group if key == "commonName"]
        return {"tls_version": secured.version(), "certificate_issuer": common_names}


def list_models(key, timeout, tls_mode="default"):
    request = urllib.request.Request(
        f"https://{HOST}/v1beta/models?pageSize=1000", headers={"x-goog-api-key": key}
    )
    with urllib.request.build_opener(*https_handlers(tls_mode)).open(
        request, timeout=timeout
    ) as response:
        raw = response.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("response too large")
    data = parse_level_json(raw)
    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        raise TypeError("invalid model listing")
    return {
        "model_ids": [
            item["name"].removeprefix("models/")
            for item in data["models"]
            if "generateContent" in item.get("supportedGenerationMethods", [])
        ],
        "listing_complete": not bool(data.get("nextPageToken")),
    }


def minimal_generation(model, timeout, tls_mode="default"):
    options = {} if tls_mode == "default" else {"tls_mode": tls_mode}
    response = gemini_call(
        {
            "contents": [{"parts": [{"text": "Reply with OK only."}]}],
            "generationConfig": {"maxOutputTokens": 256},
        },
        model=model,
        timeout=timeout,
        **options,
    )
    if not isinstance(response, dict) or not response.get("candidates"):
        raise ValueError("no candidates")
    return {
        "model": model,
        "http_response_received": True,
        "finish_reasons": [candidate.get("finishReason") for candidate in response["candidates"]],
    }


def diagnose(model, *, timeout=30, comparison_model=None, tls_mode="default"):
    if tls_mode not in {"default", "tls12"}:
        raise ValueError("tls_mode must be default or tls12")
    options = {} if tls_mode == "default" else {"tls_mode": tls_mode}
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    report = {
        "schema_version": 1,
        "openssl": ssl.OPENSSL_VERSION,
        "key_present": bool(key),
        "model": model,
        "proxy_environment_present": {
            name: bool(os.environ.get(name))
            for name in (
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "ALL_PROXY",
                "http_proxy",
                "https_proxy",
                "all_proxy",
            )
        },
        "detected_proxy_schemes": sorted(urllib.request.getproxies()),
        "timeout_seconds": timeout,
        "tls_mode": tls_mode,
        "probes": {},
    }
    probes = report["probes"]
    probes["direct_tls"] = probe(lambda: direct_tls(min(timeout, 10), **options))
    if not key:
        report["conclusion"] = "missing_api_key_in_this_terminal"
        return report
    listing = probe(lambda: list_models(key, timeout, **options))
    models = listing.pop("model_ids", [])
    listing["requested_model_listed"] = model in models if listing["status"] == "ok" else None
    probes["authenticated_models"] = listing
    probes["minimal_generation"] = probe(lambda: minimal_generation(model, timeout, **options))
    if probes["minimal_generation"]["status"] == "ok":
        report["conclusion"] = "minimal_generation_reachable_full_pipeline_still_unverified"
    else:
        report["conclusion"] = "minimal_generation_failed"
        if comparison_model and comparison_model != model:
            if (
                listing["status"] == "ok"
                and listing.get("listing_complete")
                and comparison_model not in models
            ):
                probes["comparison_generation"] = {
                    "status": "skipped",
                    "reason": "model_not_listed",
                }
            else:
                probes["comparison_generation"] = probe(
                    lambda: minimal_generation(comparison_model, timeout, **options)
                )
                if probes["comparison_generation"]["status"] == "ok":
                    report["conclusion"] = (
                        "comparison_model_responded_primary_failed_not_proof_of_root_cause"
                    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--comparison-model")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--tls-mode", choices=("default", "tls12"), default="default")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import re

    for model in (args.model, args.comparison_model):
        if model is not None and not re.fullmatch(r"gemini-[A-Za-z0-9._-]{1,100}", model):
            parser.error("use a Gemini model ID without paths or URL parameters")
    if not 0 < args.timeout <= 60:
        parser.error("timeout must be in (0,60]")
    if args.output is not None and args.output.exists():
        parser.error("output exists; choose a new path")
    report = diagnose(
        args.model,
        timeout=args.timeout,
        comparison_model=args.comparison_model,
        tls_mode=args.tls_mode,
    )
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    print(serialized, end="")
    if report["probes"].get("minimal_generation", {}).get("status") != "ok":
        parser.exit(1)


if __name__ == "__main__":
    main()
