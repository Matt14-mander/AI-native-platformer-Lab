"""Diagnostic probes must separate TLS, listing and generation; never retain keys."""

import json
import ssl
import urllib.error

from scripts import diagnose_llm_connection as diagnostic


def setup_probes(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "diagnostic-secret")
    monkeypatch.setattr(diagnostic, "direct_tls", lambda timeout: {"tls_version": "TLSv1.3"})
    monkeypatch.setattr(
        diagnostic,
        "list_models",
        lambda *a: {"model_ids": ["gemini-test", "gemini-comparison"], "listing_complete": True},
    )


def test_listing_success_is_distinct_from_tls_failure_on_post(monkeypatch):
    setup_probes(monkeypatch)

    def fail(*a, **k):
        raise urllib.error.URLError(ssl.SSLEOFError("unexpected EOF diagnostic-secret"))

    monkeypatch.setattr(diagnostic, "gemini_call", fail)
    report = diagnostic.diagnose("gemini-test", timeout=1)
    assert report["probes"]["authenticated_models"]["status"] == "ok"
    assert report["probes"]["minimal_generation"]["connection_category"] == "tls"
    assert report["probes"]["minimal_generation"]["exception_type"] == "SSLEOFError"
    assert report["conclusion"] == "minimal_generation_failed"
    assert "diagnostic-secret" not in json.dumps(report)


def test_comparison_runs_only_after_primary_failure(monkeypatch):
    setup_probes(monkeypatch)
    calls = []

    def generate(payload, *, model, timeout):
        calls.append(model)
        assert payload["generationConfig"]["maxOutputTokens"] == 256
        if model == "gemini-test":
            raise TimeoutError()
        return {"candidates": [{"finishReason": "STOP"}]}

    monkeypatch.setattr(diagnostic, "gemini_call", generate)
    report = diagnostic.diagnose("gemini-test", comparison_model="gemini-comparison")
    assert calls == ["gemini-test", "gemini-comparison"]
    assert report["probes"]["comparison_generation"]["status"] == "ok"
    assert "not_proof_of_root_cause" in report["conclusion"]


def test_primary_success_does_not_spend_on_comparison(monkeypatch):
    setup_probes(monkeypatch)
    calls = []

    def generate(payload, **kwargs):
        calls.append(kwargs["model"])
        return {"candidates": [{"finishReason": "MAX_TOKENS"}]}

    monkeypatch.setattr(diagnostic, "gemini_call", generate)
    report = diagnostic.diagnose("gemini-test", comparison_model="gemini-comparison")
    assert calls == ["gemini-test"]
    assert "full_pipeline_still_unverified" in report["conclusion"]


def test_no_key_skips_authenticated_probes(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(diagnostic, "direct_tls", lambda timeout: {})
    report = diagnostic.diagnose("gemini-test")
    assert report["conclusion"] == "missing_api_key_in_this_terminal"
    assert set(report["probes"]) == {"direct_tls"}


def test_tls12_mode_applies_to_all_network_probes(monkeypatch):
    setup_probes(monkeypatch)
    seen = []

    def direct(timeout, **options):
        seen.append(options)
        return {"tls_version": "TLSv1.2"}

    def listing(key, timeout, **options):
        seen.append(options)
        return {"model_ids": ["gemini-test"], "listing_complete": True}

    def generation(payload, **options):
        seen.append({"tls_mode": options["tls_mode"]})
        return {"candidates": [{"finishReason": "STOP"}]}

    monkeypatch.setattr(diagnostic, "direct_tls", direct)
    monkeypatch.setattr(diagnostic, "list_models", listing)
    monkeypatch.setattr(diagnostic, "gemini_call", generation)
    report = diagnostic.diagnose("gemini-test", tls_mode="tls12")
    assert report["tls_mode"] == "tls12"
    assert seen == [{"tls_mode": "tls12"}] * 3
