"""Tests for the vlm tool bundle — per-provider request shaping, all mocked.

No network, no GPU: the openrouter provider (the default) is exercised
through ``httpx.MockTransport``, and the vertex provider behind an import
guard (skipped when google-genai is absent).
"""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import httpx
import numpy as np
import pytest
from gap_core.errors import ToolError
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def vlm():
    """The vlm bundle's tools module.

    The vlm bundle serves out-of-process (``serving.protocol:
    stdio-msgpack``), so ``load_skills`` does not import its ``tools.py``
    in-process — import it directly for these in-process unit tests.
    """
    import importlib.util

    tools_path = ROOT / "tools" / "vlm" / "tools.py"
    spec = importlib.util.spec_from_file_location("vlm_tools_under_test", tools_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(4, 6, 3), dtype=np.uint8)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    for var in (
        "GAP_VLM_PROVIDER",
        "GAP_VLM_MODEL",
        "GAP_VLM_BASE_URL",
        "GAP_VLM_API_KEY",
        "GAP_VLM_TEMPERATURE",
        "GAP_VLM_SEED_CAPABILITY",
        "GAP_VLM_PROJECT_ID",
        "GAP_VLM_REGION",
        "GAP_LLM_PROVIDER",
        "GAP_LLM_MODEL",
        "OPENROUTER_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


def _mock_openrouter(vlm, monkeypatch, reply: str):
    """Install an httpx.MockTransport on the vlm bundle's http seam."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("Authorization")
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": reply}}]},
        )

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(vlm, "_http_client", lambda: httpx.Client(transport=transport))
    return captured


def _strict_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GAP_VLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("GAP_VLM_BASE_URL", "https://relay.test/v1")
    monkeypatch.setenv("GAP_VLM_API_KEY", "strict-secret-key")
    monkeypatch.setenv("GAP_VLM_MODEL", "paper/model-v1")
    monkeypatch.setenv("GAP_VLM_TEMPERATURE", "0.1")


def _mock_strict(vlm, monkeypatch: pytest.MonkeyPatch, handler):
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(vlm, "_http_client", lambda: httpx.Client(transport=transport))


# ---------------------------------------------------------------------------
# strict openai_compatible provider — formal paper evidence
# ---------------------------------------------------------------------------


def test_openai_compatible_shapes_images_and_evidence(vlm, image, monkeypatch):
    _strict_env(monkeypatch)
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            headers={"x-request-id": "relay-request-1"},
            json={
                "model": "paper/model-v1-resolved",
                "seed": 17,
                "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
                "choices": [{"message": {"content": "the cup is red"}}],
            },
        )

    _mock_strict(vlm, monkeypatch, handler)
    second = np.zeros((2, 3, 3), dtype=np.uint8)
    result = vlm.query(prompt="Describe the cup", image=image, images=[second], seed=17)

    assert result["text"] == "the cup is red"
    request = captured[0]
    assert str(request.url) == "https://relay.test/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer strict-secret-key"
    payload = json.loads(request.content)
    assert payload["model"] == "paper/model-v1"
    assert payload["temperature"] == 0.1
    assert payload["seed"] == 17
    assert payload["max_tokens"] == 1024
    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "Describe the cup"}
    assert [item["type"] for item in content] == ["text", "image_url", "image_url"]
    for item in content[1:]:
        url = item["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        assert base64.b64decode(url.split(",", 1)[1]).startswith(b"\x89PNG\r\n\x1a\n")

    evidence = result["evidence"]
    assert evidence == {
        "provider": "openai_compatible",
        "base_url": "https://relay.test/v1",
        "requested_model": "paper/model-v1",
        "resolved_model": "paper/model-v1-resolved",
        "temperature": 0.1,
        "cache_policy": "disabled",
        "randomness": {
            "requested_seed": 17,
            "provider_reported_seed": 17,
            "seed_control": "provider_confirmed",
            "deterministic_claim": False,
        },
        "provider_request_id": "relay-request-1",
        "request_sha256": evidence["request_sha256"],
        "response_sha256": evidence["response_sha256"],
        "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
        "transport_attempts": [
            {
                "attempt": 1,
                "outcome": "success",
                "status": 200,
                "provider_request_id": "relay-request-1",
            }
        ],
        "fallback_used": False,
    }
    assert evidence["request_sha256"].startswith("sha256:")
    assert evidence["response_sha256"].startswith("sha256:")


@pytest.mark.parametrize("missing", ["GAP_VLM_BASE_URL", "GAP_VLM_API_KEY", "GAP_VLM_MODEL"])
def test_openai_compatible_requires_explicit_config(vlm, monkeypatch, missing):
    _strict_env(monkeypatch)
    monkeypatch.delenv(missing)
    monkeypatch.setenv("GAP_LLM_MODEL", "must-not-fallback")
    monkeypatch.setenv("OPENROUTER_API_KEY", "must-not-fallback")
    with pytest.raises(ToolError, match=missing):
        vlm.query(prompt="secret prompt")


def test_openai_compatible_rejects_url_embedded_password(vlm, monkeypatch):
    _strict_env(monkeypatch)
    monkeypatch.setenv("GAP_VLM_BASE_URL", "https://:password@relay.test/v1")
    with pytest.raises(ToolError, match="public HTTP.*base URL"):
        vlm.query(prompt="q")


@pytest.mark.parametrize(
    "provider,base_url",
    [
        ("openai_compatible", "https://relay.test/v1?token=secret"),
        ("openai_compatible", "https://relay.test/v1/strict-secret-key"),
        ("openrouter", "https://user:secret@relay.test/v1"),
        ("openrouter", "https://relay.test/v1#secret"),
        ("openrouter", "https://relay.test/v1/strict-secret-key"),
    ],
)
def test_published_base_url_rejects_secret_or_ambiguous_components(
    vlm, monkeypatch, provider, base_url
):
    monkeypatch.setenv("GAP_VLM_PROVIDER", provider)
    monkeypatch.setenv("GAP_VLM_BASE_URL", base_url)
    monkeypatch.setenv("GAP_VLM_MODEL", "paper/model-v1")
    monkeypatch.setenv("GAP_VLM_API_KEY", "strict-secret-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-router")
    with pytest.raises(ToolError, match="public HTTP.*base URL"):
        vlm.query(prompt="q")


def test_openai_compatible_request_hash_binds_canonical_endpoint(vlm, monkeypatch):
    _strict_env(monkeypatch)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "paper/model-v1",
                "choices": [{"message": {"content": "ok"}}],
            },
        )

    _mock_strict(vlm, monkeypatch, handler)
    first = vlm.query(prompt="same", seed=3)
    monkeypatch.setenv("GAP_VLM_BASE_URL", "HTTPS://OTHER.TEST:443/a/../v1/")
    second = vlm.query(prompt="same", seed=3)
    assert second["evidence"]["base_url"] == "https://other.test/v1"
    assert first["evidence"]["request_sha256"] != second["evidence"]["request_sha256"]


@pytest.mark.parametrize(
    "configured,canonical",
    [
        ("https://relay.test/v1%3Fopaque=x", "https://relay.test/v1%3Fopaque=x"),
        ("https://relay.test/v1%23opaque", "https://relay.test/v1%23opaque"),
        ("https://relay.test/v1%2Fopaque", "https://relay.test/v1%2Fopaque"),
        ("https://例え.テスト/v1", "https://xn--r8jz45g.xn--zckzah/v1"),
    ],
)
def test_openai_compatible_evidence_matches_raw_canonical_request_endpoint(
    vlm, monkeypatch, configured, canonical
):
    _strict_env(monkeypatch)
    monkeypatch.setenv("GAP_VLM_BASE_URL", configured)
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "model": "paper/model-v1",
                "choices": [{"message": {"content": "ok"}}],
            },
        )

    _mock_strict(vlm, monkeypatch, handler)
    result = vlm.query(prompt="same", seed=3)
    assert result["evidence"]["base_url"] == canonical
    assert str(captured[0].url) == f"{canonical}/chat/completions"


def test_openai_compatible_seed_unconfirmed_and_independent_results(vlm, monkeypatch):
    _strict_env(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "paper/model-v1",
                "choices": [{"message": {"content": "ok"}}],
            },
        )

    _mock_strict(vlm, monkeypatch, handler)
    first = vlm.query(prompt="same", model="explicit/model", temperature=0.2, seed=9)
    second = vlm.query(prompt="same", model="explicit/model", temperature=0.2, seed=9)
    assert first["evidence"]["randomness"] == {
        "requested_seed": 9,
        "provider_reported_seed": None,
        "seed_control": "requested_unconfirmed",
        "deterministic_claim": False,
    }
    assert first["evidence"]["requested_model"] == "explicit/model"
    assert first["evidence"]["temperature"] == 0.2
    assert first["evidence"]["request_sha256"] == second["evidence"]["request_sha256"]
    assert first["evidence"]["response_sha256"] == second["evidence"]["response_sha256"]
    assert first["evidence"] is not second["evidence"]
    assert first["evidence"]["randomness"] is not second["evidence"]["randomness"]
    assert first["evidence"]["transport_attempts"] is not second["evidence"]["transport_attempts"]


def test_openai_compatible_response_hash_excludes_transport_metadata(vlm, monkeypatch):
    _strict_env(monkeypatch)
    monkeypatch.setattr(vlm, "_BACKOFF_S", 0.0)
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, headers={"x-request-id": "volatile-retry"})
        request_id = "volatile-first" if attempts == 2 else "volatile-second"
        return httpx.Response(
            200,
            headers={"x-request-id": request_id},
            json={
                "model": "resolved/model",
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                "choices": [{"message": {"content": "same content"}}],
            },
        )

    _mock_strict(vlm, monkeypatch, handler)
    first = vlm.query(prompt="same")
    second = vlm.query(prompt="same")
    assert len(first["evidence"]["transport_attempts"]) == 2
    assert len(second["evidence"]["transport_attempts"]) == 1
    assert first["evidence"]["provider_request_id"] != second["evidence"]["provider_request_id"]
    assert first["evidence"]["response_sha256"] == second["evidence"]["response_sha256"]


def test_openai_compatible_known_unsupported_seed_is_not_sent(vlm, monkeypatch):
    _strict_env(monkeypatch)
    monkeypatch.setenv("GAP_VLM_SEED_CAPABILITY", "unsupported")
    payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    _mock_strict(vlm, monkeypatch, handler)
    evidence = vlm.query(prompt="q", seed=4)["evidence"]
    assert "seed" not in payloads[0]
    assert evidence["randomness"] == {
        "requested_seed": None,
        "provider_reported_seed": None,
        "seed_control": "unsupported",
        "deterministic_claim": False,
    }


def test_openai_compatible_seed_mismatch_fails_closed(vlm, monkeypatch):
    _strict_env(monkeypatch)
    _mock_strict(
        vlm,
        monkeypatch,
        lambda request: httpx.Response(
            200,
            json={"seed": 8, "choices": [{"message": {"content": "unsafe"}}]},
        ),
    )
    with pytest.raises(ToolError, match="seed confirmation mismatch"):
        vlm.query(prompt="q", seed=7)


@pytest.mark.parametrize("unsafe_id", ["strict-secret-key-reflected", "sensitiveprompt"])
def test_openai_compatible_nulls_unsafe_reflected_request_id(vlm, monkeypatch, unsafe_id):
    _strict_env(monkeypatch)
    _mock_strict(
        vlm,
        monkeypatch,
        lambda request: httpx.Response(
            200,
            headers={"x-request-id": unsafe_id},
            json={"choices": [{"message": {"content": "ok"}}]},
        ),
    )
    result = vlm.query(prompt="sensitiveprompt")
    assert result["evidence"]["provider_request_id"] is None
    assert result["evidence"]["transport_attempts"][0]["provider_request_id"] is None


def test_openai_compatible_retries_then_stops_on_content(vlm, monkeypatch):
    _strict_env(monkeypatch)
    monkeypatch.setattr(vlm, "_BACKOFF_S", 0.0)
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(503, headers={"x-request-id": "retry-1"}, json={"secret": "body"})
        if len(attempts) == 2:
            raise httpx.ConnectError("secret transport detail", request=request)
        if len(attempts) == 3:
            return httpx.Response(429, headers={"x-request-id": "retry-3"})
        return httpx.Response(
            200,
            headers={"x-request-id": "success-4"},
            json={
                "choices": [{"message": {"content": "first content"}}],
            },
        )

    _mock_strict(vlm, monkeypatch, handler)
    result = vlm.query(prompt="q")
    assert result["text"] == "first content"
    assert len(attempts) == 4
    assert result["evidence"]["transport_attempts"] == [
        {"attempt": 1, "outcome": "http_error", "status": 503, "provider_request_id": "retry-1"},
        {"attempt": 2, "outcome": "transport_error", "status": None, "provider_request_id": None},
        {"attempt": 3, "outcome": "http_error", "status": 429, "provider_request_id": "retry-3"},
        {"attempt": 4, "outcome": "success", "status": 200, "provider_request_id": "success-4"},
    ]
    assert result["evidence"]["fallback_used"] is False


@pytest.mark.parametrize("status", [400, 503])
def test_openai_compatible_errors_are_bounded_and_redacted(vlm, monkeypatch, status):
    _strict_env(monkeypatch)
    monkeypatch.setattr(vlm, "_BACKOFF_S", 0.0)
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(
            status,
            headers={"x-request-id": "strict-secret-key-reflected"},
            text="secret prompt relay body strict-secret-key",
        )

    _mock_strict(vlm, monkeypatch, handler)
    with pytest.raises(ToolError) as raised:
        vlm.query(prompt="secret prompt")
    assert len(attempts) == (1 if status == 400 else 4)
    error = str(raised.value)
    assert "secret prompt" not in error
    assert "strict-secret-key" not in error
    assert "relay body" not in error


# ---------------------------------------------------------------------------
# openrouter provider (default) — OpenAI-compatible chat completions
# ---------------------------------------------------------------------------


def test_openrouter_is_default_with_data_url_image(vlm, image, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-router")
    captured = _mock_openrouter(vlm, monkeypatch, reply="a red mug")

    out = vlm.query(prompt="What is on the table?", image=image)
    assert out["text"] == "a red mug"
    assert out["evidence"]["provider"] == "openrouter"

    # Default provider targets OpenRouter with the bundle-default model.
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["auth"] == "Bearer sk-router"
    payload = captured["payload"]
    assert payload["model"] == vlm.DEFAULT_MODEL == "gemini-3.1-flash-lite-preview"
    assert payload["max_tokens"] == 1024
    assert payload["temperature"] == 0.0

    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "What is on the table?"}
    url = content[1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    raw = base64.b64decode(url.split(",", 1)[1])
    np.testing.assert_array_equal(
        np.asarray(Image.open(io.BytesIO(raw)).convert("RGB")),
        image,
    )


def test_openrouter_model_env_override_and_multiple_images(vlm, image, monkeypatch):
    monkeypatch.setenv("GAP_VLM_MODEL", "anthropic/claude-sonnet-4")
    captured = _mock_openrouter(vlm, monkeypatch, reply="compared")

    second = np.zeros((2, 3, 3), dtype=np.uint8)
    vlm.query(prompt="compare", image=image, images=[second])

    payload = captured["payload"]
    assert payload["model"] == "anthropic/claude-sonnet-4"
    content = payload["messages"][0]["content"]
    # Text first, then one image_url block per image.
    assert [b["type"] for b in content] == ["text", "image_url", "image_url"]


def test_openrouter_custom_base_url(vlm, image, monkeypatch):
    monkeypatch.setenv("GAP_VLM_BASE_URL", "http://vlm.test/v1")
    monkeypatch.setenv("GAP_VLM_API_KEY", "sk-test")
    monkeypatch.setenv("GAP_VLM_MODEL", "gcp/google/gemini-3-flash-preview")
    captured = _mock_openrouter(vlm, monkeypatch, reply="two cups")

    out = vlm.query(prompt="What objects are on the table?", image=image)
    assert out["text"] == "two cups"
    assert out["evidence"]["base_url"] == "http://vlm.test/v1"

    assert captured["url"] == "http://vlm.test/v1/chat/completions"
    assert captured["auth"] == "Bearer sk-test"
    assert captured["payload"]["model"] == "gcp/google/gemini-3-flash-preview"


def test_openrouter_text_only_query(vlm, monkeypatch):
    captured = _mock_openrouter(vlm, monkeypatch, reply="hi there")
    vlm.query(prompt="hello")
    content = captured["payload"]["messages"][0]["content"]
    assert content == [{"type": "text", "text": "hello"}]


def test_openrouter_temperature_override_matches_evidence(vlm, monkeypatch):
    captured = _mock_openrouter(vlm, monkeypatch, reply="ok")
    result = vlm.query(prompt="q", temperature=0.7)
    assert captured["payload"]["temperature"] == 0.7
    assert result["evidence"]["temperature"] == 0.7


def test_legacy_request_hash_binds_png_shape_stability_and_image_order(vlm, monkeypatch):
    _mock_openrouter(vlm, monkeypatch, reply="ok")
    raw = np.arange(6, dtype=np.uint8)
    wide = raw.reshape(1, 2, 3)
    tall = raw.reshape(2, 1, 3)

    wide_hash = vlm.query(prompt="q", image=wide)["evidence"]["request_sha256"]
    wide_copy_hash = vlm.query(prompt="q", image=wide.copy())["evidence"]["request_sha256"]
    tall_hash = vlm.query(prompt="q", image=tall)["evidence"]["request_sha256"]
    assert wide_hash == wide_copy_hash
    assert wide_hash != tall_hash

    black = np.zeros((1, 1, 3), dtype=np.uint8)
    white = np.full((1, 1, 3), 255, dtype=np.uint8)
    forward = vlm.query(prompt="q", images=[black, white])["evidence"]["request_sha256"]
    reverse = vlm.query(prompt="q", images=[white, black])["evidence"]["request_sha256"]
    assert forward != reverse


def test_openrouter_reports_actual_retry_and_response_metadata(vlm, monkeypatch):
    monkeypatch.setenv("GAP_VLM_API_KEY", "legacy-secret")
    monkeypatch.setenv("GAP_VLM_MODEL", "requested/model")
    monkeypatch.setattr(vlm, "_BACKOFF_S", 0.0)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, headers={"x-request-id": "legacy-retry"})
        return httpx.Response(
            200,
            headers={"x-request-id": "legacy-success"},
            json={
                "model": "resolved/model",
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
                "choices": [{"message": {"content": "ok"}}],
            },
        )

    _mock_strict(vlm, monkeypatch, handler)
    evidence = vlm.query(prompt="q", provider="openrouter")["evidence"]
    assert evidence["resolved_model"] == "resolved/model"
    assert evidence["provider_request_id"] == "legacy-success"
    assert evidence["usage"] == {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
    assert evidence["transport_attempts"] == [
        {
            "attempt": 1,
            "outcome": "http_error",
            "status": 503,
            "provider_request_id": "legacy-retry",
        },
        {
            "attempt": 2,
            "outcome": "success",
            "status": 200,
            "provider_request_id": "legacy-success",
        },
    ]


def test_openrouter_unattested_metadata_stays_none(vlm, monkeypatch):
    captured = _mock_openrouter(vlm, monkeypatch, reply="ok")
    evidence = vlm.query(prompt="q")["evidence"]
    assert captured
    assert evidence["resolved_model"] is None
    assert evidence["provider_request_id"] is None
    assert evidence["usage"] is None


def test_openrouter_retries_malformed_success_with_sanitized_evidence(vlm, monkeypatch):
    monkeypatch.setattr(vlm, "_BACKOFF_S", 0.0)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                headers={"x-request-id": "malformed-first"},
                text="unsafe relay body",
            )
        return httpx.Response(
            200,
            headers={"x-request-id": "valid-second"},
            json={"choices": [{"message": {"content": "ok"}}]},
        )

    _mock_strict(vlm, monkeypatch, handler)
    evidence = vlm.query(prompt="q")["evidence"]
    assert evidence["transport_attempts"] == [
        {
            "attempt": 1,
            "outcome": "transport_error",
            "status": None,
            "provider_request_id": "malformed-first",
        },
        {
            "attempt": 2,
            "outcome": "success",
            "status": 200,
            "provider_request_id": "valid-second",
        },
    ]
    assert "unsafe relay body" not in repr(evidence)


def test_openrouter_retries_non_string_content(vlm, monkeypatch):
    monkeypatch.setattr(vlm, "_BACKOFF_S", 0.0)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        content = None if calls == 1 else "ok"
        return httpx.Response(
            200,
            headers={"x-request-id": f"content-{calls}"},
            json={"choices": [{"message": {"content": content}}]},
        )

    _mock_strict(vlm, monkeypatch, handler)
    result = vlm.query(prompt="q")
    assert result["text"] == "ok"
    assert result["evidence"]["transport_attempts"][0] == {
        "attempt": 1,
        "outcome": "transport_error",
        "status": None,
        "provider_request_id": "content-1",
    }


def test_openrouter_backend_failure_raises_tool_error(vlm, monkeypatch):
    monkeypatch.setenv("GAP_VLM_MODEL", "m")
    monkeypatch.setattr(vlm, "_BACKOFF_S", 0.0)

    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(503, json={"error": "overloaded"})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(vlm, "_http_client", lambda: httpx.Client(transport=transport))

    with pytest.raises(ToolError, match="unavailable after 3 attempts"):
        vlm.query(prompt="q")
    assert len(attempts) == 3


# ---------------------------------------------------------------------------
# vertex provider (Gemini only; import-guarded)
# ---------------------------------------------------------------------------


def test_vertex_rejects_claude_model(vlm, monkeypatch):
    monkeypatch.setenv("GAP_VLM_PROVIDER", "vertex")
    monkeypatch.setenv("GAP_VLM_MODEL", "claude-opus-4-8")
    monkeypatch.setenv("GAP_VLM_PROJECT_ID", "test-project")
    with pytest.raises(ToolError, match="Gemini models only"):
        vlm.query(prompt="hi")


def test_vertex_provider_routes_gemini_models_to_genai(vlm, image, monkeypatch):
    pytest.importorskip("google.genai")
    from google import genai

    captured: dict = {}

    class _FakeClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.models = SimpleNamespace(generate_content=self._generate)

        def _generate(self, *, model, contents, config=None):
            captured["model"] = model
            captured["contents"] = contents
            captured["config"] = config
            return SimpleNamespace(text="gemini says hi")

    monkeypatch.setattr(genai, "Client", _FakeClient)
    monkeypatch.setenv("GAP_VLM_PROVIDER", "vertex")
    monkeypatch.setenv("GAP_VLM_MODEL", "gemini-3-flash-preview")
    monkeypatch.setenv("GAP_VLM_PROJECT_ID", "test-project")

    out = vlm.query(prompt="hi", image=image, temperature=0.7)
    assert out["text"] == "gemini says hi"
    assert out["evidence"]["provider"] == "vertex"
    assert out["evidence"]["base_url"] is None
    assert captured["client_kwargs"] == {
        "vertexai": True,
        "project": "test-project",
        "location": "global",
    }
    assert captured["model"] == "gemini-3-flash-preview"
    assert captured["contents"][0] == "hi"
    # Deterministic decoding — parity with the dev servicer's production
    # path (temperature 0.0, max_tokens 1024).
    assert captured["config"].temperature == 0.7
    assert captured["config"].max_output_tokens == 1024


def test_vertex_gemini_retries_transient_failures(vlm, image, monkeypatch):
    """The vertex Gemini path retries like the openrouter path (3 attempts)."""
    pytest.importorskip("google.genai")
    from google import genai

    attempts: list[int] = []

    class _FlakyClient:
        def __init__(self, **kwargs):
            self.models = SimpleNamespace(generate_content=self._generate)

        def _generate(self, *, model, contents, config=None):
            attempts.append(1)
            if len(attempts) < 3:
                raise RuntimeError("503 transient")
            return SimpleNamespace(text="third time lucky")

    monkeypatch.setattr(genai, "Client", _FlakyClient)
    monkeypatch.setattr(vlm, "_BACKOFF_S", 0.0)
    monkeypatch.setenv("GAP_VLM_PROVIDER", "vertex")
    monkeypatch.setenv("GAP_VLM_MODEL", "gemini-3-flash-preview")
    monkeypatch.setenv("GAP_VLM_PROJECT_ID", "test-project")

    out = vlm.query(prompt="hi")
    assert out["text"] == "third time lucky"
    assert len(attempts) == 3


def test_vertex_gemini_exhausted_retries_raise_tool_error(vlm, monkeypatch):
    pytest.importorskip("google.genai")
    from google import genai

    class _DeadClient:
        def __init__(self, **kwargs):
            self.models = SimpleNamespace(generate_content=self._generate)

        def _generate(self, **kwargs):
            raise RuntimeError("permanently overloaded")

    monkeypatch.setattr(genai, "Client", _DeadClient)
    monkeypatch.setattr(vlm, "_BACKOFF_S", 0.0)
    monkeypatch.setenv("GAP_VLM_PROVIDER", "vertex")
    monkeypatch.setenv("GAP_VLM_MODEL", "gemini-3-flash-preview")
    monkeypatch.setenv("GAP_VLM_PROJECT_ID", "test-project")

    with pytest.raises(ToolError, match="unavailable after 3 attempts"):
        vlm.query(prompt="hi")


def test_vertex_reports_available_metadata_and_attempts_without_sdk(vlm, monkeypatch):
    captured: dict = {}

    class _Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _Part:
        @staticmethod
        def from_bytes(**kwargs):
            return kwargs

    calls = 0

    class _FakeClient:
        def __init__(self, **kwargs):
            self.models = SimpleNamespace(generate_content=self._generate)

        def _generate(self, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("temporary")
            captured.update(kwargs)
            return SimpleNamespace(
                text="vertex ok",
                model_version="resolved-gemini",
                response_id="vertex-request",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=4,
                    candidates_token_count=2,
                    total_token_count=6,
                ),
            )

    google = ModuleType("google")
    genai = ModuleType("google.genai")
    genai.Client = _FakeClient
    genai.types = SimpleNamespace(GenerateContentConfig=_Config, Part=_Part)
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setenv("GAP_VLM_PROVIDER", "vertex")
    monkeypatch.setenv("GAP_VLM_MODEL", "requested-gemini")
    monkeypatch.setenv("GAP_VLM_PROJECT_ID", "project")
    monkeypatch.setattr(vlm, "_BACKOFF_S", 0.0)

    evidence = vlm.query(prompt="q")["evidence"]
    assert evidence["resolved_model"] == "resolved-gemini"
    assert evidence["provider_request_id"] == "vertex-request"
    assert evidence["usage"] == {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
    assert evidence["transport_attempts"] == [
        {"attempt": 1, "outcome": "transport_error", "status": None, "provider_request_id": None},
        {
            "attempt": 2,
            "outcome": "success",
            "status": None,
            "provider_request_id": "vertex-request",
        },
    ]


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------


def test_provider_kwarg_overrides_env(vlm, monkeypatch):
    # Env says vertex (and is unconfigured, so it would fail) — the kwarg wins.
    monkeypatch.setenv("GAP_VLM_PROVIDER", "vertex")
    captured = _mock_openrouter(vlm, monkeypatch, reply="a red mug")

    out = vlm.query(prompt="q", provider="openrouter")
    assert out["text"] == "a red mug"
    assert captured["payload"]["messages"][0]["content"][0]["text"] == "q"


def test_provider_does_not_inherit_gap_llm_provider(vlm, monkeypatch):
    monkeypatch.setenv("GAP_LLM_PROVIDER", "openai_compatible")
    captured = _mock_openrouter(vlm, monkeypatch, reply="default route")
    result = vlm.query(prompt="q")
    assert result["text"] == "default route"
    assert result["evidence"]["provider"] == "openrouter"
    assert captured["url"].startswith("https://openrouter.ai/")


def test_unknown_provider_raises_tool_error(vlm, monkeypatch):
    monkeypatch.setenv("GAP_VLM_PROVIDER", "bedrock")
    with pytest.raises(ToolError, match="unknown provider 'bedrock'") as raised:
        vlm.query(prompt="q")
    assert "openai_compatible" in str(raised.value)


def test_invalid_image_rejected(vlm):
    bad = np.zeros((4, 6), dtype=np.uint8)  # missing channel dim
    with pytest.raises(ValueError, match=r"uint8 \[H, W, 3\]"):
        vlm.query(prompt="q", image=bad)


# ---------------------------------------------------------------------------
# Yes/no coercion (first standalone yes/no word; legacy substring fallback)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Yes", True),
        ("yes.", True),
        ("YES — clearly visible.", True),
        ("The answer is yes", True),
        ("Yes, the object matches the description.", True),
        ("Eyes on the table", True),  # legacy substring fallback quirk
        ("No", False),
        ("no.", False),
        ("Absolutely not", False),
        ("I cannot tell", False),
        ("", False),
        # First standalone word wins — the legacy substring check would
        # mislabel both of these (the G1 verify-gate failure mode):
        ("No, although the label literally says YES on it.", False),
        ("NO. The item appears to be a small book.", False),
    ],
)
def test_query_yes_no_coercion(vlm, monkeypatch, text, expected):
    _mock_openrouter(vlm, monkeypatch, reply=text)
    out = vlm.query_yes_no(prompt="Is the sauce in the basket?")
    assert out["answer"] is expected
    assert out["text"] == text
    assert out["evidence"]["provider"] == "openrouter"


def test_query_yes_no_appends_explicit_instruction(vlm, monkeypatch):
    """query_yes_no must elicit a parseable YES/NO-first reply.

    Without the instruction (and at temperature > 0) models answer
    affirmatively in prose with no literal "yes" — which the coercion
    mislabels as False. A false "No" from the perceiving-objects verify
    gate rejects a correct exterior pick and forces the degraded
    single-view wrist fallback.
    """
    captured = _mock_openrouter(vlm, monkeypatch, reply="Yes.")

    vlm.query_yes_no(prompt="Is this a cream cheese box?")

    (text_block,) = captured["payload"]["messages"][0]["content"]
    assert text_block["type"] == "text"
    assert text_block["text"].startswith("Is this a cream cheese box?")
    assert "YES or NO first" in text_block["text"]
