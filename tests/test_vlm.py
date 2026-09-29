"""Tests for the vlm tool bundle — per-provider request shaping, all mocked.

No network, no GPU: the openrouter provider (the default) is exercised
through ``httpx.MockTransport``, as is the Gemini native REST relay; the
vertex provider is exercised behind an import guard (skipped when
google-genai is absent).
"""

from __future__ import annotations

import base64
import io
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

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
    for var in ("GAP_VLM_PROVIDER", "GAP_VLM_MODEL", "GAP_VLM_BASE_URL",
                "GAP_VLM_API_KEY", "GAP_VLM_PROJECT_ID", "GAP_VLM_REGION",
                "GAP_LLM_PROVIDER", "GAP_LLM_MODEL", "OPENROUTER_API_KEY",
                "GAP_VLM_MODEL_QUERY", "GAP_VLM_MODEL_QUERY_BATCH",
                "GAP_VLM_MODEL_QUERY_YES_NO",
                "GAP_VLM_BACKUP_BASE_URL", "GAP_VLM_BACKUP_API_KEY",
                "GAP_VLM_BACKUP_MODEL", "GAP_VLM_FAILOVER_CONSECUTIVE",
                "GAP_VLM_FAILOVER_COOLDOWN_S", "GAP_VLM_YES_NO_PARSER"):
        monkeypatch.delenv(var, raising=False)



@pytest.fixture(autouse=True)
def _fresh_route_health(vlm):
    vlm._route_health.clear()
    yield
    vlm._route_health.clear()


def _mock_openrouter(vlm, monkeypatch, reply: str):
    """Install an httpx.MockTransport on the vlm bundle's http seam."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("Authorization")
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": reply}}]},
        )

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(vlm, "_http_client", lambda: httpx.Client(transport=transport))
    return captured


# ---------------------------------------------------------------------------
# openrouter provider (default) — OpenAI-compatible chat completions
# ---------------------------------------------------------------------------


def test_openrouter_is_default_with_data_url_image(vlm, image, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-router")
    captured = _mock_openrouter(vlm, monkeypatch, reply="a red mug")

    out = vlm.query(prompt="What is on the table?", image=image)
    assert out == {"text": "a red mug"}

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
        np.asarray(Image.open(io.BytesIO(raw)).convert("RGB")), image,
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
    assert out == {"text": "two cups"}

    assert captured["url"] == "http://vlm.test/v1/chat/completions"
    assert captured["auth"] == "Bearer sk-test"
    assert captured["payload"]["model"] == "gcp/google/gemini-3-flash-preview"


def test_openrouter_text_only_query(vlm, monkeypatch):
    captured = _mock_openrouter(vlm, monkeypatch, reply="hi there")
    vlm.query(prompt="hello")
    content = captured["payload"]["messages"][0]["content"]
    assert content == [{"type": "text", "text": "hello"}]


def test_http_client_is_shared_without_closing_between_queries(vlm):
    with vlm._http_client() as first:
        with vlm._http_client() as second:
            assert first is second
        assert not first.is_closed
    assert not first.is_closed
    vlm._close_http_client()
    assert first.is_closed


def test_parallel_queries_share_client_without_serializing(vlm, image, monkeypatch):
    barrier = threading.Barrier(4)
    created = []

    def handler(request):
        barrier.wait(timeout=2)
        prompt = json.loads(request.content)["messages"][0]["content"][0]["text"]
        return httpx.Response(200, json={"choices": [{"message": {"content": prompt}}]})

    def factory():
        client = httpx.Client(transport=httpx.MockTransport(handler))
        created.append(client)
        return client

    monkeypatch.setattr(vlm, "_new_http_client", factory)
    prompts = [f"question-{i}" for i in range(8)]
    try:
        result = vlm.query_batch(prompts=prompts, images=[image] * 8)
        assert result == {"results": [{"text": prompt} for prompt in prompts]}
        assert len(created) == 1
        assert not created[0].is_closed
    finally:
        vlm._close_http_client()
    assert created[0].is_closed


def test_fork_reset_does_not_reuse_parent_pool_or_lock(vlm):
    with vlm._http_client() as parent:
        parent_lock = vlm._http_client_lock
        with parent_lock:
            vlm._reset_http_client_after_fork()
            assert vlm._http_client_lock is not parent_lock
            with vlm._http_client() as child:
                assert child is not parent
        assert not parent.is_closed
    parent.close()
    vlm._close_http_client()


@pytest.mark.parametrize("provider", ["openrouter", "gemini_rest"])
def test_sequential_queries_reuse_tcp_connection(vlm, monkeypatch, provider):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            body = json.dumps({
                "choices": [{"message": {"content": "yes"}}],
                "candidates": [{"content": {"parts": [{"text": "yes"}]}}],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    class Server(ThreadingHTTPServer):
        connections = 0

        def get_request(self):
            request = super().get_request()
            self.connections += 1
            return request

    server = Server(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    monkeypatch.setenv("GAP_VLM_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("GAP_VLM_API_KEY", "test-key")
    worker.start()
    try:
        for _ in range(8):
            out = vlm.query(prompt="hello", provider=provider)
            assert out["text"] == "yes"
            assert set(out) == ({"text", "route"} if provider == "gemini_rest" else {"text"})
        assert server.connections == 1
    finally:
        close = getattr(vlm, "_close_http_client", None)
        if close:
            close()
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_query_batch_is_bounded_concurrent_and_ordered(vlm, image, monkeypatch):
    lock = threading.Lock()
    four_started = threading.Event()
    active = 0
    max_active = 0

    def fake_query(prompt, image, images, provider, model):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            if active == 4:
                four_started.set()
        assert four_started.wait(timeout=1.0)
        time.sleep(0.01)
        with lock:
            active -= 1
        return prompt

    monkeypatch.setattr(vlm, "_query", fake_query)
    prompts = [f"question-{i}" for i in range(6)]

    out = vlm.query_batch(prompts=prompts, images=[image] * len(prompts))

    assert out == {"results": [{"text": prompt} for prompt in prompts]}
    assert max_active == 4


def test_query_batch_fails_closed(vlm, image, monkeypatch):
    def fake_query(prompt, image, images, provider, model):
        if prompt == "bad":
            raise ToolError("vlm", "provider failure")
        return prompt

    monkeypatch.setattr(vlm, "_query", fake_query)

    with pytest.raises(ToolError, match="provider failure"):
        vlm.query_batch(prompts=["ok", "bad"], images=[image, image])


def test_query_batch_rejects_mismatched_lengths(vlm, image):
    with pytest.raises(ValueError, match="one image per prompt"):
        vlm.query_batch(prompts=["one", "two"], images=[image])


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
# gemini_rest provider — Google generateContent schema over an HTTP relay
# ---------------------------------------------------------------------------


def test_gemini_rest_shapes_native_multimodal_request(vlm, image, monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["api_key"] = request.headers.get("x-goog-api-key")
        captured["authorization"] = request.headers.get("Authorization")
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "candidates": [{
                    "content": {"parts": [{"text": "YES\nInside the basket."}]}
                }]
            },
        )

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(vlm, "_http_client", lambda: httpx.Client(transport=transport))
    monkeypatch.setenv("GAP_VLM_PROVIDER", "gemini_rest")
    monkeypatch.setenv("GAP_VLM_BASE_URL", "https://relay.example/ai/genai")
    monkeypatch.setenv("GAP_VLM_API_KEY", "relay-secret")
    monkeypatch.setenv("GAP_VLM_MODEL", "gemini-3.6-flash")

    out = vlm.query_yes_no(prompt="Is the target inside?", image=image)

    route = out.pop("route")
    assert out == {"answer": True, "text": "YES\nInside the basket."}
    assert route["name"] == "primary"
    assert route["endpoint"] == "https://relay.example/ai/genai"
    assert "relay-secret" not in json.dumps(route)
    assert captured["url"] == (
        "https://relay.example/ai/genai/v1beta/models/"
        "gemini-3.6-flash:generateContent"
    )
    assert captured["api_key"] == "relay-secret"
    assert captured["authorization"] is None
    payload = captured["payload"]
    assert payload["generationConfig"] == {
        "temperature": 0.0,
        "maxOutputTokens": 8192,
    }
    parts = payload["contents"][0]["parts"]
    assert parts[0]["text"].startswith("Is the target inside?")
    assert "YES or NO first" in parts[0]["text"]
    assert parts[1]["inline_data"]["mime_type"] == "image/png"
    raw = base64.b64decode(parts[1]["inline_data"]["data"])
    np.testing.assert_array_equal(
        np.asarray(Image.open(io.BytesIO(raw)).convert("RGB")), image,
    )


def test_gemini_rest_fails_closed_on_missing_candidate_text(vlm, monkeypatch):
    monkeypatch.setenv("GAP_VLM_PROVIDER", "gemini_rest")
    monkeypatch.setenv("GAP_VLM_BASE_URL", "https://relay.example/ai/genai")
    monkeypatch.setenv("GAP_VLM_API_KEY", "relay-secret")
    monkeypatch.setattr(vlm, "_BACKOFF_S", 0.0)

    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(200, json={"candidates": []})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(vlm, "_http_client", lambda: httpx.Client(transport=transport))

    with pytest.raises(ToolError, match="unavailable after 3 attempts"):
        vlm.query(prompt="q")
    assert len(attempts) == 3


_THOUGHT_FIXTURES = json.loads(
    (ROOT / "tests" / "fixtures" / "vlm_gemini_rest_thought_responses.json").read_text()
)["responses"]


def _mock_gemini_rest(vlm, monkeypatch, bodies):
    """Serve ``bodies`` in order (last one repeats); capture every request."""
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append({
            "url": str(request.url),
            "payload": json.loads(request.content),
        })
        return httpx.Response(200, json=bodies[min(len(captured), len(bodies)) - 1])

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(vlm, "_http_client", lambda: httpx.Client(transport=transport))
    monkeypatch.setenv("GAP_VLM_PROVIDER", "gemini_rest")
    monkeypatch.setenv("GAP_VLM_BASE_URL", "https://relay.example/ai/genai")
    monkeypatch.setenv("GAP_VLM_API_KEY", "relay-secret")
    monkeypatch.setenv("GAP_VLM_MODEL", "gemini-3.8-flash")
    return captured


@pytest.mark.parametrize(
    "fixture", _THOUGHT_FIXTURES,
    ids=[f"{f['source']['model']}-{f['source']['pid']}" for f in _THOUGHT_FIXTURES],
)
def test_gemini_rest_answer_excludes_real_thought_parts(vlm, image, monkeypatch, fixture):
    _mock_gemini_rest(vlm, monkeypatch, [fixture["response"]])
    expected = fixture["expected_answer_text"]

    if fixture["source"]["kind"] == "yes_no":
        out = vlm.query_yes_no(prompt="Is it?", image=image)
        assert out["answer"] == vlm._coerce_yes_no(expected)
    else:
        out = vlm.query(prompt="A or B?", image=image)
    assert out["text"] == expected
    assert out["route"]["name"] == "primary"
    usage = fixture["response"].get("usageMetadata") or {}
    assert out["route"]["usage"]["thoughts_tokens"] == usage.get("thoughtsTokenCount", 0)
    has_thought = any(
        part.get("thought") for part in fixture["response"]["candidates"][0]["content"]["parts"]
    )
    assert (fixture["legacy_joined_text"] != expected) == has_thought


def test_gemini_rest_thought_leak_no_longer_flips_real_yes_no(vlm, image, monkeypatch):
    """Real 3.8-flash reply: its thought summary mentions "yes" before the NO answer."""
    fixture = next(f for f in _THOUGHT_FIXTURES if f["source"]["pid"] == "d4ea468e318e78d1")
    assert vlm._coerce_yes_no(fixture["legacy_joined_text"]) is True
    _mock_gemini_rest(vlm, monkeypatch, [fixture["response"]])

    out = vlm.query_yes_no(prompt="Is it?", image=image)

    assert out["answer"] is False
    assert out["text"].startswith("NO")


def test_gemini_rest_thought_only_response_fails_closed(vlm, monkeypatch):
    monkeypatch.setattr(vlm, "_BACKOFF_S", 0.0)
    body = {"candidates": [{"content": {"parts": [
        {"text": "**Thinking**\n\nStill deciding", "thought": True},
    ]}, "finishReason": "MAX_TOKENS"}]}
    captured = _mock_gemini_rest(vlm, monkeypatch, [body])

    with pytest.raises(ToolError, match="no candidate answer text"):
        vlm.query(prompt="q")
    assert len(captured) == 3


def test_gemini_rest_logs_thought_usage_without_secret(vlm, image, monkeypatch, caplog):
    fixture = next(f for f in _THOUGHT_FIXTURES if f["source"]["pid"] == "b33da5db74eff02f")
    _mock_gemini_rest(vlm, monkeypatch, [fixture["response"]])

    with caplog.at_level("INFO"):
        vlm.query(prompt="A or B?", image=image)

    usage = fixture["response"]["usageMetadata"]
    [record] = [r for r in caplog.records if "usage" in r.getMessage()]
    assert record.levelname == "WARNING"  # finish=MAX_TOKENS
    message = record.getMessage()
    assert f"thoughts_tokens={usage['thoughtsTokenCount']}" in message
    assert "thought_parts=1" in message
    assert "model=gemini-3.8-flash" in message
    assert "relay-secret" not in caplog.text


def test_call_kind_models_route_tournament_and_verify(vlm, image, monkeypatch):
    body = {"candidates": [{"content": {"parts": [{"text": "YES"}]}}]}
    captured = _mock_gemini_rest(vlm, monkeypatch, [body])
    monkeypatch.setenv("GAP_VLM_MODEL_QUERY_BATCH", "gemini-robotics-er-2-preview")
    monkeypatch.setenv("GAP_VLM_MODEL_QUERY_YES_NO", "gemini-robotics-er-2-preview")

    vlm.query(prompt="free", image=image)
    vlm.query_batch(prompts=["A or B?"], images=[image])
    vlm.query_yes_no(prompt="Is it?", image=image)
    vlm.query_yes_no(prompt="Is it?", image=image, model="explicit-model")

    models = [c["url"].rsplit("/", 1)[1].split(":")[0] for c in captured]
    assert models == [
        "gemini-3.8-flash",
        "gemini-robotics-er-2-preview",
        "gemini-robotics-er-2-preview",
        "explicit-model",
    ]


def test_call_kind_models_default_to_global_model(vlm, image, monkeypatch):
    body = {"candidates": [{"content": {"parts": [{"text": "A"}]}}]}
    captured = _mock_gemini_rest(vlm, monkeypatch, [body])
    monkeypatch.setenv("GAP_VLM_MODEL_QUERY", "")

    vlm.query(prompt="q", image=image)
    vlm.query_batch(prompts=["q"], images=[image])
    vlm.query_yes_no(prompt="q", image=image)

    assert {c["url"] for c in captured} == {
        "https://relay.example/ai/genai/v1beta/models/gemini-3.8-flash:generateContent"
    }


# Real response shape from the backup relay (2026-09-28): the answer part also
# carries a thoughtSignature and usage reports thinking tokens.
_THINKING_BODY = {
    "candidates": [{
        "content": {"role": "model", "parts": [
            {"text": "NO\nThe mug is outside the basket.", "thoughtSignature": "c2lnbmF0dXJl"},
        ]},
        "finishReason": "STOP",
    }],
    "usageMetadata": {
        "promptTokenCount": 1290, "candidatesTokenCount": 11,
        "thoughtsTokenCount": 187, "totalTokenCount": 1488,
        "promptTokensDetails": [
            {"modality": "TEXT", "tokenCount": 32}, {"modality": "IMAGE", "tokenCount": 1258},
        ],
    },
}


def _two_routes(vlm, monkeypatch, primary_handler, backup_handler):
    calls = {"primary": 0, "backup": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "primary.example":
            calls["primary"] += 1
            assert request.headers["x-goog-api-key"] == "primary-secret"
            return primary_handler(request)
        calls["backup"] += 1
        assert request.headers["x-goog-api-key"] == "backup-secret"
        return backup_handler(request)

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(vlm, "_http_client", lambda: httpx.Client(transport=transport))
    monkeypatch.setattr(vlm, "_BACKOFF_S", 0.0)
    monkeypatch.setenv("GAP_VLM_PROVIDER", "gemini_rest")
    monkeypatch.setenv("GAP_VLM_BASE_URL", "https://primary.example")
    monkeypatch.setenv("GAP_VLM_API_KEY", "primary-secret")
    monkeypatch.setenv("GAP_VLM_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("GAP_VLM_BACKUP_BASE_URL", "https://backup.example/ai/genai")
    monkeypatch.setenv("GAP_VLM_BACKUP_API_KEY", "backup-secret")
    return calls


def test_gemini_rest_thought_signature_text_and_thinking_usage(vlm, monkeypatch):
    calls = _two_routes(
        vlm, monkeypatch,
        lambda request: httpx.Response(200, json=_THINKING_BODY),
        lambda request: pytest.fail("healthy primary must not fail over"),
    )
    out = vlm.query(prompt="q")
    assert out["text"] == "NO\nThe mug is outside the basket."
    assert out["route"] == {
        "name": "primary", "endpoint": "https://primary.example",
        "model": "gemini-3.8-flash",
        "usage": {
            "input_tokens": 1290, "output_tokens": 198, "thoughts_tokens": 187,
            "prompt_tokens_details": {"IMAGE": 1258, "TEXT": 32},
        },
    }
    assert calls == {"primary": 1, "backup": 0}


@pytest.mark.parametrize("status", [429, 503])
def test_gemini_rest_provider_outage_fails_over_to_backup(vlm, monkeypatch, status):
    calls = _two_routes(
        vlm, monkeypatch,
        lambda request: httpx.Response(status, json={"error": "busy"}),
        lambda request: httpx.Response(200, json=_THINKING_BODY),
    )
    out = vlm.query_batch(prompts=["a"], images=[np.zeros((2, 2, 3), np.uint8)])
    (result,) = out["results"]
    assert result["route"]["name"] == "backup"
    assert result["route"]["endpoint"] == "https://backup.example/ai/genai"
    assert result["route"]["model"] == "gemini-3.8-flash"
    assert "secret" not in json.dumps(out)
    assert calls == {"primary": 3, "backup": 1}


def test_gemini_rest_transport_fault_fails_over(vlm, monkeypatch):
    def refuse(request):
        raise httpx.ConnectTimeout("handshake timed out", request=request)

    calls = _two_routes(
        vlm, monkeypatch, refuse, lambda request: httpx.Response(200, json=_THINKING_BODY)
    )
    assert vlm.query_yes_no(prompt="q")["route"]["name"] == "backup"
    assert calls == {"primary": 3, "backup": 1}


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(400, json={"error": "bad request"}),
        httpx.Response(200, json={"candidates": []}),
    ],
)
def test_gemini_rest_content_failure_never_fails_over(vlm, monkeypatch, response):
    calls = _two_routes(
        vlm, monkeypatch,
        lambda request: response,
        lambda request: pytest.fail("content failures must not fail over"),
    )
    with pytest.raises(ToolError, match="gemini_rest backend unavailable after 3 attempts"):
        vlm.query(prompt="q")
    assert calls == {"primary": 3, "backup": 0}


def test_gemini_rest_parks_failed_primary_then_recovers(vlm, monkeypatch):
    monkeypatch.setenv("GAP_VLM_FAILOVER_CONSECUTIVE", "1")
    monkeypatch.setenv("GAP_VLM_FAILOVER_COOLDOWN_S", "30")
    state = {"primary_down": True}
    clock = {"now": 1000.0}
    monkeypatch.setattr(vlm.time, "monotonic", lambda: clock["now"])
    calls = _two_routes(
        vlm, monkeypatch,
        lambda request: (httpx.Response(503) if state["primary_down"]
                         else httpx.Response(200, json=_THINKING_BODY)),
        lambda request: httpx.Response(200, json=_THINKING_BODY),
    )
    assert vlm.query(prompt="q")["route"]["name"] == "backup"
    state["primary_down"] = False
    # Parked primary is skipped during its cooldown.
    assert vlm.query(prompt="q")["route"]["name"] == "backup"
    assert calls == {"primary": 3, "backup": 2}
    clock["now"] += 31
    assert vlm.query(prompt="q")["route"]["name"] == "primary"
    assert calls == {"primary": 4, "backup": 2}


def test_gemini_rest_both_routes_down_keeps_the_outage_wrapper(vlm, monkeypatch):
    _two_routes(
        vlm, monkeypatch,
        lambda request: httpx.Response(503),
        lambda request: httpx.Response(503),
    )
    with pytest.raises(ToolError) as caught:
        vlm.query(prompt="q")
    message = str(caught.value)
    assert "gemini_rest backend unavailable after 3 attempts: generate_url=" in message
    assert "503" in message
    assert "secret" not in message


def test_gemini_rest_without_backup_is_unchanged(vlm, monkeypatch):
    monkeypatch.setattr(vlm, "_BACKOFF_S", 0.0)
    monkeypatch.setenv("GAP_VLM_PROVIDER", "gemini_rest")
    monkeypatch.setenv("GAP_VLM_BASE_URL", "https://primary.example")
    monkeypatch.setenv("GAP_VLM_API_KEY", "primary-secret")
    monkeypatch.setenv("GAP_VLM_BACKUP_BASE_URL", "https://backup.example")  # no key
    attempts = []

    def handler(request):
        attempts.append(request.url.host)
        return httpx.Response(503)

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(vlm, "_http_client", lambda: httpx.Client(transport=transport))
    with pytest.raises(ToolError, match="unavailable after 3 attempts"):
        vlm.query(prompt="q")
    assert attempts == ["primary.example"] * 3


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

    out = vlm.query(prompt="hi", image=image)
    assert out == {"text": "gemini says hi"}
    assert captured["client_kwargs"] == {
        "vertexai": True, "project": "test-project", "location": "global",
    }
    assert captured["model"] == "gemini-3-flash-preview"
    assert captured["contents"][0] == "hi"
    # Deterministic decoding — parity with the dev servicer's production
    # path (temperature 0.0, max_tokens 1024).
    assert captured["config"].temperature == 0.0
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
    assert out == {"text": "third time lucky"}
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


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------


def test_provider_kwarg_overrides_env(vlm, monkeypatch):
    # Env says vertex (and is unconfigured, so it would fail) — the kwarg wins.
    monkeypatch.setenv("GAP_VLM_PROVIDER", "vertex")
    captured = _mock_openrouter(vlm, monkeypatch, reply="a red mug")

    out = vlm.query(prompt="q", provider="openrouter")
    assert out == {"text": "a red mug"}
    assert captured["payload"]["messages"][0]["content"][0]["text"] == "q"


def test_unknown_provider_raises_tool_error(vlm, monkeypatch):
    monkeypatch.setenv("GAP_VLM_PROVIDER", "bedrock")
    with pytest.raises(ToolError, match="unknown provider 'bedrock'"):
        vlm.query(prompt="q")


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
    assert out == {"answer": expected, "text": text}


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


# ---------------------------------------------------------------------------
# Yes/no verdict parser v2 (GAP_VLM_YES_NO_PARSER=2)
# ---------------------------------------------------------------------------

_VERDICT_FIXTURE = json.loads(
    (ROOT / "tests" / "fixtures" / "vlm_yes_no_verdict_replies.json").read_text()
)["cases"]


def _mock_openrouter_replies(vlm, monkeypatch, replies: list[str]):
    """Serve ``replies`` in order; record every request payload."""
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        reply = replies[len(payloads) - 1]
        return httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(vlm, "_http_client", lambda: httpx.Client(transport=transport))
    return payloads


@pytest.mark.parametrize(
    "case", _VERDICT_FIXTURE, ids=[f"{i}" for i in range(len(_VERDICT_FIXTURE))])
def test_recorded_replies_legacy_coercion_is_unchanged(vlm, case):
    """The recorded answers were produced by the legacy first-word rule."""
    assert vlm._coerce_yes_no(case["text"]) is case["legacy_answer"]


@pytest.mark.parametrize(
    "case", _VERDICT_FIXTURE, ids=[f"{i}" for i in range(len(_VERDICT_FIXTURE))])
def test_recorded_replies_v2_verdict(vlm, case):
    assert vlm._yes_no_verdict(case["text"]) is case["expected"]


def test_corpus_s19_reply_is_misread_by_legacy_and_read_by_v2(vlm, monkeypatch):
    """memory-corpus-v1 temp_y0_2/3 seed 19: 'no doubt' precedes the final YES."""
    (case,) = [c for c in _VERDICT_FIXTURE if "s19 parent" in c["note"]]
    _mock_openrouter_replies(vlm, monkeypatch, [case["text"], case["text"]])
    assert vlm.query_yes_no(prompt="q")["answer"] is False  # legacy default
    monkeypatch.setenv("GAP_VLM_YES_NO_PARSER", "2")
    payloads = _mock_openrouter_replies(vlm, monkeypatch, [case["text"]])
    assert vlm.query_yes_no(prompt="q") == {"answer": True, "text": case["text"]}
    assert len(payloads) == 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Yes", True),
        ("no.", False),
        ("YES — clearly visible.", True),
        ("NO The main object is a mug.", False),
        ("**YES**\nIt is a basket.", True),
        ("The answer is yes", True),
        ("Final answer: **NO**. It is a book.", False),
        ("Therefore, the answer is: NO. It is a book.", False),
        ("Reasoning first.\n\nYES\nIt matches.", True),
        ("No need to overthink it.\n\nYES. It matches.", True),
        # Not a verdict: prose, echoes, alternatives and contradictions.
        ("No doubt it is a basket.", None),
        ("No-brainer, it is a basket.", None),
        ("The answer is no doubt YES.", None),
        ("The answer is no longer in doubt: YES", None),
        ("The answer is YES The label says so.", True),
        ("Yes - it is a basket.", True),
        ("Yes the object matches.", None),
        ("Eyes on the table", None),
        ("Absolutely not", None),
        ("", None),
        ("It appears to be a match.", None),
        ('Answer with the single word YES or NO first.', None),
        ('"YES" or "NO" first, then a sentence.', None),
        ("YES. It is pudding.\n\nNO\nIt is a gripper.", None),
        ("UNRESOLVED. The base is out of frame.", None),
    ],
)
def test_yes_no_verdict_rules(vlm, text, expected):
    assert vlm._yes_no_verdict(text) is expected


def test_v2_ambiguous_reply_is_reasked_once_with_the_same_prompt(vlm, monkeypatch):
    monkeypatch.setenv("GAP_VLM_YES_NO_PARSER", "2")
    payloads = _mock_openrouter_replies(
        vlm, monkeypatch, ["It appears to be a match.", "YES. It matches."])
    assert vlm.query_yes_no(prompt="Is it a basket?") == {
        "answer": True, "text": "YES. It matches."}
    assert len(payloads) == 2
    assert payloads[0] == payloads[1]
    assert "YES or NO first" in payloads[1]["messages"][0]["content"][0]["text"]


def test_v2_still_ambiguous_after_retry_fails_the_call(vlm, monkeypatch):
    monkeypatch.setenv("GAP_VLM_YES_NO_PARSER", "2")
    payloads = _mock_openrouter_replies(
        vlm, monkeypatch, ["UNRESOLVED. Out of frame.", "YES. Pudding.\n\nNO\nGripper."])
    with pytest.raises(ToolError, match="no unambiguous YES/NO verdict"):
        vlm.query_yes_no(prompt="Is it lifted?")
    assert len(payloads) == 2


def test_v2_retry_failure_does_not_leak_a_route_record(vlm, monkeypatch):
    monkeypatch.setenv("GAP_VLM_YES_NO_PARSER", "2")
    monkeypatch.setenv("GAP_VLM_PROVIDER", "gemini_rest")
    monkeypatch.setenv("GAP_VLM_BASE_URL", "https://relay.invalid")
    monkeypatch.setenv("GAP_VLM_API_KEY", "fixture-key")
    replies = iter(["maybe", "unclear", "YES."])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": next(replies)}]},
                            "finishReason": "STOP"}]})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(vlm, "_http_client", lambda: httpx.Client(transport=transport))
    with pytest.raises(ToolError):
        vlm.query_yes_no(prompt="q")
    assert vlm._take_route() is None
    out = vlm.query_yes_no(prompt="q")
    assert out["answer"] is True and out["route"]["name"] == "primary"


@pytest.mark.parametrize("value", ["", "1"])
def test_parser_selector_unset_or_1_is_legacy(vlm, monkeypatch, value):
    monkeypatch.setenv("GAP_VLM_YES_NO_PARSER", value)
    payloads = _mock_openrouter_replies(vlm, monkeypatch, ["maybe"])
    assert vlm.query_yes_no(prompt="q") == {"answer": False, "text": "maybe"}
    assert len(payloads) == 1


@pytest.mark.parametrize("value", ["3", "v2", "0", "2.0"])
def test_parser_selector_rejects_unknown_versions(vlm, monkeypatch, value):
    monkeypatch.setenv("GAP_VLM_YES_NO_PARSER", value)
    payloads = _mock_openrouter_replies(vlm, monkeypatch, ["YES"])
    with pytest.raises(ToolError, match="GAP_VLM_YES_NO_PARSER"):
        vlm.query_yes_no(prompt="q")
    assert payloads == []


def test_gemini_rest_usage_omits_absent_prompt_token_details(vlm):
    body = {"usageMetadata": {
        "promptTokenCount": 5, "candidatesTokenCount": 2,
        "promptTokensDetails": [{"modality": "TEXT"}, "bad", {"tokenCount": 3}],
    }}
    assert vlm._gemini_usage(body) == {
        "input_tokens": 5, "output_tokens": 2, "thoughts_tokens": 0,
    }


def _perceive_script(skill, monkeypatch):
    import importlib.util
    import sys

    name = f"perceive_dino_vlm_under_test_{skill.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "skills" / skill / "scripts" / "perceive_dino_vlm.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("skill", ["perceiving-objects", "perceiving-next-item"])
def test_perception_cache_key_binds_only_a_non_legacy_parser(skill, monkeypatch):
    """A v2 verify decision must not be served from a legacy-parser cache entry."""
    module = _perceive_script(skill, monkeypatch)
    legacy = module._make_cache_key([], {})
    monkeypatch.setenv("GAP_VLM_YES_NO_PARSER", "1")
    assert module._make_cache_key([], {}) == legacy
    monkeypatch.setenv("GAP_VLM_YES_NO_PARSER", "2")
    assert module._make_cache_key([], {}) != legacy


class _VerifyCtx:
    """Minimal ctx: query_yes_no raises ``error``; vlm.query answers YES."""

    def __init__(self, error: Exception):
        self.error = error
        self.calls: list[str] = []

    def tool(self, name, **kwargs):
        self.calls.append(name)
        if name == "vlm.query_yes_no":
            raise self.error
        return {"text": "YES"}


_V2_REFUSALS = [
    ToolError("vlm", "query_yes_no: no unambiguous YES/NO verdict after 2 replies; x"),
    # The out-of-process bundle surfaces remote errors by message.
    RuntimeError("vlm.query_yes_no: ToolError: GAP_VLM_YES_NO_PARSER must be 1 or 2"),
]


@pytest.mark.parametrize("skill", ["perceiving-objects", "perceiving-next-item"])
@pytest.mark.parametrize("error", _V2_REFUSALS, ids=["ambiguous", "selector"])
def test_verify_pick_never_answers_a_v2_refusal_from_the_query_fallback(
    skill, error, monkeypatch,
):
    module = _perceive_script(skill, monkeypatch)
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    box = {"x1": 10, "y1": 10, "x2": 40, "y2": 40}
    ctx = _VerifyCtx(error)
    with pytest.raises(type(error), match="YES/NO verdict|GAP_VLM_YES_NO_PARSER"):
        module._verify_pick(ctx, rgb, box, "basket", "", True)
    assert ctx.calls == ["vlm.query_yes_no"]


@pytest.mark.parametrize("skill", ["perceiving-objects", "perceiving-next-item"])
def test_verify_pick_keeps_the_query_fallback_for_other_failures(skill, monkeypatch):
    module = _perceive_script(skill, monkeypatch)
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    box = {"x1": 10, "y1": 10, "x2": 40, "y2": 40}
    ctx = _VerifyCtx(ToolError("vlm", "relay unavailable"))
    assert module._verify_pick(ctx, rgb, box, "basket", "", False) is True
    assert ctx.calls == ["vlm.query_yes_no", "vlm.query"]
