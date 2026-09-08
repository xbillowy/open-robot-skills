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
                "GAP_LLM_PROVIDER", "GAP_LLM_MODEL", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)


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
            assert vlm.query(prompt="hello", provider=provider) == {"text": "yes"}
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

    assert out == {"answer": True, "text": "YES\nInside the basket."}
    assert captured["url"] == (
        "https://relay.example/ai/genai/v1beta/models/"
        "gemini-3.6-flash:generateContent"
    )
    assert captured["api_key"] == "relay-secret"
    assert captured["authorization"] is None
    payload = captured["payload"]
    assert payload["generationConfig"] == {
        "temperature": 0.0,
        "maxOutputTokens": 1024,
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
