"""VLM tool bundle — hosted vision-language Q&A behind a provider switch.

In-process ``@tool`` functions: ``Query`` / ``QueryYesNo`` semantics with a
free-form prompt and one or more images (``images=`` carries several
context frames in one request). The tool signatures deliberately expose no
system prompt and no temperature knob — prompts are self-contained and
sampling is pinned for determinism.

Providers — selected by ``GAP_VLM_PROVIDER`` (default ``"openrouter"``); a
per-call ``provider=`` kwarg overrides the env. Every ``GAP_VLM_*`` knob
inherits from the matching ``GAP_LLM_*`` / google-SDK env var when unset
(see :func:`_resolve_provider`, :func:`_resolve_model`,
:func:`_resolve_vertex_project`, :func:`_resolve_vertex_region`) — so a
user who configures the agent's LLM doesn't have to re-configure the VLM
bundle separately. Set ``GAP_VLM_*`` explicitly only to route the VLM to
a different provider/model than the agent.

- ``openrouter`` (default) — OpenRouter's OpenAI-compatible
  chat-completions API (data-URL image blocks, ``temperature: 0.0``, 3
  retries with exponential backoff). Base URL defaults to
  ``https://openrouter.ai/api/v1`` (override with ``GAP_VLM_BASE_URL`` for
  any other OpenAI-compatible server, e.g. a local vLLM). Key from
  ``GAP_VLM_API_KEY`` (else ``OPENROUTER_API_KEY``); model from
  ``GAP_VLM_MODEL`` (else ``GAP_LLM_MODEL`` else :data:`DEFAULT_MODEL`).
- ``gemini_rest`` — Google's native ``generateContent`` REST schema through
  a compatible relay. ``GAP_VLM_BASE_URL`` is required and is extended with
  ``/v1beta/models/<model>:generateContent``. The API key comes from
  ``GAP_VLM_API_KEY`` and is sent only as ``x-goog-api-key``. Images use
  native inline PNG parts rather than OpenAI data-URL blocks. An optional
  backup route (``GAP_VLM_BACKUP_BASE_URL`` + ``GAP_VLM_BACKUP_API_KEY``,
  model ``GAP_VLM_BACKUP_MODEL`` else the primary model) serves a call only
  after the primary exhausted its retries on a provider outage (transport
  fault, HTTP 5xx, HTTP 429); content/schema failures never fail over. A
  route that fails ``GAP_VLM_FAILOVER_CONSECUTIVE`` calls in a row is
  parked for ``GAP_VLM_FAILOVER_COOLDOWN_S`` seconds. Each gemini_rest
  result carries a key-free ``route`` record (name, endpoint, model, usage
  incl. thought tokens and per-modality prompt tokens) so the trace shows
  which route answered.
- ``vertex`` — Vertex AI via ``google-genai`` (Gemini models). Lazy
  import; install the vertex extra
  (``pip install "graph-as-policy[vertex]"``). Config:
  ``GAP_VLM_MODEL`` (else ``GAP_LLM_MODEL`` else :data:`DEFAULT_MODEL`) +
  ``GAP_VLM_PROJECT_ID`` (else ``GOOGLE_CLOUD_PROJECT``) +
  ``GAP_VLM_REGION`` (else ``GOOGLE_CLOUD_REGION`` else
  ``GOOGLE_CLOUD_LOCATION`` else ``"global"``).

Generation config: perception callers (the pairwise tournament, the
yes/no verify gate) are binary judgments that depend on deterministic
decoding, so all providers pin ``temperature: 0.0`` with 3 retries +
exponential backoff. ``openrouter`` and ``vertex`` keep ``max_tokens:
1024``. ``gemini_rest`` sends ``maxOutputTokens``
:data:`_GEMINI_REST_MAX_OUTPUT_TOKENS`: Gemini thinking models count
thought tokens against that limit, and the former 1024 cap let thinking
exhaust it and truncate the final answer. The answer is assembled only
from response parts that are not marked ``"thought": true``.

Per-call-kind model: ``GAP_VLM_MODEL_QUERY``, ``GAP_VLM_MODEL_QUERY_BATCH``
and ``GAP_VLM_MODEL_QUERY_YES_NO`` select the model for ``vlm.query``,
``vlm.query_batch`` (the perceiving-objects pairwise tournament) and
``vlm.query_yes_no`` (its verify_pick gate). Resolution order: per-call
``model=`` > per-kind env > ``GAP_VLM_MODEL`` > ``GAP_LLM_MODEL`` >
:data:`DEFAULT_MODEL`. Unset per-kind variables leave behavior unchanged.

Yes/no parser version: ``GAP_VLM_YES_NO_PARSER`` selects how
``vlm.query_yes_no`` reads the reply. Unset or ``1`` keeps the legacy first
standalone yes/no word (byte-identical behavior). ``2`` accepts only an
explicit verdict — a line that starts with YES/NO or a marked
``answer/verdict/decision/conclusion: yes|no`` — and requires every such
verdict in the reply to agree; an ambiguous reply is re-asked once with the
same prompt and, if still ambiguous, the call fails with ``ToolError``.

All functions are synchronous — the gap runtime is threaded, not async.
"""

from __future__ import annotations

import atexit
import base64
import concurrent.futures
from contextlib import nullcontext
import io
import logging
import os
import re
import threading
import time
from typing import TypedDict

import httpx
import numpy as np
from gap_core.errors import ToolError
from gap_core.tools import tool
from PIL import Image

logger = logging.getLogger(__name__)

#: Default model when none is resolved (used by both providers). Override
#: per-call with ``model=`` or globally with ``GAP_VLM_MODEL`` /
#: ``GAP_LLM_MODEL``. On ``openrouter`` the slug may need a ``google/``
#: prefix depending on the account.
DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"

#: Provider used when neither ``provider=`` nor ``GAP_VLM_PROVIDER`` nor
#: ``GAP_LLM_PROVIDER`` is set.
DEFAULT_PROVIDER = "openrouter"


def _envstr(name: str) -> str:
    """``os.environ.get(name, "").strip()`` — empty string if unset/blank."""
    return os.environ.get(name, "").strip()


def _resolve_provider(provider: str | None) -> str:
    """Per-call override > ``GAP_VLM_PROVIDER`` > ``GAP_LLM_PROVIDER`` >
    :data:`DEFAULT_PROVIDER`. The ``GAP_LLM_*`` inheritance lets a user
    who's already configured the agent's LLM run the VLM bundle through
    the same provider without re-exporting a parallel set of env vars
    (the silent ``GAP_VLM_*`` defaults caused the dev-era milk-vs-soup
    mispick: missing creds → tournament fell back to "box 0 wins")."""
    return (
        (provider or "").strip().lower()
        or _envstr("GAP_VLM_PROVIDER").lower()
        or _envstr("GAP_LLM_PROVIDER").lower()
        or DEFAULT_PROVIDER
    )


def _resolve_model(model: str | None) -> str:
    """Per-call override > ``GAP_VLM_MODEL`` > ``GAP_LLM_MODEL`` >
    :data:`DEFAULT_MODEL`."""
    return (
        (model or "").strip()
        or _envstr("GAP_VLM_MODEL")
        or _envstr("GAP_LLM_MODEL")
        or DEFAULT_MODEL
    )


def _call_kind_model(model: str | None, kind: str) -> str | None:
    """Per-call ``model=`` > the call kind's ``GAP_VLM_MODEL_<KIND>``; ``None``
    defers to :func:`_resolve_model`'s global chain."""
    return (model or "").strip() or _envstr(_CALL_KIND_MODEL_ENV[kind]) or None


def _resolve_vertex_project() -> str:
    """``GAP_VLM_PROJECT_ID`` > ``GOOGLE_CLOUD_PROJECT`` (the documented
    google-genai knob). Empty string when unset — the caller raises with
    the install hint."""
    return (
        _envstr("GAP_VLM_PROJECT_ID")
        or _envstr("GOOGLE_CLOUD_PROJECT")
    )


def _resolve_vertex_region() -> str:
    """``GAP_VLM_REGION`` > ``GOOGLE_CLOUD_REGION`` > ``GOOGLE_CLOUD_LOCATION``
    > ``"global"`` (the documented Vertex default)."""
    return (
        _envstr("GAP_VLM_REGION")
        or _envstr("GOOGLE_CLOUD_REGION")
        or _envstr("GOOGLE_CLOUD_LOCATION")
        or "global"
    )

_MAX_TOKENS = 1024  # ported from the source servicer
#: Output limit for ``gemini_rest``; it includes thinking tokens. Measured on
#: real perception pairs: gemini-3.8-flash thinks at most ~5k tokens even when
#: uncapped (1024 truncated 7.5% of its answers), while
#: gemini-robotics-er-2-preview ignores ``thinkingBudget`` and on degenerate
#: pairs thinks up to ~96% of this limit before answering (~31k tokens and
#: ~105 s per call at 32768). 8192 leaves flash headroom and bounds ER at ~30 s.
_GEMINI_REST_MAX_OUTPUT_TOKENS = 8192
#: Tool name -> per-call-kind model selector (see module docstring).
_CALL_KIND_MODEL_ENV = {
    "query": "GAP_VLM_MODEL_QUERY",
    "query_batch": "GAP_VLM_MODEL_QUERY_BATCH",
    "query_yes_no": "GAP_VLM_MODEL_QUERY_YES_NO",
}
_MAX_RETRIES = 3
_BACKOFF_S = 1.0
#: Deterministic decoding for the binary perception judgments (tournament
#: A/B picks, yes/no verify). The dev servicer's proxy path always sent
#: ``"temperature": 0.0``; this port applies it to every provider.
_TEMPERATURE = 0.0


class QueryResult(TypedDict):
    text: str


class QueryBatchResult(TypedDict):
    results: list[QueryResult]


class YesNoResult(TypedDict):
    answer: bool
    text: str


# ---------------------------------------------------------------------------
# Image helpers (numpy-first: gap images are uint8 [H, W, 3], no byte packing)
# ---------------------------------------------------------------------------


def _validate_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"vlm expects a uint8 [H, W, 3] RGB array, got dtype={arr.dtype} "
            f"shape={arr.shape}"
        )
    return arr


def _png_b64(image: np.ndarray) -> str:
    """Encode a uint8 [H, W, 3] RGB array as a raw base64 PNG string."""
    arr = _validate_image(image)
    pil_image = Image.fromarray(arr, "RGB")
    with io.BytesIO() as buf:
        pil_image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def _gather_images(
    image: np.ndarray | None, images: list | None
) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    if image is not None:
        arrays.append(image)
    if images:
        arrays.extend(images)
    return [_validate_image(a) for a in arrays]


# ---------------------------------------------------------------------------
# Provider: openrouter (OpenRouter's OpenAI-compatible chat-completions API)
# ---------------------------------------------------------------------------


_shared_http_client: httpx.Client | None = None
_http_client_lock = threading.Lock()


def _new_http_client() -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0))


def _http_client() -> nullcontext[httpx.Client]:
    """Borrow the thread-safe process pool; request contexts never close it.

    Proxy settings are fixed at process startup. Credentials stay on each
    request, not on the shared client. HTTPX replaces expired/broken sockets.
    """
    global _shared_http_client
    with _http_client_lock:
        if _shared_http_client is None:
            _shared_http_client = _new_http_client()
        return nullcontext(_shared_http_client)


def _close_http_client() -> None:
    """Close at process shutdown, after query workers have settled."""
    global _shared_http_client
    with _http_client_lock:
        if _shared_http_client is not None:
            _shared_http_client.close()
            _shared_http_client = None


def _reset_http_client_after_fork() -> None:
    # Never reuse a parent's sockets or possibly held thread lock in a child.
    global _shared_http_client, _http_client_lock
    _shared_http_client = None
    _http_client_lock = threading.Lock()


atexit.register(_close_http_client)
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_http_client_after_fork)


def _query_openrouter(prompt: str, images: list[np.ndarray], model: str | None) -> str:
    base_url = _envstr("GAP_VLM_BASE_URL") or "https://openrouter.ai/api/v1"
    model = _resolve_model(model)
    api_key = _envstr("GAP_VLM_API_KEY") or _envstr("OPENROUTER_API_KEY")
    chat_url = f"{base_url.rstrip('/')}/chat/completions"

    content: list[dict] = [{"type": "text", "text": prompt}]
    for arr in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{_png_b64(arr)}"},
        })
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": _MAX_TOKENS,
        "temperature": 0.0,
    }
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    last_exc: Exception | None = None
    with _http_client() as client:
        for attempt in range(_MAX_RETRIES):
            try:
                resp = client.post(chat_url, json=payload, headers=headers)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "VLM request failed (attempt %d/%d, chat_url=%s): %s",
                    attempt + 1, _MAX_RETRIES, chat_url, exc,
                )
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_BACKOFF_S * (2 ** attempt))

    raise ToolError(
        "vlm",
        f"backend unavailable after {_MAX_RETRIES} attempts: "
        f"chat_url={chat_url}, error={last_exc}",
    )


# ---------------------------------------------------------------------------
# Provider: gemini_rest (Google generateContent schema through an HTTP relay)
# ---------------------------------------------------------------------------


def _gemini_answer_text(body: dict) -> str:
    """Join the first candidate's answer text parts, excluding thought parts.

    Thinking models may return their reasoning summary as parts marked
    ``"thought": true``; those are not the answer and must not reach the
    caller's A/B or yes/no parser.
    """
    candidate = body["candidates"][0]
    response_parts = (candidate.get("content") or {}).get("parts") or []
    return "".join(
        part["text"] for part in response_parts
        if isinstance(part, dict)
        and isinstance(part.get("text"), str)
        and part.get("thought") is not True
    )


def _log_gemini_usage(model: str, body: dict, latency_s: float) -> None:
    """Log non-secret per-request usage, including thought tokens."""
    candidate = (body.get("candidates") or [{}])[0]
    usage = body.get("usageMetadata") or {}
    finish = candidate.get("finishReason")
    parts = (candidate.get("content") or {}).get("parts") or []
    thought_parts = sum(
        1 for part in parts if isinstance(part, dict) and part.get("thought") is True
    )
    logger.log(
        logging.WARNING if finish == "MAX_TOKENS" else logging.INFO,
        "VLM gemini_rest usage model=%s finish=%s prompt_tokens=%s "
        "candidates_tokens=%s thoughts_tokens=%s thought_parts=%d latency_s=%.3f",
        model, finish, usage.get("promptTokenCount"),
        usage.get("candidatesTokenCount"), usage.get("thoughtsTokenCount"),
        thought_parts, latency_s,
    )


class _ProviderOutage(Exception):
    """A route exhausted its retries on a provider (not content) failure."""


#: Per-thread record of the route that served the latest gemini_rest call.
_route_state = threading.local()
#: Process-local route health: consecutive outage calls and parked-until time.
_route_health: dict[str, dict[str, float]] = {}
_route_health_lock = threading.Lock()


def _is_provider_outage(exc: Exception) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return False


def _failover_policy() -> tuple[int, float]:
    try:
        consecutive = int(_envstr("GAP_VLM_FAILOVER_CONSECUTIVE") or 1)
        cooldown = float(_envstr("GAP_VLM_FAILOVER_COOLDOWN_S") or 60.0)
    except ValueError as exc:
        raise ToolError("vlm", "invalid gemini_rest failover policy") from exc
    if consecutive < 1 or not cooldown > 0:
        raise ToolError("vlm", "invalid gemini_rest failover policy")
    return consecutive, cooldown


def _gemini_rest_routes(model: str | None) -> list[tuple[str, str, str, str]]:
    """``(name, base_url, api_key, model)`` in preference order."""
    base_url = _envstr("GAP_VLM_BASE_URL")
    api_key = _envstr("GAP_VLM_API_KEY")
    if not base_url:
        raise ToolError(
            "vlm",
            "gemini_rest requires GAP_VLM_BASE_URL",
        )
    if not api_key:
        raise ToolError(
            "vlm",
            "gemini_rest requires GAP_VLM_API_KEY",
        )
    resolved = _resolve_model(model)
    routes = [("primary", base_url, api_key, resolved)]
    backup_url = _envstr("GAP_VLM_BACKUP_BASE_URL")
    backup_key = _envstr("GAP_VLM_BACKUP_API_KEY")
    if backup_url and backup_key:
        # A per-call model override applies to every route.
        backup_model = (model or "").strip() or _envstr("GAP_VLM_BACKUP_MODEL") or resolved
        routes.append(("backup", backup_url, backup_key, backup_model))
    return routes


def _ordered_routes(routes, now: float):
    with _route_health_lock:
        ready = [r for r in routes if _route_health.get(r[0], {}).get("parked_until", 0.0) <= now]
    # All parked: still attempt every route rather than failing without a call.
    return ready or list(routes)


def _record_route_result(name: str, *, outage: bool, now: float) -> None:
    consecutive, cooldown = _failover_policy() if outage else (1, 0.0)
    with _route_health_lock:
        health = _route_health.setdefault(name, {"failures": 0.0, "parked_until": 0.0})
        if not outage:
            health["failures"] = 0.0
            health["parked_until"] = 0.0
            return
        health["failures"] += 1
        if health["failures"] >= consecutive:
            health["parked_until"] = now + cooldown
            logger.warning("VLM gemini_rest route %s parked for %.0fs", name, cooldown)


def _gemini_usage(body: dict) -> dict:
    usage = body.get("usageMetadata") if isinstance(body, dict) else None
    usage = usage if isinstance(usage, dict) else {}

    def count(key: str) -> int:
        value = usage.get(key)
        return value if isinstance(value, int) and value >= 0 else 0

    thoughts = count("thoughtsTokenCount")
    result: dict = {
        "input_tokens": count("promptTokenCount"),
        # Thinking tokens are billed output even though no text part carries them.
        "output_tokens": count("candidatesTokenCount") + thoughts,
        "thoughts_tokens": thoughts,
    }
    # Upstream prompt tokens per modality: relays differ in hidden prompt text
    # (e.g. an injected preamble), which only this breakdown makes visible.
    details = usage.get("promptTokensDetails")
    by_modality = {
        str(item["modality"]): item["tokenCount"]
        for item in (details if isinstance(details, list) else ())
        if isinstance(item, dict)
        and isinstance(item.get("modality"), str)
        and isinstance(item.get("tokenCount"), int)
        and item["tokenCount"] >= 0
    }
    if by_modality:
        result["prompt_tokens_details"] = dict(sorted(by_modality.items()))
    return result


def _take_route() -> dict | None:
    route = getattr(_route_state, "route", None)
    _route_state.route = None
    return route


def _with_route(result: dict) -> dict:
    route = _take_route()
    if route is not None:
        result["route"] = route
    return result


def _query_gemini_rest(
    prompt: str, images: list[np.ndarray], model: str | None
) -> str:
    routes = _gemini_rest_routes(model)
    _route_state.route = None
    last: _ProviderOutage | None = None
    for name, base_url, api_key, route_model in _ordered_routes(routes, time.monotonic()):
        try:
            text, usage = _query_gemini_rest_route(
                prompt, images, base_url=base_url, api_key=api_key, model=route_model,
            )
        except _ProviderOutage as outage:
            _record_route_result(name, outage=True, now=time.monotonic())
            last = outage
            continue
        _record_route_result(name, outage=False, now=time.monotonic())
        _route_state.route = {
            "name": name, "endpoint": base_url, "model": route_model, "usage": usage,
        }
        return text
    assert last is not None
    # Same public wrapper as before the backup route existed.
    raise ToolError("vlm", str(last))


def _query_gemini_rest_route(
    prompt: str, images: list[np.ndarray], *, base_url: str, api_key: str, model: str
) -> tuple[str, dict]:
    generate_url = (
        f"{base_url.rstrip('/')}/v1beta/models/{model}:generateContent"
    )
    parts: list[dict] = [{"text": prompt}]
    for arr in images:
        parts.append({
            "inline_data": {
                "mime_type": "image/png",
                "data": _png_b64(arr),
            }
        })
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": _TEMPERATURE,
            "maxOutputTokens": _GEMINI_REST_MAX_OUTPUT_TOKENS,
        },
    }
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    last_exc: Exception | None = None
    with _http_client() as client:
        for attempt in range(_MAX_RETRIES):
            try:
                started = time.monotonic()
                response = client.post(generate_url, json=payload, headers=headers)
                response.raise_for_status()
                body = response.json()
                text = _gemini_answer_text(body)
                _log_gemini_usage(model, body, time.monotonic() - started)
                if not text:
                    raise ValueError("Gemini response contained no candidate answer text")
                return text, _gemini_usage(body)
            except Exception as exc:  # noqa: BLE001 — transient API/schema errors
                last_exc = exc
                logger.warning(
                    "VLM gemini_rest request failed "
                    "(attempt %d/%d, generate_url=%s): %s",
                    attempt + 1, _MAX_RETRIES, generate_url, exc,
                )
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_BACKOFF_S * (2 ** attempt))

    message = (
        f"gemini_rest backend unavailable after {_MAX_RETRIES} attempts: "
        f"generate_url={generate_url}, error={last_exc}"
    )
    if last_exc is not None and _is_provider_outage(last_exc):
        raise _ProviderOutage(message)
    raise ToolError("vlm", message)


# ---------------------------------------------------------------------------
# Provider: vertex (google-genai; Gemini models only)
# ---------------------------------------------------------------------------


def _is_claude_model(model: str) -> bool:
    """Check if a model name refers to a Claude model."""
    return "claude" in model.lower()


def _query_vertex(prompt: str, images: list[np.ndarray], model: str | None) -> str:
    model = _resolve_model(model)
    if _is_claude_model(model):
        raise ToolError(
            "vlm",
            f"vertex serves Gemini models only (got {model!r}); "
            "Claude-on-Vertex was removed with the anthropic dependency. "
            "Use a gemini-* model, or route Claude via the openrouter provider.",
        )
    project_id = _resolve_vertex_project()
    region = _resolve_vertex_region()

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ToolError(
            "vlm",
            "google-genai is not installed in the vlm bundle's venv "
            "(needed to route gemini-* via Vertex). Re-sync the bundle: "
            "`uv sync --project open-robot-skills/tools/vlm` "
            "(google-genai is declared in tools/vlm/pyproject.toml).",
        ) from exc

    client = genai.Client(vertexai=True, project=project_id, location=region)
    parts: list = [prompt]
    for arr in images:
        parts.append(types.Part.from_bytes(
            data=base64.b64decode(_png_b64(arr)), mime_type="image/png",
        ))
    config = types.GenerateContentConfig(
        temperature=_TEMPERATURE, max_output_tokens=_MAX_TOKENS,
    )
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model, contents=parts, config=config,
            )
            return response.text or ""
        except Exception as exc:  # noqa: BLE001 — transient API errors
            last_exc = exc
            logger.warning(
                "VLM vertex request failed (attempt %d/%d, model=%s): %s",
                attempt + 1, _MAX_RETRIES, model, exc,
            )
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_BACKOFF_S * (2 ** attempt))
    raise ToolError(
        "vlm",
        f"vertex backend unavailable after {_MAX_RETRIES} attempts: "
        f"model={model}, error={last_exc}",
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


_PROVIDERS = {
    "gemini_rest": _query_gemini_rest,
    "openrouter": _query_openrouter,
    "vertex": _query_vertex,
}


def _query(
    prompt: str,
    image: np.ndarray | None,
    images: list | None,
    provider: str | None,
    model: str | None,
) -> str:
    name = _resolve_provider(provider)
    fn = _PROVIDERS.get(name)
    if fn is None:
        raise ToolError(
            "vlm",
            f"unknown provider {name!r} (valid: {sorted(_PROVIDERS)}); set "
            f"GAP_VLM_PROVIDER (or GAP_LLM_PROVIDER — VLM inherits from "
            f"LLM when unset) or pass provider=",
        )
    return fn(prompt, _gather_images(image, images), model)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool(
    name="vlm.query",
    summary="Free-form visual question answering via a hosted VLM.",
    tags=("perception",),
)
def query(
    prompt: str,
    image: np.ndarray | None = None,
    images: list | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> QueryResult:
    """Ask the configured VLM a free-form question, optionally about images.

    Args:
        prompt: Free-form question.
        image: Optional uint8 [H, W, 3] RGB context image.
        images: Optional additional context images (same dtype/shape).
        provider: Per-call provider override
            (``openrouter``/``gemini_rest``/``vertex``).
        model: Per-call model override.

    Returns:
        ``{"text": <model response>}``.
    """
    return _with_route({"text": _query(
        prompt, image, images, provider, _call_kind_model(model, "query"),
    )})


@tool(
    name="vlm.query_batch",
    summary="Run one independent round of visual questions concurrently.",
    tags=("perception", "guard_cost_input:prompts"),
)
def query_batch(
    prompts: list[str],
    images: list,
    provider: str | None = None,
    model: str | None = None,
) -> QueryBatchResult:
    """Run independent single-image queries concurrently, preserving order.

    Callers must wait for the full result before constructing a dependent
    next round. Any child failure fails the whole batch rather than fabricating
    a result. Concurrency is bounded at four hosted-model requests.
    """
    if len(prompts) != len(images):
        raise ValueError(
            "vlm.query_batch requires one image per prompt "
            f"(got {len(prompts)} prompts, {len(images)} images)"
        )
    if not prompts:
        return {"results": []}
    model = _call_kind_model(model, "query_batch")

    def _one(item: tuple[str, np.ndarray]) -> QueryResult:
        prompt, image = item
        return _with_route({"text": _query(prompt, image, None, provider, model)})

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(4, len(prompts))
    ) as pool:
        results = list(pool.map(_one, zip(prompts, images, strict=True)))
    return {"results": results}


#: Appended to every ``query_yes_no`` prompt so the reply is machine-checkable.
#: The dev servicer relied on temperature-0 replies leading with "Yes,"/"No,"
#: and coerced with a bare ``"yes" in text.lower()``; without an explicit
#: instruction, models sometimes answer affirmatively in prose that contains
#: no literal "yes" ("...it appears to be a match.") which the substring
#: check silently mislabels as False. In the perceiving-objects safe gate
#: such a false "No" rejects a correct exterior pick and forces a degraded
#: single-view wrist fallback — the G1 cream-cheese failure mode.
_YES_NO_INSTRUCTION = (
    " Answer with the single word YES or NO first, then one short "
    "sentence of justification."
)

_YES_NO_WORD = re.compile(r"\b(yes|no)\b")


def _coerce_yes_no(text: str) -> bool:
    """First standalone yes/no word wins; legacy substring check as fallback."""
    m = _YES_NO_WORD.search(text.lower())
    if m:
        return m.group(1) == "yes"
    return "yes" in text.lower()


#: ``GAP_VLM_YES_NO_PARSER`` versions: 1 = legacy first word, 2 = explicit verdict.
_YES_NO_PARSERS = {"": 1, "1": 1, "2": 2}

_VERDICT_MARKUP = re.compile(r"[*_`#>]+")
#: A verdict token: a bare/punctuated yes|no in any case ("No, ...", "yes."),
#: or an upper-case YES/NO followed by text ("NO The object is ..."); never
#: "No doubt ...", "No-brainer" or the option list "YES or NO" / "yes/no".
_VERDICT_TOKEN = (
    r"(?:(yes|no)(?=\s*$|\s*[.,!:;)\]\"'\u2013\u2014]|\s+-)|(YES|NO)(?=\s))"
    r"(?!\W{0,3}\s*(?:or|and|/)\s*\W{0,3}(?:yes|no)\b)"
)
#: A line that starts with the verdict.
_LINE_VERDICT = re.compile(r"^[\s\-\u2022\"'(\[]*" + _VERDICT_TOKEN, re.IGNORECASE)
#: An explicitly marked verdict anywhere in a line ("Final answer: NO").
_MARKED_VERDICT = re.compile(
    r"\b(?:answer|verdict|decision|conclusion)\b(?:\s*(?:is|:|=|-))+\s*"
    r"(?:a\s+|an\s+)?(?:definitive|clear|firm|simple|resounding)?\s*[\"']?"
    + _VERDICT_TOKEN,
    re.IGNORECASE,
)


def _yes_no_parser() -> int:
    """``GAP_VLM_YES_NO_PARSER``: unset/``1`` legacy, ``2`` explicit verdict."""
    raw = _envstr("GAP_VLM_YES_NO_PARSER")
    version = _YES_NO_PARSERS.get(raw)
    if version is None:
        raise ToolError("vlm", "GAP_VLM_YES_NO_PARSER must be 1 or 2")
    return version


def _yes_no_verdict(text: str) -> bool | None:
    """Explicit verdict of a yes/no reply, or ``None`` when there is none.

    Collects every line-leading YES/NO and every marked verdict (markdown
    emphasis ignored). Returns the verdict only when at least one exists and
    all agree; prose mentions ("no doubt"), option echoes ("YES or NO") and
    contradictory replies return ``None`` rather than a guess.
    """
    found: set[bool] = set()
    for line in _VERDICT_MARKUP.sub("", text).splitlines():
        matches = [_LINE_VERDICT.match(line), *_MARKED_VERDICT.finditer(line)]
        for m in matches:
            # Group 2 matched case-insensitively; only a real upper-case
            # YES/NO may be followed directly by prose.
            if m and (m.group(1) or m.group(2).isupper()):
                found.add((m.group(1) or m.group(2)).lower() == "yes")
    return found.pop() if len(found) == 1 else None


@tool(
    name="vlm.query_yes_no",
    summary="Yes/no visual question answering; coerces the model reply to a bool.",
    tags=("perception",),
)
def query_yes_no(
    prompt: str,
    image: np.ndarray | None = None,
    images: list | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> YesNoResult:
    """Ask the configured VLM a yes/no question, optionally about images.

    The prompt is suffixed with an explicit "answer YES or NO first"
    instruction (see :data:`_YES_NO_INSTRUCTION`) and ``answer`` is the
    first standalone ``yes``/``no`` word in the lowercased reply, falling
    back to the source servicer's verbatim ``"yes" in text.lower()``
    substring check when neither word appears.

    With ``GAP_VLM_YES_NO_PARSER=2`` the answer is the reply's explicit
    verdict (:func:`_yes_no_verdict`). An ambiguous reply is re-asked once
    with the identical request; if that reply is also ambiguous the call
    raises :class:`ToolError` instead of guessing. ``text`` (and ``route``)
    are those of the reply that supplied the answer.

    Returns:
        ``{"answer": <bool>, "text": <raw model response>}``.
    """
    parser = _yes_no_parser()
    if parser == 1:
        text = _query(
            prompt + _YES_NO_INSTRUCTION, image, images, provider,
            _call_kind_model(model, "query_yes_no"),
        )
        return _with_route({"answer": _coerce_yes_no(text), "text": text})

    model = _call_kind_model(model, "query_yes_no")
    for attempt in (1, 2):
        text = _query(prompt + _YES_NO_INSTRUCTION, image, images, provider, model)
        answer = _yes_no_verdict(text)
        if answer is not None:
            return _with_route({"answer": answer, "text": text})
        # The ambiguous reply did not decide anything; drop its route record.
        _take_route()
        logger.warning(
            "vlm.query_yes_no: no unambiguous YES/NO verdict (reply %d/2): %r",
            attempt, text[:200])
    raise ToolError(
        "vlm",
        "query_yes_no: no unambiguous YES/NO verdict after 2 replies; "
        f"last reply starts {text[:120]!r}",
    )
