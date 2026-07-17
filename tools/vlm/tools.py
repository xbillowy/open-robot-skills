"""VLM tool bundle — hosted vision-language Q&A behind a provider switch.

In-process ``@tool`` functions: ``Query`` / ``QueryYesNo`` semantics with a
free-form prompt and one or more images (``images=`` carries several
context frames in one request). Prompts are self-contained; callers may set
temperature and a provider sampling seed. Every result retains ``text`` and
adds a raw-wire ``evidence`` mapping for provenance adapters.

Providers — selected by ``GAP_VLM_PROVIDER`` (default ``"openrouter"``); a
per-call ``provider=`` kwarg overrides the env. Provider selection is isolated
from ``GAP_LLM_PROVIDER`` so agent configuration cannot activate the formal
VLM boundary. Ordinary providers still inherit compatible model/project
settings where documented below.

- ``openrouter`` (default) — OpenRouter's OpenAI-compatible
  chat-completions API (data-URL image blocks, ``temperature: 0.0``, 3
  retries with exponential backoff). Base URL defaults to
  ``https://openrouter.ai/api/v1`` (override with ``GAP_VLM_BASE_URL`` for
  any other OpenAI-compatible server, e.g. a local vLLM). Key from
  ``GAP_VLM_API_KEY`` (else ``OPENROUTER_API_KEY``); model from
  ``GAP_VLM_MODEL`` (else ``GAP_LLM_MODEL`` else :data:`DEFAULT_MODEL`).
- ``vertex`` — Vertex AI via ``google-genai`` (Gemini models). Lazy
  import; install the vertex extra
  (``pip install "graph-as-policy[vertex]"``). Config:
  ``GAP_VLM_MODEL`` (else ``GAP_LLM_MODEL`` else :data:`DEFAULT_MODEL`) +
  ``GAP_VLM_PROJECT_ID`` (else ``GOOGLE_CLOUD_PROJECT``) +
  ``GAP_VLM_REGION`` (else ``GOOGLE_CLOUD_REGION`` else
  ``GOOGLE_CLOUD_LOCATION`` else ``"global"``).
- ``openai_compatible`` — strict paper path. Requires explicit
  ``GAP_VLM_BASE_URL``, ``GAP_VLM_API_KEY``, and ``GAP_VLM_MODEL``; never
  falls back to OpenRouter, Vertex, or agent LLM configuration. It records
  canonical request/response hashes and all bounded transport attempts.

Generation config defaults to temperature 0.0 and max_tokens 1024. Set
``GAP_VLM_TEMPERATURE`` (the paper environment uses 0.1), or pass an explicit
``temperature=``. A seed is evidence, never a deterministic claim.

All functions are synchronous — the gap runtime is threaded, not async.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import math
import os
import re
import time
from typing import Literal, TypedDict
from urllib.parse import unquote

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

#: Provider used when neither ``provider=`` nor ``GAP_VLM_PROVIDER`` is set.
DEFAULT_PROVIDER = "openrouter"


def _envstr(name: str) -> str:
    """``os.environ.get(name, "").strip()`` — empty string if unset/blank."""
    return os.environ.get(name, "").strip()


def _resolve_provider(provider: str | None) -> str:
    """Per-call override > ``GAP_VLM_PROVIDER`` > product default.

    Provider selection intentionally does not inherit ``GAP_LLM_PROVIDER``:
    agent configuration must never activate the strict paper VLM boundary.
    """
    return (
        (provider or "").strip().lower() or _envstr("GAP_VLM_PROVIDER").lower() or DEFAULT_PROVIDER
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


def _resolve_vertex_project() -> str:
    """``GAP_VLM_PROJECT_ID`` > ``GOOGLE_CLOUD_PROJECT`` (the documented
    google-genai knob). Empty string when unset — the caller raises with
    the install hint."""
    return _envstr("GAP_VLM_PROJECT_ID") or _envstr("GOOGLE_CLOUD_PROJECT")


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
_MAX_RETRIES = 3
_BACKOFF_S = 1.0
#: Deterministic decoding for the binary perception judgments (tournament
#: A/B picks, yes/no verify). The dev servicer's proxy path always sent
#: ``"temperature": 0.0``; this port applies it to every provider.
_TEMPERATURE = 0.0


Digest = str


class ModelRandomnessEvidenceWire(TypedDict):
    requested_seed: int | None
    provider_reported_seed: int | None
    seed_control: Literal[
        "provider_confirmed", "requested_unconfirmed", "unsupported", "uncontrolled"
    ]
    deterministic_claim: Literal[False]


class ModelCallEvidence(TypedDict):
    provider: str
    base_url: str | None
    requested_model: str
    resolved_model: str | None
    temperature: float
    cache_policy: Literal["disabled"]
    randomness: ModelRandomnessEvidenceWire
    provider_request_id: str | None
    request_sha256: Digest
    response_sha256: Digest
    usage: dict[str, int] | None
    transport_attempts: list[dict[str, object]]
    fallback_used: bool


class _ProviderResult(TypedDict):
    text: str
    base_url: str | None
    resolved_model: str | None
    provider_request_id: str | None
    usage: dict[str, int] | None
    transport_attempts: list[dict[str, object]]


class QueryResult(TypedDict):
    text: str
    evidence: ModelCallEvidence


class YesNoResult(TypedDict):
    answer: bool
    text: str
    evidence: ModelCallEvidence


# ---------------------------------------------------------------------------
# Image helpers (numpy-first: gap images are uint8 [H, W, 3], no byte packing)
# ---------------------------------------------------------------------------


def _validate_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"vlm expects a uint8 [H, W, 3] RGB array, got dtype={arr.dtype} shape={arr.shape}"
        )
    return arr


def _png_b64(image: np.ndarray) -> str:
    """Encode a uint8 [H, W, 3] RGB array as a raw base64 PNG string."""
    arr = _validate_image(image)
    pil_image = Image.fromarray(arr, "RGB")
    with io.BytesIO() as buf:
        pil_image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def _gather_images(image: np.ndarray | None, images: list | None) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    if image is not None:
        arrays.append(image)
    if images:
        arrays.extend(images)
    return [_validate_image(a) for a in arrays]


def _canonical_sha256(value: object) -> Digest:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _resolve_temperature(temperature: float | None) -> float:
    raw: object = (
        temperature if temperature is not None else (_envstr("GAP_VLM_TEMPERATURE") or _TEMPERATURE)
    )
    try:
        resolved = float(raw)
    except (TypeError, ValueError) as exc:
        raise ToolError("vlm", "GAP_VLM_TEMPERATURE must be a finite nonnegative number") from exc
    if not math.isfinite(resolved) or resolved < 0:
        raise ToolError("vlm", "GAP_VLM_TEMPERATURE must be a finite nonnegative number")
    return resolved


def _safe_request_id(value: object, *, sensitive_values: tuple[str, ...]) -> str | None:
    if not isinstance(value, str) or not value or len(value) > 128:
        return None
    if any(
        secret and (value == secret or (len(secret) >= 8 and secret in value))
        for secret in sensitive_values
    ):
        return None
    if re.fullmatch(r"[A-Za-z0-9._:-]+", value) is None:
        return None
    return value


def _canonical_usage(value: object) -> dict[str, int] | None:
    if not isinstance(value, dict) or not value:
        return None
    usage: dict[str, int] = {}
    for key, count in value.items():
        if (
            not isinstance(key, str)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
        ):
            return None
        usage[key] = count
    return usage


def _openai_content(prompt: str, images: list[np.ndarray]) -> list[dict[str, object]]:
    content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
    for arr in images:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_png_b64(arr)}"},
            }
        )
    return content


def _randomness_evidence(
    *, requested_seed: int | None, reported_seed: object, seed_supported: bool
) -> ModelRandomnessEvidenceWire:
    if not seed_supported:
        return {
            "requested_seed": None,
            "provider_reported_seed": None,
            "seed_control": "unsupported",
            "deterministic_claim": False,
        }
    if requested_seed is None:
        return {
            "requested_seed": None,
            "provider_reported_seed": None,
            "seed_control": "uncontrolled",
            "deterministic_claim": False,
        }
    if reported_seed is None:
        return {
            "requested_seed": requested_seed,
            "provider_reported_seed": None,
            "seed_control": "requested_unconfirmed",
            "deterministic_claim": False,
        }
    if (
        isinstance(reported_seed, bool)
        or not isinstance(reported_seed, int)
        or reported_seed != requested_seed
    ):
        raise ToolError("vlm", "openai_compatible seed confirmation mismatch")
    return {
        "requested_seed": requested_seed,
        "provider_reported_seed": reported_seed,
        "seed_control": "provider_confirmed",
        "deterministic_claim": False,
    }


# ---------------------------------------------------------------------------
# Provider: openrouter (OpenRouter's OpenAI-compatible chat-completions API)
# ---------------------------------------------------------------------------


def _http_client() -> httpx.Client:
    """Build the HTTP client for the OpenAI-compatible path (test seam)."""
    return httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0))


def _canonical_public_base_url(value: str, *, sensitive_values: tuple[str, ...]) -> str:
    """Return the exact public request base URL or reject ambiguous/secret input."""

    try:
        url = httpx.URL(value)
        port = url.port
    except (TypeError, ValueError) as exc:
        raise ToolError("vlm", "provider requires a valid public HTTP(S) base URL") from exc
    if (
        url.scheme not in {"http", "https"}
        or not url.host
        or url.userinfo
        or url.query
        or url.fragment
    ):
        raise ToolError("vlm", "provider requires a valid public HTTP(S) base URL")
    raw_host = url.raw_host.decode("ascii")
    host = f"[{raw_host}]" if ":" in raw_host else raw_host
    default_port = 443 if url.scheme == "https" else 80
    authority = host if port in {None, default_port} else f"{host}:{port}"
    path = url.raw_path.rstrip(b"/").decode("ascii")
    canonical = f"{url.scheme}://{authority}{path}"
    decoded = unquote(canonical)
    if any(secret and (secret in canonical or secret in decoded) for secret in sensitive_values):
        raise ToolError("vlm", "provider requires a valid public HTTP(S) base URL")
    final = httpx.URL(canonical)
    if final.userinfo or final.query or final.fragment or str(final) != canonical:
        raise ToolError("vlm", "provider requires a valid public HTTP(S) base URL")
    return canonical


def _query_openai_compatible(
    prompt: str,
    images: list[np.ndarray],
    model: str | None,
    temperature: float | None,
    seed: int | None,
) -> QueryResult:
    base_url = _envstr("GAP_VLM_BASE_URL")
    api_key = _envstr("GAP_VLM_API_KEY")
    requested_model = (model or "").strip() or _envstr("GAP_VLM_MODEL")
    for env_name, value in (
        ("GAP_VLM_BASE_URL", base_url),
        ("GAP_VLM_API_KEY", api_key),
        ("GAP_VLM_MODEL", requested_model),
    ):
        if not value:
            raise ToolError("vlm", f"openai_compatible requires explicit {env_name}")
    base_url = _canonical_public_base_url(base_url, sensitive_values=(api_key,))
    resolved_temperature = _resolve_temperature(temperature)
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ToolError("vlm", "seed must be an integer or null")
    capability = _envstr("GAP_VLM_SEED_CAPABILITY").lower() or "supported"
    if capability not in {"supported", "unsupported"}:
        raise ToolError("vlm", "GAP_VLM_SEED_CAPABILITY must be supported or unsupported")
    seed_supported = capability == "supported"

    payload: dict[str, object] = {
        "model": requested_model,
        "messages": [{"role": "user", "content": _openai_content(prompt, images)}],
        "max_tokens": _MAX_TOKENS,
        "temperature": resolved_temperature,
    }
    sent_seed = seed if seed_supported else None
    if sent_seed is not None:
        payload["seed"] = sent_seed
    request_sha256 = _canonical_sha256(
        {"provider": "openai_compatible", "base_url": base_url, "payload": payload}
    )
    headers = {"Authorization": f"Bearer {api_key}"}
    chat_url = f"{base_url.rstrip('/')}/chat/completions"
    attempts: list[dict[str, object]] = []

    with _http_client() as client:
        for attempt_index in range(1, _MAX_RETRIES + 2):
            try:
                response = client.post(chat_url, json=payload, headers=headers)
            except httpx.TransportError:
                attempts.append(
                    {
                        "attempt": attempt_index,
                        "outcome": "transport_error",
                        "status": None,
                        "provider_request_id": None,
                    }
                )
                if attempt_index <= _MAX_RETRIES:
                    time.sleep(_BACKOFF_S * (2 ** (attempt_index - 1)))
                    continue
                raise ToolError(
                    "vlm", f"openai_compatible transport exhausted after {attempt_index} attempts"
                ) from None

            header_request_id = _safe_request_id(
                response.headers.get("x-request-id"), sensitive_values=(api_key, prompt)
            )
            if response.status_code == 429 or response.status_code >= 500:
                attempts.append(
                    {
                        "attempt": attempt_index,
                        "outcome": "http_error",
                        "status": response.status_code,
                        "provider_request_id": header_request_id,
                    }
                )
                if attempt_index <= _MAX_RETRIES:
                    time.sleep(_BACKOFF_S * (2 ** (attempt_index - 1)))
                    continue
                raise ToolError(
                    "vlm",
                    f"openai_compatible HTTP {response.status_code} after {attempt_index} attempts",
                )
            if response.status_code >= 400:
                raise ToolError("vlm", f"openai_compatible HTTP {response.status_code}")

            try:
                parsed = response.json()
                text = parsed["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise ToolError(
                    "vlm", "openai_compatible returned invalid response content"
                ) from exc
            if not isinstance(text, str) or not text:
                raise ToolError("vlm", "openai_compatible returned invalid response content")
            body_request_id = parsed.get("id") if isinstance(parsed, dict) else None
            provider_request_id = header_request_id or _safe_request_id(
                body_request_id, sensitive_values=(api_key, prompt)
            )
            attempts.append(
                {
                    "attempt": attempt_index,
                    "outcome": "success",
                    "status": response.status_code,
                    "provider_request_id": provider_request_id,
                }
            )
            resolved_model = parsed.get("model")
            if not isinstance(resolved_model, str) or not resolved_model:
                resolved_model = None
            usage = _canonical_usage(parsed.get("usage"))
            randomness = _randomness_evidence(
                requested_seed=sent_seed,
                reported_seed=parsed.get("seed"),
                seed_supported=seed_supported,
            )
            response_semantics = {
                "text": text,
                "resolved_model": resolved_model,
                "randomness": randomness,
                "usage": usage,
            }
            evidence: ModelCallEvidence = {
                "provider": "openai_compatible",
                "base_url": base_url,
                "requested_model": requested_model,
                "resolved_model": resolved_model,
                "temperature": resolved_temperature,
                "cache_policy": "disabled",
                "randomness": randomness,
                "provider_request_id": provider_request_id,
                "request_sha256": request_sha256,
                "response_sha256": _canonical_sha256(response_semantics),
                "usage": usage,
                "transport_attempts": attempts,
                "fallback_used": False,
            }
            return {"text": text, "evidence": evidence}

    raise AssertionError("unreachable")


def _query_openrouter(
    prompt: str,
    images: list[np.ndarray],
    model: str | None,
    temperature: float | None,
) -> _ProviderResult:
    base_url = _envstr("GAP_VLM_BASE_URL") or "https://openrouter.ai/api/v1"
    model = _resolve_model(model)
    api_key = _envstr("GAP_VLM_API_KEY") or _envstr("OPENROUTER_API_KEY")
    base_url = _canonical_public_base_url(base_url, sensitive_values=(api_key,))
    chat_url = f"{base_url.rstrip('/')}/chat/completions"

    content: list[dict] = [{"type": "text", "text": prompt}]
    for arr in images:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_png_b64(arr)}"},
            }
        )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": _MAX_TOKENS,
        "temperature": _resolve_temperature(temperature),
    }
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    attempts: list[dict[str, object]] = []
    with _http_client() as client:
        for attempt_index in range(1, _MAX_RETRIES + 1):
            try:
                resp = client.post(chat_url, json=payload, headers=headers)
            except Exception:  # noqa: BLE001 — preserve legacy retry behavior
                attempts.append(
                    {
                        "attempt": attempt_index,
                        "outcome": "transport_error",
                        "status": None,
                        "provider_request_id": None,
                    }
                )
                if attempt_index < _MAX_RETRIES:
                    time.sleep(_BACKOFF_S * (2 ** (attempt_index - 1)))
                    continue
                break

            request_id = _safe_request_id(
                resp.headers.get("x-request-id"), sensitive_values=(api_key, prompt)
            )
            if resp.status_code >= 400:
                attempts.append(
                    {
                        "attempt": attempt_index,
                        "outcome": "http_error",
                        "status": resp.status_code,
                        "provider_request_id": request_id,
                    }
                )
                if attempt_index < _MAX_RETRIES:
                    time.sleep(_BACKOFF_S * (2 ** (attempt_index - 1)))
                    continue
                break

            try:
                parsed = resp.json()
                text = parsed["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError, ValueError):
                parsed = None
                text = None
            if not isinstance(text, str):
                attempts.append(
                    {
                        "attempt": attempt_index,
                        "outcome": "transport_error",
                        "status": None,
                        "provider_request_id": request_id,
                    }
                )
                if attempt_index < _MAX_RETRIES:
                    time.sleep(_BACKOFF_S * (2 ** (attempt_index - 1)))
                    continue
                raise ToolError(
                    "vlm", "openrouter returned invalid response content after 3 attempts"
                ) from None
            request_id = request_id or _safe_request_id(
                parsed.get("id"), sensitive_values=(api_key, prompt)
            )
            attempts.append(
                {
                    "attempt": attempt_index,
                    "outcome": "success",
                    "status": resp.status_code,
                    "provider_request_id": request_id,
                }
            )
            resolved_model = parsed.get("model")
            if not isinstance(resolved_model, str) or not resolved_model:
                resolved_model = None
            return {
                "text": text,
                "base_url": base_url,
                "resolved_model": resolved_model,
                "provider_request_id": request_id,
                "usage": _canonical_usage(parsed.get("usage")),
                "transport_attempts": attempts,
            }

    raise ToolError(
        "vlm",
        f"openrouter backend unavailable after {_MAX_RETRIES} attempts",
    )


# ---------------------------------------------------------------------------
# Provider: vertex (google-genai; Gemini models only)
# ---------------------------------------------------------------------------


def _is_claude_model(model: str) -> bool:
    """Check if a model name refers to a Claude model."""
    return "claude" in model.lower()


def _vertex_usage(value: object) -> dict[str, int] | None:
    if value is None:
        return None
    raw = {
        "prompt_tokens": getattr(value, "prompt_token_count", None),
        "completion_tokens": getattr(value, "candidates_token_count", None),
        "total_tokens": getattr(value, "total_token_count", None),
    }
    if isinstance(value, dict):
        raw = {
            "prompt_tokens": value.get("prompt_token_count"),
            "completion_tokens": value.get("candidates_token_count"),
            "total_tokens": value.get("total_token_count"),
        }
    if any(count is None for count in raw.values()):
        return None
    return _canonical_usage(raw)


def _query_vertex(
    prompt: str,
    images: list[np.ndarray],
    model: str | None,
    temperature: float | None,
) -> _ProviderResult:
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
        parts.append(
            types.Part.from_bytes(
                data=base64.b64decode(_png_b64(arr)),
                mime_type="image/png",
            )
        )
    config = types.GenerateContentConfig(
        temperature=_resolve_temperature(temperature),
        max_output_tokens=_MAX_TOKENS,
    )
    attempts: list[dict[str, object]] = []
    for attempt_index in range(1, _MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=parts,
                config=config,
            )
            text = response.text or ""
            request_id = _safe_request_id(
                getattr(response, "response_id", None), sensitive_values=(prompt,)
            )
            attempts.append(
                {
                    "attempt": attempt_index,
                    "outcome": "success",
                    "status": None,
                    "provider_request_id": request_id,
                }
            )
            resolved_model = getattr(response, "model_version", None)
            if not isinstance(resolved_model, str) or not resolved_model:
                resolved_model = None
            return {
                "text": text,
                "base_url": None,
                "resolved_model": resolved_model,
                "provider_request_id": request_id,
                "usage": _vertex_usage(getattr(response, "usage_metadata", None)),
                "transport_attempts": attempts,
            }
        except Exception:  # noqa: BLE001 — transient API errors
            attempts.append(
                {
                    "attempt": attempt_index,
                    "outcome": "transport_error",
                    "status": None,
                    "provider_request_id": None,
                }
            )
            logger.warning(
                "VLM vertex request failed (attempt %d/%d, model=%s)",
                attempt_index,
                _MAX_RETRIES,
                model,
            )
            if attempt_index < _MAX_RETRIES:
                time.sleep(_BACKOFF_S * (2 ** (attempt_index - 1)))
    raise ToolError(
        "vlm",
        f"vertex backend unavailable after {_MAX_RETRIES} attempts: model={model}",
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


_PROVIDERS = {
    "openrouter": _query_openrouter,
    "vertex": _query_vertex,
}


def _legacy_result(
    *,
    provider_result: _ProviderResult,
    provider: str,
    prompt: str,
    images: list[np.ndarray],
    model: str | None,
    temperature: float | None,
    seed: int | None,
) -> QueryResult:
    requested_model = _resolve_model(model)
    resolved_temperature = _resolve_temperature(temperature)
    image_wire = [f"data:image/png;base64,{_png_b64(arr)}" for arr in images]
    randomness: ModelRandomnessEvidenceWire = {
        "requested_seed": None,
        "provider_reported_seed": None,
        "seed_control": "unsupported" if seed is not None else "uncontrolled",
        "deterministic_claim": False,
    }
    response_semantics = {
        "text": provider_result["text"],
        "resolved_model": provider_result["resolved_model"],
        "randomness": randomness,
        "usage": provider_result["usage"],
    }
    evidence: ModelCallEvidence = {
        "provider": provider,
        "base_url": provider_result["base_url"],
        "requested_model": requested_model,
        "resolved_model": provider_result["resolved_model"],
        "temperature": resolved_temperature,
        "cache_policy": "disabled",
        "randomness": randomness,
        "provider_request_id": provider_result["provider_request_id"],
        "request_sha256": _canonical_sha256(
            {
                "provider": provider,
                "base_url": provider_result["base_url"],
                "prompt": prompt,
                "images": image_wire,
                "model": requested_model,
                "temperature": resolved_temperature,
            }
        ),
        "response_sha256": _canonical_sha256(response_semantics),
        "usage": provider_result["usage"],
        "transport_attempts": provider_result["transport_attempts"],
        "fallback_used": False,
    }
    return {"text": provider_result["text"], "evidence": evidence}


def _query(
    prompt: str,
    image: np.ndarray | None,
    images: list | None,
    provider: str | None,
    model: str | None,
    temperature: float | None,
    seed: int | None,
) -> QueryResult:
    name = _resolve_provider(provider)
    gathered_images = _gather_images(image, images)
    if name == "openai_compatible":
        return _query_openai_compatible(prompt, gathered_images, model, temperature, seed)
    fn = _PROVIDERS.get(name)
    if fn is None:
        raise ToolError(
            "vlm",
            f"unknown provider {name!r} "
            f"(valid: {sorted([*_PROVIDERS, 'openai_compatible'])}); set "
            "GAP_VLM_PROVIDER or pass provider=",
        )
    provider_result = fn(prompt, gathered_images, model, temperature)
    return _legacy_result(
        provider_result=provider_result,
        provider=name,
        prompt=prompt,
        images=gathered_images,
        model=model,
        temperature=temperature,
        seed=seed,
    )


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
    temperature: float | None = None,
    seed: int | None = None,
) -> QueryResult:
    """Ask the configured VLM a free-form question, optionally about images.

    Args:
        prompt: Free-form question.
        image: Optional uint8 [H, W, 3] RGB context image.
        images: Optional additional context images (same dtype/shape).
        provider: Per-call provider override (``openrouter``/``vertex``).
        model: Per-call model override.
        temperature: Per-call sampling temperature override.
        seed: Optional provider sampling seed (strict provider only).

    Returns:
        ``{"text": <model response>, "evidence": <call evidence>}``.
    """
    return _query(prompt, image, images, provider, model, temperature, seed)


#: Appended to every ``query_yes_no`` prompt so the reply is machine-checkable.
#: The dev servicer relied on temperature-0 replies leading with "Yes,"/"No,"
#: and coerced with a bare ``"yes" in text.lower()``; without an explicit
#: instruction, models sometimes answer affirmatively in prose that contains
#: no literal "yes" ("...it appears to be a match.") which the substring
#: check silently mislabels as False. In the perceiving-objects safe gate
#: such a false "No" rejects a correct exterior pick and forces a degraded
#: single-view wrist fallback — the G1 cream-cheese failure mode.
_YES_NO_INSTRUCTION = (
    " Answer with the single word YES or NO first, then one short sentence of justification."
)

_YES_NO_WORD = re.compile(r"\b(yes|no)\b")


def _coerce_yes_no(text: str) -> bool:
    """First standalone yes/no word wins; legacy substring check as fallback."""
    m = _YES_NO_WORD.search(text.lower())
    if m:
        return m.group(1) == "yes"
    return "yes" in text.lower()


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
    temperature: float | None = None,
    seed: int | None = None,
) -> YesNoResult:
    """Ask the configured VLM a yes/no question, optionally about images.

    The prompt is suffixed with an explicit "answer YES or NO first"
    instruction (see :data:`_YES_NO_INSTRUCTION`) and ``answer`` is the
    first standalone ``yes``/``no`` word in the lowercased reply, falling
    back to the source servicer's verbatim ``"yes" in text.lower()``
    substring check when neither word appears.

    Returns:
        ``{"answer": <bool>, "text": <raw model response>}``.
    """
    result = _query(
        prompt + _YES_NO_INSTRUCTION,
        image,
        images,
        provider,
        model,
        temperature,
        seed,
    )
    return {
        "answer": _coerce_yes_no(result["text"]),
        "text": result["text"],
        "evidence": result["evidence"],
    }
