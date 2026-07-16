---
name: vlm
description: Free-form and yes/no visual question answering against a hosted
  vision-language model (OpenRouter by default, strict OpenAI-compatible and
  Vertex AI Gemini selectable by config). Use when a workflow needs scene descriptions,
  semantic checks ("is the drawer open?"), or LLM-judged verification of a
  camera frame — no GPU required.
license: MIT
compatibility: requires gap>=0.1
metadata: {category: perception, tags: [perception, vlm, api]}
gap:
  # Provider selection uses only GAP_VLM_PROVIDER (or the product default),
  # never GAP_LLM_PROVIDER. Ordinary providers may still inherit compatible
  # model/project settings. `gap check` reports the resolved configuration.
  requires: {env_any: [
    OPENROUTER_API_KEY, GAP_VLM_API_KEY, GAP_VLM_BASE_URL,
    GAP_VLM_PROJECT_ID, GOOGLE_CLOUD_PROJECT,
  ]}
  serving:
    command: ["python", "-m", "gap_core.rpc.server", "--bundle", "vlm"]
    protocol: stdio-msgpack
  tools:
    - vlm.query: Free-form visual question answering via a hosted VLM.
    - vlm.query_yes_no: Yes/no visual question answering; coerces the model reply to a bool.
---

# vlm

API-backed vision-language Q&A. Zero GPU: every provider is a remote
endpoint. Images are gap-native `uint8 [H, W, 3]` numpy arrays, PNG-encoded
on the wire.

## Providers

Selected by `GAP_VLM_PROVIDER` (default `openrouter`); each tool also accepts
a per-call `provider=` override.

| Provider     | Backend                                             | Config (env)                                              |
|--------------|-----------------------------------------------------|-----------------------------------------------------------|
| `openrouter` | OpenRouter's OpenAI-compatible chat-completions API | `OPENROUTER_API_KEY` (or `GAP_VLM_API_KEY`); `GAP_VLM_MODEL` (default `gemini-3.1-flash-lite-preview`, see `DEFAULT_MODEL` in `tools.py`); set `GAP_VLM_BASE_URL` to point at another OpenAI-compatible server (e.g. a local vLLM) |
| `openai_compatible` | Strict OpenAI chat-completions relay for formal runs | Explicit `GAP_VLM_BASE_URL`, `GAP_VLM_API_KEY`, and `GAP_VLM_MODEL`; optional `GAP_VLM_TEMPERATURE` (paper default injection: `0.1`) and `GAP_VLM_SEED_CAPABILITY=supported\|unsupported` |
| `vertex`     | Vertex AI via google-genai (Gemini models)          | `GAP_VLM_MODEL`, `GAP_VLM_PROJECT_ID`, `GAP_VLM_REGION`   |

The `vertex` provider lazy-imports google-genai — install the engine's vertex
extra first: `pip install "graph-as-policy[vertex]"`.

## When to use

- Semantic scene checks and checkpoint verification (`vlm.query_yes_no`).
- Free-form scene descriptions or attribute queries (`vlm.query`).
- Prefer `gemini-er.detect` when you need pixel-space bounding boxes, and
  `molmo.point_prompt` when you need a single click point.

## Notes

- Both tools accept `provider=`, `model=`, `temperature=`, and `seed=`. Explicit
  arguments override their corresponding VLM environment settings. The strict
  provider never inherits a model/key/base URL from `GAP_LLM_*`, OpenRouter,
  or Vertex and never falls back to another provider.
- Results preserve `result["text"]` and include `result["evidence"]`. Evidence
  contains provider/requested and resolved model, temperature, disabled cache
  policy, seed-control status, request ID, canonical `sha256:` request/response
  digests, token usage, every transport attempt, and `fallback_used: false`.
  Nested evidence is fresh per call and is suitable for immediate adapter
  validation/freezing.
- Strict requests use OpenAI text and PNG data-URL image content. Three retries
  means at most four attempts, limited to transport failures, HTTP 429, and
  HTTP 5xx before valid model content. Other 4xx responses fail once. The first
  valid content stops retrying; errors and request IDs are sanitized.
- Caching is always reported disabled. Seed control is evidence-based:
  `provider_confirmed` requires an exact response confirmation;
  `requested_unconfirmed` is not deterministic; known-unsupported relays do
  not receive a seed. `deterministic_claim` is always false.
- `vlm.query_yes_no` coerces with the source-verbatim rule: answer is true
  iff `"yes"` appears in the lowercased reply.
- Requests carry no system prompt. The ordinary product default temperature is
  `0.0`; formal paper runs inject `GAP_VLM_TEMPERATURE=0.1`.
