"""grounding-dino tool bundle — zero-shot object detection.

Extracted from the original Grounding DINO gRPC servicer in the dev
tree. The transformers pipeline
(processor → model → post_process_grounded_object_detection) is verbatim;
the proto byte decode/encode is replaced by numpy arrays + gap.types dicts.

The model loads lazily on first call (module-level singleton); importing
this module never pulls torch/transformers. Knobs via env:

- ``GAP_DINO_DEVICE`` — torch device (default ``cuda``).
The model identity is fixed to the checked paper artifact manifest in
``SKILL.md``; only the device is configurable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np
from gap_core.errors import PerceptionFailed, ToolError
from gap_core.tools import tool
from gap_core.types import BoundingBox2D

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_NAME = "IDEA-Research/grounding-dino-base"
_MODEL_REVISION = "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
_WEIGHTS_SHA256 = "sha256:5548f844c928c4b6f411fa8cbcc2bfa8dbbba437cb1d513975519f93c2a9ed21"
_DEFAULT_BOX_THRESHOLD = 0.20
_DEFAULT_TEXT_THRESHOLD = 0.20

_DEVICE = os.environ.get("GAP_DINO_DEVICE", "cuda")
_MODEL_NAME = _DEFAULT_MODEL_NAME

_load_lock = threading.Lock()
_model: Any = None
_processor: Any = None


class Detection(TypedDict):
    box: BoundingBox2D  # [x1, y1, x2, y2] in pixels
    label: str  # matched text label
    score: float  # confidence [0, 1]


class LearnedServiceEvidence(TypedDict):
    kind: Literal["learned_model"]
    requested_model: str
    resolved_revision: str
    weights_sha256: str
    input_sha256: str
    output_sha256: str
    fallback_used: bool


class DetectResult(TypedDict):
    detections: list[Detection]
    evidence: LearnedServiceEvidence


def _validate_digest(value: str) -> str:
    if re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise ValueError("digest must be canonical SHA256 (sha256:<64 lowercase hex>)")
    return value


def _validate_git_object_id(value: str) -> str:
    if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", value) is None:
        raise ValueError("revision must be a 40- or 64-hex Git object ID")
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            "dtype": str(contiguous.dtype),
            "shape": list(contiguous.shape),
            "bytes_sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


@lru_cache(maxsize=1)
def _weight_file_sha256(path: str) -> str:
    with Path(path).open("rb") as stream:
        return f"sha256:{hashlib.file_digest(stream, 'sha256').hexdigest()}"


def paper_model_artifact(*, verify_local: bool = False) -> dict[str, str]:
    """Return the checked immutable model identity derived from the local HF blob."""

    artifact = {
        "requested_model": _MODEL_NAME,
        "resolved_revision": _validate_git_object_id(_MODEL_REVISION),
        "weights_sha256": _validate_digest(_WEIGHTS_SHA256),
    }
    manifest = Path(__file__).with_name("SKILL.md").read_text(encoding="utf-8")
    for key, value in artifact.items():
        if f"{key}: {value}" not in manifest:
            raise RuntimeError(f"grounding-dino checked model manifest drift: {key}")
    if verify_local:
        hub = Path(
            os.environ.get(
                "HUGGINGFACE_HUB_CACHE",
                Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub",
            )
        )
        selected = (
            hub
            / f"models--{_MODEL_NAME.replace('/', '--')}"
            / "snapshots"
            / _MODEL_REVISION
            / "model.safetensors"
        )
        if not selected.is_symlink():
            raise RuntimeError("grounding-dino checked model artifact is not cached")
        blob_name = selected.resolve(strict=True).name
        if f"sha256:{blob_name}" != artifact["weights_sha256"]:
            raise RuntimeError("grounding-dino selected model blob differs from manifest")
        if _weight_file_sha256(str(selected.resolve())) != artifact["weights_sha256"]:
            raise RuntimeError("grounding-dino selected model bytes differ from manifest")
    return artifact


def _learned_evidence(
    inputs: Any,
    output: Any,
    *,
    fallback_used: bool = False,
) -> LearnedServiceEvidence:
    artifact = paper_model_artifact()
    return {
        "kind": "learned_model",
        **artifact,
        "input_sha256": _canonical_sha256(inputs),
        "output_sha256": _canonical_sha256(output),
        "fallback_used": fallback_used,
    }


def weights_cached() -> bool | None:
    """Filesystem-only weight-cache probe for ``gap check``.

    Checks the Hugging Face cache for the configured model's config.json
    (the canonical presence marker) — never downloads, never imports
    torch. ``None`` when huggingface_hub is unavailable ("unknown").
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    try:
        result = try_to_load_from_cache(_MODEL_NAME, "config.json", revision=_MODEL_REVISION)
    except Exception:
        return None
    return isinstance(result, str)


def prefetch() -> None:
    """Snapshot-download the configured GDINO weights into the HF cache.

    Called by ``gap skills check --download``. Idempotent: re-running
    against an already-cached snapshot is a near-no-op (HF revision
    check + symlink refresh). Logs the bytes added so the user sees
    progress; raises on network / auth / disk errors so ``gap skills
    check --download`` exits non-zero.

    Uses ``snapshot_download`` rather than ``AutoModel.from_pretrained``
    so we never load torch / instantiate the model at prefetch time —
    important for CI lanes and for the bare-engine venv.
    """
    from huggingface_hub import snapshot_download

    logger.info("[grounding-dino] prefetching weights for %s ...", _MODEL_NAME)
    snapshot_download(
        repo_id=_MODEL_NAME,
        repo_type="model",
        revision=_MODEL_REVISION,
    )
    logger.info("[grounding-dino] prefetch complete (cached at HF default)")


def _get_model() -> tuple[Any, Any]:
    """Load the Grounding DINO model + processor once (lazy singleton)."""
    global _model, _processor
    with _load_lock:
        if _model is None:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            paper_model_artifact(verify_local=True)
            logger.info("Loading Grounding DINO model: %s on %s ...", _MODEL_NAME, _DEVICE)
            _processor = AutoProcessor.from_pretrained(
                _MODEL_NAME,
                revision=_MODEL_REVISION,
                local_files_only=True,
            )
            model = AutoModelForZeroShotObjectDetection.from_pretrained(
                _MODEL_NAME,
                revision=_MODEL_REVISION,
                local_files_only=True,
            )
            model = model.to(_DEVICE)
            model.eval()
            _model = model
            logger.info("Grounding DINO model loaded successfully on %s.", _DEVICE)
        return _model, _processor


@tool(
    name="grounding-dino.detect",
    summary="Zero-shot object detection from a text prompt; returns labeled boxes with confidence scores.",
    tags=("perception",),
)
def detect(
    image: np.ndarray,
    query: str,
    box_threshold: float = _DEFAULT_BOX_THRESHOLD,
    text_threshold: float = _DEFAULT_TEXT_THRESHOLD,
) -> DetectResult:
    """Detect objects matching ``query`` in an RGB uint8 [H, W, 3] image.

    Grounding DINO expects period-separated phrases (``"red cube. green
    cube."``); a missing trailing period is appended automatically. Returns
    an empty detections list when nothing clears the thresholds — select the
    best box downstream (e.g. closest to a pointing-model pixel) and feed it
    to ``sam3.segment_box`` for a pixel-accurate mask.
    """
    import torch
    from PIL import Image

    model, processor = _get_model()

    arr = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ToolError(
            "grounding-dino.detect",
            f"expected RGB uint8 [H, W, 3] image, got shape {arr.shape}",
        )
    pil_image = Image.fromarray(arr, "RGB")

    # GDINO requires period-terminated phrases
    text_prompt = query
    if not text_prompt.endswith("."):
        text_prompt = text_prompt + "."

    # Defaults if zero/unset (mirrors the servicer's proto-default handling)
    threshold_fallback = box_threshold <= 0 or text_threshold <= 0
    box_threshold = box_threshold if box_threshold > 0 else _DEFAULT_BOX_THRESHOLD
    text_threshold = text_threshold if text_threshold > 0 else _DEFAULT_TEXT_THRESHOLD

    try:
        inputs = processor(images=pil_image, text=text_prompt, return_tensors="pt").to(_DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[pil_image.size[::-1]],
        )[0]
    except Exception as e:
        raise PerceptionFailed(f"Grounding DINO detection failed: {e}") from e

    detections: list[Detection] = []
    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results["labels"]

    for box, score, label in zip(boxes, scores, labels, strict=False):
        detections.append(
            {
                "box": {
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3]),
                },
                "label": str(label),
                "score": float(score),
            }
        )

    logger.info("grounding-dino.detect returning %d detections.", len(detections))
    output = {"detections": detections}
    return {
        **output,
        "evidence": _learned_evidence(
            {
                "image": arr,
                "query": text_prompt,
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
            },
            output,
            fallback_used=threshold_fallback,
        ),
    }
