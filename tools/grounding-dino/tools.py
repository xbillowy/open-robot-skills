"""grounding-dino tool bundle — zero-shot object detection.

Extracted from the original Grounding DINO gRPC servicer in the dev
tree. The transformers pipeline
(processor → model → post_process_grounded_object_detection) is verbatim;
the proto byte decode/encode is replaced by numpy arrays + gap.types dicts.

The model loads lazily on first call (module-level singleton); importing
this module never pulls torch/transformers. Knobs via env:

- ``GAP_DINO_DEVICE`` — torch device (default ``cuda``).

The public model identity is immutable so runtime provenance always names
the exact snapshot that was loaded.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, TypedDict

import numpy as np
from gap_core.errors import PerceptionFailed, ToolError
from gap_core.tools import tool
from gap_core.types import BoundingBox2D

logger = logging.getLogger(__name__)

MODEL_PROVIDER = "huggingface"
MODEL_NAME = "IDEA-Research/grounding-dino-base"
MODEL_REVISION = "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
DEFAULT_BOX_THRESHOLD = 0.20
DEFAULT_TEXT_THRESHOLD = 0.20
REQUIRED_OFFLINE_FILES = (
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

_DEVICE = os.environ.get("GAP_DINO_DEVICE", "cuda")

_load_lock = threading.Lock()
_model: Any = None
_processor: Any = None


class Detection(TypedDict):
    box: BoundingBox2D    # [x1, y1, x2, y2] in pixels
    label: str            # matched text label
    score: float          # confidence [0, 1]


class DetectResult(TypedDict):
    detections: list[Detection]


def _validate_model_override() -> None:
    configured = os.environ.get("GAP_DINO_MODEL")
    if configured is not None and configured != MODEL_NAME:
        raise RuntimeError(
            "GAP_DINO_MODEL must name the exact pinned model "
            f"{MODEL_NAME!r}; unset it or set it to that value"
        )


def weights_cached() -> bool | None:
    """Filesystem-only weight-cache probe for ``gap check``.

    Checks the pinned Hugging Face revision for the complete declared
    offline load set: model config and safetensors, image preprocessor
    config, tokenizer config, and tokenizer JSON. Never downloads or
    imports torch. ``None`` when huggingface_hub is unavailable ("unknown").
    """
    _validate_model_override()
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    for filename in REQUIRED_OFFLINE_FILES:
        try:
            result = try_to_load_from_cache(
                MODEL_NAME,
                filename,
                revision=MODEL_REVISION,
            )
        except Exception:
            return None
        if not isinstance(result, str):
            return False
    return True


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
    _validate_model_override()
    from huggingface_hub import snapshot_download

    logger.info(
        "[grounding-dino] prefetching weights for %s@%s ...",
        MODEL_NAME,
        MODEL_REVISION,
    )
    snapshot_download(
        repo_id=MODEL_NAME,
        repo_type="model",
        revision=MODEL_REVISION,
    )
    logger.info("[grounding-dino] prefetch complete (cached at HF default)")


def _get_model() -> tuple[Any, Any]:
    """Load the Grounding DINO model + processor once (lazy singleton)."""
    global _model, _processor
    _validate_model_override()
    with _load_lock:
        if _model is None:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            logger.info(
                "Loading Grounding DINO model: %s@%s on %s ...",
                MODEL_NAME,
                MODEL_REVISION,
                _DEVICE,
            )
            _processor = AutoProcessor.from_pretrained(
                MODEL_NAME,
                revision=MODEL_REVISION,
            )
            model = AutoModelForZeroShotObjectDetection.from_pretrained(
                MODEL_NAME,
                revision=MODEL_REVISION,
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
    metadata={
        "provider": MODEL_PROVIDER,
        "model": MODEL_NAME,
        "revision": MODEL_REVISION,
        "snapshot": MODEL_REVISION,
        "parameters": {
            "box_threshold": DEFAULT_BOX_THRESHOLD,
            "text_threshold": DEFAULT_TEXT_THRESHOLD,
        },
    },
)
def detect(
    image: np.ndarray,
    query: str,
    box_threshold: float = DEFAULT_BOX_THRESHOLD,
    text_threshold: float = DEFAULT_TEXT_THRESHOLD,
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
    box_threshold = box_threshold if box_threshold > 0 else DEFAULT_BOX_THRESHOLD
    text_threshold = text_threshold if text_threshold > 0 else DEFAULT_TEXT_THRESHOLD

    try:
        inputs = processor(
            images=pil_image, text=text_prompt, return_tensors="pt"
        ).to(_DEVICE)
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
        detections.append({
            "box": {
                "x1": float(box[0]),
                "y1": float(box[1]),
                "x2": float(box[2]),
                "y2": float(box[3]),
            },
            "label": str(label),
            "score": float(score),
        })

    logger.info("grounding-dino.detect returning %d detections.", len(detections))
    return {"detections": detections}
