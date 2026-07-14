---
name: grounding-dino
description: Grounding DINO zero-shot object detection — natural-language
  queries to labeled 2D bounding boxes with confidence scores. Use when a
  workflow needs to locate named objects in an RGB image before segmenting
  or grasping them.
license: MIT
compatibility: requires gap>=0.1
metadata: {category: perception, tags: [perception, detection, gpu]}
gap:
  requires: {gpu: true, weights: true}
  serving:
    command: ["python", "-m", "gap_core.rpc.server", "--bundle", "grounding-dino"]
    protocol: stdio-msgpack
    requires_gpu: true
  tools:
    - grounding-dino.detect: Zero-shot detection from a text prompt (labeled boxes + scores).
---

# grounding-dino

The Grounding DINO servicer (`IDEA-Research/grounding-dino-base` via
transformers) as one in-process tool. Image in: RGB uint8 `[H, W, 3]` numpy
array; out: `{detections: [{box, label, score}, ...]}`.

## When to use

- Locating a named object in a camera frame: `grounding-dino.detect(rgb,
  "cream cheese box.")`, pick the best box (highest score, or closest to a
  pointing-model pixel), then `sam3.segment_box` for a pixel-accurate mask.
- Empty `detections` means nothing cleared the thresholds — treat as
  not-found, don't retry blindly with the same prompt.

## Install

```bash
uv sync --extra grounding-dino   # torch + transformers
# (pip: pip install -e ".[grounding-dino]")
```

Weights download from Hugging Face on first call. The runtime pins
`IDEA-Research/grounding-dino-base` at revision
`12bdfa3120f3e7ec7b434d90674b3396eccf88eb`; `GAP_DINO_DEVICE` selects the
device (default `cuda`; CPU works but is slow) without changing model identity.
For compatibility with older deployments, `GAP_DINO_MODEL` may still be set
to that exact model name. Any other value is rejected before cache checks or
model loading; unset the variable instead of using it to override the pin.

## Gotchas (carried over from the servicer)

- **Period-separated phrases**: GDINO's text encoder expects each object
  phrase terminated with `.` (`"red cube. green cube."`). A missing final
  period is appended automatically, but separate multiple objects yourself.
- Default thresholds are deliberately low (0.20/0.20) for recall on
  household objects; raise them when false positives leak through. Zero or
  negative thresholds fall back to the defaults (proto-default semantics).
- `label` strings are the matched text spans, not your full query — when
  querying multiple phrases, group detections by label.
- The model is a lazy module-level singleton; the first call pays the
  weights-load latency, subsequent calls don't.
