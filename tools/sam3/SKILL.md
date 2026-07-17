---
name: sam3
description: Segment Anything 3 — text-, point-, and box-prompted instance
  segmentation, plus a stateful streaming video tracker that carries object
  identity through SAM3's memory bank. Use when a workflow needs open-vocabulary
  masks from an RGB image or needs to follow one object across frames.
license: MIT
compatibility: requires gap>=0.1
metadata: {category: perception, tags: [perception, segmentation, tracking, gpu]}
gap:
  requires: {gpu: true, weights: true}
  serving:
    command: ["python", "-m", "gap_core.rpc.server", "--bundle", "sam3"]
    protocol: stdio-msgpack
    requires_gpu: true
  tools:
    - sam3.segment_text: Segment all instances matching a text description (masks, scores, boxes best-first).
    - sam3.segment_point: Segment the object at a pixel coordinate (multimask, best-first).
    - sam3.segment_box: Segment within a bounding box, optionally refined by a foreground point.
    - sam3.tracker_init: Open a tracker session seeded with one frame + a text/box/point prompt.
    - sam3.tracker_update: Advance a tracker session by one frame (mask, box, confidence).
    - sam3.tracker_close: Free a tracker session (idempotent).
---

# sam3

The SAM3 image servicer + video-tracker servicer as in-process tools. Images
are RGB uint8 `[H, W, 3]` numpy arrays; masks come back as gap `Mask`
(uint8 `[H, W]`, 0 background / 255 foreground), score-sorted best-first.

## When to use

- `segment_text` for open-vocabulary "find the X" masks (one mask per
  instance; check `scores[0]` — callers typically reject below ~0.3).
- `segment_box` after a detector (e.g. `grounding-dino.detect`) for a
  pixel-accurate mask inside the detection box; add the point prompt
  (`use_point=True`) when a pointing model supplies one.
- `tracker_init` / `tracker_update` / `tracker_close` to follow a single
  target across an observation stream (e.g. for visual servoing).

## Install

```bash
uv sync --extra sam3       # torch + torchvision + the upstream sam3 package
# (pip: pip install -e ".[sam3]")
```

Use `prefetch()` / `gap skills check --download` to download the gated model
before sealing its checked manifest. Model construction never falls back to an
unpinned Hugging Face lookup. Device is taken from `GAP_SAM3_DEVICE` (default
`cuda`); the image model also runs on `cpu` (slow), while the video tracker is
CUDA-only in practice.

## Paper model admission

Paper admission is bound to the checked `paper_model_manifest.json` committed
with this bundle. `paper_model_artifact()` validates the exact local bytes and
fails closed on any missing file or identity drift; no revision or digest is
inferred from the mutable model name. Image and tracker singleton creation also
fails closed without this authority, and neither builder may silently download
from an unpinned revision. Tool results still carry `evidence: null` until the
paper evidence plumbing promotes this validated identity.

The admitted identity is:

- `requested_model`: `facebook/sam3`
- `resolved_revision`: `3c879f39826c281e95690f02c7821c4de09afae7`
- `sam3.pt` SHA256:
  `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`

The checked-manifest pipeline binds all of the following:

- the exact 40- or 64-hex Hugging Face snapshot revision for `facebook/sam3`;
- the pinned upstream loader source revision
  `b26a5f330e05d321afb39d01d3d4881f258f65ff`;
- byte size and SHA256 for `config.json` and `sam3.pt`;
- the normal Hugging Face cache layout, where both selected files are symlinks
  into the same repository's `blobs/` directory.

Those are the two files fetched by pinned upstream
`sam3/model_builder.py::download_ckpt_from_hf()`. The helper discards the
`config.json` path without parsing its contents, so config is validated as part
of the official two-file download closure, not claimed as model-construction
input. The actual checkpoint authority is `sam3.pt`, shared by the image and
video builders. The repository's separate `model.safetensors` artifact is not
selected by these paper services, so it is not admitted as runtime authority.

After an account has been authorized for the gated repository, download the
snapshot normally and retain the exact path returned by `snapshot_download`:

```python
from huggingface_hub import snapshot_download

snapshot = snapshot_download(repo_id="facebook/sam3", repo_type="model")
print(snapshot)  # .../models--facebook--sam3/snapshots/<immutable-commit>
```

Then seal those local bytes. Both paths are mandatory, the output must not
already exist, and sealing never contacts Hugging Face:

```bash
tools/sam3/.venv/bin/python tools/sam3/tools.py \
  --snapshot /path/to/models--facebook--sam3/snapshots/<immutable-commit> \
  --output tools/sam3/paper_model_manifest.json
```

Review and commit the generated JSON. At admission time,
`paper_model_artifact()` strictly parses that checked file, reconstructs the
exact snapshot path under the configured Hugging Face cache, rejects regular
snapshot files, dangling/escaping symlinks, incomplete layouts, schema drift,
and loader/revision drift, and re-hashes both config and checkpoint bytes.
The singleton loaders receive the already-validated resolved `sam3.pt` blob;
they hold one `O_NOFOLLOW` file descriptor across pre-load hashing, model
construction through `/proc/self/fd/<fd>`, and post-load hashing. Thus pathname
replacement cannot redirect a builder, and in-place mutation during loading
fails before a singleton is published. A `(device, inode, size, mtime_ns,
ctime_ns)` fingerprint must also remain identical across the entire load
window; unlike mtime, an ordinary same-user writer cannot restore ctime with
`utime` after a mutate-read-restore attack. The image builder additionally sets
`load_from_HF=False`, while the tracker predictor's pinned API suppresses its
internal HF download whenever an explicit checkpoint path is present. Hosts
without a valid `/proc/self/fd` view fail closed.
Synthetic tests write manifests only under pytest temporary directories; they
never mint production authority.

## Gotchas (carried over from the servicers)

- **Lazy singletons**: the image model and the video predictor each load on
  first call and stay resident; importing the bundle never imports torch.
- `segment_text` caps results at `max_results=5` by default — cluttered
  scenes emit 100+ instances (~1 MB/mask at 720p) and downstream consumes
  only the top mask. Pass `max_results<=0` for everything.
- The video tracker JIT-compiles **Triton NMS kernels** via the `CC` env
  var; a stale `CC` (e.g. a Ray env pointing at a non-existent gcc-13)
  surfaces as `FileNotFoundError` inside `tracker_init`. The bundle forces
  `CC` to a real compiler before tracker use (`_ensure_cc_compiler`).
- Tracker prompt precedence is **box > point > text**; a point prompt is
  converted to a small (10% of image) box because the predictor's box path
  is more reliable for init than a single point.
- The tracker is built with `apply_temporal_disambiguation=False` — the
  default hotstart heuristics silently delete the masklet around frame 3 in
  streaming mode (no fresh text re-detection per frame).
- Drift handling in `tracker_update`: a mask-area jump >1.5x the running
  median or confidence <0.30 keeps the LAST GOOD mask and reports
  `confidence=0.0` with `object_present=True` (skip this frame); after 5
  consecutive drift hits `object_present=False` — re-init the tracker.
- Sessions idle longer than 120 s are evicted lazily on the next tracker
  call; an evicted/unknown `tracker_id` raises `ToolError`.
- `tracker_init` returns `object_present=False` with an empty `tracker_id`
  (no exception) when the initial detection finds nothing.
