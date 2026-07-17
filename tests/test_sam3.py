"""sam3 bundle: signature/schema units + GPU smoke.

Collection must work without torch/sam3 installed — all heavy imports stay
behind ``pytest.importorskip`` inside the gpu-marked tests.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import typing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from gap_core.errors import ToolError
from gap_core.tools import ToolRegistry
from gap_core.tools._registry import _PENDING_TOOLS

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_TOOLS = {
    "sam3.segment_text",
    "sam3.segment_point",
    "sam3.segment_box",
    "sam3.tracker_init",
    "sam3.tracker_update",
    "sam3.tracker_close",
}

_SYNTHETIC_REVISION = "a" * 40
_SYNTHETIC_FILES = {
    "config.json": b'{"model_type":"synthetic-sam3"}\n',
    "sam3.pt": b"synthetic checkpoint bytes\n",
}


def _sha256(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _synthetic_hf_snapshot(tmp_path: Path) -> tuple[Path, Path]:
    cache = tmp_path / "hub"
    repo = cache / "models--facebook--sam3"
    snapshot = repo / "snapshots" / _SYNTHETIC_REVISION
    blobs = repo / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    for filename, data in _SYNTHETIC_FILES.items():
        blob = blobs / hashlib.sha256(data).hexdigest()
        blob.write_bytes(data)
        (snapshot / filename).symlink_to(Path("../../blobs") / blob.name)
    return cache, snapshot


def _seal_synthetic_manifest(
    sam3_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path]:
    cache, snapshot = _synthetic_hf_snapshot(tmp_path)
    manifest = tmp_path / "sam3-paper-model.json"
    sam3_module.seal_paper_model_manifest(snapshot_path=snapshot, output_path=manifest)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(cache))
    monkeypatch.setattr(sam3_module, "_CHECKED_MANIFEST_PATH", manifest)
    return manifest, snapshot


@pytest.fixture(scope="module")
def sam3_module():
    name = "gap_skills.tools.sam3.tools"
    sys.modules.pop(name, None)
    _PENDING_TOOLS[:] = [e for e in _PENDING_TOOLS if e["name"] not in EXPECTED_TOOLS]
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools/sam3/tools.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool_registry(sam3_module):
    del sam3_module
    registry = ToolRegistry()
    registry.discover_pending()
    return registry


def test_all_tools_registered(tool_registry):
    for name in EXPECTED_TOOLS:
        assert name in tool_registry


def test_segment_text_schema(tool_registry):
    schema = tool_registry.get("sam3.segment_text").schema
    assert set(schema.inputs) == {"image", "query", "max_results"}
    assert schema.inputs["image"].required
    assert schema.inputs["query"].required
    assert schema.inputs["max_results"].required is False
    assert schema.inputs["max_results"].default == 5
    assert set(schema.outputs) == {"masks", "scores", "boxes", "evidence"}


def test_segment_box_schema(tool_registry):
    schema = tool_registry.get("sam3.segment_box").schema
    assert set(schema.inputs) == {"image", "box", "pixel_x", "pixel_y", "use_point"}
    assert schema.inputs["use_point"].default is False
    assert set(schema.outputs) == {"masks", "scores", "evidence"}


def test_tracker_schemas(tool_registry):
    init = tool_registry.get("sam3.tracker_init").schema
    assert set(init.inputs) == {
        "image",
        "text",
        "box",
        "pixel_x",
        "pixel_y",
        "use_point",
        "object_name",
    }
    assert set(init.outputs) == {
        "tracker_id",
        "initial_mask",
        "initial_box",
        "score",
        "object_present",
        "evidence",
    }

    update = tool_registry.get("sam3.tracker_update").schema
    assert set(update.inputs) == {"tracker_id", "image"}
    assert set(update.outputs) == {"mask", "box", "confidence", "object_present", "evidence"}

    close = tool_registry.get("sam3.tracker_close").schema
    assert set(close.inputs) == {"tracker_id"}
    assert set(close.outputs) == {"closed", "evidence"}


def test_invalid_image_shape_rejected_before_model_load(tool_registry, sam3_module):
    """The RGB shape check fires in _to_pil — but only after the model
    singleton loads, so exercise the helper directly (no GPU needed)."""
    from gap_core.errors import ToolError

    mod = sam3_module
    with pytest.raises(ToolError):
        mod._to_pil(np.zeros((4, 4), dtype=np.uint8))
    pil = mod._to_pil(np.zeros((4, 4, 3), dtype=np.uint8))
    assert pil.size == (4, 4)


def test_sam3_paper_artifact_is_unavailable_without_checked_weights(sam3_module):
    fields = typing.get_type_hints(sam3_module.LearnedServiceEvidence)
    assert "weights_sha256" in fields
    with pytest.raises(ToolError, match="paper|artifact|weights"):
        sam3_module.paper_model_artifact()


def test_sam3_seals_and_validates_an_immutable_synthetic_snapshot(
    sam3_module, monkeypatch, tmp_path
):
    production_manifest = ROOT / "tools/sam3/paper_model_manifest.json"
    assert not production_manifest.exists()
    manifest, _ = _seal_synthetic_manifest(sam3_module, monkeypatch, tmp_path)

    assert sam3_module.paper_model_artifact() == {
        "requested_model": "facebook/sam3",
        "resolved_revision": _SYNTHETIC_REVISION,
        "weights_sha256": _sha256(_SYNTHETIC_FILES["sam3.pt"]),
    }
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload == {
        "schema_version": 1,
        "requested_model": "facebook/sam3",
        "resolved_revision": _SYNTHETIC_REVISION,
        "loader_revision": "b26a5f330e05d321afb39d01d3d4881f258f65ff",
        "files": {
            name: {"sha256": _sha256(data), "size_bytes": len(data)}
            for name, data in sorted(_SYNTHETIC_FILES.items())
        },
    }
    assert not production_manifest.exists(), "synthetic tests must not mint production authority"


def _fake_torch_module() -> ModuleType:
    module = ModuleType("torch")
    module.cuda = SimpleNamespace(is_available=lambda: False, current_device=lambda: 0)
    module.backends = SimpleNamespace(
        cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
        cudnn=SimpleNamespace(allow_tf32=False),
    )
    return module


def test_sam3_image_singleton_loads_only_the_validated_checkpoint(
    sam3_module, monkeypatch, tmp_path
):
    _seal_synthetic_manifest(sam3_module, monkeypatch, tmp_path)
    calls: list[dict[str, object]] = []

    class FakeModel:
        def to(self, device):
            calls.append({"model_to": device})
            return self

    class FakeProcessor:
        def __init__(self, model, confidence_threshold):
            calls.append({"processor_model": model, "threshold": confidence_threshold})

    def build_image(**kwargs):
        calls.append(kwargs)
        return FakeModel()

    model_builder = ModuleType("sam3.model_builder")
    model_builder.build_sam3_image_model = build_image
    processor_module = ModuleType("sam3.model.sam3_image_processor")
    processor_module.Sam3Processor = FakeProcessor
    monkeypatch.setitem(sys.modules, "torch", _fake_torch_module())
    monkeypatch.setitem(sys.modules, "sam3", ModuleType("sam3"))
    monkeypatch.setitem(sys.modules, "sam3.model", ModuleType("sam3.model"))
    monkeypatch.setitem(sys.modules, "sam3.model_builder", model_builder)
    monkeypatch.setitem(sys.modules, "sam3.model.sam3_image_processor", processor_module)
    monkeypatch.setattr(sam3_module, "_image_model", None)
    monkeypatch.setattr(sam3_module, "_image_processor", None)

    sam3_module._get_model(device="cpu")

    image_call = calls[0]
    checkpoint_path = Path(str(image_call.pop("checkpoint_path")))
    assert image_call == {
        "enable_inst_interactivity": True,
        "load_from_HF": False,
    }
    assert checkpoint_path.parent == Path("/proc/self/fd")
    assert checkpoint_path.name.isdigit()


def test_sam3_tracker_singleton_loads_only_the_validated_checkpoint(
    sam3_module, monkeypatch, tmp_path
):
    _seal_synthetic_manifest(sam3_module, monkeypatch, tmp_path)
    calls: list[dict[str, object]] = []

    def build_predictor(**kwargs):
        calls.append(kwargs)
        return object()

    model_builder = ModuleType("sam3.model_builder")
    model_builder.build_sam3_video_predictor = build_predictor
    monkeypatch.setitem(sys.modules, "torch", _fake_torch_module())
    monkeypatch.setitem(sys.modules, "sam3", ModuleType("sam3"))
    monkeypatch.setitem(sys.modules, "sam3.model_builder", model_builder)
    monkeypatch.setattr(sam3_module, "_tracker_predictor", None)
    monkeypatch.setattr(sam3_module, "_ensure_cc_compiler", lambda: None)

    sam3_module._get_tracker_predictor()

    assert len(calls) == 1
    tracker_call = calls[0]
    checkpoint_path = Path(str(tracker_call.pop("checkpoint_path")))
    assert tracker_call == {
        "gpus_to_use": [0],
        "apply_temporal_disambiguation": False,
    }
    assert checkpoint_path.parent == Path("/proc/self/fd")
    assert checkpoint_path.name.isdigit()


def test_sam3_image_loader_keeps_attested_inode_across_path_replacement(
    sam3_module, monkeypatch, tmp_path
):
    _, snapshot = _seal_synthetic_manifest(sam3_module, monkeypatch, tmp_path)
    blob = (snapshot / "sam3.pt").resolve(strict=True)
    loaded: list[bytes] = []

    class FakeModel:
        def to(self, device):
            return self

    class FakeProcessor:
        def __init__(self, model, confidence_threshold):
            pass

    def build_image(**kwargs):
        displaced = blob.with_name(f"{blob.name}.attested")
        blob.rename(displaced)
        blob.write_bytes(b"unattested replacement bytes")
        loaded.append(Path(kwargs["checkpoint_path"]).read_bytes())
        return FakeModel()

    model_builder = ModuleType("sam3.model_builder")
    model_builder.build_sam3_image_model = build_image
    processor_module = ModuleType("sam3.model.sam3_image_processor")
    processor_module.Sam3Processor = FakeProcessor
    monkeypatch.setitem(sys.modules, "torch", _fake_torch_module())
    monkeypatch.setitem(sys.modules, "sam3", ModuleType("sam3"))
    monkeypatch.setitem(sys.modules, "sam3.model", ModuleType("sam3.model"))
    monkeypatch.setitem(sys.modules, "sam3.model_builder", model_builder)
    monkeypatch.setitem(sys.modules, "sam3.model.sam3_image_processor", processor_module)
    monkeypatch.setattr(sam3_module, "_image_model", None)
    monkeypatch.setattr(sam3_module, "_image_processor", None)

    with pytest.raises(ToolError, match="sam3.pt.*changed"):
        sam3_module._get_model(device="cpu")

    assert loaded == [_SYNTHETIC_FILES["sam3.pt"]]
    assert sam3_module._image_model is None
    assert sam3_module._image_processor is None


def test_sam3_tracker_rejects_checkpoint_mutated_during_load(sam3_module, monkeypatch, tmp_path):
    _, snapshot = _seal_synthetic_manifest(sam3_module, monkeypatch, tmp_path)
    blob = (snapshot / "sam3.pt").resolve(strict=True)

    def build_predictor(**kwargs):
        assert Path(kwargs["checkpoint_path"]).read_bytes() == _SYNTHETIC_FILES["sam3.pt"]
        blob.write_bytes(b"mutated while tracker was loading")
        return object()

    model_builder = ModuleType("sam3.model_builder")
    model_builder.build_sam3_video_predictor = build_predictor
    monkeypatch.setitem(sys.modules, "torch", _fake_torch_module())
    monkeypatch.setitem(sys.modules, "sam3", ModuleType("sam3"))
    monkeypatch.setitem(sys.modules, "sam3.model_builder", model_builder)
    monkeypatch.setattr(sam3_module, "_tracker_predictor", None)
    monkeypatch.setattr(sam3_module, "_ensure_cc_compiler", lambda: None)

    with pytest.raises(ToolError, match="sam3.pt.*(changed|digest|size)"):
        sam3_module._get_tracker_predictor()

    assert sam3_module._tracker_predictor is None


def test_sam3_tracker_rejects_mutate_read_restore_during_load(sam3_module, monkeypatch, tmp_path):
    _, snapshot = _seal_synthetic_manifest(sam3_module, monkeypatch, tmp_path)
    blob = (snapshot / "sam3.pt").resolve(strict=True)
    original = blob.read_bytes()
    original_stat = blob.stat()
    malicious = b"x" * len(original)
    loaded: list[bytes] = []

    def build_predictor(**kwargs):
        blob.write_bytes(malicious)
        loaded.append(Path(kwargs["checkpoint_path"]).read_bytes())
        blob.write_bytes(original)
        os.utime(blob, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        return object()

    model_builder = ModuleType("sam3.model_builder")
    model_builder.build_sam3_video_predictor = build_predictor
    monkeypatch.setitem(sys.modules, "torch", _fake_torch_module())
    monkeypatch.setitem(sys.modules, "sam3", ModuleType("sam3"))
    monkeypatch.setitem(sys.modules, "sam3.model_builder", model_builder)
    monkeypatch.setattr(sam3_module, "_tracker_predictor", None)
    monkeypatch.setattr(sam3_module, "_ensure_cc_compiler", lambda: None)

    with pytest.raises(ToolError, match="sam3.pt.*changed"):
        sam3_module._get_tracker_predictor()

    assert loaded == [malicious]
    assert blob.read_bytes() == original
    assert blob.stat().st_mtime_ns == original_stat.st_mtime_ns
    assert sam3_module._tracker_predictor is None


def test_sam3_concurrent_sealers_publish_exactly_one_complete_manifest(sam3_module, tmp_path):
    _, snapshot = _synthetic_hf_snapshot(tmp_path)
    output = tmp_path / "raced-manifest.json"
    barrier = Barrier(2)

    def seal():
        barrier.wait()
        return sam3_module.seal_paper_model_manifest(
            snapshot_path=snapshot,
            output_path=output,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(seal) for _ in range(2)]
    successes = [future.result() for future in futures if future.exception() is None]
    failures = [future.exception() for future in futures if future.exception() is not None]

    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], ToolError)
    assert "already exists" in str(failures[0])
    assert json.loads(output.read_text(encoding="utf-8")) == successes[0]


@pytest.mark.parametrize("filename", ["config.json", "sam3.pt"])
def test_sam3_rejects_content_changed_after_manifest_sealing(
    sam3_module, monkeypatch, tmp_path, filename
):
    _, snapshot = _seal_synthetic_manifest(sam3_module, monkeypatch, tmp_path)
    selected = snapshot / filename
    selected.resolve(strict=True).write_bytes(b"changed after sealing")

    with pytest.raises(ToolError, match=rf"{filename}.*(digest|size)|(?:digest|size).*{filename}"):
        sam3_module.paper_model_artifact()


def test_sam3_rejects_snapshot_file_that_is_not_a_symlink(sam3_module, monkeypatch, tmp_path):
    cache, snapshot = _synthetic_hf_snapshot(tmp_path)
    selected = snapshot / "sam3.pt"
    data = selected.read_bytes()
    selected.unlink()
    selected.write_bytes(data)
    manifest = tmp_path / "manifest.json"
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(cache))

    with pytest.raises(ToolError, match="symlink|mutable"):
        sam3_module.seal_paper_model_manifest(snapshot_path=snapshot, output_path=manifest)


def test_sam3_rejects_snapshot_symlink_that_escapes_the_repo_blobs(
    sam3_module, monkeypatch, tmp_path
):
    manifest, snapshot = _seal_synthetic_manifest(sam3_module, monkeypatch, tmp_path)
    outside = tmp_path / "outside.pt"
    outside.write_bytes(_SYNTHETIC_FILES["sam3.pt"])
    selected = snapshot / "sam3.pt"
    selected.unlink()
    selected.symlink_to(outside)

    with pytest.raises(ToolError, match="escape|blobs"):
        sam3_module.paper_model_artifact()
    assert manifest.exists()


def test_sam3_rejects_incomplete_snapshot(sam3_module, tmp_path):
    _, snapshot = _synthetic_hf_snapshot(tmp_path)
    (snapshot / "config.json").unlink()

    with pytest.raises(ToolError, match="config.json|incomplete"):
        sam3_module.seal_paper_model_manifest(
            snapshot_path=snapshot,
            output_path=tmp_path / "manifest.json",
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.pop("files"),
        lambda payload: payload.update({"unexpected": True}),
        lambda payload: payload.update({"resolved_revision": "main"}),
        lambda payload: payload.update({"loader_revision": "b" * 40}),
        lambda payload: payload.update({"requested_model": "mutable/sam3"}),
        lambda payload: payload["files"]["config.json"].update({"sha256": "bad"}),
        lambda payload: payload["files"]["sam3.pt"].update({"size_bytes": -1}),
    ],
)
def test_sam3_rejects_invalid_checked_manifest_schema(sam3_module, monkeypatch, tmp_path, mutation):
    manifest, _ = _seal_synthetic_manifest(sam3_module, monkeypatch, tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    mutation(payload)
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ToolError, match="manifest|schema|revision|size"):
        sam3_module.paper_model_artifact()


def test_sam3_rejects_symlinked_checked_manifest(sam3_module, monkeypatch, tmp_path):
    manifest, _ = _seal_synthetic_manifest(sam3_module, monkeypatch, tmp_path)
    link = tmp_path / "manifest-link.json"
    link.symlink_to(manifest)
    monkeypatch.setattr(sam3_module, "_CHECKED_MANIFEST_PATH", link)

    with pytest.raises(ToolError, match="manifest.*symlink|symlink.*manifest"):
        sam3_module.paper_model_artifact()


def test_sam3_sealer_refuses_to_overwrite_existing_authority(sam3_module, tmp_path):
    _, snapshot = _synthetic_hf_snapshot(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("existing authority", encoding="utf-8")

    with pytest.raises(ToolError, match="already exists"):
        sam3_module.seal_paper_model_manifest(
            snapshot_path=snapshot,
            output_path=manifest,
        )
    assert manifest.read_text(encoding="utf-8") == "existing authority"


def test_sam3_manifest_sealer_cli_writes_only_the_requested_path(sam3_module, tmp_path, capsys):
    _, snapshot = _synthetic_hf_snapshot(tmp_path)
    output = tmp_path / "explicit-output.json"
    production_manifest = ROOT / "tools/sam3/paper_model_manifest.json"

    assert (
        sam3_module._manifest_sealer_main(["--snapshot", str(snapshot), "--output", str(output)])
        == 0
    )

    assert json.loads(output.read_text(encoding="utf-8"))["resolved_revision"] == (
        _SYNTHETIC_REVISION
    )
    assert str(output.resolve()) in capsys.readouterr().out
    assert not production_manifest.exists()


@pytest.mark.parametrize("invalid", ["a" * 64, "sha256:abc", "SHA256:" + "a" * 64])
def test_sam3_rejects_noncanonical_weight_hash(sam3_module, invalid):
    with pytest.raises(ValueError, match="canonical SHA256"):
        sam3_module._validate_digest(invalid)


def _smoke_image() -> np.ndarray:
    img = np.full((128, 128, 3), 255, dtype=np.uint8)
    img[32:96, 32:96] = (220, 30, 30)  # red square on white
    return img


@pytest.mark.gpu
def test_segment_text_gpu_smoke(tool_registry):
    torch = pytest.importorskip("torch")
    pytest.importorskip("sam3")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")

    out = tool_registry.invoke("sam3.segment_text", image=_smoke_image(), query="red square")
    assert set(out) == {"masks", "scores", "boxes", "evidence"}
    assert out["evidence"] is None
    assert len(out["masks"]) == len(out["scores"]) == len(out["boxes"])
    for mask in out["masks"]:
        assert mask.dtype == np.uint8
        assert mask.shape == (128, 128)
        assert set(np.unique(mask)) <= {0, 255}
    # Scores are sorted best-first.
    assert out["scores"] == sorted(out["scores"], reverse=True)


@pytest.mark.gpu
def test_segment_box_gpu_smoke(tool_registry):
    torch = pytest.importorskip("torch")
    pytest.importorskip("sam3")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")

    out = tool_registry.invoke(
        "sam3.segment_box",
        image=_smoke_image(),
        box={"x1": 28.0, "y1": 28.0, "x2": 100.0, "y2": 100.0},
    )
    assert out["masks"], "box prompt on a clear square should segment something"
    assert out["masks"][0].shape == (128, 128)


@pytest.mark.gpu
def test_tracker_roundtrip_gpu_smoke(tool_registry):
    torch = pytest.importorskip("torch")
    pytest.importorskip("sam3")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")

    frame0 = _smoke_image()
    init = tool_registry.invoke(
        "sam3.tracker_init",
        image=frame0,
        box={"x1": 28.0, "y1": 28.0, "x2": 100.0, "y2": 100.0},
        object_name="red_square",
    )
    if not init["object_present"]:
        pytest.skip("tracker did not lock onto the synthetic square")

    # Shift the square a few pixels and track it.
    frame1 = np.full_like(frame0, 255)
    frame1[36:100, 36:100] = (220, 30, 30)
    upd = tool_registry.invoke("sam3.tracker_update", tracker_id=init["tracker_id"], image=frame1)
    assert set(upd) == {"mask", "box", "confidence", "object_present", "evidence"}
    assert upd["evidence"] is None

    closed = tool_registry.invoke("sam3.tracker_close", tracker_id=init["tracker_id"])
    assert closed["closed"] is True
    assert closed["evidence"] is None
    # Idempotent close.
    closed2 = tool_registry.invoke("sam3.tracker_close", tracker_id=init["tracker_id"])
    assert closed2["closed"] is False
