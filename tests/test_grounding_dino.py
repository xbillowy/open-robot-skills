"""grounding-dino bundle: signature/schema units + GPU smoke."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_MODULE_NAME = "gap_skills.tools.grounding-dino.tools"
if _MODULE_NAME in sys.modules:
    _DINO_TOOLS = sys.modules[_MODULE_NAME]
else:
    _TOOLS_PATH = Path(__file__).resolve().parents[1] / "tools" / "grounding-dino" / "tools.py"
    _SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, _TOOLS_PATH)
    assert _SPEC is not None and _SPEC.loader is not None
    _DINO_TOOLS = importlib.util.module_from_spec(_SPEC)
    sys.modules[_MODULE_NAME] = _DINO_TOOLS
    _SPEC.loader.exec_module(_DINO_TOOLS)


def test_detect_registered(tool_registry):
    assert "grounding-dino.detect" in tool_registry
    desc = tool_registry.get("grounding-dino.detect")
    assert desc.tags == ("perception",)


def test_detect_schema(tool_registry):
    schema = tool_registry.get("grounding-dino.detect").schema
    assert set(schema.inputs) == {"image", "query", "box_threshold", "text_threshold"}
    assert schema.inputs["image"].required
    assert schema.inputs["query"].required
    assert schema.inputs["box_threshold"].default == pytest.approx(0.20)
    assert schema.inputs["text_threshold"].default == pytest.approx(0.20)
    assert set(schema.outputs) == {"detections"}


def test_runtime_model_provenance_constants_are_public_and_exact():
    assert _DINO_TOOLS.MODEL_PROVIDER == "huggingface"
    assert _DINO_TOOLS.MODEL_NAME == "IDEA-Research/grounding-dino-base"
    assert _DINO_TOOLS.MODEL_REVISION == "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
    assert _DINO_TOOLS.DEFAULT_BOX_THRESHOLD == pytest.approx(0.20)
    assert _DINO_TOOLS.DEFAULT_TEXT_THRESHOLD == pytest.approx(0.20)


def test_model_and_processor_load_the_pinned_snapshot(monkeypatch):
    calls: list[tuple[str, str, dict[str, object]]] = []

    class _FakeProcessor:
        @classmethod
        def from_pretrained(cls, model: str, **kwargs: object):
            calls.append(("processor", model, kwargs))
            return object()

    class _FakeModel:
        @classmethod
        def from_pretrained(cls, model: str, **kwargs: object):
            calls.append(("model", model, kwargs))
            return cls()

        def to(self, _device: str):
            return self

        def eval(self) -> None:
            return None

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForZeroShotObjectDetection=_FakeModel,
            AutoProcessor=_FakeProcessor,
        ),
    )
    monkeypatch.setenv("GAP_DINO_MODEL", _DINO_TOOLS.MODEL_NAME)
    monkeypatch.setattr(_DINO_TOOLS, "_model", None)
    monkeypatch.setattr(_DINO_TOOLS, "_processor", None)

    _DINO_TOOLS._get_model()

    expected = {
        "revision": _DINO_TOOLS.MODEL_REVISION,
    }
    assert calls == [
        ("processor", _DINO_TOOLS.MODEL_NAME, expected),
        ("model", _DINO_TOOLS.MODEL_NAME, expected),
    ]


def test_prefetch_downloads_the_pinned_snapshot(monkeypatch):
    calls: list[dict[str, object]] = []
    monkeypatch.setenv("GAP_DINO_MODEL", _DINO_TOOLS.MODEL_NAME)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=lambda **kwargs: calls.append(kwargs)),
    )

    _DINO_TOOLS.prefetch()

    assert calls == [{
        "repo_id": _DINO_TOOLS.MODEL_NAME,
        "repo_type": "model",
        "revision": _DINO_TOOLS.MODEL_REVISION,
    }]


_REQUIRED_OFFLINE_FILES = {
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
}


@pytest.mark.parametrize(
    ("present", "expected"),
    [
        ({"config.json"}, False),
        *[
            (_REQUIRED_OFFLINE_FILES - {missing}, False)
            for missing in sorted(_REQUIRED_OFFLINE_FILES)
        ],
        (_REQUIRED_OFFLINE_FILES, True),
    ],
    ids=(
        "config-only",
        *[f"missing-{filename}" for filename in sorted(_REQUIRED_OFFLINE_FILES)],
        "complete",
    ),
)
def test_weight_probe_requires_complete_offline_snapshot(monkeypatch, present, expected):
    calls: list[tuple[str, str, str]] = []

    def _probe(model: str, filename: str, *, revision: str):
        calls.append((model, filename, revision))
        return f"/cache/{filename}" if filename in present else None

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(try_to_load_from_cache=_probe),
    )
    monkeypatch.setenv("GAP_DINO_MODEL", _DINO_TOOLS.MODEL_NAME)

    assert _DINO_TOOLS.weights_cached() is expected
    assert calls == [
        (_DINO_TOOLS.MODEL_NAME, filename, _DINO_TOOLS.MODEL_REVISION)
        for filename in _DINO_TOOLS.REQUIRED_OFFLINE_FILES
    ][: len(calls)]


@pytest.mark.parametrize(
    "override",
    ["attacker/model", "../local-checkpoint", "IDEA-Research/grounding-dino-large"],
)
def test_model_override_drift_fails_before_cache_check(monkeypatch, override):
    calls: list[object] = []
    monkeypatch.setenv("GAP_DINO_MODEL", override)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(try_to_load_from_cache=lambda *args, **kwargs: calls.append((args, kwargs))),
    )

    with pytest.raises(RuntimeError, match="GAP_DINO_MODEL.*exact pinned model"):
        _DINO_TOOLS.weights_cached()
    assert calls == []


def test_model_override_drift_fails_before_model_load(monkeypatch):
    calls: list[object] = []
    monkeypatch.setenv("GAP_DINO_MODEL", "attacker/model")
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForZeroShotObjectDetection=object(),
            AutoProcessor=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: calls.append((args, kwargs))
            ),
        ),
    )
    monkeypatch.setattr(_DINO_TOOLS, "_model", None)

    with pytest.raises(RuntimeError, match="GAP_DINO_MODEL.*exact pinned model"):
        _DINO_TOOLS._get_model()
    assert calls == []


def test_booted_rpc_descriptor_attests_runtime_model_and_defaults(monkeypatch):
    from gap.runtime.provenance import snapshot_tool_registry
    from gap.runtime.tool_bundle_manager import ToolBundleManager
    from gap.skills import load_skills
    from gap_core.tools import ToolRegistry

    monkeypatch.delenv("GAP_DINO_MODEL", raising=False)
    root = Path(__file__).resolve().parents[1]
    skills = load_skills(root, only=["grounding-dino"])
    registry = ToolRegistry()
    manager = ToolBundleManager(skills, registry, startup_timeout_s=120)
    try:
        manager.boot_all(["grounding-dino"])
        descriptor = registry.get("grounding-dino.detect")
        snapshot = snapshot_tool_registry(registry)["grounding-dino.detect"]
    finally:
        manager.shutdown_all()

    assert descriptor.transport == "rpc"
    assert descriptor.metadata == {
        "bundle": "grounding-dino",
        "model": "IDEA-Research/grounding-dino-base",
        "parameters": {
            "box_threshold": 0.20,
            "text_threshold": 0.20,
        },
        "provider": "huggingface",
        "revision": "12bdfa3120f3e7ec7b434d90674b3396eccf88eb",
        "snapshot": "12bdfa3120f3e7ec7b434d90674b3396eccf88eb",
    }
    assert descriptor.schema.inputs["box_threshold"].default == pytest.approx(0.20)
    assert descriptor.schema.inputs["text_threshold"].default == pytest.approx(0.20)
    assert snapshot["metadata"] == descriptor.metadata
    assert snapshot["descriptor_hash"] == (
        "sha256:bbe7eb97818855c49f8475a695cc33c9856d7a6b18ad97756d4405b687549ccc"
    )


@pytest.mark.gpu
def test_detect_gpu_smoke(tool_registry, monkeypatch):
    monkeypatch.setenv("GAP_DINO_MODEL", _DINO_TOOLS.MODEL_NAME)
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")

    img = np.full((160, 160, 3), 255, dtype=np.uint8)
    img[40:120, 40:120] = (200, 30, 30)
    out = tool_registry.invoke(
        "grounding-dino.detect", image=img, query="red square"
    )
    assert set(out) == {"detections"}
    for det in out["detections"]:
        assert set(det) == {"box", "label", "score"}
        assert 0.0 <= det["score"] <= 1.0
        box = det["box"]
        assert box["x2"] >= box["x1"] and box["y2"] >= box["y1"]
