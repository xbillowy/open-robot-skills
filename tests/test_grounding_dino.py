"""grounding-dino bundle: signature/schema units + GPU smoke."""

from __future__ import annotations

import importlib.util
import sys
import typing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from gap_core.tools import ToolRegistry
from gap_core.tools._registry import _PENDING_TOOLS

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def grounding_dino_module():
    name = "gap_skills.tools.grounding-dino.tools"
    sys.modules.pop(name, None)
    _PENDING_TOOLS[:] = [e for e in _PENDING_TOOLS if e["name"] != "grounding-dino.detect"]
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools/grounding-dino/tools.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool_registry(grounding_dino_module):
    del grounding_dino_module
    registry = ToolRegistry()
    registry.discover_pending()
    return registry


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
    assert set(schema.outputs) == {"detections", "evidence"}


def test_grounding_dino_uses_real_checked_model_artifact(grounding_dino_module):
    artifact = grounding_dino_module.paper_model_artifact()
    assert artifact == {
        "requested_model": "IDEA-Research/grounding-dino-base",
        "resolved_revision": "12bdfa3120f3e7ec7b434d90674b3396eccf88eb",
        "weights_sha256": (
            "sha256:5548f844c928c4b6f411fa8cbcc2bfa8dbbba437cb1d513975519f93c2a9ed21"
        ),
    }
    assert len(artifact["resolved_revision"]) == 40
    assert grounding_dino_module.paper_model_artifact(verify_local=True) == artifact
    manifest = (ROOT / "tools/grounding-dino/SKILL.md").read_text(encoding="utf-8")
    assert f"resolved_revision: {artifact['resolved_revision']}" in manifest
    assert f"weights_sha256: {artifact['weights_sha256']}" in manifest


def test_grounding_dino_evidence_is_discriminated_and_canonical(grounding_dino_module):
    fields = typing.get_type_hints(grounding_dino_module.LearnedServiceEvidence)
    assert list(fields) == [
        "kind",
        "requested_model",
        "resolved_revision",
        "weights_sha256",
        "input_sha256",
        "output_sha256",
        "fallback_used",
    ]
    evidence = grounding_dino_module._learned_evidence(
        {"image": np.zeros((2, 2, 3), dtype=np.uint8), "query": "cup"},
        {"detections": []},
    )
    assert evidence["kind"] == "learned_model"
    assert evidence["fallback_used"] is False
    assert evidence["input_sha256"].startswith("sha256:")
    assert evidence["output_sha256"].startswith("sha256:")
    with pytest.raises(ValueError, match="canonical SHA256"):
        grounding_dino_module._validate_digest("a" * 64)


def test_grounding_dino_prefetch_uses_checked_revision(monkeypatch, grounding_dino_module):
    calls = []

    def cached(*args, **kwargs):
        calls.append({"cache_args": args, **kwargs})
        return "/cached/config.json"

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            snapshot_download=lambda **kwargs: calls.append(kwargs),
            try_to_load_from_cache=cached,
        ),
    )

    grounding_dino_module.prefetch()
    assert grounding_dino_module.weights_cached() is True

    assert calls == [
        {
            "repo_id": "IDEA-Research/grounding-dino-base",
            "repo_type": "model",
            "revision": "12bdfa3120f3e7ec7b434d90674b3396eccf88eb",
        },
        {
            "cache_args": ("IDEA-Research/grounding-dino-base", "config.json"),
            "revision": "12bdfa3120f3e7ec7b434d90674b3396eccf88eb",
        },
    ]


def test_grounding_dino_loader_never_resolves_checked_artifact_remotely(
    monkeypatch, grounding_dino_module
):
    calls: list[tuple[str, str, dict[str, object]]] = []

    class FakeModel:
        def to(self, device):
            calls.append(("model.to", device, {}))
            return self

        def eval(self):
            calls.append(("model.eval", "", {}))

    class FakeLoader:
        def __init__(self, kind: str, result: object):
            self.kind = kind
            self.result = result

        def from_pretrained(self, model_name: str, **kwargs):
            calls.append((self.kind, model_name, kwargs))
            return self.result

    fake_transformers = SimpleNamespace(
        AutoProcessor=FakeLoader("processor", object()),
        AutoModelForZeroShotObjectDetection=FakeLoader("model", FakeModel()),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(
        grounding_dino_module,
        "paper_model_artifact",
        lambda *, verify_local=False: {"verified": verify_local},
    )
    monkeypatch.setattr(grounding_dino_module, "_model", None)
    monkeypatch.setattr(grounding_dino_module, "_processor", None)

    grounding_dino_module._get_model()

    expected_selection = {
        "revision": "12bdfa3120f3e7ec7b434d90674b3396eccf88eb",
        "local_files_only": True,
    }
    assert calls[:2] == [
        ("processor", "IDEA-Research/grounding-dino-base", expected_selection),
        ("model", "IDEA-Research/grounding-dino-base", expected_selection),
    ]


def test_grounding_dino_threshold_substitution_is_fallback(grounding_dino_module):
    evidence = grounding_dino_module._learned_evidence({}, {}, fallback_used=True)
    assert evidence["fallback_used"] is True


@pytest.mark.gpu
def test_detect_gpu_smoke(tool_registry):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")

    img = np.full((160, 160, 3), 255, dtype=np.uint8)
    img[40:120, 40:120] = (200, 30, 30)
    out = tool_registry.invoke("grounding-dino.detect", image=img, query="red square")
    assert set(out) == {"detections", "evidence"}
    assert out["evidence"]["fallback_used"] is False
    for det in out["detections"]:
        assert set(det) == {"box", "label", "score"}
        assert 0.0 <= det["score"] <= 1.0
        box = det["box"]
        assert box["x2"] >= box["x1"] and box["y2"] >= box["y1"]
