"""Opt-in VLM choice-prompt v2 (GAP_VLM_CHOICE_PROMPT=2).

v2 changes two perception questions: the parent-box pick of
perceiving-object-parts ("which box CONTAINS" + NONE + JSON) and left/right
names in perceiving-objects (full-scene set-of-marks before the crop
tournament). Unset or "1" must keep the legacy requests byte-identical.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ENV = "GAP_VLM_CHOICE_PROMPT"


def _module(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


parts = _module("skills/perceiving-object-parts/scripts/perceive_subpart.py", "_choice_parts")
objs = _module("skills/perceiving-objects/scripts/perceive_dino_vlm.py", "_choice_objs")


def _det(x1, y1, x2, y2, score=0.5):
    return {"box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}, "label": "x", "score": score}


DETS = [_det(10, 10, 60, 60, 0.9), _det(70, 10, 150, 90, 0.8), _det(5, 5, 155, 115, 0.7)]


class Ctx:
    """Records every tool call; vlm.query replies come from a queue."""

    def __init__(self, replies, detections=DETS):
        self.replies = list(replies)
        self.detections = detections
        self.calls: list[tuple[str, dict]] = []

    def tool(self, name, **kw):
        self.calls.append((name, kw))
        if name == "grounding-dino.detect":
            return {"detections": list(self.detections)}
        if name == "vlm.query":
            reply = self.replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return {"text": reply}
        raise AssertionError(name)

    def prompts(self):
        return [kw["prompt"] for name, kw in self.calls if name == "vlm.query"]


RGB = np.zeros((120, 160, 3), np.uint8)


def _parent(ctx):
    return parts._pick_parent_box(ctx, RGB, "top drawer of the cabinet", 0.3, 0.25)


@pytest.mark.parametrize("raw,want", [("", 1), ("1", 1), ("2", 2), (" 2 ", 2)])
def test_version_selector(monkeypatch, raw, want):
    monkeypatch.setenv(ENV, raw)
    assert parts._choice_prompt_version() == want
    assert objs._choice_prompt_version() == want


def test_version_selector_rejects_unknown(monkeypatch):
    monkeypatch.setenv(ENV, "3")
    with pytest.raises(ValueError):
        parts._choice_prompt_version()
    with pytest.raises(ValueError):
        objs._choice_prompt_version()


def test_parent_legacy_prompt_unchanged(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    ctx = Ctx(["B"])
    assert _parent(ctx) == DETS[1]["box"]
    assert ctx.prompts() == [
        "The image on the right shows 3 candidate bounding boxes labeled A, B, C. "
        "Which box is the actual top drawer of the cabinet? Reply with a single letter."]


def test_parent_contain_v2_prompt_and_pick(monkeypatch):
    monkeypatch.setenv(ENV, "2")
    ctx = Ctx(['{"box": "C"}'])
    assert _parent(ctx) == DETS[2]["box"]
    (prompt,) = ctx.prompts()
    assert "Which box best CONTAINS the top drawer of the cabinet?" in prompt
    assert "labeled A, B, C." in prompt and 'the left panel ("Original")' in prompt
    assert prompt.endswith('{"box": "<letter>"} or {"box": "NONE"}.')


@pytest.mark.parametrize("reply", ['{"box": "NONE"}', "NONE", "neither"])
def test_parent_contain_v2_none_means_no_parent(monkeypatch, reply):
    monkeypatch.setenv(ENV, "2")
    ctx = Ctx([reply])
    assert _parent(ctx) is None
    assert len(ctx.prompts()) == 1


def test_parent_contain_v2_reasks_once_then_legacy_fallback(monkeypatch):
    monkeypatch.setenv(ENV, "2")
    ctx = Ctx(["I am not sure about this.", "B"])
    assert _parent(ctx) == DETS[1]["box"]
    ctx = Ctx(["hmm", "still unsure"])
    assert _parent(ctx) == DETS[0]["box"]          # top-confidence fallback
    assert len(ctx.prompts()) == 2
    ctx = Ctx([RuntimeError("transport")])
    assert _parent(ctx) == DETS[0]["box"]


def test_spatial_name_scene_marks_v2(monkeypatch):
    monkeypatch.setenv(ENV, "2")
    ctx = Ctx(['{"box": "B"}'])
    assert objs._scene_marks_pick(ctx, RGB, DETS, "left bowl") == 1
    (name, kw), = ctx.calls
    assert kw["image"].shape == (120, 320, 3)      # original | annotated
    assert "Which box is the left bowl? If no box shows it, answer NONE." in kw["prompt"]
    for reply in (['{"box": "NONE"}'], ["?", "??"]):
        ctx = Ctx(reply)
        assert objs._scene_marks_pick(ctx, RGB, DETS, "right bowl") is None
    assert objs._scene_marks_pick(Ctx([]), RGB, DETS[:1], "left bowl") is None


def test_spatial_name_regex():
    hit = objs._SPATIAL_NAME.search
    assert hit("left bowl") and hit("the Right plate") and hit("leftmost mug")
    assert not hit("bowl") and not hit("tray") and not hit("bright red mug")


def test_cache_key_legacy_unchanged(monkeypatch):
    cams = [{"name": "agentview", "rgb": RGB, "depth": np.zeros((120, 160), np.float32),
             "intrinsics": np.eye(3), "pose": None}]
    monkeypatch.delenv(ENV, raising=False)
    legacy = objs._make_cache_key(cams, {"object_name": "bowl"})
    monkeypatch.setenv(ENV, "1")
    assert objs._make_cache_key(cams, {"object_name": "bowl"}) == legacy
    monkeypatch.setenv(ENV, "2")
    assert objs._make_cache_key(cams, {"object_name": "bowl"}) != legacy
