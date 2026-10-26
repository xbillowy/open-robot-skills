"""Toolfix c1e393d catalog v1: tools that cannot run in the frozen pipeline are not discoverable.

- gemini-er: needs GOOGLE_API_KEY / GEMINI_API_KEY (absent from the run env), calls Google directly (outside the
  VLM route policy) and its frozen bundle venv was never provisioned -> moved to ``disabled/gemini-er``.
- curobo.validate_joint_trajectory_robot / _grasped: v0.7-only MotionGen path (NameError TensorDeviceType under
  curobo v0.8.0) -> not registered and not advertised.
"""

from __future__ import annotations

from pathlib import Path

from gap.skills import load_skills

ROOT = Path(__file__).resolve().parents[1]
REMOVED = {"gemini-er.detect", "curobo.validate_joint_trajectory_robot", "curobo.validate_joint_trajectory_grasped"}


def test_removed_tools_are_not_discoverable():
    reg = load_skills(ROOT)
    names = {i.meta.name for i in reg.list_skills()}
    assert "gemini-er" not in names
    advertised = {t for i in reg.list_skills() for t in (i.meta.tools or {})}
    assert not advertised & REMOVED
    assert {"curobo.plan_to_grasp_poses", "vlm.query_yes_no", "molmo.point_prompt"} <= advertised


def test_no_visible_text_points_at_removed_tools():
    for md in sorted(m for kind in ("skills", "tools", "policies") for m in (ROOT / kind).glob("*/SKILL.md")):
        text = md.read_text(encoding="utf-8")
        for name in REMOVED | {"gemini-er bundle"}:
            assert name not in text, (md, name)
    src = (ROOT / "tools/curobo/tools.py").read_text(encoding="utf-8")
    assert 'name="curobo.validate_joint_trajectory' not in src


def test_no_bundle_dir_outside_the_catalog_is_globbed():
    """Every consumer that globs ``<kind>/*/SKILL.md`` (public catalog, packs) must not see the disabled bundle."""
    for kind in ("skills", "tools", "policies"):
        for md in (ROOT / kind).glob("*/SKILL.md"):
            assert "gemini" not in md.parent.name, md
