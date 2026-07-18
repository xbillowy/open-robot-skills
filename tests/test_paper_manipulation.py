from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
PRESET = {
    "schema_version": "recast.paper_manipulation.v1",
    "parameters": [
        {
            "kind": "positive_distance_m",
            "name": "approach_distance_m",
            "paper_value": 0.12,
            "runtime_value": 0.12,
            "mapping": "exact",
            "evidence_level": "paper_explicit",
            "paper_locator": "Appendix D.2 / PlanGraspMotion / approach distance",
            "ambiguity_reason": None,
        },
        {
            "kind": "multiple_candidate_count",
            "name": "grasp_candidate_count",
            "paper_value": None,
            "runtime_value": 29,
            "mapping": "reproduction_choice",
            "evidence_level": "reproduction_choice",
            "paper_locator": "Appendix D.1 / grasp curobo obb / several top-down OBB candidates",
            "ambiguity_reason": "Appendix D.1 gives no numeric OBB count; runtime pins the maximum 29-candidate top-down set including pitch variants rather than importing GraspGen's unrelated 200/50 defaults.",
        },
        {
            "kind": "positive_count",
            "name": "ik_seed_count",
            "paper_value": 32,
            "runtime_value": 32,
            "mapping": "exact",
            "evidence_level": "paper_explicit",
            "paper_locator": "Appendix D.2 / SolveIK / num seeds",
            "ambiguity_reason": None,
        },
        {
            "kind": "positive_distance_m",
            "name": "lift_distance_m",
            "paper_value": 0.2,
            "runtime_value": 0.2,
            "mapping": "exact",
            "evidence_level": "paper_explicit",
            "paper_locator": "Appendix D.2 / PlanGraspMotion / lift distance",
            "ambiguity_reason": None,
        },
        {
            "kind": "positive_count",
            "name": "trajectory_waypoint_count",
            "paper_value": 20,
            "runtime_value": 20,
            "mapping": "exact",
            "evidence_level": "paper_explicit",
            "paper_locator": "Appendix D.2 / cuRobo PlanLinear and PyRoKI PlanLinear / waypoints",
            "ambiguity_reason": None,
        },
    ],
    "preset_sha256": "sha256:8f6f81c9f2880fe0e3f786e276511868142ca8033255d6b598d82baad22b77d9",
}
PRESET_JSON = json.dumps(
    PRESET, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
)
DIGEST = "sha256:" + "a" * 64
OBB_TARGET = {
    "center": {"x": 0.1, "y": 0.2, "z": 0.3},
    "extent": {"x": 0.04, "y": 0.03, "z": 0.08},
    "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
}
OBB_DEST = {
    "center": {"x": 0.6, "y": 0.2, "z": 0.1},
    "extent": {"x": 0.2, "y": 0.2, "z": 0.02},
    "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
}
POSE = {
    "position": {"x": 0.1, "y": 0.2, "z": 0.3},
    "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
}
TRAJECTORY = {"waypoints": [{"positions": [0.0] * 7}, {"positions": [0.1] * 7}]}

PAPER_SCRIPTS = {
    "perceive": ROOT / "skills/perceiving-objects/scripts/perceive_disambiguate_segment.py",
    "plan": ROOT / "skills/grasping-with-planner/scripts/plan_validate_grasp.py",
    "execute": ROOT / "skills/grasping-with-planner/scripts/execute_verify_grasp.py",
    "transport": ROOT / "skills/transporting-objects/scripts/plan_validate_transport.py",
}


def _literal_string_set(tree: ast.Module, name: str) -> frozenset[str]:
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    )
    assert (
        isinstance(assignment.value, ast.Call)
        and isinstance(assignment.value.func, ast.Name)
        and assignment.value.func.id == "frozenset"
        and len(assignment.value.args) == 1
        and not assignment.value.keywords
    )
    value = frozenset(ast.literal_eval(assignment.value.args[0]))
    assert all(isinstance(item, str) for item in value)
    return value


def _runtime_value_reads(tree: ast.Module) -> frozenset[str]:
    return frozenset(
        node.slice.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "values"
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    )


def _module(relative: str) -> ModuleType:
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _learned() -> dict[str, Any]:
    return {
        "kind": "learned_model",
        "requested_model": "m",
        "resolved_revision": "b" * 40,
        "weights_sha256": DIGEST,
        "input_sha256": DIGEST,
        "output_sha256": DIGEST,
        "fallback_used": False,
    }


def _algorithm() -> dict[str, Any]:
    return {
        "kind": "algorithm_service",
        "source_commit": "b" * 40,
        "uv_lock_sha256": DIGEST,
        "config_sha256": DIGEST,
        "runtime_environment_sha256": DIGEST,
        "input_sha256": DIGEST,
        "output_sha256": DIGEST,
        "fallback_used": False,
    }


def _vlm() -> dict[str, Any]:
    return {
        "provider": "fake",
        "requested_model": "fake",
        "resolved_model": "fake",
        "temperature": 0.0,
        "cache_policy": "disabled",
        "randomness": {
            "requested_seed": 0,
            "provider_reported_seed": 0,
            "seed_control": "provider_confirmed",
            "deterministic_claim": False,
        },
        "provider_request_id": "1",
        "request_sha256": DIGEST,
        "response_sha256": DIGEST,
        "usage": None,
        "transport_attempts": [],
        "fallback_used": False,
    }


class FakeContext:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.observations = 0
        self.obb_calls = 0

    def tool(self, name: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((name, kwargs))
        if name == "robot.get_observation":
            self.observations += 1
            return {
                "cameras": [
                    {
                        "name": "agentview",
                        "rgb": np.zeros((32, 32, 3), dtype=np.uint8),
                        "depth": np.ones((32, 32), dtype=np.float32),
                        "intrinsics": np.eye(3),
                        "pose": POSE,
                    }
                ],
                "arms": [
                    {
                        "joint_state": {"positions": [0.0] * 7},
                        "ee_pose": POSE,
                        "gripper": {"is_holding": self.observations >= 4},
                    }
                ],
            }
        if name == "grounding-dino.detect":
            return {
                "detections": [
                    {"box": [1, 1, 10, 10], "label": "object", "score": 0.9},
                    {"box": [12, 1, 20, 10], "label": "object", "score": 0.8},
                ],
                "evidence": _learned(),
            }
        if name == "vlm.query":
            return {
                "text": "YES" if "visibly holding" in kwargs["prompt"] else "B",
                "evidence": _vlm(),
            }
        if name == "sam3.segment_box":
            return {"masks": ["mask"], "scores": [0.9], "evidence": _learned()}
        if name == "geometry.mask_to_world_points":
            return {
                "points": [[0.1, 0.2, 0.3]],
                "evidence": {**_algorithm(), "valid_depth_count": 1, "total_mask_count": 1},
            }
        if name == "geometry.filter_and_compute_obb":
            self.obb_calls += 1
            return {
                "obb": OBB_TARGET if self.obb_calls == 1 else OBB_DEST,
                "evidence": _algorithm(),
            }
        if name == "geometry.top_down_grasp_candidates":
            return {
                "candidates": {
                    "poses": [
                        dict(POSE, position={"x": 0.1, "y": 0.2, "z": 0.3 + i / 1000})
                        for i in range(29)
                    ]
                },
                "evidence": _algorithm(),
            }
        if name == "curobo.batch_grasp_feasibility":
            return {
                "feasible": [True] + [False] * 28,
                "grasp_ik_ok": [True] + [False] * 28,
                "approach_ik_ok": [True] + [False] * 28,
                "corridor_collision_fraction": [0.0] + [1.0] * 28,
                "evidence": _algorithm(),
            }
        if name in {
            "curobo.plan_to_grasp_poses",
            "curobo.plan_with_grasped_object",
            "curobo.plan_directed_linear",
        }:
            return {"success": True, "trajectory": TRAJECTORY, "evidence": _algorithm()}
        if name in {
            "curobo.validate_joint_trajectory_robot",
            "curobo.validate_joint_trajectory_grasped",
        }:
            return {
                "success": True,
                "failure_reason": "",
                "first_collision_waypoint": -1,
                "collision_status_detail": "",
                "evidence": _algorithm(),
            }
        if name == "geometry.compute_drop_position":
            return {"position": {"x": 0.6, "y": 0.2, "z": 0.15}, "evidence": _algorithm()}
        if name == "robot.open_gripper":
            return {"position": 1.0}
        return {"success": True}


class FailingContext(FakeContext):
    def __init__(self, fail_name: str, *, occurrence: int = 1, mode: str = "raise") -> None:
        super().__init__()
        self.fail_name = fail_name
        self.occurrence = occurrence
        self.mode = mode
        self.seen = 0

    def tool(self, name: str, **kwargs: Any) -> dict[str, Any]:
        if name == self.fail_name:
            self.seen += 1
            if self.seen == self.occurrence:
                self.calls.append((name, kwargs))
                if self.mode == "raise":
                    raise RuntimeError("unstructured provider detail")
                if self.mode == "vlm_no":
                    return {"text": "NO", "evidence": _vlm()}
                if self.mode == "release_closed":
                    return {"position": 0.0}
        return super().tool(name, **kwargs)


class NegativeResultContext(FakeContext):
    def __init__(self, tool_name: str, response: dict[str, Any]) -> None:
        super().__init__()
        self.tool_name = tool_name
        self.response = response

    def tool(self, name: str, **kwargs: Any) -> dict[str, Any]:
        if name == self.tool_name:
            self.calls.append((name, kwargs))
            return self.response
        return super().tool(name, **kwargs)


def _happy_artifacts() -> tuple[dict[str, ModuleType], dict[str, Any]]:
    modules = {
        "perceive": _module("skills/perceiving-objects/scripts/perceive_disambiguate_segment.py"),
        "plan": _module("skills/grasping-with-planner/scripts/plan_validate_grasp.py"),
        "execute": _module("skills/grasping-with-planner/scripts/execute_verify_grasp.py"),
        "transport": _module("skills/transporting-objects/scripts/plan_validate_transport.py"),
    }
    ctx = FakeContext()
    target = modules["perceive"].run(
        ctx, query="红色 mug", semantic_role="target", preset_json=PRESET_JSON
    )
    destination = modules["perceive"].run(
        ctx, query="tray", semantic_role="destination", preset_json=PRESET_JSON
    )
    world = {"meshes": [{"name": "red_mug"}]}
    grasp = modules["plan"].run(
        ctx,
        target_obb=target["obb"],
        target_lineage_json=target["lineage_json"],
        world_config=world,
        target_name="red_mug",
        preset_json=PRESET_JSON,
    )
    held = modules["execute"].run(
        ctx,
        validated_grasp_json=grasp["validated_grasp_json"],
        target_lineage_json=target["lineage_json"],
        world_config=world,
        target_name="red_mug",
        preset_json=PRESET_JSON,
    )
    return modules, {
        "target": target,
        "destination": destination,
        "grasp": grasp,
        "held": held,
        "world": world,
    }


def _assert_failure(result: dict[str, Any], code: str, *, released: bool | None = None) -> None:
    assert result["success"] is False
    assert result["error_code"] == code
    if released is not None:
        assert result["released"] is released


def test_paper_manipulation_exact_admitted_path_and_distinct_lineages() -> None:
    perceive = _module("skills/perceiving-objects/scripts/perceive_disambiguate_segment.py")
    plan_grasp = _module("skills/grasping-with-planner/scripts/plan_validate_grasp.py")
    execute_grasp = _module("skills/grasping-with-planner/scripts/execute_verify_grasp.py")
    transport = _module("skills/transporting-objects/scripts/plan_validate_transport.py")
    ctx = FakeContext()

    target = perceive.run(ctx, query="红色 mug", semantic_role="target", preset_json=PRESET_JSON)
    destination = perceive.run(
        ctx, query="wooden tray", semantic_role="destination", preset_json=PRESET_JSON
    )
    grasp = plan_grasp.run(
        ctx,
        target_obb=target["obb"],
        target_lineage_json=target["lineage_json"],
        world_config={"meshes": [{"name": "red_mug"}]},
        target_name="red_mug",
        preset_json=PRESET_JSON,
    )
    held = execute_grasp.run(
        ctx,
        validated_grasp_json=grasp["validated_grasp_json"],
        target_lineage_json=target["lineage_json"],
        world_config={"meshes": [{"name": "red_mug"}]},
        target_name="red_mug",
        preset_json=PRESET_JSON,
    )
    result = transport.run(
        ctx,
        held_grasp_json=held["held_grasp_json"],
        target_lineage_json=target["lineage_json"],
        destination_obb=destination["obb"],
        destination_lineage_json=destination["lineage_json"],
        world_config={"meshes": [{"name": "red_mug"}]},
        target_name="red_mug",
        preset_json=PRESET_JSON,
    )

    assert target["semantic_role"] == "target"
    assert destination["semantic_role"] == "destination"
    for output in (target, destination, grasp, held, result):
        assert output["paper_outcome"] == {
            "status": "success",
            "failure_code": None,
            "source": "canonical_script",
        }
        assert isinstance(output["semantic_evidence"], list)
    assert (
        target["lineage_record"]["lineage_sha256"]
        != destination["lineage_record"]["lineage_sha256"]
    )
    assert result["status"] == "terminal"
    assert result["released"] is True
    assert (
        json.dumps(
            json.loads(result["terminal_result_json"]),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        == result["terminal_result_json"]
    )
    assert result["fallback_used"] is False
    assert result["decision_path"][-1] == "terminal_return"
    assert target["lineage_record"]["valid_depth_ratio"] == 1.0
    assert len(target["lineage_record"]["detector_candidates"]) == 2
    assert len(target["lineage_record"]["crop_mapping"]) == 2
    assert len(grasp["candidate_records"]) == 29
    assert grasp["candidate_records"][0]["status"] == "accepted"
    assert all(
        record["rejection_code"] == "IK_OR_APPROACH_INFEASIBLE"
        for record in grasp["candidate_records"][1:]
    )
    assert grasp["approach_pose"]["position"]["z"] == pytest.approx(
        grasp["grasp_pose"]["position"]["z"] + 0.12
    )
    assert grasp["retreat_pose"]["position"]["z"] == pytest.approx(
        grasp["grasp_pose"]["position"]["z"] + 0.2
    )
    assert (
        result["drop_input_binding"]["destination_obb_sha256"] == result["destination_obb_sha256"]
    )
    assert (
        result["transport_input_binding"]["destination_obb_sha256"]
        == result["destination_obb_sha256"]
    )
    assert result["drop_input_sha256"] == transport._hash(result["drop_input_binding"])
    assert result["transport_input_sha256"] == transport._hash(result["transport_input_binding"])
    assert {
        value["name"]: value["runtime_value"] for value in result["preset_trace"]["parameters"]
    } == {value["name"]: value["runtime_value"] for value in PRESET["parameters"]}
    names = [name for name, _ in ctx.calls]
    assert names == [
        "robot.get_observation",
        "grounding-dino.detect",
        "vlm.query",
        "sam3.segment_box",
        "geometry.mask_to_world_points",
        "geometry.filter_and_compute_obb",
        "robot.get_observation",
        "grounding-dino.detect",
        "vlm.query",
        "sam3.segment_box",
        "geometry.mask_to_world_points",
        "geometry.filter_and_compute_obb",
        "geometry.top_down_grasp_candidates",
        "robot.get_observation",
        "curobo.batch_grasp_feasibility",
        "curobo.plan_to_grasp_poses",
        "curobo.validate_joint_trajectory_robot",
        "curobo.validate_joint_trajectory_grasped",
        "robot.get_observation",
        "robot.execute_trajectory",
        "robot.close_gripper",
        "robot.get_observation",
        "vlm.query",
        "curobo.plan_directed_linear",
        "curobo.validate_joint_trajectory_robot",
        "curobo.validate_joint_trajectory_grasped",
        "robot.execute_trajectory",
        "robot.get_observation",
        "vlm.query",
        "geometry.compute_drop_position",
        "robot.get_observation",
        "curobo.plan_with_grasped_object",
        "curobo.validate_joint_trajectory_robot",
        "curobo.validate_joint_trajectory_grasped",
        "robot.execute_trajectory",
        "curobo.plan_directed_linear",
        "curobo.validate_joint_trajectory_robot",
        "curobo.validate_joint_trajectory_grasped",
        "robot.execute_trajectory",
        "robot.open_gripper",
        "curobo.plan_directed_linear",
        "curobo.validate_joint_trajectory_robot",
        "robot.execute_trajectory",
    ]
    assert "runner.check_success" not in names
    assert not any("native" in name or "wrist" in name or "fallback" in name for name in names)
    candidate_call = next(
        kwargs for name, kwargs in ctx.calls if name == "geometry.top_down_grasp_candidates"
    )
    assert "candidate_count" not in candidate_call
    assert (
        next(kwargs for name, kwargs in ctx.calls if name == "curobo.batch_grasp_feasibility")[
            "num_ik_seeds"
        ]
        == 32
    )
    grasp_plan_call = next(
        kwargs for name, kwargs in ctx.calls if name == "curobo.plan_to_grasp_poses"
    )
    assert grasp_plan_call["num_ik_seeds"] == 32
    assert grasp_plan_call["use_grasp_approach"] is True
    assert grasp_plan_call["grasp_approach_offset"] == 0.12
    grasp_validations = [(name, kwargs) for name, kwargs in ctx.calls[16:18]]
    assert grasp_validations[0][1]["trajectory"]["waypoints"][0]["positions"] == [0.0] * 7
    assert grasp_validations[1][1]["trajectory"]["waypoints"][0]["positions"] == [0.1] * 7
    linear_calls = [kwargs for name, kwargs in ctx.calls if name == "curobo.plan_directed_linear"]
    assert linear_calls[0]["distance"] == 0.2
    assert linear_calls[-1]["distance"] == 0.12
    executed = [
        kwargs["trajectory"] for name, kwargs in ctx.calls if name == "robot.execute_trajectory"
    ]
    assert all(len(trajectory["waypoints"]) == 20 for trajectory in executed)
    transport_plan_call = next(
        kwargs for name, kwargs in ctx.calls if name == "curobo.plan_with_grasped_object"
    )
    assert transport_plan_call["num_ik_seeds"] == 32
    drop_call = next(
        kwargs for name, kwargs in ctx.calls if name == "geometry.compute_drop_position"
    )
    assert drop_call["clearance"] == 0.12
    assert linear_calls[1]["distance"] == 0.12
    assert names[-1] == "robot.execute_trajectory"


@pytest.mark.parametrize(
    ("relative", "kwargs", "code"),
    [
        (
            "skills/perceiving-objects/scripts/perceive_disambiguate_segment.py",
            {"query": "mug", "semantic_role": "target"},
            "PRESET_HASH_MISMATCH",
        ),
        (
            "skills/grasping-with-planner/scripts/plan_validate_grasp.py",
            {
                "target_obb": OBB_TARGET,
                "target_lineage_json": "{}",
                "world_config": {"meshes": []},
                "target_name": "mug",
            },
            "PRESET_HASH_MISMATCH",
        ),
    ],
)
def test_sealed_preset_is_mandatory(relative: str, kwargs: dict[str, Any], code: str) -> None:
    module = _module(relative)
    bad = {**PRESET, "preset_sha256": "sha256:" + "0" * 64}
    result = module.run(
        FakeContext(),
        preset_json=json.dumps(bad, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        **kwargs,
    )
    _assert_failure(result, code)


def test_resealed_mutated_preset_is_rejected() -> None:
    module = _module("skills/perceiving-objects/scripts/perceive_disambiguate_segment.py")
    mutated = json.loads(json.dumps(PRESET))
    mutated["parameters"][0]["runtime_value"] = 0.13
    payload = json.dumps(
        {key: value for key, value in mutated.items() if key != "preset_sha256"},
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    mutated["preset_sha256"] = "sha256:" + hashlib.sha256(payload).hexdigest()
    result = module.run(
        FakeContext(),
        query="mug",
        semantic_role="target",
        preset_json=json.dumps(mutated, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )
    _assert_failure(result, "PRESET_HASH_MISMATCH")


def test_noncanonical_json_envelope_is_rejected_before_tools() -> None:
    module = _module("skills/perceiving-objects/scripts/perceive_disambiguate_segment.py")
    ctx = FakeContext()
    result = module.run(
        ctx, query="mug", semantic_role="target", preset_json=json.dumps(PRESET, indent=2)
    )
    _assert_failure(result, "PRESET_SCHEMA_INVALID")
    assert ctx.calls == []


def test_skill_frontmatter_declares_all_canonical_json_bindings() -> None:
    from gap.skills import load_skills

    registry = load_skills(ROOT)
    expected = {
        "perceiving-objects": (
            {"preset_json": "str"},
            {"target_lineage_json": "str", "destination_lineage_json": "str"},
        ),
        "grasping-with-planner": (
            {"target_lineage_json": "str", "preset_json": "str"},
            {"held_grasp_json": "str", "world_config": "WorldConfig"},
        ),
        "transporting-objects": (
            {
                "held_grasp_json": "str",
                "target_lineage_json": "str",
                "destination_lineage_json": "str",
                "preset_json": "str",
            },
            {"terminal_result_json": "str"},
        ),
    }
    for bundle, (required, produced) in expected.items():
        meta = registry.get(bundle).meta
        assert all(meta.required_inputs[name] == type_name for name, type_name in required.items())
        assert all(meta.produces_outputs[name] == type_name for name, type_name in produced.items())

    scripts = {
        "perceiving-objects": {
            "perceive_disambiguate_segment": ({"preset_json"}, {"lineage_json"})
        },
        "grasping-with-planner": {
            "plan_validate_grasp": (
                {"target_lineage_json", "preset_json"},
                {"validated_grasp_json"},
            ),
            "execute_verify_grasp": (
                {"validated_grasp_json", "target_lineage_json", "preset_json"},
                {"held_grasp_json"},
            ),
        },
        "transporting-objects": {
            "plan_validate_transport": (
                {
                    "held_grasp_json",
                    "target_lineage_json",
                    "destination_lineage_json",
                    "preset_json",
                },
                {"terminal_result_json"},
            )
        },
    }
    for bundle, contracts in scripts.items():
        loaded = registry.get(bundle).canonical_scripts
        for name, (inputs, outputs) in contracts.items():
            assert inputs <= set(loaded[name].schema.inputs)
            assert outputs <= set(loaded[name].schema.outputs)


def test_missing_learned_evidence_fails_closed_with_stable_code() -> None:
    module = _module("skills/perceiving-objects/scripts/perceive_disambiguate_segment.py")
    ctx = FakeContext()
    original = ctx.tool

    def tool(name: str, **kwargs: Any) -> dict[str, Any]:
        result = original(name, **kwargs)
        if name == "sam3.segment_box":
            result["evidence"] = None
        return result

    ctx.tool = tool  # type: ignore[method-assign]
    result = module.run(ctx, query="mug", semantic_role="target", preset_json=PRESET_JSON)
    _assert_failure(result, "SEGMENTATION_EVIDENCE_UNAVAILABLE")


def test_detector_exception_is_mapped_to_stable_stage_code() -> None:
    module = _module("skills/perceiving-objects/scripts/perceive_disambiguate_segment.py")
    ctx = FakeContext()
    original = ctx.tool

    def tool(name: str, **kwargs: Any) -> dict[str, Any]:
        if name == "grounding-dino.detect":
            raise RuntimeError("provider details must not escape")
        return original(name, **kwargs)

    ctx.tool = tool  # type: ignore[method-assign]
    result = module.run(ctx, query="mug", semantic_role="target", preset_json=PRESET_JSON)
    _assert_failure(result, "DETECTOR_SERVICE_ERROR")
    assert result["paper_outcome"]["failure_code"] == "detector_service_error"


@pytest.mark.parametrize(
    ("tool_name", "error_code"),
    [
        ("vlm.query", "VLM_SERVICE_ERROR"),
        ("geometry.mask_to_world_points", "SERVICE_UNAVAILABLE"),
        ("geometry.filter_and_compute_obb", "SERVICE_UNAVAILABLE"),
    ],
)
def test_perception_tool_failures_have_stage_codes(tool_name: str, error_code: str) -> None:
    module = _module("skills/perceiving-objects/scripts/perceive_disambiguate_segment.py")
    result = module.run(
        FailingContext(tool_name), query="mug", semantic_role="target", preset_json=PRESET_JSON
    )
    _assert_failure(result, error_code)
    assert result["paper_outcome"]["failure_code"] != "protocol_contract_violation"


@pytest.mark.parametrize(
    ("tool_name", "response", "internal_code", "paper_code"),
    [
        (
            "grounding-dino.detect",
            {"detections": [], "evidence": _learned()},
            "DETECTION_CANDIDATES_INSUFFICIENT",
            "detector_no_candidate",
        ),
        (
            "vlm.query",
            {"text": "not-a-label", "evidence": _vlm()},
            "VLM_DISAMBIGUATION_FAILED",
            "vlm_no_valid_selection",
        ),
        (
            "sam3.segment_box",
            {"masks": [], "scores": [], "evidence": _learned()},
            "SEGMENTATION_EMPTY",
            "segmentation_empty_mask",
        ),
    ],
)
def test_perception_domain_negatives_are_not_service_errors(
    tool_name: str, response: dict[str, Any], internal_code: str, paper_code: str
) -> None:
    module = _module("skills/perceiving-objects/scripts/perceive_disambiguate_segment.py")
    result = module.run(
        NegativeResultContext(tool_name, response),
        query="mug",
        semantic_role="target",
        preset_json=PRESET_JSON,
    )
    _assert_failure(result, internal_code)
    assert result["paper_outcome"]["failure_code"] == paper_code


@pytest.mark.parametrize(
    ("tool_name", "error_code"),
    [
        ("curobo.batch_grasp_feasibility", "SERVICE_UNAVAILABLE"),
        ("curobo.plan_to_grasp_poses", "SERVICE_UNAVAILABLE"),
        ("curobo.validate_joint_trajectory_robot", "SERVICE_UNAVAILABLE"),
        ("curobo.validate_joint_trajectory_grasped", "SERVICE_UNAVAILABLE"),
    ],
)
def test_grasp_planning_failures_have_stage_codes(tool_name: str, error_code: str) -> None:
    modules, values = _happy_artifacts()
    result = modules["plan"].run(
        FailingContext(tool_name),
        target_obb=values["target"]["obb"],
        target_lineage_json=values["target"]["lineage_json"],
        world_config=values["world"],
        target_name="red_mug",
        preset_json=PRESET_JSON,
    )
    _assert_failure(result, error_code)
    assert result["paper_outcome"]["failure_code"] == "service_unavailable"


@pytest.mark.parametrize(
    ("tool_name", "response", "internal_code", "paper_code"),
    [
        (
            "curobo.batch_grasp_feasibility",
            {
                "feasible": [False] * 29,
                "grasp_ik_ok": [False] * 29,
                "approach_ik_ok": [False] * 29,
                "corridor_collision_fraction": [1.0] * 29,
                "evidence": _algorithm(),
            },
            "IK_UNSOLVED",
            "ik_unsolved",
        ),
        (
            "curobo.batch_grasp_feasibility",
            {
                "feasible": [False] * 29,
                "grasp_ik_ok": [True] * 29,
                "approach_ik_ok": [True] * 29,
                "corridor_collision_fraction": [1.0] * 29,
                "evidence": _algorithm(),
            },
            "TRAJECTORY_INVALID",
            "trajectory_invalid",
        ),
        (
            "curobo.plan_to_grasp_poses",
            {"success": False, "trajectory": None, "evidence": _algorithm()},
            "MOTION_PLAN_FAILED",
            "motion_plan_failed",
        ),
        (
            "curobo.validate_joint_trajectory_robot",
            {"success": False, "evidence": _algorithm()},
            "TRAJECTORY_INVALID",
            "trajectory_invalid",
        ),
    ],
)
def test_grasp_domain_negatives_are_not_service_errors(
    tool_name: str, response: dict[str, Any], internal_code: str, paper_code: str
) -> None:
    modules, values = _happy_artifacts()
    result = modules["plan"].run(
        NegativeResultContext(tool_name, response),
        target_obb=values["target"]["obb"],
        target_lineage_json=values["target"]["lineage_json"],
        world_config=values["world"],
        target_name="red_mug",
        preset_json=PRESET_JSON,
    )
    _assert_failure(result, internal_code)
    assert result["paper_outcome"]["failure_code"] == paper_code


@pytest.mark.parametrize(
    ("tool_name", "occurrence", "mode", "error_code"),
    [
        ("robot.execute_trajectory", 1, "raise", "SERVICE_UNAVAILABLE"),
        ("robot.close_gripper", 1, "raise", "SERVICE_UNAVAILABLE"),
        ("vlm.query", 1, "vlm_no", "TARGET_NOT_HELD"),
        ("vlm.query", 2, "vlm_no", "POST_LIFT_TARGET_NOT_HELD"),
    ],
)
def test_grasp_execution_failures_have_stage_codes(
    tool_name: str, occurrence: int, mode: str, error_code: str
) -> None:
    modules, values = _happy_artifacts()
    result = modules["execute"].run(
        FailingContext(tool_name, occurrence=occurrence, mode=mode),
        validated_grasp_json=values["grasp"]["validated_grasp_json"],
        target_lineage_json=values["target"]["lineage_json"],
        world_config=values["world"],
        target_name="red_mug",
        preset_json=PRESET_JSON,
    )
    _assert_failure(result, error_code)
    if mode == "raise":
        assert result["paper_outcome"]["failure_code"] == "service_unavailable"


def test_visual_hold_without_exterior_camera_is_not_a_negative_hold() -> None:
    module = _module("skills/grasping-with-planner/scripts/execute_verify_grasp.py")

    with pytest.raises(module.PaperManipulationError) as caught:
        module._visual_hold(
            FakeContext(),
            {"cameras": [{"name": "wrist", "rgb": "image"}]},
            "red_mug",
            "TARGET_NOT_HELD",
        )

    assert caught.value.code == "EXTERNAL_CAMERA_UNAVAILABLE"
    assert module._PAPER_FAILURE_CODE[caught.value.code] == "service_unavailable"


@pytest.mark.parametrize(
    "evidence",
    [
        None,
        {**_vlm(), "fallback_used": True},
        {**_vlm(), "request_sha256": "sha256:malformed"},
    ],
)
def test_visual_hold_invalid_vlm_evidence_is_not_a_negative_hold(
    evidence: dict[str, Any] | None,
) -> None:
    module = _module("skills/grasping-with-planner/scripts/execute_verify_grasp.py")
    response: dict[str, Any] = {"text": "NO"}
    if evidence is not None:
        response["evidence"] = evidence

    with pytest.raises(module.PaperManipulationError) as caught:
        module._visual_hold(
            NegativeResultContext("vlm.query", response),
            {"cameras": [{"name": "exterior", "rgb": "image"}]},
            "red_mug",
            "TARGET_NOT_HELD",
        )

    assert caught.value.code == "VLM_EVIDENCE_UNAVAILABLE"
    assert module._PAPER_FAILURE_CODE[caught.value.code] == "vlm_service_error"


@pytest.mark.parametrize(
    ("tool_name", "occurrence", "mode", "error_code", "released"),
    [
        ("curobo.plan_with_grasped_object", 1, "raise", "SERVICE_UNAVAILABLE", False),
        ("curobo.plan_directed_linear", 1, "raise", "SERVICE_UNAVAILABLE", False),
        ("robot.open_gripper", 1, "release_closed", "RELEASE_FAILED", False),
        ("curobo.plan_directed_linear", 2, "raise", "SERVICE_UNAVAILABLE", True),
        ("robot.execute_trajectory", 3, "raise", "SERVICE_UNAVAILABLE", True),
    ],
)
def test_transport_failures_record_stage_and_release_state(
    tool_name: str, occurrence: int, mode: str, error_code: str, released: bool
) -> None:
    modules, values = _happy_artifacts()
    result = modules["transport"].run(
        FailingContext(tool_name, occurrence=occurrence, mode=mode),
        held_grasp_json=values["held"]["held_grasp_json"],
        target_lineage_json=values["target"]["lineage_json"],
        destination_obb=values["destination"]["obb"],
        destination_lineage_json=values["destination"]["lineage_json"],
        world_config=values["world"],
        target_name="red_mug",
        preset_json=PRESET_JSON,
    )
    _assert_failure(result, error_code, released=released)
    if mode == "raise":
        assert result["paper_outcome"]["failure_code"] == "service_unavailable"


def test_release_call_has_explicit_destination_drop_annotation() -> None:
    modules, values = _happy_artifacts()
    ctx = FakeContext()
    result = modules["transport"].run(
        ctx,
        held_grasp_json=values["held"]["held_grasp_json"],
        target_lineage_json=values["target"]["lineage_json"],
        destination_obb=values["destination"]["obb"],
        destination_lineage_json=values["destination"]["lineage_json"],
        world_config=values["world"],
        target_name="red_mug",
        preset_json=PRESET_JSON,
    )

    assert result["success"] is True
    release = next(kwargs for name, kwargs in ctx.calls if name == "robot.open_gripper")
    assert release["_paper_evidence"] == {
        "branch": "destination",
        "operation": "drop_release",
    }


def test_fewer_than_sealed_grasp_candidates_fails_without_top_one() -> None:
    module = _module("skills/grasping-with-planner/scripts/plan_validate_grasp.py")
    ctx = FakeContext()
    original = ctx.tool
    lineage = {
        "semantic_role": "target",
        "obb_sha256": module._hash(OBB_TARGET),
        "preset_trace": {"preset_sha256": PRESET["preset_sha256"]},
    }
    lineage["lineage_sha256"] = module._hash(lineage)

    def tool(name: str, **kwargs: Any) -> dict[str, Any]:
        result = original(name, **kwargs)
        if name == "geometry.top_down_grasp_candidates":
            result["candidates"]["poses"] = result["candidates"]["poses"][:25]
        return result

    ctx.tool = tool  # type: ignore[method-assign]
    result = module.run(
        ctx,
        target_obb=OBB_TARGET,
        target_lineage_json=module._canonical(lineage),
        world_config={"meshes": []},
        target_name="mug",
        preset_json=PRESET_JSON,
    )
    _assert_failure(result, "GRASP_CANDIDATES_INSUFFICIENT")
    assert [name for name, _ in ctx.calls] == ["geometry.top_down_grasp_candidates"]


def test_transport_rejects_reused_target_lineage_before_any_tool_call() -> None:
    module = _module("skills/transporting-objects/scripts/plan_validate_transport.py")
    ctx = FakeContext()
    reused = {
        "semantic_role": "target",
        "obb_sha256": module._hash(OBB_TARGET),
        "preset_trace": {"preset_sha256": PRESET["preset_sha256"]},
    }
    reused["lineage_sha256"] = module._hash(reused)
    held = {"held": True, "target_name": "mug", "target_lineage_sha256": reused["lineage_sha256"]}
    result = module.run(
        ctx,
        held_grasp_json=module._canonical(held),
        target_lineage_json=module._canonical(reused),
        destination_obb=OBB_DEST,
        destination_lineage_json=module._canonical(reused),
        world_config={"meshes": []},
        target_name="mug",
        preset_json=PRESET_JSON,
    )
    _assert_failure(result, "DESTINATION_LINEAGE_INVALID", released=False)
    assert ctx.calls == []


def test_paper_scripts_statically_exclude_native_success_and_route_inputs() -> None:
    paths = [
        ROOT / "skills/perceiving-objects/scripts/perceive_disambiguate_segment.py",
        ROOT / "skills/grasping-with-planner/scripts/plan_validate_grasp.py",
        ROOT / "skills/grasping-with-planner/scripts/execute_verify_grasp.py",
        ROOT / "skills/transporting-objects/scripts/plan_validate_transport.py",
    ]
    source = "\n".join(path.read_text() for path in paths)
    for forbidden in (
        "sim.check_success",
        "conn.check_success",
        "native predicate",
        "checkpoint(",
        "go_to_pose",
        "go_home",
    ):
        assert forbidden not in source
    assert "predicate" not in source
    assert "retry" not in source
    required_stage_codes = {
        "DETECTION_FAILED",
        "VLM_DISAMBIGUATION_FAILED",
        "SEGMENTATION_FAILED",
        "DEPTH_PROJECTION_FAILED",
        "OBB_FIT_FAILED",
        "GRASP_CANDIDATE_GENERATION_FAILED",
        "IK_FEASIBILITY_FAILED",
        "GRASP_PLANNING_FAILED",
        "ROBOT_TRAJECTORY_VALIDATION_FAILED",
        "HELD_OBJECT_TRAJECTORY_VALIDATION_FAILED",
        "GRASP_EXECUTION_FAILED",
        "GRASP_CLOSE_FAILED",
        "TARGET_NOT_HELD",
        "POST_LIFT_TARGET_NOT_HELD",
        "TRANSPORT_PLANNING_FAILED",
        "PLACEMENT_PLANNING_FAILED",
        "RELEASE_FAILED",
        "RETREAT_EXECUTION_FAILED",
    }
    assert all(f'"{code}"' in source for code in required_stage_codes)


def test_paper_scripts_have_one_static_run_entrypoint() -> None:
    for name, path in PAPER_SCRIPTS.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
        run_definitions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == "run"
        ]
        assert len(run_definitions) == 1, name


def test_paper_scripts_use_only_literal_tool_dispatch() -> None:
    for name, path in PAPER_SCRIPTS.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "tool"
        ]
        assert calls, name
        assert all(
            call.args
            and isinstance(call.args[0], ast.Constant)
            and isinstance(call.args[0].value, str)
            for call in calls
        ), name


def test_preset_field_declarations_match_runtime_reads_and_unique_ownership() -> None:
    expected_responsibility = {
        "perceive": frozenset(),
        "plan": frozenset({"approach_distance_m", "grasp_candidate_count", "ik_seed_count"}),
        "execute": frozenset({"lift_distance_m", "trajectory_waypoint_count"}),
        "transport": frozenset(),
    }
    all_responsible: list[str] = []
    for name, path in PAPER_SCRIPTS.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
        consumed = _literal_string_set(tree, "CONSUMED_PRESET_FIELDS")
        responsible = _literal_string_set(tree, "RESPONSIBLE_PRESET_FIELDS")
        assert consumed == _runtime_value_reads(tree), name
        assert responsible == expected_responsibility[name]
        assert responsible <= consumed
        all_responsible.extend(responsible)

    expected_fields = {item["name"] for item in PRESET["parameters"]}
    assert set(all_responsible) == expected_fields
    assert len(all_responsible) == len(expected_fields)
