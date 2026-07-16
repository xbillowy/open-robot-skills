"""Execute a validated grasp and verify hold before and after a planned lift."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, TypedDict

from gap import NodeContext
from gap_core.types import OrientedBoundingBox, Se3Pose, WorldConfig

EXPECTED = {
    "approach_distance_m",
    "grasp_candidate_count",
    "ik_seed_count",
    "lift_distance_m",
    "trajectory_waypoint_count",
}
DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
SEALED_PRESET_SHA256 = "sha256:8f6f81c9f2880fe0e3f786e276511868142ca8033255d6b598d82baad22b77d9"


class PaperManipulationError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class Output(TypedDict, total=False):
    success: bool
    error_code: str | None
    held: bool
    target_obb: OrientedBoundingBox
    target_name: str
    target_lineage_sha256: str
    post_lift_joint_state: dict[str, Any]
    post_lift_ee_pose: Se3Pose
    validation_evidence: dict[str, Any]
    decision_path: list[str]
    fallback_used: bool
    preset_trace: dict[str, Any]
    held_grasp_json: str


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    )


def _hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode()).hexdigest()


def _decode_record(value: str, code: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise PaperManipulationError(code) from error
    if not isinstance(decoded, dict) or _canonical(decoded) != value:
        raise PaperManipulationError(code)
    return decoded


def _preset(preset_json: str) -> tuple[dict[str, Any], dict[str, Any]]:
    preset = _decode_record(preset_json, "PRESET_SCHEMA_INVALID")
    rows = preset.get("parameters")
    if preset.get("schema_version") != "recast.paper_manipulation.v1" or not isinstance(rows, list):
        raise PaperManipulationError("PRESET_SCHEMA_INVALID")
    if preset.get("preset_sha256") != SEALED_PRESET_SHA256 or preset.get("preset_sha256") != _hash(
        {k: v for k, v in preset.items() if k != "preset_sha256"}
    ):
        raise PaperManipulationError("PRESET_HASH_MISMATCH")
    if len(rows) != 5 or {row.get("name") for row in rows} != EXPECTED:
        raise PaperManipulationError("PRESET_PARAMETERS_INVALID")
    return (
        {row["name"]: row["runtime_value"] for row in rows},
        {
            "preset_sha256": preset["preset_sha256"],
            "parameters": [
                {
                    k: row[k]
                    for k in ("name", "runtime_value", "mapping", "evidence_level", "paper_locator")
                }
                for row in rows
            ],
        },
    )


def _admitted(result: dict[str, Any], code: str) -> dict[str, Any]:
    evidence = result.get("evidence")
    required = (
        "uv_lock_sha256",
        "config_sha256",
        "runtime_environment_sha256",
        "input_sha256",
        "output_sha256",
    )
    if (
        not isinstance(evidence, dict)
        or evidence.get("kind") != "algorithm_service"
        or evidence.get("fallback_used") is not False
        or not all(DIGEST.fullmatch(str(evidence.get(k, ""))) for k in required)
    ):
        raise PaperManipulationError(code)
    return evidence


def _tool(ctx: NodeContext, name: str, code: str, **kwargs: Any) -> dict[str, Any]:
    try:
        return ctx.tool(name, **kwargs)
    except Exception as error:
        raise PaperManipulationError(code) from error


def _visual_hold(
    ctx: NodeContext, observation: dict[str, Any], target_name: str, code: str
) -> dict[str, Any]:
    cameras = observation.get("cameras", [])
    exterior = [
        camera
        for camera in cameras
        if not any(
            token in str(camera.get("name", "")).lower()
            for token in ("wrist", "eye_in_hand", "hand_camera")
        )
    ]
    if not exterior:
        raise PaperManipulationError(code)
    result = _tool(
        ctx,
        "vlm.query",
        code,
        prompt=f"Is the robot visibly holding {target_name}? Answer exactly YES or NO.",
        image=exterior[0]["rgb"],
    )
    evidence = result.get("evidence")
    if (
        not isinstance(evidence, dict)
        or evidence.get("fallback_used") is not False
        or not DIGEST.fullmatch(str(evidence.get("request_sha256", "")))
        or not DIGEST.fullmatch(str(evidence.get("response_sha256", "")))
    ):
        raise PaperManipulationError(code)
    if str(result.get("text", "")).strip().upper() != "YES":
        raise PaperManipulationError(code)
    return evidence


def _resample(trajectory: dict[str, Any], count: int) -> dict[str, Any]:
    waypoints = trajectory.get("waypoints", [])
    if not waypoints:
        raise PaperManipulationError("TRAJECTORY_EMPTY")
    if len(waypoints) == count:
        return trajectory
    positions = [point["positions"] for point in waypoints]
    sampled = []
    for output_index in range(count):
        coordinate = output_index * (len(positions) - 1) / (count - 1)
        left = int(coordinate)
        right = min(left + 1, len(positions) - 1)
        alpha = coordinate - left
        sampled.append(
            {
                "positions": [
                    (1.0 - alpha) * a + alpha * b
                    for a, b in zip(positions[left], positions[right], strict=True)
                ]
            }
        )
    return {**trajectory, "waypoints": sampled}


def _lineage_valid(lineage: dict[str, Any], role: str) -> bool:
    claimed = lineage.get("lineage_sha256")
    return lineage.get("semantic_role") == role and claimed == _hash(
        {key: value for key, value in lineage.items() if key != "lineage_sha256"}
    )


def run(
    ctx: NodeContext,
    validated_grasp_json: str,
    target_lineage_json: str,
    world_config: WorldConfig,
    target_name: str,
    preset_json: str,
) -> Output:
    values, preset_trace = _preset(preset_json)
    validated_grasp = _decode_record(validated_grasp_json, "VALIDATED_GRASP_INVALID")
    target_lineage = _decode_record(target_lineage_json, "VALIDATED_GRASP_INVALID")
    if (
        not validated_grasp.get("validated")
        or validated_grasp.get("target_lineage_sha256") != target_lineage.get("lineage_sha256")
        or validated_grasp.get("target_name") != target_name
        or not _lineage_valid(target_lineage, "target")
        or target_lineage.get("preset_trace", {}).get("preset_sha256")
        != preset_trace["preset_sha256"]
    ):
        raise PaperManipulationError("VALIDATED_GRASP_INVALID")
    if validated_grasp.get("world_config_sha256") != _hash(world_config):
        raise PaperManipulationError("COLLISION_WORLD_STALE")
    pre_execute = _tool(ctx, "robot.get_observation", "PRE_EXECUTION_OBSERVATION_FAILED")
    if _hash(pre_execute["arms"][0]["joint_state"]) != validated_grasp.get(
        "planned_start_joint_state_sha256"
    ):
        raise PaperManipulationError("PLANNED_START_STATE_STALE")
    _tool(
        ctx,
        "robot.execute_trajectory",
        "GRASP_EXECUTION_FAILED",
        trajectory=validated_grasp["trajectory"],
    )
    _tool(ctx, "robot.close_gripper", "GRASP_CLOSE_FAILED")
    closed = _tool(ctx, "robot.get_observation", "HOLD_OBSERVATION_FAILED")
    closed_evidence = _visual_hold(ctx, closed, target_name, "TARGET_NOT_HELD")
    lift = _tool(
        ctx,
        "curobo.plan_directed_linear",
        "LIFT_PLANNING_FAILED",
        start_joint_position=closed["arms"][0]["joint_state"],
        allowed_axes=["Z"],
        explicit_direction={"x": 0.0, "y": 0.0, "z": 1.0},
        distance=values["lift_distance_m"],
        endpoint_mode="DISTANCE",
        orientation_mode="LOCK",
    )
    plan_evidence = _admitted(lift, "LIFT_PLAN_EVIDENCE_UNAVAILABLE")
    if not lift.get("success"):
        raise PaperManipulationError("LIFT_PLANNING_FAILED")
    trajectory = _resample(lift["trajectory"], values["trajectory_waypoint_count"])
    robot_validation = _tool(
        ctx,
        "curobo.validate_joint_trajectory_robot",
        "ROBOT_TRAJECTORY_VALIDATION_FAILED",
        world_config=world_config,
        trajectory=trajectory,
        ignore_obstacle_names=[target_name],
    )
    robot_evidence = _admitted(robot_validation, "ROBOT_VALIDATION_EVIDENCE_UNAVAILABLE")
    held_validation = _tool(
        ctx,
        "curobo.validate_joint_trajectory_grasped",
        "HELD_OBJECT_TRAJECTORY_VALIDATION_FAILED",
        world_config=world_config,
        trajectory=trajectory,
        object_name=target_name,
    )
    held_evidence = _admitted(held_validation, "HELD_OBJECT_VALIDATION_EVIDENCE_UNAVAILABLE")
    if not robot_validation.get("success") or not held_validation.get("success"):
        raise PaperManipulationError("LIFT_TRAJECTORY_INVALID")
    _tool(ctx, "robot.execute_trajectory", "LIFT_EXECUTION_FAILED", trajectory=trajectory)
    post_lift = _tool(ctx, "robot.get_observation", "POST_LIFT_OBSERVATION_FAILED")
    post_lift_evidence = _visual_hold(ctx, post_lift, target_name, "POST_LIFT_TARGET_NOT_HELD")
    arm = post_lift["arms"][0]
    if not isinstance(arm.get("ee_pose"), dict):
        raise PaperManipulationError("POST_LIFT_POSE_UNAVAILABLE")
    result = {
        "success": True,
        "error_code": None,
        "held": True,
        "target_obb": validated_grasp["target_obb"],
        "target_name": target_name,
        "target_lineage_sha256": target_lineage["lineage_sha256"],
        "post_lift_joint_state": arm["joint_state"],
        "post_lift_ee_pose": arm["ee_pose"],
        "validation_evidence": {
            "closed_visual": closed_evidence,
            "lift_plan": plan_evidence,
            "robot": robot_evidence,
            "held_object": held_evidence,
            "post_lift_visual": post_lift_evidence,
        },
        "decision_path": [
            "execute_grasp",
            "close",
            "visual_hold_verify",
            "plan_lift",
            "validate_lift",
            "execute_lift",
            "post_lift_visual_verify",
        ],
        "fallback_used": False,
        "preset_trace": preset_trace,
    }
    result["held_grasp_json"] = _canonical(result)
    return result


_run = run


def run(
    ctx: NodeContext,
    validated_grasp_json: str,
    target_lineage_json: str,
    world_config: WorldConfig,
    target_name: str,
    preset_json: str,
) -> Output:
    try:
        return _run(
            ctx, validated_grasp_json, target_lineage_json, world_config, target_name, preset_json
        )
    except PaperManipulationError as error:
        code = error.code
    except Exception:
        code = "PAPER_MANIPULATION_INTERNAL_ERROR"
    return {
        "success": False,
        "error_code": code,
        "held": False,
        "target_obb": None,
        "target_name": target_name,
        "target_lineage_sha256": "",
        "post_lift_joint_state": {},
        "post_lift_ee_pose": None,
        "validation_evidence": {},
        "decision_path": [],
        "fallback_used": False,
        "preset_trace": {},
        "held_grasp_json": "{}",
    }
