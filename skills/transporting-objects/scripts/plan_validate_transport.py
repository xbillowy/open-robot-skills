"""Plan, validate, execute, release, and retreat to a terminal paper result."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, TypedDict

from gap import NodeContext
from gap_core.types import OrientedBoundingBox, WorldConfig

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
    def __init__(self, code: str, *, released: bool = False) -> None:
        self.code = code
        self.released = released
        super().__init__(code)


class Output(TypedDict, total=False):
    success: bool
    error_code: str | None
    status: str
    released: bool
    target_lineage_sha256: str
    destination_lineage_sha256: str
    destination_obb_sha256: str
    drop_input_binding: dict[str, Any]
    drop_input_sha256: str
    transport_input_binding: dict[str, Any]
    transport_input_sha256: str
    validation_evidence: dict[str, Any]
    decision_path: list[str]
    fallback_used: bool
    preset_trace: dict[str, Any]
    terminal_result_json: str


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


def _validate(
    ctx: NodeContext,
    world_config: dict[str, Any],
    trajectory: dict[str, Any],
    target_name: str,
    *,
    held: bool,
) -> dict[str, Any]:
    robot = _tool(
        ctx,
        "curobo.validate_joint_trajectory_robot",
        "TRANSPORT_ROBOT_VALIDATION_FAILED",
        world_config=world_config,
        trajectory=trajectory,
        ignore_obstacle_names=[target_name] if held else None,
    )
    robot_evidence = _admitted(robot, "ROBOT_VALIDATION_EVIDENCE_UNAVAILABLE")
    if not robot.get("success"):
        raise PaperManipulationError("TRANSPORT_ROBOT_COLLISION_INVALID")
    evidence = {"robot": robot_evidence}
    if held:
        attached = _tool(
            ctx,
            "curobo.validate_joint_trajectory_grasped",
            "TRANSPORT_HELD_VALIDATION_FAILED",
            world_config=world_config,
            trajectory=trajectory,
            object_name=target_name,
        )
        evidence["held_object"] = _admitted(attached, "HELD_OBJECT_VALIDATION_EVIDENCE_UNAVAILABLE")
        if not attached.get("success"):
            raise PaperManipulationError("TRANSPORT_HELD_OBJECT_COLLISION_INVALID")
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


def _last_state(trajectory: dict[str, Any]) -> dict[str, Any]:
    return trajectory["waypoints"][-1]


def _z_extent(obb: dict[str, Any]) -> float:
    extent = obb["extent"]
    return float(extent["z"] if isinstance(extent, dict) else extent[2])


def _lineage_valid(lineage: dict[str, Any], role: str) -> bool:
    claimed = lineage.get("lineage_sha256")
    return lineage.get("semantic_role") == role and claimed == _hash(
        {key: value for key, value in lineage.items() if key != "lineage_sha256"}
    )


def run(
    ctx: NodeContext,
    held_grasp_json: str,
    target_lineage_json: str,
    destination_obb: OrientedBoundingBox,
    destination_lineage_json: str,
    world_config: WorldConfig,
    target_name: str,
    preset_json: str,
) -> Output:
    values, preset_trace = _preset(preset_json)
    held_grasp = _decode_record(held_grasp_json, "TARGET_LINEAGE_INVALID")
    target_lineage = _decode_record(target_lineage_json, "TARGET_LINEAGE_INVALID")
    destination_lineage = _decode_record(destination_lineage_json, "DESTINATION_LINEAGE_INVALID")
    target_id = target_lineage.get("lineage_sha256")
    destination_id = destination_lineage.get("lineage_sha256")
    if (
        not _lineage_valid(target_lineage, "target")
        or held_grasp.get("target_lineage_sha256") != target_id
        or held_grasp.get("target_name") != target_name
        or not held_grasp.get("held")
        or target_lineage.get("preset_trace", {}).get("preset_sha256")
        != preset_trace["preset_sha256"]
    ):
        raise PaperManipulationError("TARGET_LINEAGE_INVALID")
    if (
        not _lineage_valid(destination_lineage, "destination")
        or destination_lineage.get("obb_sha256") != _hash(destination_obb)
        or destination_lineage.get("preset_trace", {}).get("preset_sha256")
        != preset_trace["preset_sha256"]
        or destination_id == target_id
    ):
        raise PaperManipulationError("DESTINATION_LINEAGE_INVALID")
    target_obb = held_grasp.get("target_obb")
    post_lift_pose = held_grasp.get("post_lift_ee_pose")
    if not isinstance(target_obb, dict) or not isinstance(post_lift_pose, dict):
        raise PaperManipulationError("HELD_GEOMETRY_INVALID")
    drop = _tool(
        ctx,
        "geometry.compute_drop_position",
        "PLACEMENT_GEOMETRY_FAILED",
        container_obb=destination_obb,
        clearance=values["approach_distance_m"],
        object_z_extent=_z_extent(target_obb),
    )
    drop_evidence = _admitted(drop, "PLACEMENT_GEOMETRY_EVIDENCE_UNAVAILABLE")
    observation = _tool(ctx, "robot.get_observation", "TRANSPORT_OBSERVATION_FAILED")
    start_state = observation["arms"][0]["joint_state"]
    drop_pose = {"position": drop["position"], "rotation": post_lift_pose["rotation"]}
    transport_inputs = {
        "world_config": world_config,
        "start_joint_position": start_state,
        "target_pose": drop_pose,
        "object_name": target_name,
        "num_ik_seeds": values["ik_seed_count"],
        "destination_obb_sha256": _hash(destination_obb),
    }
    transport = _tool(
        ctx,
        "curobo.plan_with_grasped_object",
        "TRANSPORT_PLANNING_FAILED",
        **{
            key: value for key, value in transport_inputs.items() if key != "destination_obb_sha256"
        },
    )
    transport_evidence = _admitted(transport, "TRANSPORT_PLAN_EVIDENCE_UNAVAILABLE")
    if not transport.get("success"):
        raise PaperManipulationError("TRANSPORT_PLANNING_FAILED")
    transport_trajectory = _resample(transport["trajectory"], values["trajectory_waypoint_count"])
    transport_validation = _validate(
        ctx, world_config, transport_trajectory, target_name, held=True
    )
    _tool(
        ctx,
        "robot.execute_trajectory",
        "TRANSPORT_EXECUTION_FAILED",
        trajectory=transport_trajectory,
    )
    descend = _tool(
        ctx,
        "curobo.plan_directed_linear",
        "PLACEMENT_PLANNING_FAILED",
        start_joint_position=_last_state(transport_trajectory),
        allowed_axes=["Z"],
        explicit_direction={"x": 0.0, "y": 0.0, "z": -1.0},
        distance=values["approach_distance_m"],
        endpoint_mode="DISTANCE",
        orientation_mode="LOCK",
    )
    descend_evidence = _admitted(descend, "PLACEMENT_PLAN_EVIDENCE_UNAVAILABLE")
    if not descend.get("success"):
        raise PaperManipulationError("PLACEMENT_PLANNING_FAILED")
    descend_trajectory = _resample(descend["trajectory"], values["trajectory_waypoint_count"])
    descend_validation = _validate(ctx, world_config, descend_trajectory, target_name, held=True)
    _tool(
        ctx, "robot.execute_trajectory", "PLACEMENT_EXECUTION_FAILED", trajectory=descend_trajectory
    )
    released = _tool(ctx, "robot.open_gripper", "RELEASE_FAILED")
    if not isinstance(released.get("position"), int | float) or released["position"] <= 0:
        raise PaperManipulationError("RELEASE_FAILED")
    try:
        retreat = _tool(
            ctx,
            "curobo.plan_directed_linear",
            "RETREAT_PLANNING_FAILED",
            start_joint_position=_last_state(descend_trajectory),
            allowed_axes=["Z"],
            explicit_direction={"x": 0.0, "y": 0.0, "z": 1.0},
            distance=values["approach_distance_m"],
            endpoint_mode="DISTANCE",
            orientation_mode="LOCK",
        )
        retreat_evidence = _admitted(retreat, "RETREAT_PLAN_EVIDENCE_UNAVAILABLE")
        if not retreat.get("success"):
            raise PaperManipulationError("RETREAT_PLANNING_FAILED")
        retreat_trajectory = _resample(retreat["trajectory"], values["trajectory_waypoint_count"])
        retreat_validation = _validate(
            ctx, world_config, retreat_trajectory, target_name, held=False
        )
        _tool(
            ctx,
            "robot.execute_trajectory",
            "RETREAT_EXECUTION_FAILED",
            trajectory=retreat_trajectory,
        )
    except PaperManipulationError as error:
        raise PaperManipulationError(error.code, released=True) from error
    drop_input_binding = {
        "destination_obb": destination_obb,
        "destination_obb_sha256": _hash(destination_obb),
        "target_obb": target_obb,
        "clearance": values["approach_distance_m"],
    }
    result = {
        "success": True,
        "error_code": None,
        "status": "terminal",
        "released": True,
        "target_lineage_sha256": target_id,
        "destination_lineage_sha256": destination_id,
        "destination_obb_sha256": _hash(destination_obb),
        "drop_input_binding": drop_input_binding,
        "drop_input_sha256": _hash(drop_input_binding),
        "transport_input_binding": transport_inputs,
        "transport_input_sha256": _hash(transport_inputs),
        "validation_evidence": {
            "drop": drop_evidence,
            "transport_plan": transport_evidence,
            "transport": transport_validation,
            "descend_plan": descend_evidence,
            "descend": descend_validation,
            "retreat_plan": retreat_evidence,
            "retreat": retreat_validation,
        },
        "decision_path": [
            "destination_drop_geometry",
            "held_transport_plan",
            "transport_validation",
            "transport_execute",
            "placement_plan",
            "placement_validation",
            "placement_execute",
            "release",
            "retreat_plan",
            "retreat_validation",
            "retreat_execute",
            "terminal_return",
        ],
        "fallback_used": False,
        "preset_trace": preset_trace,
    }
    result["terminal_result_json"] = _canonical(result)
    return result


_run = run


def run(
    ctx: NodeContext,
    held_grasp_json: str,
    target_lineage_json: str,
    destination_obb: OrientedBoundingBox,
    destination_lineage_json: str,
    world_config: WorldConfig,
    target_name: str,
    preset_json: str,
) -> Output:
    try:
        return _run(
            ctx,
            held_grasp_json,
            target_lineage_json,
            destination_obb,
            destination_lineage_json,
            world_config,
            target_name,
            preset_json,
        )
    except PaperManipulationError as error:
        code, released = error.code, error.released
    except Exception:
        code, released = "PAPER_MANIPULATION_INTERNAL_ERROR", False
    failure = {
        "success": False,
        "error_code": code,
        "status": "failed",
        "released": released,
        "target_lineage_sha256": "",
        "destination_lineage_sha256": "",
        "destination_obb_sha256": _hash(destination_obb),
        "drop_input_binding": {},
        "drop_input_sha256": "",
        "transport_input_binding": {},
        "transport_input_sha256": "",
        "validation_evidence": {},
        "decision_path": [],
        "fallback_used": False,
        "preset_trace": {},
    }
    failure["terminal_result_json"] = _canonical(failure)
    return failure
