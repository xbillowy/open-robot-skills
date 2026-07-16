"""Generate multiple grasps and admit only a fully validated paper plan."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, TypedDict

from gap import NodeContext
from gap_core.types import OrientedBoundingBox, Se3Pose, Trajectory, WorldConfig

EXPECTED = {
    "approach_distance_m",
    "grasp_candidate_count",
    "ik_seed_count",
    "lift_distance_m",
    "trajectory_waypoint_count",
}
CONSUMED_PRESET_FIELDS = frozenset(
    {
        "approach_distance_m",
        "grasp_candidate_count",
        "ik_seed_count",
        "lift_distance_m",
        "trajectory_waypoint_count",
    }
)
RESPONSIBLE_PRESET_FIELDS = frozenset(
    {"approach_distance_m", "grasp_candidate_count", "ik_seed_count"}
)
DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
SEALED_PRESET_SHA256 = "sha256:8f6f81c9f2880fe0e3f786e276511868142ca8033255d6b598d82baad22b77d9"


class PaperManipulationError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class Output(TypedDict, total=False):
    success: bool
    error_code: str | None
    validated: bool
    candidate_index: int
    grasp_pose: Se3Pose
    approach_pose: Se3Pose
    retreat_pose: Se3Pose
    trajectory: Trajectory
    validated_retreat_trajectory: Trajectory
    planned_start_joint_state: dict[str, Any]
    planned_start_joint_state_sha256: str
    world_config_sha256: str
    target_obb: OrientedBoundingBox
    target_name: str
    target_lineage_sha256: str
    validation_evidence: dict[str, Any]
    candidate_records: list[dict[str, Any]]
    decision_path: list[str]
    fallback_used: bool
    preset_trace: dict[str, Any]
    validated_grasp_json: str


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
        {
            row["name"]: row["runtime_value"]
            for row in rows
            if row["name"] in CONSUMED_PRESET_FIELDS
        },
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
        if name == "geometry.top_down_grasp_candidates":
            return ctx.tool("geometry.top_down_grasp_candidates", **kwargs)
        if name == "robot.get_observation":
            return ctx.tool("robot.get_observation", **kwargs)
        if name == "curobo.batch_grasp_feasibility":
            return ctx.tool("curobo.batch_grasp_feasibility", **kwargs)
        if name == "curobo.plan_to_grasp_poses":
            return ctx.tool("curobo.plan_to_grasp_poses", **kwargs)
        if name == "curobo.validate_joint_trajectory_robot":
            return ctx.tool("curobo.validate_joint_trajectory_robot", **kwargs)
        if name == "curobo.validate_joint_trajectory_grasped":
            return ctx.tool("curobo.validate_joint_trajectory_grasped", **kwargs)
        raise PaperManipulationError("UNDECLARED_TOOL_DISPATCH")
    except Exception as error:
        raise PaperManipulationError(code) from error


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


def _offset_pose(pose: dict[str, Any], z_offset: float) -> dict[str, Any]:
    result = {**pose, "position": dict(pose["position"])}
    result["position"]["z"] = float(result["position"]["z"]) + z_offset
    return result


def _reverse_trajectory(trajectory: dict[str, Any]) -> dict[str, Any]:
    """Start attachment validation at the grasp endpoint, then retreat."""
    return {**trajectory, "waypoints": list(reversed(trajectory["waypoints"]))}


def _lineage_valid(lineage: dict[str, Any], role: str) -> bool:
    claimed = lineage.get("lineage_sha256")
    payload = {key: value for key, value in lineage.items() if key != "lineage_sha256"}
    return lineage.get("semantic_role") == role and claimed == _hash(payload)


def _run_impl(
    ctx: NodeContext,
    target_obb: OrientedBoundingBox,
    target_lineage_json: str,
    world_config: WorldConfig,
    target_name: str,
    preset_json: str,
) -> Output:
    values, preset_trace = _preset(preset_json)
    target_lineage = _decode_record(target_lineage_json, "TARGET_LINEAGE_INVALID")
    if (
        not _lineage_valid(target_lineage, "target")
        or target_lineage.get("obb_sha256") != _hash(target_obb)
        or target_lineage.get("preset_trace", {}).get("preset_sha256")
        != preset_trace["preset_sha256"]
    ):
        raise PaperManipulationError("TARGET_LINEAGE_INVALID")
    candidate_count = values["grasp_candidate_count"]
    generated = _tool(
        ctx,
        "geometry.top_down_grasp_candidates",
        "GRASP_CANDIDATE_GENERATION_FAILED",
        obb=target_obb,
    )
    candidate_evidence = _admitted(generated, "GRASP_CANDIDATE_EVIDENCE_UNAVAILABLE")
    candidates_value = generated.get("candidates", generated.get("poses", []))
    candidates = (
        candidates_value.get("poses", [])
        if isinstance(candidates_value, dict)
        else candidates_value
    )
    if len(candidates) < candidate_count or candidate_count <= 1:
        raise PaperManipulationError("GRASP_CANDIDATES_INSUFFICIENT")
    candidates = candidates[:candidate_count]
    observation = _tool(ctx, "robot.get_observation", "GRASP_OBSERVATION_FAILED")
    start_state = observation["arms"][0]["joint_state"]
    if target_name not in {
        mesh.get("name") for mesh in world_config.get("meshes", []) if isinstance(mesh, dict)
    }:
        raise PaperManipulationError("TARGET_MESH_MISSING")
    feasibility = _tool(
        ctx,
        "curobo.batch_grasp_feasibility",
        "IK_FEASIBILITY_FAILED",
        world_config=world_config,
        start_state=start_state,
        grasp_poses=candidates,
        approach_offset_m=values["approach_distance_m"],
        num_ik_seeds=values["ik_seed_count"],
        ignore_obstacle_names=[target_name],
    )
    ik_evidence = _admitted(feasibility, "IK_EVIDENCE_UNAVAILABLE")
    feasible = feasibility.get("feasible", [])
    candidate_records: list[dict[str, Any]] = []
    admitted_candidates: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        record = {
            "candidate_index": index,
            "candidate_sha256": _hash(candidate),
            "approach_pose": _offset_pose(candidate, values["approach_distance_m"]),
            "retreat_pose": _offset_pose(candidate, values["lift_distance_m"]),
        }
        if index >= len(feasible) or not feasible[index]:
            candidate_records.append(
                {**record, "status": "rejected", "rejection_code": "IK_OR_APPROACH_INFEASIBLE"}
            )
            continue
        planned = _tool(
            ctx,
            "curobo.plan_to_grasp_poses",
            "GRASP_PLANNING_FAILED",
            world_config=world_config,
            start_joint_position=start_state,
            grasp_poses=[candidate],
            grasp_pose_is_fingertip=True,
            num_ik_seeds=values["ik_seed_count"],
            use_grasp_approach=True,
            grasp_approach_offset=values["approach_distance_m"],
            use_world_collision=True,
            ignore_obstacle_names=[target_name],
        )
        plan_evidence = _admitted(planned, "GRASP_PLAN_EVIDENCE_UNAVAILABLE")
        if not planned.get("success"):
            candidate_records.append(
                {**record, "status": "rejected", "rejection_code": "GRASP_PLANNING_FAILED"}
            )
            continue
        trajectory = _resample(planned["trajectory"], values["trajectory_waypoint_count"])
        robot_validation = _tool(
            ctx,
            "curobo.validate_joint_trajectory_robot",
            "ROBOT_TRAJECTORY_VALIDATION_FAILED",
            world_config=world_config,
            trajectory=trajectory,
            ignore_obstacle_names=[target_name],
        )
        robot_evidence = _admitted(robot_validation, "ROBOT_VALIDATION_EVIDENCE_UNAVAILABLE")
        if not robot_validation.get("success"):
            candidate_records.append(
                {**record, "status": "rejected", "rejection_code": "ROBOT_COLLISION_INVALID"}
            )
            continue
        retreat_trajectory = _reverse_trajectory(trajectory)
        held_validation = _tool(
            ctx,
            "curobo.validate_joint_trajectory_grasped",
            "HELD_OBJECT_TRAJECTORY_VALIDATION_FAILED",
            world_config=world_config,
            trajectory=retreat_trajectory,
            object_name=target_name,
        )
        held_evidence = _admitted(held_validation, "HELD_OBJECT_VALIDATION_EVIDENCE_UNAVAILABLE")
        if not held_validation.get("success"):
            candidate_records.append(
                {**record, "status": "rejected", "rejection_code": "HELD_OBJECT_COLLISION_INVALID"}
            )
            continue
        accepted = {
            **record,
            "status": "accepted",
            "rejection_code": None,
            "trajectory": trajectory,
            "validated_retreat_trajectory": retreat_trajectory,
            "validation_evidence": {
                "plan": plan_evidence,
                "robot": robot_evidence,
                "held_object": held_evidence,
            },
        }
        candidate_records.append(accepted)
        admitted_candidates.append(accepted)
    if admitted_candidates:
        selected = admitted_candidates[0]
        index = selected["candidate_index"]
        result = {
            "success": True,
            "error_code": None,
            "validated": True,
            "candidate_index": index,
            "grasp_pose": candidates[index],
            "approach_pose": selected["approach_pose"],
            "retreat_pose": selected["retreat_pose"],
            "trajectory": selected["trajectory"],
            "validated_retreat_trajectory": selected["validated_retreat_trajectory"],
            "planned_start_joint_state": start_state,
            "planned_start_joint_state_sha256": _hash(start_state),
            "world_config_sha256": _hash(world_config),
            "target_obb": target_obb,
            "target_name": target_name,
            "target_lineage_sha256": target_lineage["lineage_sha256"],
            "validation_evidence": {
                "candidate": candidate_evidence,
                "ik": ik_evidence,
                **selected["validation_evidence"],
            },
            "candidate_records": candidate_records,
            "decision_path": [
                "generate_candidates",
                "batch_ik",
                "motion_plan",
                "robot_validation",
                "held_object_validation",
                "trajectory_admission",
            ],
            "fallback_used": False,
            "preset_trace": preset_trace,
        }
        result["validated_grasp_json"] = _canonical(result)
        return result
    raise PaperManipulationError("NO_FULLY_VALIDATED_GRASP")


def run(
    ctx: NodeContext,
    target_obb: OrientedBoundingBox,
    target_lineage_json: str,
    world_config: WorldConfig,
    target_name: str,
    preset_json: str,
) -> Output:
    try:
        return _run_impl(
            ctx, target_obb, target_lineage_json, world_config, target_name, preset_json
        )
    except PaperManipulationError as error:
        code = error.code
    except Exception:
        code = "PAPER_MANIPULATION_INTERNAL_ERROR"
    return {
        "success": False,
        "error_code": code,
        "validated": False,
        "candidate_index": -1,
        "grasp_pose": None,
        "approach_pose": None,
        "retreat_pose": None,
        "trajectory": None,
        "validated_retreat_trajectory": None,
        "planned_start_joint_state": {},
        "planned_start_joint_state_sha256": "",
        "world_config_sha256": "",
        "target_obb": target_obb,
        "target_name": target_name,
        "target_lineage_sha256": "",
        "validation_evidence": {},
        "candidate_records": [],
        "decision_path": [],
        "fallback_used": False,
        "preset_trace": {},
        "validated_grasp_json": "{}",
    }
