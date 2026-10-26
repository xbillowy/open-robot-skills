"""Verified top-down grasp descent with one bounded planner fallback.

Every commanded segment requires a fresh public panda_hand pose within
0.02 m of its world-frame TCP target plus the 0.1029 m vertical offset,
and within 0.15 rad of the requested orientation. Verification times out
after three observation reads; there is no wall-clock timeout. A failed
observation is UNKNOWN and immediately exits without recovery or fallback.

Only a known primary motion/planning failure or observed endpoint mismatch
can enter recovery. Recovery refreshes hand pose and target perception,
then completes a verified vertical rise before the shared budget can permit
one planner fallback. Success means a verified grasp pose, not a verified
held object. This script never opens or closes the gripper.
"""

import math
from typing import TypedDict

import numpy as np
from scipy.spatial.transform import Rotation

from gap import NodeContext
from gap_core.errors import PlanningFailed, PerceptionFailed, VerificationFailed
from gap_core.types import OrientedBoundingBox, Se3Pose

_HAND_OFFSET = 0.1029
_POSITION_TOLERANCE = 0.02
_ROTATION_TOLERANCE = 0.15
_OBSERVATION_READS = 3
_CLEARANCE = 0.15
_DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}


class Output(TypedDict):
    done: bool


def _fallback_budget(ctx, limit: int = 3) -> bool:
    """Episode-shared fallback budget (Apply v6 sec. 2.2): True (and spend one) when a fallback may still run."""
    import builtins
    import os

    reg = builtins.__dict__.setdefault("_ors_fallback_budget", {})
    key = (os.getpid(), id(getattr(ctx, "_trace", None) or ctx))
    left = reg.get(key, limit)
    if left <= 0:
        return False
    reg[key] = left - 1
    return True


class _PrimaryFailure(Exception):
    """Known primary failure eligible for the single recovery transition."""


def _xyz(value):
    result = np.array([float(value[k]) for k in ("x", "y", "z")])
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite position")
    return result


def _rotation(value):
    q = np.array([float(value[k]) for k in ("x", "y", "z", "w")])
    if not np.isfinite(q).all() or abs(float(np.linalg.norm(q)) - 1.0) > 0.01:
        raise ValueError("Invalid quaternion")
    return Rotation.from_quat(q)


def _pose(position, rotation):
    result = {
        "position": dict(zip(("x", "y", "z"), map(float, position))),
        "rotation": dict(rotation),
    }
    return result


def _refreshed_obb(center, extent):
    return {"center": dict(zip(("x", "y", "z"), map(float, center))),
            "extent": dict(extent)}


def _top_down(rotation):
    return float(_rotation(rotation).as_matrix()[2, 2]) <= -0.98


def _read_hand(ctx):
    try:
        hand = ctx.tool("robot.get_ee_pose")["pose"]
        _xyz(hand["position"])
        _rotation(hand["rotation"])
        return hand
    except Exception as exc:
        raise VerificationFailed("Hand-pose observation is unknown; stopping without fallback") from exc


def _check(ctx, target):
    """Return (pass/fail/unknown, last hand); unknown stops on its first read."""
    expected = _xyz(target["position"]) + np.array([0.0, 0.0, _HAND_OFFSET])
    desired = _rotation(target["rotation"])
    for _ in range(_OBSERVATION_READS):
        try:
            hand = ctx.tool("robot.get_ee_pose")["pose"]
            actual = _xyz(hand["position"])
            orientation = _rotation(hand["rotation"])
            position_error = float(np.linalg.norm(actual - expected))
            rotation_error = float((desired.inv() * orientation).magnitude())
        except Exception:
            return "unknown", None
        if position_error <= _POSITION_TOLERANCE and rotation_error <= _ROTATION_TOLERANCE:
            return "pass", hand
    return "fail", hand


def _command(ctx, target, tool, **kwargs):
    motion_failed = False
    try:
        ctx.tool(tool, **kwargs)
    except Exception:
        motion_failed = True
    # Even a motion that reports an error may have moved the robot. Its
    # endpoint must be observed before deciding whether recovery is allowed.
    result, hand = _check(ctx, target)
    if result == "unknown":
        raise VerificationFailed("Post-motion observation is unknown; stopping without fallback")
    if result == "fail":
        raise _PrimaryFailure("Endpoint mismatch after three public pose reads")
    if motion_failed:
        raise _PrimaryFailure("Motion tool reported a failure despite an observed endpoint")
    return hand


def _cartesian(ctx, target):
    return _command(ctx, target, "robot.go_to_pose_cartesian", pose=target)


def _observation(ctx):
    try:
        obs = ctx.tool("robot.get_observation")
        joints = obs["arms"][0]["joint_state"]
        if joints is None:
            raise ValueError("Missing joint state")
        return obs
    except Exception as exc:
        raise VerificationFailed("Robot observation is unknown; stopping without fallback") from exc


def _geometry(obb):
    center = _xyz(obb["center"])
    extent = _xyz(obb["extent"])
    if np.any(extent <= 0.0):
        raise ValueError("Target extents must be positive")
    orientation = obb.get("orientation", {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0})
    matrix = _rotation(orientation).as_matrix()
    half_height = float(np.abs(matrix[2, :]) @ extent)
    return center, extent, matrix, half_height


def _grasp_target(pose, obb):
    center, _, _, half_height = _geometry(obb)
    height = 2.0 * half_height
    top, base = center[2] + half_height, center[2] - half_height
    deepen = min(0.035, 0.5 * height)
    floor = min(0.012, max(0.006, 0.35 * height))
    position = _xyz(pose["position"])
    position[2] = max(top - deepen, base + floor)
    rotation = pose.get("rotation") or _DOWN
    if not _top_down(rotation):
        raise PlanningFailed("Grasp orientation must be top-down")
    return _pose(position, rotation)


def _clearance_z(obb, hover_z):
    center, _, _, half_height = _geometry(obb)
    value = float(hover_z)
    if not math.isfinite(value):
        raise ValueError("Invalid hover height")
    return max(value, float(center[2]) + half_height + _CLEARANCE)


def _refresh_target(ctx, obb):
    """One fresh wrist segmentation; ambiguity or unavailable data is terminal.

    Preserve dimensions and orientation, updating the center from the top slab.
    No original-target or unsegmented-cloud fallback is allowed.
    """
    obs = _observation(ctx)
    try:
        center, extent, matrix, half_height = _geometry(obb)
        cameras = obs["cameras"]
        if isinstance(cameras, dict):
            cameras = list(cameras.values())
        cam = next(c for c in cameras if "eye_in_hand" in (c.get("name") or ""))
        depth = np.asarray(cam["depth"])
        top = center.copy()
        top[2] += half_height
        projected = _rotation(cam["pose"]["rotation"]).inv().apply(
            top - _xyz(cam["pose"]["position"])
        )
        if projected[2] <= 0.01:
            raise ValueError("Target is behind the wrist camera")
        pixel = np.asarray(cam["intrinsics"], dtype=float) @ projected
        u, v = float(pixel[0] / pixel[2]), float(pixel[1] / pixel[2])
        if not (math.isfinite(u) and math.isfinite(v)
                and 0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
            raise ValueError("Target is outside the wrist image")
        masks = ctx.tool("sam3.segment_point", image=cam["rgb"], pixel_x=u, pixel_y=v)["masks"]
        if len(masks) == 0:
            raise ValueError("Empty target segmentation")
        cloud = ctx.tool(
            "geometry.mask_to_world_points", mask=masks[0], depth=cam["depth"],
            intrinsics=cam["intrinsics"], camera_pose=cam["pose"],
        )["points"]["points"]
        points = np.asarray(cloud, dtype=float).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) < 30:
            raise ValueError("Sparse target cloud")
        local = (points - center) @ matrix
        inside = np.all(np.abs(local) <= extent + 0.03, axis=1)
        if float(np.mean(inside)) < 0.8:
            raise ValueError("Segmentation does not match the target region")
        points = points[inside]
        if len(points) < 30:
            raise ValueError("Sparse target region")
        observed_top = float(np.percentile(points[:, 2], 95.0))
        slab = points[points[:, 2] >= observed_top - 0.02]
        if len(slab) < 15:
            raise ValueError("Sparse target top slab")
        refreshed_center = center.copy()
        refreshed_center[:2] = np.mean(slab[:, :2], axis=0)
        refreshed_center[2] += observed_top - top[2]
        if float(np.linalg.norm(refreshed_center[:2] - center[:2])) > 0.03:
            raise ValueError("Ambiguous target horizontal displacement")
        if abs(float(refreshed_center[2] - center[2])) > 0.04:
            raise ValueError("Ambiguous target height displacement")
        refreshed = _refreshed_obb(refreshed_center, obb["extent"])
        if "orientation" in obb:
            refreshed["orientation"] = dict(obb["orientation"])
        return refreshed
    except Exception as exc:
        raise PerceptionFailed("Target refresh failed; planner fallback is prohibited") from exc


def _primary(ctx, target, obb, hover_z):
    hand = _read_hand(ctx)
    if not _top_down(hand["rotation"]):
        raise VerificationFailed("Initial hand is not top-down; TCP offset is unverified")
    current = _xyz(hand["position"]) - np.array([0.0, 0.0, _HAND_OFFSET])
    safe_z = max(_clearance_z(obb, hover_z), float(current[2]))
    rise = current.copy()
    rise[2] = safe_z
    _cartesian(ctx, _pose(rise, hand["rotation"]))
    above = _xyz(target["position"])
    above[2] = safe_z
    approach_target = _pose(above, target["rotation"])
    _cartesian(ctx, approach_target)
    obs = _observation(ctx)
    result, hand = _check(ctx, approach_target)
    if result == "unknown":
        raise VerificationFailed("Approach observation is unknown; descent is prohibited")
    if result == "fail":
        raise _PrimaryFailure("Approach no longer matches its verified target")
    distance = float(hand["position"]["z"]) - _HAND_OFFSET - float(target["position"]["z"])
    if distance <= 0.0:
        raise _PrimaryFailure("No valid downward descent remains")
    try:
        plan = ctx.tool(
            "curobo.plan_directed_linear",
            start_joint_position=obs["arms"][0]["joint_state"],
            endpoint_mode="DISTANCE",
            explicit_direction={"x": 0.0, "y": 0.0, "z": -1.0},
            distance=distance, allowed_axes=["Z"], orientation_mode="LOCK",
        )
        if not plan.get("success") or not plan.get("trajectory"):
            raise _PrimaryFailure("Directed descent planning failed")
    except _PrimaryFailure:
        raise
    except Exception as exc:
        raise _PrimaryFailure("Directed descent planner failed") from exc
    _command(ctx, target, "robot.execute_trajectory", trajectory=plan["trajectory"])


def _recover_and_fallback(ctx, candidate_poses, obb, hover_z):
    _read_hand(ctx)
    refreshed = _refresh_target(ctx, obb)
    hand = _read_hand(ctx)
    if not _top_down(hand["rotation"]):
        raise VerificationFailed("Recovery hand is not top-down; vertical recovery is prohibited")
    current = _xyz(hand["position"]) - np.array([0.0, 0.0, _HAND_OFFSET])
    recovery = current.copy()
    recovery[2] = max(float(current[2]), _clearance_z(refreshed, hover_z))
    recovery_target = _pose(recovery, hand["rotation"])
    try:
        _cartesian(ctx, recovery_target)
    except _PrimaryFailure as exc:
        raise VerificationFailed("Vertical recovery failed or its clearance was not verified") from exc

    old_center, _, _, _ = _geometry(obb)
    new_center, _, _, _ = _geometry(refreshed)
    shift = new_center - old_center
    poses = []
    for pose in list(candidate_poses)[:8]:
        translated = _pose(_xyz(pose["position"]) + shift, pose.get("rotation") or _DOWN)
        poses.append(_grasp_target(translated, refreshed))
    if not poses:
        raise PlanningFailed("No refreshed grasp candidates are available")

    if not _fallback_budget(ctx):
        raise PlanningFailed("Episode-shared fallback budget is exhausted")
    obs = _observation(ctx)
    result, _ = _check(ctx, recovery_target)
    if result != "pass":
        raise VerificationFailed("Recovery clearance could not be reconfirmed; fallback is prohibited")
    try:
        world = ctx.tool(
            "geometry.build_world_config", cameras=obs["cameras"],
            robot_joint_state=obs["arms"][0]["joint_state"],
            target_obb=refreshed, target_obb_name="target",
        )["config"]
    except Exception as exc:
        raise PerceptionFailed("Fresh collision-world construction failed") from exc

    # Candidate planning is one fallback attempt. At most one trajectory
    # is executed; an execution error or mismatch never tries another pose.
    for target in poses:
        try:
            plan = ctx.tool(
                "curobo.plan_to_grasp_poses", world_config=world,
                start_joint_position=obs["arms"][0]["joint_state"],
                grasp_poses=[target], grasp_pose_is_fingertip=True,
                use_world_collision=True, use_cuda_graph=False,
                robot_collision_sphere_buffer=-0.01,
                collision_activation_distance=0.005,
                ignore_obstacle_names=["target"], use_grasp_approach=False,
                num_ik_seeds=128,
            )
            success = plan.get("success") and plan.get("trajectory")
        except Exception as exc:
            raise PlanningFailed("Planner fallback tool failed") from exc
        if not success:
            continue
        try:
            _command(ctx, target, "robot.execute_trajectory", trajectory=plan["trajectory"])
        except _PrimaryFailure as exc:
            raise VerificationFailed("Planner fallback execution failed or its endpoint was not verified") from exc
        return
    raise PlanningFailed("Planner fallback found no feasible refreshed candidate")


def run(
    ctx: NodeContext,
    grasp_pose: Se3Pose,
    candidate_poses: list[Se3Pose],
    target_obb: OrientedBoundingBox,
    hover_z: float = 0.2,
) -> Output:
    try:
        target = _grasp_target(grasp_pose, target_obb)
        try:
            _primary(ctx, target, target_obb, hover_z)
        except _PrimaryFailure:
            _recover_and_fallback(ctx, candidate_poses, target_obb, hover_z)
        return {"done": True}
    except (PlanningFailed, PerceptionFailed, VerificationFailed):
        raise
    except Exception as exc:
        raise PlanningFailed("Verified grasp procedure failed; downstream closure is prohibited") from exc
