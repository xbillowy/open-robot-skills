"""Verified transport and straight-Z descent, without releasing the object.

Every commanded segment must pass fresh world-frame panda_hand position
checks before another segment runs. An observed mismatch permits one
budgeted retry after destination re-perception and verified vertical recovery.
Observation failures abort immediately. Motion failures never trigger an
implicit alternate planner or an unverified advance.
"""

import math
from typing import TypedDict

import numpy as np
from gap import NodeContext
from gap_core.errors import PerceptionFailed, PlanningFailed, VerificationFailed
from gap_core.types import OrientedBoundingBox, Vec3

_DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
_HAND_OFFSET = 0.1029
_XY_TOL = 0.02
_Z_TOL = 0.015
_CHECK_READS = 3
_RECOVERY_CLEARANCE = 0.10


class Output(TypedDict):
    place_position: Vec3


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


def _position(value) -> Vec3:
    result = {axis: float(value[axis]) for axis in ("x", "y", "z")}
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError("position contains non-finite values")
    return result


def _geometry(obb):
    center = _position(obb["center"])
    extent = _position(obb["extent"])
    if any(v <= 0.0 for v in extent.values()):
        raise ValueError("container extents must be positive")
    return center, extent, center["z"] + extent["z"]


def _hand(ctx: NodeContext) -> Vec3:
    try:
        return _position(ctx.tool("robot.get_ee_pose")["pose"]["position"])
    except Exception as exc:
        raise VerificationFailed(
            "Transport observation unknown: a fresh world-frame hand position could not be read."
        ) from exc


def _matches(hand: Vec3, tcp: Vec3) -> bool:
    return (
        math.hypot(hand["x"] - tcp["x"], hand["y"] - tcp["y"]) <= _XY_TOL
        and abs(hand["z"] - (tcp["z"] + _HAND_OFFSET)) <= _Z_TOL
    )


def _verify(ctx: NodeContext, tcp: Vec3) -> bool:
    """pass on a matching read, fail after three mismatches, unknown raises.

    Timeout is a bounded observation window of three synchronous reads,
    with no polling, sleeps, or additional reads after an observation fails.
    Individual tool-call deadlines remain owned by the tool runtime.
    """
    for _ in range(_CHECK_READS):
        if _matches(_hand(ctx), tcp):
            return True
    return False


def _cartesian(ctx: NodeContext, tcp: Vec3, stage: str) -> bool:
    try:
        ctx.tool("robot.go_to_pose_cartesian", pose={
            "position": dict(tcp), "rotation": dict(_DOWN),
        })
    except Exception as exc:
        raise PlanningFailed(
            f"Transport {stage} motion failed; holding the object and aborting before release."
        ) from exc
    return _verify(ctx, tcp)


def _descend(ctx: NodeContext, above: Vec3, target: Vec3) -> bool:
    # Refresh both the start joints and actual hand position. Do not derive
    # travel distance from a commanded height or from uncorrected hand Z.
    try:
        obs = ctx.tool("robot.get_observation")
        arm = obs["arms"][0]
        hand = _position(arm["ee_pose"]["position"])
        joints = arm["joint_state"]
        if joints is None:
            raise ValueError("missing joint state")
    except Exception as exc:
        raise VerificationFailed(
            "Transport descent observation unknown: fresh hand pose and joints are unavailable."
        ) from exc
    if not _matches(hand, above):
        return False
    distance = hand["z"] - _HAND_OFFSET - target["z"]
    if distance < 0.0:
        # A fresh, verified pose can lie slightly below the intended target.
        # Accept only if that pose independently passes the final check.
        return _verify(ctx, target)
    if distance > 0.0:
        try:
            plan = ctx.tool(
                "curobo.plan_directed_linear",
                start_joint_position=joints,
                endpoint_mode="DISTANCE",
                explicit_direction={"x": 0.0, "y": 0.0, "z": -1.0},
                distance=float(distance),
                allowed_axes=["Z"],
                orientation_mode="LOCK",
            )
            if not plan.get("success") or not plan.get("trajectory"):
                raise PlanningFailed(
                    "Transport straight-Z descent has no executable trajectory."
                )
            ctx.tool("robot.execute_trajectory", trajectory=plan["trajectory"])
        except PlanningFailed:
            raise
        except Exception as exc:
            raise PlanningFailed(
                "Transport descent planning or execution failed; aborting before release."
            ) from exc
    return _verify(ctx, target)


def _staged(ctx: NodeContext, obb: OrientedBoundingBox,
            transport_z: float, lift_z: float, place_offset: float):
    center, extent, rim = _geometry(obb)
    hand = _hand(ctx)
    current_tcp_z = hand["z"] - _HAND_OFFSET
    target = {"x": center["x"], "y": center["y"],
              "z": rim - place_offset}
    # Preserve the legacy altitude inputs, but never descend during the
    # initial lift or translate below the rim clearance / final target.
    safe_z = max(transport_z, lift_z, current_tcp_z,
                 rim + _RECOVERY_CLEARANCE, target["z"] + _RECOVERY_CLEARANCE)
    lift = {"x": hand["x"], "y": hand["y"], "z": safe_z}
    if not _cartesian(ctx, lift, "vertical lift"):
        return None
    above = {"x": center["x"], "y": center["y"], "z": safe_z}
    if not _cartesian(ctx, above, "lateral transit"):
        return None
    if not _descend(ctx, above, target):
        return None
    return target


def _refresh_destination(ctx: NodeContext, previous: OrientedBoundingBox):
    """Re-perceive a uniquely associated container from fresh RGBD.

    Use a generic category query, then associate by the supplied container's
    geometry. Ambiguity, missing observations, or any tool failure aborts;
    the old OBB is never substituted for failed fresh perception.
    """
    try:
        old_center, old_extent, old_rim = _geometry(previous)
        obs = ctx.tool("robot.get_observation")
        cameras = list(obs["cameras"])
        cam = next(c for c in cameras
                   if "eye_in_hand" not in str(c.get("name", "")))
        segmentation = ctx.tool(
            "sam3.segment_text", image=cam["rgb"],
            query="container. basket. bin. box. tote. cup.",
        )
        masks = list(segmentation["masks"])
        scores = list(segmentation["scores"])
        if len(masks) != len(scores):
            raise ValueError("inconsistent segmentation output")
        candidates = []
        association_radius = max(old_extent["x"], old_extent["y"]) + 0.05
        for mask, score in zip(masks, scores):
            score = float(score)
            if not math.isfinite(score):
                raise ValueError("non-finite segmentation score")
            if score < 0.30:
                continue
            cloud = ctx.tool(
                "geometry.mask_to_world_points", mask=mask,
                depth=cam["depth"], intrinsics=cam["intrinsics"],
                camera_pose=cam["pose"],
            )["points"]
            points = np.asarray(cloud["points"], dtype=float)
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError("invalid world point cloud")
            if not np.isfinite(points).all():
                raise ValueError("non-finite world point cloud")
            if len(points) < 30:
                continue
            obb = ctx.tool("geometry.filter_and_compute_obb", points=cloud)["obb"]
            center, extent, rim = _geometry(obb)
            displacement = math.hypot(center["x"] - old_center["x"],
                                      center["y"] - old_center["y"])
            old_widths = sorted((old_extent["x"], old_extent["y"]))
            widths = sorted((extent["x"], extent["y"]))
            size_matches = all(0.5 <= new / old <= 2.0
                               for new, old in zip(widths, old_widths))
            if (displacement <= association_radius
                    and abs(rim - old_rim) <= 0.10 and size_matches):
                candidates.append((displacement, obb))
        candidates.sort(key=lambda item: item[0])
        if not candidates:
            raise PerceptionFailed("Destination refresh found no associated container.")
        if len(candidates) > 1 and candidates[1][0] - candidates[0][0] <= 0.02:
            raise PerceptionFailed("Destination refresh is ambiguous between container candidates.")
        return candidates[0][1]
    except PerceptionFailed:
        raise
    except Exception as exc:
        raise PerceptionFailed(
            "Destination refresh failed; transport cannot retry using stale geometry."
        ) from exc


def run(
    ctx: NodeContext,
    container_obb: OrientedBoundingBox,
    transport_z: float = 0.353,
    lift_z: float = 0.25,
    place_offset: float = 0.06,
) -> Output:
    try:
        transport_z, lift_z, place_offset = (
            float(transport_z), float(lift_z), float(place_offset)
        )
        if not all(math.isfinite(v) for v in (transport_z, lift_z, place_offset)):
            raise PlanningFailed("Transport altitude inputs must be finite.")
        _geometry(container_obb)
        result = _staged(ctx, container_obb, transport_z, lift_z, place_offset)
        if result is not None:
            return {"place_position": result}

        # Only an observed position mismatch reaches this transition.
        # Unknown observations and motion exceptions have already raised.
        if not _fallback_budget(ctx):
            raise VerificationFailed(
                "Transport pose mismatch; the episode fallback budget is exhausted."
            )
        refreshed = _refresh_destination(ctx, container_obb)
        hand = _hand(ctx)
        _, _, rim = _geometry(refreshed)
        recovery = {
            "x": hand["x"], "y": hand["y"],
            "z": max(hand["z"] - _HAND_OFFSET + _RECOVERY_CLEARANCE,
                     rim + _RECOVERY_CLEARANCE,
                     rim - place_offset + _RECOVERY_CLEARANCE),
        }
        if not _cartesian(ctx, recovery, "vertical recovery"):
            raise VerificationFailed(
                "Transport vertical recovery did not reach its verified clearance pose."
            )
        result = _staged(ctx, refreshed, transport_z, lift_z, place_offset)
        if result is None:
            raise VerificationFailed(
                "Transport pose mismatch persisted during the single refreshed retry."
            )
        return {"place_position": result}
    except (PlanningFailed, PerceptionFailed, VerificationFailed):
        raise
    except Exception as exc:
        raise PlanningFailed(
            "Transport could not complete safely; aborting before downstream release."
        ) from exc
