"""Descend to a TCP drop pose and verify world-frame XY before release.

Frozen sequence: descend -> check -> optional refresh and one retry ->
check -> release -> vertical clearance lift -> home. Arrival requires a
finite public panda_hand XY measurement within 0.012 m of commanded TCP
XY. For a top-down hand these XY coordinates are directly comparable.
Checks take one immediate snapshot, with zero additional convergence-wait
or polling time. Observation failures are unknown and abort immediately.

The retry uses a fresh pose observation and the unchanged drop target.
No recovery motion is required before this direct Cartesian reissue; the
Cartesian tool plans from the robot's current configuration. Motion errors
abort rather than triggering a retry. A failed final check or exhausted
shared budget aborts without opening the gripper. Ordinary exceptions map
to the skill failure exit; BaseException safety guards propagate unchanged.

Inputs and returned keys match descend_release.
"""

import math
from typing import TypedDict

from gap import NodeContext
from gap_core.errors import PlanningFailed, VerificationFailed
from gap_core.types import Quaternion, Se3Pose, Vec3


class Output(TypedDict):
    drop_position: Vec3


_DOWN: Quaternion = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
_XY_TOLERANCE_M = 0.012


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


def _refresh_xy_error(ctx: NodeContext, target: Vec3) -> float:
    """Refresh public pose before retry; unknown observations never retry."""
    try:
        position = ctx.tool("robot.get_ee_pose")["pose"]["position"]
        x = float(position["x"])
        y = float(position["y"])
        if not all(math.isfinite(value) for value in (x, y)):
            raise ValueError("Non-finite public hand XY")
        error = math.hypot(x - target["x"], y - target["y"])
        if not math.isfinite(error):
            raise ValueError("Non-finite XY error")
        return error
    except Exception as exc:
        raise VerificationFailed(
            "descend_release_linear: arrival observation unknown during "
            "retry refresh; stopped without releasing."
        ) from exc


def run(
    ctx: NodeContext,
    drop_position: Vec3,
    drop_rotation: Quaternion | None = None,
) -> Output:
    stage = "drop pose preparation"
    try:
        target: Vec3 = {
            axis: float(drop_position[axis]) for axis in ("x", "y", "z")
        }
        if not all(math.isfinite(value) for value in target.values()):
            raise ValueError("Drop position must be finite")
        source_rotation = drop_rotation if drop_rotation is not None else _DOWN
        rotation: Quaternion = {
            axis: float(source_rotation[axis]) for axis in ("w", "x", "y", "z")
        }
        if not all(math.isfinite(value) for value in rotation.values()):
            raise ValueError("Drop rotation must be finite")
        if sum(value * value for value in rotation.values()) <= 0.0:
            raise ValueError("Drop rotation must be nonzero")
        end_pose: Se3Pose = {"position": dict(target), "rotation": dict(rotation)}

        stage = "Cartesian descent"
        ctx.tool("robot.go_to_pose_cartesian", pose=end_pose)

        stage = "pre-release arrival verification"
        verified = False
        for _ in range(2):
            try:
                cur = ctx.tool("robot.get_ee_pose")["pose"]["position"]
                if not all(math.isfinite(float(cur[axis])) for axis in ("x", "y")):
                    raise ValueError("Non-finite public hand XY")
                err_xy = math.hypot(
                    float(cur["x"]) - float(drop_position["x"]),
                    float(cur["y"]) - float(drop_position["y"]),
                )
                if not math.isfinite(err_xy):
                    raise ValueError("Non-finite XY error")
            except Exception as exc:
                raise VerificationFailed(
                    "descend_release_linear: arrival observation unknown; "
                    "stopped without releasing or retrying."
                ) from exc
            if err_xy <= 0.012:
                verified = True
                break

            if _ == 1:
                raise VerificationFailed(
                    "descend_release_linear: world-frame hand XY remains "
                    "more than 0.012 m from drop XY after one retry; "
                    "stopped without releasing."
                )

            stage = "retry input refresh"
            refreshed_error = _refresh_xy_error(ctx, target)
            if refreshed_error <= _XY_TOLERANCE_M:
                verified = True
                break
            if not _fallback_budget(ctx):
                raise VerificationFailed(
                    "descend_release_linear: arrival failed and episode "
                    "fallback budget is exhausted; stopped without releasing."
                )

            # Refresh completed. No recovery motion precedes this direct
            # retry; reconstruct the same target and plan from live state.
            end_pose = {"position": dict(target), "rotation": dict(rotation)}
            stage = "single Cartesian descent retry"
            ctx.tool("robot.go_to_pose_cartesian", pose=end_pose)
            stage = "post-retry arrival verification"

        if not verified:
            raise VerificationFailed(
                "descend_release_linear: arrival was not verified; "
                "stopped without releasing."
            )

        stage = "gripper opening and settling"
        ctx.tool("robot.open_gripper", settle_steps=60)

        stage = "vertical clearance lift"
        retract_pose: Se3Pose = {
            "position": {
                "x": target["x"],
                "y": target["y"],
                "z": target["z"] + 0.14,
            },
            "rotation": rotation,
        }
        ctx.tool("robot.go_to_pose_cartesian", pose=retract_pose)

        stage = "home retract"
        ctx.tool("robot.go_home")
        return {"drop_position": drop_position}
    except VerificationFailed:
        raise
    except Exception as exc:
        raise PlanningFailed(
            f"descend_release_linear: {stage} failed; placement sequence stopped."
        ) from exc
