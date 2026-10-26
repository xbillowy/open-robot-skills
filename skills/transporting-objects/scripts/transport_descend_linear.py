"""Transport with verified lift, translation, and axis-locked descent.

Frozen sequence per leg: command -> immediate public arrival snapshot ->
pass, unknown/abort, or fail -> refresh -> at most one bounded correction
across this entire script call -> immediate snapshot -> pass or abort.
Only successful final verification publishes place_position for release.

Positions are world-frame TCP positions. Public panda_hand positions are
converted by subtracting 0.1029 m in world Z. Arrival requires position
error < 0.014 m and normalized abs(quaternion.x) > 0.995. Each check is
one immediate snapshot with zero additional polling/convergence wait.
Observation errors are unknown and abort without fallback. Motion errors
also abort, including errors after partial execution. Safety guards are
not caught.

The supplied move implementation is preserved verbatim. Its context adapter
terminates each invocation at its first observation and delegates checks to
the frozen wrapper, preventing its internal four-attempt loop from spending
extra retries. The adapter supplies the wrapper's bounded command and uses
an axis-locked planner for descent. Corrections use freshly observed state;
no separate recovery motion is required because each motion replans from
current robot state. There is no planned-pose or skipped-lift fallback.
"""

import math
from typing import TypedDict

import numpy as np
from gap import NodeContext
from gap_core.errors import PlanningFailed, VerificationFailed
from gap_core.types import OrientedBoundingBox, Vec3

_DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
_HAND_TO_TCP_M = 0.1029
_POSITION_TOLERANCE_M = 0.014
_MAX_CORRECTION_M = 0.04
_ORIENTATION_X_MIN = 0.995


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


def move(ctx, x, y, z):
    target = np.array([x,y,z], dtype=float)
    command = target.copy()
    for _ in range(4):
        ctx.tool("robot.go_to_pose_cartesian", pose={"position": dict(zip(('x','y','z'),map(float,command))), "rotation": _DOWN})
        actual = ctx.tool("robot.get_ee_pose")["pose"]
        p = np.array([actual['position'][k] for k in ('x','y','z')]);p[2] -= 0.10
        error = target-p
        if np.linalg.norm(error)<0.014 and abs(actual['rotation']['x'])>0.995:
            return
        command += error
    raise PlanningFailed("Placement arrival failed")


def _decode_pose(pose):
    position = np.array([
        float(pose["position"][axis]) for axis in ("x", "y", "z")
    ], dtype=float)
    quaternion = np.array([
        float(pose["rotation"][axis]) for axis in ("w", "x", "y", "z")
    ], dtype=float)
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(quaternion)):
        raise ValueError("Public hand pose is non-finite")
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("Public hand quaternion is invalid")
    position[2] -= _HAND_TO_TCP_M
    aligned = abs(float(quaternion[1]) / norm) > _ORIENTATION_X_MIN
    return position, aligned


def _snapshot(ctx):
    try:
        return _decode_pose(ctx.tool("robot.get_ee_pose")["pose"])
    except Exception as exc:
        raise VerificationFailed(
            "transport_descend_linear: public hand observation unknown; "
            "stopped without fallback or release."
        ) from exc


def _passes(position, aligned, target):
    error = float(np.linalg.norm(target - position))
    if not math.isfinite(error):
        raise VerificationFailed(
            "transport_descend_linear: arrival error is invalid; stopped before release."
        )
    return error < _POSITION_TOLERANCE_M and aligned


def _issue_motion(ctx, command, vertical):
    if not vertical:
        ctx.tool(
            "robot.go_to_pose_cartesian",
            pose={
                "position": dict(zip(("x", "y", "z"), map(float, command))),
                "rotation": dict(_DOWN),
            },
        )
        return

    # Refresh both start joints and TCP height together. A directed linear
    # distance must use measured TCP height, never the nominal hover height.
    try:
        arm = ctx.tool("robot.get_observation")["arms"][0]
        position, aligned = _decode_pose(arm["ee_pose"])
        joints = arm["joint_state"]
    except Exception as exc:
        raise VerificationFailed(
            "transport_descend_linear: descent start observation unknown; "
            "stopped without fallback or release."
        ) from exc
    if not aligned or float(np.linalg.norm(position[:2] - command[:2])) >= _POSITION_TOLERANCE_M:
        raise VerificationFailed(
            "transport_descend_linear: descent start is not aligned with the "
            "verified container column; stopped before release."
        )
    delta = float(command[2] - position[2])
    if abs(delta) <= 0.002:
        return
    plan = ctx.tool(
        "curobo.plan_directed_linear",
        start_joint_position=joints,
        endpoint_mode="DISTANCE",
        explicit_direction={"x": 0.0, "y": 0.0, "z": 1.0 if delta > 0.0 else -1.0},
        distance=abs(delta),
        allowed_axes=["Z"],
        orientation_mode="LOCK",
    )
    if not plan.get("success") or not plan.get("trajectory"):
        raise PlanningFailed(
            "transport_descend_linear: axis-locked descent planning failed; "
            "stopped before release."
        )
    ctx.tool("robot.execute_trajectory", trajectory=plan["trajectory"])


class _ArrivalChecked(Exception):
    def __init__(self, position, aligned):
        super().__init__("Public arrival snapshot completed")
        self.position = position
        self.aligned = aligned


class _CheckedContext:
    def __init__(self, ctx, command, vertical):
        self.ctx = ctx
        self.command = command.copy()
        self.vertical = vertical
        self.commanded = False

    def tool(self, name, **kwargs):
        if name == "robot.go_to_pose_cartesian" and not self.commanded:
            self.commanded = True
            _issue_motion(self.ctx, self.command, self.vertical)
            return None
        if name == "robot.get_ee_pose" and self.commanded:
            position, aligned = _snapshot(self.ctx)
            # This private transition prevents the preserved core from
            # continuing its retry loop or using its approximate TCP offset.
            raise _ArrivalChecked(position, aligned)
        raise PlanningFailed(
            "transport_descend_linear: invalid internal motion transition; stopped."
        )


def _command_and_check(ctx, target, command, vertical):
    adapter = _CheckedContext(ctx, command, vertical)
    try:
        move(adapter, float(target[0]), float(target[1]), float(target[2]))
    except _ArrivalChecked as result:
        return _passes(result.position, result.aligned, target)
    raise VerificationFailed(
        "transport_descend_linear: motion completed without an arrival check; stopped."
    )


def _verified_leg(ctx, target, vertical, fallback_state):
    if _command_and_check(ctx, target, target, vertical):
        return

    # Unknown snapshots raise immediately. Refresh before deciding whether
    # a fallback can run, and compute its inputs from this fresh measurement.
    position, aligned = _snapshot(ctx)
    if _passes(position, aligned, target):
        return
    if fallback_state["used"]:
        raise VerificationFailed(
            "transport_descend_linear: arrival failed after this call's single "
            "fallback was already used; stopped before release."
        )
    error = target - position
    if float(np.linalg.norm(error)) > _MAX_CORRECTION_M:
        raise VerificationFailed(
            "transport_descend_linear: arrival residual exceeds the 0.04 m "
            "correction bound; stopped before release."
        )
    if vertical and not aligned:
        raise VerificationFailed(
            "transport_descend_linear: descent orientation failed verification; "
            "axis-locked correction cannot restore it; stopped before release."
        )
    command = target + error
    if vertical:
        # Preserve the live XY column inside the walls; only correct Z.
        command[:2] = position[:2]
    if not np.all(np.isfinite(command)):
        raise VerificationFailed(
            "transport_descend_linear: corrective command is invalid; stopped."
        )
    if not _fallback_budget(ctx):
        raise VerificationFailed(
            "transport_descend_linear: episode fallback budget exhausted; "
            "stopped before release."
        )
    fallback_state["used"] = True
    if not _command_and_check(ctx, target, command, vertical):
        raise VerificationFailed(
            "transport_descend_linear: arrival still failed after one bounded "
            "correction; stopped before release."
        )


def run(
    ctx: NodeContext,
    container_obb: OrientedBoundingBox,
    transport_z: float = 0.353,
    lift_z: float = 0.25,
    place_offset: float = 0.06,
) -> Output:
    stage = "input validation"
    try:
        center = container_obb["center"]
        extent = container_obb["extent"]
        bx, by = float(center["x"]), float(center["y"])
        top = float(center["z"]) + float(extent["z"])
        place_z = top - float(place_offset)
        transport_height = float(transport_z)
        lift_height = float(lift_z)
        values = (bx, by, top, place_z, transport_height, lift_height)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Transport inputs must be finite")
        if float(extent["z"]) < 0.0:
            raise ValueError("Container vertical extent must be nonnegative")

        stage = "initial hand observation"
        current, _ = _snapshot(ctx)
        # Retain caller-provided heights, but never lower during the lift
        # or start transport below the destination rim clearance.
        lift_height = max(lift_height, float(current[2]))
        transport_height = max(transport_height, lift_height, top + 0.05, place_z)
        fallback_state = {"used": False}

        stage = "vertical lift"
        lift_target = np.array([current[0], current[1], lift_height], dtype=float)
        _verified_leg(ctx, lift_target, False, fallback_state)

        stage = "transport above container"
        hover_target = np.array([bx, by, transport_height], dtype=float)
        _verified_leg(ctx, hover_target, False, fallback_state)

        stage = "axis-locked container descent"
        place_target = np.array([bx, by, place_z], dtype=float)
        _verified_leg(ctx, place_target, True, fallback_state)

        return {"place_position": {"x": bx, "y": by, "z": place_z}}
    except (PlanningFailed, VerificationFailed):
        raise
    except Exception as exc:
        raise PlanningFailed(
            f"transport_descend_linear: {stage} failed; stopped before release."
        ) from exc
