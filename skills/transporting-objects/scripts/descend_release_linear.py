"""Descend to a TCP drop pose, release, lift vertically, then retract home.

Cartesian targets are fingertip poses in the world frame. After opening
with a settle, lift at the drop XY with the release orientation held fixed
before retracting home. Any ordinary exception raises PlanningFailed and
stops the sequence; safety guard exceptions propagate unchanged.

Inputs and returned keys match descend_release: drop_position (Vec3) and
optional drop_rotation (Quaternion); returns drop_position unchanged.
"""

from typing import TypedDict

from gap import NodeContext
from gap_core.errors import PlanningFailed
from gap_core.types import Quaternion, Se3Pose, Vec3


class Output(TypedDict):
    drop_position: Vec3


# Canonical top-down orientation (gripper Z aligned with world -Z).
_DOWN: Quaternion = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}


def run(
    ctx: NodeContext,
    drop_position: Vec3,
    drop_rotation: Quaternion | None = None,
) -> Output:
    stage = "drop pose preparation"
    try:
        rotation = drop_rotation if drop_rotation is not None else _DOWN
        end_pose: Se3Pose = {"position": drop_position, "rotation": rotation}

        stage = "Cartesian descent"
        ctx.tool("robot.go_to_pose_cartesian", pose=end_pose)

        # Open and hold so the released object settles before withdrawal.
        stage = "gripper opening and settling"
        ctx.tool("robot.open_gripper", settle_steps=60)

        stage = "vertical clearance lift"
        # Lift straight up before retracting home so open fingers clear the bowl rim.
        retract_pose: Se3Pose = {
            "position": {
                "x": float(drop_position["x"]),
                "y": float(drop_position["y"]),
                "z": max(float(drop_position["z"]) + 0.14, 0.22),
            },
            "rotation": rotation,
        }
        ctx.tool("robot.go_to_pose_cartesian", pose=retract_pose)

        # Start the free-space retract only after the vertical lift completes.
        stage = "home retract"
        ctx.tool("robot.go_home")

        return {"drop_position": drop_position}
    except Exception as exc:
        raise PlanningFailed(
            f"descend_release_linear: {stage} failed; placement sequence stopped."
        ) from exc
