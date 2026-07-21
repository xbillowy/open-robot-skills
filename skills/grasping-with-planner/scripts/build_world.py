"""Thin adapter: extract joint state from observation, call build_world_config.

Calls ``geometry.build_world_config`` to reconstruct the scene as a set of
collision meshes, excluding the robot body. Unwraps
``observation["arms"][0]["joint_state"]`` (the ``$ref`` DSL cannot index
into repeated fields via bracket notation) and forwards the rest.

Target geometry has two distinct roles:

1. ``target_mask``: the pixel-accurate segmentation mask from
   perception (e.g. SAM3 output via ``perceiving-*`` skills). Wrapped into
   an object-mask entry ``{name: target_name, mask, camera_index: 0}`` so
   target points are excluded from the background ``scene`` mesh.

2. ``target_obb``: supplies stable local-frame attachment geometry for
   held-object collision validation. When no mask is available, the geometry
   bundle also projects this OBB to derive a fallback exclusion mask.

Pass BOTH when available: the mask isolates the target from the scene, while
the OBB represents the uniquely named object attached by CuRobo.
"""

from typing import TypedDict

from gap import NodeContext
from gap_core.types import Mask, Observation, OrientedBoundingBox, WorldConfig


class Output(TypedDict):
    config: WorldConfig


def run(
    ctx: NodeContext,
    observation: Observation,
    target_mask: Mask | None = None,
    target_obb: OrientedBoundingBox | None = None,
    target_name: str = "target",
    robot_distance_threshold: float = 0.15,
    robot_file: str = "",
) -> Output:
    kwargs = dict(
        cameras=observation["cameras"],
        robot_distance_threshold=robot_distance_threshold,
    )
    if observation.get("arms"):
        kwargs["robot_joint_state"] = observation["arms"][0]["joint_state"]

    if target_mask is not None:
        kwargs["object_masks"] = [{"name": target_name, "mask": target_mask, "camera_index": 0}]
    if target_obb is not None:
        kwargs["target_obb"] = target_obb
        kwargs["target_obb_name"] = target_name

    if robot_file:
        kwargs["robot_file"] = robot_file

    resp = ctx.tool("geometry.build_world_config", **kwargs)
    return {"config": resp["config"]}
