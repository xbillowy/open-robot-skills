"""geometry tool bundle — pure-math perception/planning geometry ops.

Each operation is exposed as a ``@tool`` function, including the two
scalar helpers ``geometry.iou`` / ``geometry.pose_distance``. The math
lives in ``_impl.py``; this module is the typed boundary: numpy arrays +
:mod:`gap.types` TypedDicts in and out.

No model, no GPU — everything here is CPU numpy/scipy/Open3D/sklearn/cv2.
Heavy optional imports (open3d, sklearn, cv2) happen inside the functions
that need them, so importing this module is always cheap.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import threading
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np
from gap_core.tools import tool
from gap_core.types import (
    CameraFrame,
    GraspCandidates,
    JointState,
    Mask,
    OrientedBoundingBox,
    PointCloud,
    Quaternion,
    Se3Pose,
    Vec3,
    WorldConfig,
    pose_to_matrix,
)

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class AlgorithmServiceEvidence(TypedDict):
    kind: Literal["algorithm_service"]
    source_commit: str
    uv_lock_sha256: str
    config_sha256: str
    runtime_environment_sha256: str
    input_sha256: str
    output_sha256: str
    fallback_used: bool


class MaskWorldEvidence(AlgorithmServiceEvidence):
    valid_depth_count: int
    total_mask_count: int
    depth_bounds_m: tuple[float, float]
    camera_to_world_sha256: str
    world_points_sha256: str


class PointCloudResult(TypedDict):
    points: PointCloud


class MaskWorldResult(PointCloudResult):
    evidence: MaskWorldEvidence


class EvidencedPointCloudResult(PointCloudResult):
    evidence: AlgorithmServiceEvidence


class FilterEvidence(AlgorithmServiceEvidence):
    eps: float
    min_samples: int
    input_points_sha256: str
    filtered_points_sha256: str


class FilterPointCloudResult(PointCloudResult):
    evidence: FilterEvidence


class ObbEvidence(AlgorithmServiceEvidence):
    input_points_sha256: str
    obb_sha256: str
    random_seed: int


class ObbResult(TypedDict):
    obb: OrientedBoundingBox


class EvidencedObbResult(ObbResult):
    evidence: ObbEvidence


class PoseResult(TypedDict):
    pose: Se3Pose


class PointResult(TypedDict):
    point: Vec3


class PositionResult(TypedDict):
    position: Vec3


class QuatResult(TypedDict):
    quat: Quaternion


class DistanceResult(TypedDict):
    distance: float


class IouResult(TypedDict):
    iou: float


class GraspCandidatesResult(TypedDict):
    candidates: GraspCandidates


class FrontGraspResult(TypedDict):
    grasp_pose: Se3Pose
    pre_grasp_pose: Se3Pose
    approach_direction: Vec3
    slide_axis: Vec3


class ObjectMaskEntry(TypedDict):
    """Named segmentation mask for build_world_config."""

    name: str
    mask: Mask
    camera_index: int


class WorldConfigResult(TypedDict):
    config: WorldConfig
    mesh_names: list[str]
    evidence: AlgorithmServiceEvidence


_SOURCE_COMMIT = "158ddeef24a9b5f39ff481eb2a63f15eb858dae6"
_UV_LOCK_SHA256 = "sha256:53e83f9b1a5db267b6210de7aa1c45f9526eef40b82db850738aa6b309cee49d"
_OBB_RANDOM_SEED = 0
_OBB_RANDOM_LOCK = threading.Lock()


def _validate_digest(value: str) -> str:
    if re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise ValueError("digest must be canonical SHA256 (sha256:<64 lowercase hex>)")
    return value


def _validate_git_object_id(value: str) -> str:
    if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", value) is None:
        raise ValueError("source commit must be a 40- or 64-hex Git object ID")
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            "dtype": str(contiguous.dtype),
            "shape": list(contiguous.shape),
            "bytes_sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


@lru_cache(maxsize=1)
def _runtime_environment_sha256() -> str:
    distributions = {}
    for name in ("numpy", "scipy", "scikit-learn", "open3d", "opencv-python"):
        try:
            distributions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            distributions[name] = "unavailable"
    return _canonical_sha256(
        {
            "python": sys.version,
            "platform": platform.platform(),
            "distributions": distributions,
        }
    )


@lru_cache(maxsize=1)
def _verified_uv_lock_sha256() -> str:
    actual = (
        f"sha256:{hashlib.sha256(Path(__file__).with_name('uv.lock').read_bytes()).hexdigest()}"
    )
    if actual != _validate_digest(_UV_LOCK_SHA256):
        raise RuntimeError("geometry uv.lock hash drift")
    return actual


def _source_state() -> tuple[str, bool]:
    root = Path(__file__).resolve().parents[2]
    git_env = {
        "PATH": os.environ.get("PATH", ""),
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_OPTIONAL_LOCKS": "0",
    }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD^{commit}"],
            cwd=root,
            env=git_env,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                [
                    "git",
                    "status",
                    "--porcelain",
                    "--",
                    "tools/geometry/tools.py",
                    "tools/geometry/SKILL.md",
                    "tools/geometry/uv.lock",
                ],
                cwd=root,
                env=git_env,
                check=True,
                text=True,
                capture_output=True,
            ).stdout
        )
        return _validate_git_object_id(commit), dirty
    except (OSError, subprocess.SubprocessError, ValueError):
        return _validate_git_object_id(_SOURCE_COMMIT), True


def _algorithm_evidence(
    inputs: Any,
    output: Any,
    config: Any,
    *,
    algorithm_fallback: bool = False,
) -> AlgorithmServiceEvidence:
    source_commit, source_unsealed = _source_state()
    return {
        "kind": "algorithm_service",
        "source_commit": source_commit,
        "uv_lock_sha256": _verified_uv_lock_sha256(),
        "config_sha256": _canonical_sha256(config),
        "runtime_environment_sha256": _runtime_environment_sha256(),
        "input_sha256": _canonical_sha256(inputs),
        "output_sha256": _canonical_sha256(output),
        "fallback_used": source_unsealed or algorithm_fallback,
    }


def _pc(points: np.ndarray) -> PointCloud:
    return {"points": np.asarray(points, dtype=np.float32).reshape(-1, 3)}


# ---------------------------------------------------------------------------
# Back-projection / transforms
# ---------------------------------------------------------------------------


@tool(
    name="geometry.depth_to_point_cloud",
    summary="Convert a metric depth image to a 3D point cloud in the camera frame.",
    tags=("perception",),
)
def depth_to_point_cloud(depth: np.ndarray, intrinsics: np.ndarray) -> PointCloudResult:
    """Back-project ``depth`` (float32 [H, W], meters) through the pinhole
    ``intrinsics`` (float64 [3, 3]). Pixels with depth <= 0 are dropped."""
    from gap_skills.tools.geometry import _impl

    points = _impl._depth_to_points(
        np.asarray(depth, dtype=np.float32), np.asarray(intrinsics, dtype=np.float64)
    )
    return {"points": _pc(points)}


@tool(
    name="geometry.mask_to_world_points",
    summary="Back-project a 2D segmentation mask to 3D world points using depth + camera calibration.",
    tags=("perception",),
)
def mask_to_world_points(
    mask: Mask,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    camera_pose: Se3Pose,
) -> MaskWorldResult:
    """Foreground pixels of ``mask`` (uint8 0/255 [H, W]) with valid depth in
    [0.015, 20.0] m (HyRL bounds) become world-frame points via the
    camera-to-world ``camera_pose``."""
    from gap_skills.tools.geometry import _impl

    mask_bool = _impl.as_mask_bool(mask)
    depth_array = np.asarray(depth, dtype=np.float32)
    intrinsics_array = np.asarray(intrinsics, dtype=np.float64)
    camera_to_world = pose_to_matrix(camera_pose)
    points = _impl.mask_to_world_points(mask_bool, depth_array, intrinsics_array, camera_to_world)
    output = {"points": _pc(points)}
    valid = mask_bool & np.isfinite(depth_array) & (depth_array >= 0.015) & (depth_array <= 20.0)
    evidence: MaskWorldEvidence = {
        **_algorithm_evidence(
            {
                "mask": np.asarray(mask),
                "depth": depth_array,
                "intrinsics": intrinsics_array,
                "camera_pose": camera_pose,
            },
            output,
            {"depth_bounds_m": (0.015, 20.0)},
        ),
        "valid_depth_count": int(np.count_nonzero(valid)),
        "total_mask_count": int(np.count_nonzero(mask_bool)),
        "depth_bounds_m": (0.015, 20.0),
        "camera_to_world_sha256": _canonical_sha256(camera_to_world),
        "world_points_sha256": _canonical_sha256(points),
    }
    return {**output, "evidence": evidence}


@tool(
    name="geometry.pixel_to_world_point",
    summary="Back-project a single pixel to a 3D world point using depth + camera calibration.",
    tags=("perception",),
)
def pixel_to_world_point(
    pixel_x: float,
    pixel_y: float,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    camera_pose: Se3Pose,
) -> PointResult:
    """Raises ToolError when the pixel is out of bounds or has invalid depth."""
    from gap_skills.tools.geometry import _impl

    pt = _impl.pixel_to_world_point(
        pixel_x,
        pixel_y,
        np.asarray(depth, dtype=np.float32),
        np.asarray(intrinsics, dtype=np.float64),
        pose_to_matrix(camera_pose),
    )
    return {"point": _impl.vec3(pt)}


@tool(
    name="geometry.transform_points",
    summary="Apply a rigid SE(3) transform to a set of 3D points.",
    tags=("perception",),
)
def transform_points(points: PointCloud, transform: Se3Pose) -> PointCloudResult:
    from gap_skills.tools.geometry import _impl

    pts = _impl.as_points(points)
    if len(pts) == 0:
        return {"points": _pc(pts)}
    out = _impl._transform_points(pts, pose_to_matrix(transform))
    return {"points": _pc(out)}


# ---------------------------------------------------------------------------
# Filtering + OBB fitting
# ---------------------------------------------------------------------------


@tool(
    name="geometry.exclude_robot_points",
    summary="Remove points near the robot body via FK-based sphere exclusion "
    "(7-DOF Franka; other arms pass through unchanged).",
    tags=("perception",),
)
def exclude_robot_points(
    points: PointCloud,
    joint_positions: JointState,
    distance_threshold: float = 0.05,
) -> EvidencedPointCloudResult:
    """Strip robot-body points from a perception cloud (HyRL RobotSegmenter
    concept, simplified FK + link spheres). Essential when the perceived
    object sits against the robot base — the segmentation mask bleeds onto
    robot pixels and the merged cloud yields a wildly oversized OBB."""
    import numpy as np
    from gap_skills.tools.geometry import _impl

    pts = _impl.as_points(points)
    joints = np.asarray(joint_positions["positions"], dtype=np.float64).reshape(-1)
    if joints.shape[0] != 7:
        output = {"points": _pc(pts)}
        return {
            **output,
            "evidence": _algorithm_evidence(
                {"points": pts, "joint_positions": joints},
                output,
                {"distance_threshold": distance_threshold},
                algorithm_fallback=True,
            ),
        }
    filtered = _impl._exclude_robot_points(pts, joints, distance_threshold)
    output = {"points": _pc(filtered)}
    return {
        **output,
        "evidence": _algorithm_evidence(
            {"points": pts, "joint_positions": joints},
            output,
            {"distance_threshold": distance_threshold},
        ),
    }


@tool(
    name="geometry.filter_noise",
    summary="Filter point-cloud noise with DBSCAN clustering (keeps all non-noise points).",
    tags=("perception",),
)
def filter_noise(
    points: PointCloud,
    eps: float = 0.005,
    min_samples: int = 10,
) -> FilterPointCloudResult:
    """Mirrors HyRL filter_noise: keeps ALL non-noise points (labels != -1),
    not just the largest cluster. If everything is classified as noise the
    original cloud is returned unchanged."""
    from gap_skills.tools.geometry import _impl

    pts = _impl.as_points(points)
    filtered = _impl.filter_noise(pts, eps, min_samples)
    output = {"points": _pc(filtered)}
    all_noise_fallback = bool(len(pts) and filtered is pts)
    evidence: FilterEvidence = {
        **_algorithm_evidence(
            {"points": pts},
            output,
            {"eps": eps, "min_samples": min_samples},
            algorithm_fallback=all_noise_fallback,
        ),
        "eps": eps,
        "min_samples": min_samples,
        "input_points_sha256": _canonical_sha256(pts),
        "filtered_points_sha256": _canonical_sha256(filtered),
    }
    return {**output, "evidence": evidence}


@tool(
    name="geometry.compute_obb",
    summary="Fit an oriented bounding box to 3D points (HyRL contour-based min-width fit, upright in Z).",
    tags=("perception",),
)
def compute_obb(points: PointCloud) -> EvidencedObbResult:
    """Statistical outlier removal → XY rasterization → contour polygon →
    min-width rectangle search → 2nd/98th percentile extents. The returned
    OBB is upright (rotation only around world Z); ``extent`` holds
    HALF-extents per gap.types. Raises PerceptionFailed on < 4 points."""
    from gap_skills.tools.geometry import _impl

    pts = _impl.as_points(points)
    with _OBB_RANDOM_LOCK:
        state = np.random.get_state()
        try:
            np.random.seed(_OBB_RANDOM_SEED)
            obb = _impl.compute_obb(pts)
        finally:
            np.random.set_state(state)
    output = {"obb": obb}
    evidence: ObbEvidence = {
        **_algorithm_evidence({"points": pts}, output, {"random_seed": _OBB_RANDOM_SEED}),
        "input_points_sha256": _canonical_sha256(pts),
        "obb_sha256": _canonical_sha256(obb),
        "random_seed": _OBB_RANDOM_SEED,
    }
    return {**output, "evidence": evidence}


@tool(
    name="geometry.filter_and_compute_obb",
    summary="DBSCAN-filter a point cloud then fit its oriented bounding box in one call.",
    tags=("perception",),
)
def filter_and_compute_obb(
    points: PointCloud,
    eps: float = 0.005,
    min_samples: int = 10,
) -> EvidencedObbResult:
    """Sequences geometry.filter_noise + geometry.compute_obb (the servicer
    offered this fusion to avoid two round trips; kept for workflow parity)."""
    from gap_skills.tools.geometry import _impl

    pts = _impl.as_points(points)
    filtered = _impl.filter_noise(pts, eps, min_samples)
    with _OBB_RANDOM_LOCK:
        state = np.random.get_state()
        try:
            np.random.seed(_OBB_RANDOM_SEED)
            obb = _impl.compute_obb(filtered)
        finally:
            np.random.set_state(state)
    output = {"obb": obb}
    all_noise_fallback = bool(len(pts) and filtered is pts)
    evidence: ObbEvidence = {
        **_algorithm_evidence(
            {"points": pts},
            output,
            {
                "eps": eps,
                "min_samples": min_samples,
                "random_seed": _OBB_RANDOM_SEED,
            },
            algorithm_fallback=all_noise_fallback,
        ),
        "input_points_sha256": _canonical_sha256(pts),
        "obb_sha256": _canonical_sha256(obb),
        "random_seed": _OBB_RANDOM_SEED,
    }
    return {**output, "evidence": evidence}


# ---------------------------------------------------------------------------
# Grasp-pose derivation
# ---------------------------------------------------------------------------


@tool(
    name="geometry.top_down_grasp_from_obb",
    summary="Compute a single world-aligned top-down grasp pose from an oriented bounding box.",
    tags=("planning",),
)
def top_down_grasp_from_obb(obb: OrientedBoundingBox, z_offset: float = 0.0) -> PoseResult:
    """Gripper points straight down world -Z above the OBB centre; Z lands on
    the world-frame top surface plus ``z_offset`` (negative = lower, typical
    -0.06 for bottles), clamped to 5 cm below the table plane."""
    from gap_skills.tools.geometry import _impl

    return {"pose": _impl.compute_top_down_grasp_world_aligned(obb, z_offset)}


@tool(
    name="geometry.top_down_grasp_candidates",
    summary="Fan out 29 grasp candidates (canonical primary+alt, yaw/depth fan, then shape-safe completion).",
    tags=("planning",),
)
def top_down_grasp_candidates(
    obb: OrientedBoundingBox,
    z_offset: float = -0.04,
) -> GraspCandidatesResult:
    """poses[0]/poses[1] reproduce the legacy 2-pose RPC exactly; the rest are
    enriched candidates for a planner goalset. Default ``z_offset=-0.04``
    puts the fingertip 4 cm below the OBB top — with 0.0 the fingers close
    above the object (silent empty grip)."""
    from gap_skills.tools.geometry import _impl

    poses = _impl.top_down_grasp_candidates(obb, z_offset)
    output = {"candidates": {"poses": poses}}
    return {
        **output,
        "evidence": _algorithm_evidence(
            {"obb": obb, "z_offset": z_offset}, output, {}
        ),
        "paper_outcome": {
            "status": "success" if poses else "failure",
            "failure_code": None if poses else "no_valid_grasp_candidate",
            "source": "service",
        },
    }


@tool(
    name="geometry.select_top_down_grasp",
    summary="Select the most top-down oriented grasp from candidates (gripper distance as tiebreaker).",
    tags=("planning",),
)
def select_top_down_grasp(
    grasp_poses: list[Se3Pose],
    gripper_position: Vec3 | None = None,
) -> PoseResult:
    from gap_skills.tools.geometry import _impl

    return {"pose": _impl.select_top_down_grasp(grasp_poses, gripper_position)}


@tool(
    name="geometry.front_grasp_from_obb",
    summary="Compute front-approach grasp + pre-grasp poses for a handle from its OBB (drawers, doors).",
    tags=("planning",),
)
def front_grasp_from_obb(
    obb: OrientedBoundingBox,
    approach_offset: float = 0.08,
    approach_hint: Vec3 | None = None,
    z_offset: float = 0.0,
) -> FrontGraspResult:
    """Derives approach direction and slide axis from the OBB orientation.
    ``approach_hint`` points from the handle toward the robot (default:
    OBB centre → origin, XY only). Raises PlanningFailed when the approach
    direction is near-vertical — use top_down_grasp_from_obb instead."""
    from gap_skills.tools.geometry import _impl

    out = _impl.front_grasp_from_obb(obb, approach_offset, approach_hint, z_offset)
    return {
        "grasp_pose": out["grasp_pose"],
        "pre_grasp_pose": out["pre_grasp_pose"],
        "approach_direction": out["approach_direction"],
        "slide_axis": out["slide_axis"],
    }


# ---------------------------------------------------------------------------
# World reconstruction
# ---------------------------------------------------------------------------


@tool(
    name="geometry.build_world_config",
    summary="Build a planner-agnostic collision world (alpha-shape scene mesh) from RGB-D camera frames.",
    tags=("planning",),
)
def build_world_config(
    cameras: list[CameraFrame],
    object_masks: list[ObjectMaskEntry] | None = None,
    voxel_size: float = 0.005,
    noise_eps: float = 0.01,
    noise_min_samples: int = 5,
    table_z_threshold: float = 0.0,
    mesh_alpha: float = 0.03,
    robot_joint_state: JointState | None = None,
    robot_distance_threshold: float = 0.15,
    robot_file: str = "franka.yml",
    target_obb: OrientedBoundingBox | None = None,
    target_obb_name: str = "target",
) -> WorldConfigResult:
    """Pipeline: depth → merged world cloud → voxel downsample → optional
    FK-based robot-point exclusion → DBSCAN largest-cluster filter → table
    removal (when ``table_z_threshold`` != 0; typical -0.01) → alpha-shape
    ``scene`` mesh, with all ``object_masks`` (or a projected ``target_obb``)
    points excluded. Only ``target_obb_name`` is emitted as a named OBB mesh
    for held-object attachment; receptacle masks remain exclusion-only so a
    drop pose inside a container is not treated as a solid-hull collision.

    ``robot_file`` is accepted for parity with the service request but the
    FK exclusion is Franka-only (simplified DH model); non-7-DOF joint
    states skip exclusion. Returns an empty WorldConfig if no geometry can
    be reconstructed."""
    from gap_skills.tools.geometry import _impl

    config, mesh_names = _impl.build_world_config(
        cameras,
        list(object_masks or []),
        voxel_size=voxel_size if voxel_size > 0 else 0.005,
        noise_eps=noise_eps if noise_eps > 0 else 0.01,
        noise_min_samples=noise_min_samples if noise_min_samples > 0 else 5,
        table_z_threshold=table_z_threshold,
        mesh_alpha=mesh_alpha if mesh_alpha > 0 else 0.03,
        robot_joint_state=robot_joint_state,
        robot_distance_threshold=(
            robot_distance_threshold if robot_distance_threshold > 0 else 0.15
        ),
        target_obb=target_obb,
        target_obb_name=target_obb_name,
    )
    output = {"config": config, "mesh_names": mesh_names}
    used_projection_fallback = not object_masks and target_obb is not None
    invalid_robot_shape = bool(
        robot_joint_state is not None
        and len(np.asarray(robot_joint_state.get("positions", [])).reshape(-1)) != 7
    )
    empty_reconstruction = not mesh_names
    return {
        **output,
        "evidence": _algorithm_evidence(
            {
                "cameras": cameras,
                "object_masks": object_masks,
                "robot_joint_state": robot_joint_state,
                "target_obb": target_obb,
            },
            output,
            {
                "voxel_size": voxel_size,
                "noise_eps": noise_eps,
                "noise_min_samples": noise_min_samples,
                "table_z_threshold": table_z_threshold,
                "mesh_alpha": mesh_alpha,
                "robot_distance_threshold": robot_distance_threshold,
                "robot_file": robot_file,
                "target_obb_name": target_obb_name,
            },
            algorithm_fallback=(
                used_projection_fallback or invalid_robot_shape or empty_reconstruction
            ),
        ),
    }


# ---------------------------------------------------------------------------
# Utility ops (merged from the geometry_utils skill, same as the servicer)
# ---------------------------------------------------------------------------


@tool(
    name="geometry.rotate_quat_z90",
    summary="Rotate a wxyz quaternion by 90 degrees around the world Z axis.",
    tags=("planning",),
)
def rotate_quat_z90(quat: Quaternion) -> QuatResult:
    s = math.sqrt(2.0) / 2.0
    zw, zx, zy, zz = s, 0.0, 0.0, s
    q = quat
    rw = q["w"] * zw - q["x"] * zx - q["y"] * zy - q["z"] * zz
    rx = q["w"] * zx + q["x"] * zw + q["y"] * zz - q["z"] * zy
    ry = q["w"] * zy - q["x"] * zz + q["y"] * zw + q["z"] * zx
    rz = q["w"] * zz + q["x"] * zy - q["y"] * zx + q["z"] * zw
    return {"quat": {"w": rw, "x": rx, "y": ry, "z": rz}}


@tool(
    name="geometry.compute_drop_position",
    summary="Compute a drop position above a container from its oriented bounding box.",
    tags=("planning",),
)
def compute_drop_position(
    container_obb: OrientedBoundingBox,
    clearance: float = 0.05,
    object_z_extent: float = 0.0,
) -> PositionResult:
    obb = container_obb
    clearance = clearance or 0.05
    obj_z = object_z_extent or 0.0
    c = obb["center"]
    e = obb["extent"]
    drop_z = c["z"] + e["z"] / 2.0 + obj_z + clearance
    output = {"position": {"x": c["x"], "y": c["y"], "z": drop_z}}
    return {
        **output,
        "evidence": _algorithm_evidence(
            {
                "container_obb": container_obb,
                "clearance": clearance,
                "object_z_extent": object_z_extent,
            },
            output,
            {},
        ),
        "paper_outcome": {
            "status": "success",
            "failure_code": None,
            "source": "service",
        },
    }


@tool(
    name="geometry.compute_xy_distance",
    summary="Euclidean distance between two 3D points projected onto the XY plane.",
    tags=("perception",),
)
def compute_xy_distance(point_a: Vec3, point_b: Vec3) -> DistanceResult:
    a, b = point_a, point_b
    dist = math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2)
    return {"distance": dist}


# ---------------------------------------------------------------------------
# Legacy canary tools (ported from the dev tree's geometry_iou / pose_distance tools)
# ---------------------------------------------------------------------------


@tool(
    name="geometry.iou",
    summary="Compute IoU of two axis-aligned 2D boxes [x1, y1, x2, y2]. Returns 0 if boxes don't overlap.",
    tags=("perception",),
)
def iou(box_a: list[float], box_b: list[float]) -> IouResult:
    """Pure-Python intersection-over-union for axis-aligned 2D boxes.

    Args:
        box_a: ``[x1, y1, x2, y2]`` corners of box A.
        box_b: ``[x1, y1, x2, y2]`` corners of box B.

    Returns:
        ``{"iou": float}`` in ``[0, 1]``. Zero when the boxes don't overlap
        or either has zero area.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return {"iou": inter / union if union > 0 else 0.0}


@tool(
    name="geometry.pose_distance",
    summary="Euclidean distance between two 3D positions [x, y, z].",
    tags=("perception",),
)
def pose_distance(a: list[float], b: list[float]) -> DistanceResult:
    """Returns the Euclidean distance between two ``[x, y, z]`` points."""
    if len(a) != 3 or len(b) != 3:
        raise ValueError(f"expected 3-vectors, got len(a)={len(a)}, len(b)={len(b)}")
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return {"distance": math.sqrt(dx * dx + dy * dy + dz * dz)}
