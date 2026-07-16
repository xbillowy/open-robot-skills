"""Paper-admitted target or destination perception with sealed lineage."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal, TypedDict

from gap import NodeContext
from gap_core.types import OrientedBoundingBox

EXPECTED_PARAMETERS = {
    "approach_distance_m",
    "grasp_candidate_count",
    "ik_seed_count",
    "lift_distance_m",
    "trajectory_waypoint_count",
}
CONSUMED_PRESET_FIELDS = frozenset(())
RESPONSIBLE_PRESET_FIELDS = frozenset(())
DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
SEALED_PRESET_SHA256 = "sha256:8f6f81c9f2880fe0e3f786e276511868142ca8033255d6b598d82baad22b77d9"


class PaperManipulationError(RuntimeError):
    """Stable, machine-owned paper manipulation failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class Output(TypedDict, total=False):
    success: bool
    error_code: str | None
    semantic_role: Literal["target", "destination"]
    obb: OrientedBoundingBox
    target_obb: OrientedBoundingBox
    destination_obb: OrientedBoundingBox
    lineage_json: str
    lineage_record: dict[str, Any]
    decision_path: list[str]
    fallback_used: bool
    preset_trace: dict[str, Any]


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def _hash(value: Any) -> str:
    payload = _canonical(value).encode()
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


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
    payload = {key: value for key, value in preset.items() if key != "preset_sha256"}
    if preset.get("preset_sha256") != SEALED_PRESET_SHA256 or preset.get("preset_sha256") != _hash(
        payload
    ):
        raise PaperManipulationError("PRESET_HASH_MISMATCH")
    if {row.get("name") for row in rows} != EXPECTED_PARAMETERS or len(rows) != 5:
        raise PaperManipulationError("PRESET_PARAMETERS_INVALID")
    values = {
        row["name"]: row["runtime_value"] for row in rows if row["name"] in CONSUMED_PRESET_FIELDS
    }
    trace = {
        "preset_sha256": preset["preset_sha256"],
        "parameters": [
            {
                key: row[key]
                for key in ("name", "runtime_value", "mapping", "evidence_level", "paper_locator")
            }
            for row in rows
        ],
    }
    return values, trace


def _evidence(result: dict[str, Any], kind: str, code: str) -> dict[str, Any]:
    evidence = result.get("evidence")
    if not isinstance(evidence, dict) or evidence.get("fallback_used") is not False:
        raise PaperManipulationError(code)
    if kind == "learned_model":
        required = ("weights_sha256", "input_sha256", "output_sha256")
        if evidence.get("kind") != kind or not all(
            DIGEST.fullmatch(str(evidence.get(key, ""))) for key in required
        ):
            raise PaperManipulationError(code)
    elif kind == "algorithm_service":
        required = (
            "uv_lock_sha256",
            "config_sha256",
            "runtime_environment_sha256",
            "input_sha256",
            "output_sha256",
        )
        if evidence.get("kind") != kind or not all(
            DIGEST.fullmatch(str(evidence.get(key, ""))) for key in required
        ):
            raise PaperManipulationError(code)
    elif not DIGEST.fullmatch(str(evidence.get("request_sha256", ""))) or not DIGEST.fullmatch(
        str(evidence.get("response_sha256", ""))
    ):
        raise PaperManipulationError(code)
    return evidence


def _tool(ctx: NodeContext, name: str, code: str, **kwargs: Any) -> dict[str, Any]:
    try:
        if name == "robot.get_observation":
            return ctx.tool("robot.get_observation", **kwargs)
        if name == "grounding-dino.detect":
            return ctx.tool("grounding-dino.detect", **kwargs)
        if name == "vlm.query":
            return ctx.tool("vlm.query", **kwargs)
        if name == "sam3.segment_box":
            return ctx.tool("sam3.segment_box", **kwargs)
        if name == "geometry.mask_to_world_points":
            return ctx.tool("geometry.mask_to_world_points", **kwargs)
        if name == "geometry.filter_and_compute_obb":
            return ctx.tool("geometry.filter_and_compute_obb", **kwargs)
        raise PaperManipulationError("UNDECLARED_TOOL_DISPATCH")
    except Exception as error:
        raise PaperManipulationError(code) from error


def _selected_index(text: str, candidate_count: int) -> int:
    token = text.strip().upper()
    if len(token) == 1 and "A" <= token <= "Z":
        index = ord(token) - ord("A")
    elif token.isdigit():
        index = int(token) - 1
    else:
        raise PaperManipulationError("VLM_DISAMBIGUATION_FAILED")
    if not 0 <= index < candidate_count:
        raise PaperManipulationError("VLM_DISAMBIGUATION_FAILED")
    return index


def _crop_tournament(image: Any, candidates: list[dict[str, Any]]) -> Any:
    """Build one labeled crop sheet; failure never degrades to the full image."""
    import numpy as np
    from PIL import Image, ImageDraw

    source = Image.fromarray(np.asarray(image, dtype=np.uint8))
    crops = []
    for index, candidate in enumerate(candidates):
        box = candidate["box"]
        if isinstance(box, dict):
            xyxy = (box["x1"], box["y1"], box["x2"], box["y2"])
        else:
            xyxy = tuple(box)
        crop = source.crop(xyxy).convert("RGB")
        crop.thumbnail((256, 256))
        tile = Image.new("RGB", (256, 280), "white")
        tile.paste(crop, ((256 - crop.width) // 2, 24))
        ImageDraw.Draw(tile).text((8, 4), chr(65 + index), fill="black")
        crops.append(tile)
    sheet = Image.new("RGB", (256 * len(crops), 280), "white")
    for index, crop in enumerate(crops):
        sheet.paste(crop, (256 * index, 0))
    return np.asarray(sheet)


def _exterior_camera(cameras: list[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    exterior = [
        (index, camera)
        for index, camera in enumerate(cameras)
        if not any(
            token in str(camera.get("name", "")).lower()
            for token in ("wrist", "eye_in_hand", "hand_camera")
        )
    ]
    if not exterior:
        raise PaperManipulationError("EXTERIOR_RGBD_UNAVAILABLE")
    return exterior[0]


def _valid_box(box: Any, image: Any) -> bool:
    try:
        if isinstance(box, dict):
            x1, y1, x2, y2 = (float(box[key]) for key in ("x1", "y1", "x2", "y2"))
        else:
            x1, y1, x2, y2 = (float(value) for value in box)
        height, width = image.shape[:2]
        return (
            all(value == value and abs(value) != float("inf") for value in (x1, y1, x2, y2))
            and 0 <= x1 < x2 <= width
            and 0 <= y1 < y2 <= height
        )
    except (KeyError, TypeError, ValueError):
        return False


def _run_impl(
    ctx: NodeContext,
    query: str,
    semantic_role: Literal["target", "destination"],
    preset_json: str,
) -> Output:
    """Observe, detect broadly, disambiguate, segment, and fit a role-bound OBB."""
    _, preset_trace = _preset(preset_json)
    if semantic_role not in ("target", "destination"):
        raise PaperManipulationError("SEMANTIC_ROLE_INVALID")
    observation = _tool(ctx, "robot.get_observation", "RGBD_OBSERVATION_FAILED")
    cameras = observation.get("cameras", [])
    if not cameras:
        raise PaperManipulationError("RGBD_OBSERVATION_UNAVAILABLE")
    camera_index, camera = _exterior_camera(cameras)
    detected = _tool(
        ctx, "grounding-dino.detect", "DETECTION_FAILED", image=camera["rgb"], query=query
    )
    detector_evidence = _evidence(detected, "learned_model", "DETECTION_EVIDENCE_UNAVAILABLE")
    candidates = detected.get("detections", [])
    if len(candidates) < 2:
        raise PaperManipulationError("DETECTION_CANDIDATES_INSUFFICIENT")
    if not all(_valid_box(candidate.get("box"), camera["rgb"]) for candidate in candidates):
        raise PaperManipulationError("DETECTION_BOX_INVALID")
    try:
        crop_sheet = _crop_tournament(camera["rgb"], candidates)
    except Exception as error:
        raise PaperManipulationError("VLM_CROP_TOURNAMENT_FAILED") from error
    tournament = _tool(
        ctx,
        "vlm.query",
        "VLM_DISAMBIGUATION_FAILED",
        prompt=f"Select exactly one crop label for {semantic_role} '{query}': "
        + ", ".join(chr(65 + i) for i in range(len(candidates))),
        image=crop_sheet,
    )
    vlm_evidence = _evidence(tournament, "vlm", "VLM_EVIDENCE_UNAVAILABLE")
    selected_index = _selected_index(str(tournament.get("text", "")), len(candidates))
    selected = candidates[selected_index]
    segmented = _tool(
        ctx, "sam3.segment_box", "SEGMENTATION_FAILED", image=camera["rgb"], box=selected["box"]
    )
    segment_evidence = _evidence(segmented, "learned_model", "SEGMENTATION_EVIDENCE_UNAVAILABLE")
    masks = segmented.get("masks", [])
    if not masks:
        raise PaperManipulationError("SEGMENTATION_EMPTY")
    projected = _tool(
        ctx,
        "geometry.mask_to_world_points",
        "DEPTH_PROJECTION_FAILED",
        mask=masks[0],
        depth=camera["depth"],
        intrinsics=camera["intrinsics"],
        camera_pose=camera["pose"],
    )
    depth_evidence = _evidence(projected, "algorithm_service", "DEPTH_EVIDENCE_UNAVAILABLE")
    if depth_evidence.get("valid_depth_count", 0) <= 0 or not projected.get("points"):
        raise PaperManipulationError("DEPTH_POINTS_EMPTY")
    fitted = _tool(
        ctx, "geometry.filter_and_compute_obb", "OBB_FIT_FAILED", points=projected["points"]
    )
    obb_evidence = _evidence(fitted, "algorithm_service", "OBB_EVIDENCE_UNAVAILABLE")
    obb = fitted.get("obb")
    if not isinstance(obb, dict):
        raise PaperManipulationError("OBB_FIT_FAILED")
    lineage = {
        "semantic_role": semantic_role,
        "source_camera_index": camera_index,
        "query": query,
        "observation_sha256": _hash(observation),
        "detector_candidates": candidates,
        "candidate_boxes_sha256": _hash([item["box"] for item in candidates]),
        "crop_mapping": [
            {
                "crop_label": chr(65 + index),
                "candidate_index": index,
                "box_sha256": _hash(item["box"]),
            }
            for index, item in enumerate(candidates)
        ],
        "vlm_decision": {
            "raw_response_sha256": _hash(tournament.get("text", "")),
            "selected_candidate_index": selected_index,
        },
        "selected_candidate_index": selected_index,
        "selected_box_sha256": _hash(selected["box"]),
        "mask_metadata": {
            "mask_sha256": _hash(masks[0]),
            "selected_score": segmented.get("scores", [None])[0],
            "mask_count": len(masks),
        },
        "valid_depth_ratio": depth_evidence["valid_depth_count"]
        / max(1, depth_evidence.get("total_mask_count", 0)),
        "world_points_sha256": _hash(projected["points"]),
        "obb_sha256": _hash(obb),
        "evidence": {
            "detector": detector_evidence,
            "vlm": vlm_evidence,
            "segmenter": segment_evidence,
            "depth": depth_evidence,
            "obb": obb_evidence,
        },
        "preset_trace": preset_trace,
        "fallback_used": False,
        "decision_path": [
            "observe_rgbd",
            "broad_detection",
            "crop_tournament",
            "box_segmentation",
            "depth_to_world",
            "obb_fit",
        ],
    }
    lineage["lineage_sha256"] = _hash(lineage)
    lineage_json = _canonical(lineage)
    return {
        "success": True,
        "error_code": None,
        "semantic_role": semantic_role,
        "obb": obb,
        "target_obb": obb if semantic_role == "target" else None,
        "destination_obb": obb if semantic_role == "destination" else None,
        "lineage_json": lineage_json,
        "lineage_record": lineage,
        "decision_path": lineage["decision_path"],
        "fallback_used": False,
        "preset_trace": preset_trace,
    }


def run(
    ctx: NodeContext, query: str, semantic_role: Literal["target", "destination"], preset_json: str
) -> Output:
    try:
        return _run_impl(ctx, query, semantic_role, preset_json)
    except PaperManipulationError as error:
        code = error.code
    except Exception:
        code = "PAPER_MANIPULATION_INTERNAL_ERROR"
    return {
        "success": False,
        "error_code": code,
        "semantic_role": semantic_role,
        "obb": None,
        "target_obb": None,
        "destination_obb": None,
        "lineage_json": "{}",
        "lineage_record": {},
        "decision_path": [],
        "fallback_used": False,
        "preset_trace": {},
    }
