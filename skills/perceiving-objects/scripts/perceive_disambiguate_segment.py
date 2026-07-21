"""Paper-admitted target or destination perception with sealed lineage."""

from __future__ import annotations

import hashlib
import json
import math
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


_SERVICE_FAILURE_CODE = {
    "grounding-dino.detect": "DETECTOR_SERVICE_ERROR",
    "vlm.query": "VLM_SERVICE_ERROR",
    "sam3.segment_box": "SEGMENTATION_SERVICE_ERROR",
    "robot.get_observation": "SERVICE_UNAVAILABLE",
    "geometry.mask_to_world_points": "SERVICE_UNAVAILABLE",
    "geometry.filter_and_compute_obb": "SERVICE_UNAVAILABLE",
}

_PAPER_FAILURE_CODE = {
    "DETECTOR_SERVICE_ERROR": "detector_service_error",
    "VLM_SERVICE_ERROR": "vlm_service_error",
    "SEGMENTATION_SERVICE_ERROR": "segmentation_service_error",
    "SERVICE_UNAVAILABLE": "service_unavailable",
    "DETECTION_CANDIDATES_INSUFFICIENT": "detector_no_candidate",
    "VLM_DISAMBIGUATION_FAILED": "vlm_no_valid_selection",
    "SEGMENTATION_EMPTY": "segmentation_empty_mask",
    "SEGMENTATION_LOW_CONFIDENCE": "segmentation_empty_mask",
    "DEPTH_POINTS_EMPTY": "depth_invalid",
    "OBB_FIT_FAILED": "obb_invalid",
    "DETECTION_EVIDENCE_UNAVAILABLE": "detector_service_error",
    "VLM_EVIDENCE_UNAVAILABLE": "vlm_service_error",
    "SEGMENTATION_EVIDENCE_UNAVAILABLE": "segmentation_service_error",
    "DEPTH_EVIDENCE_UNAVAILABLE": "service_unavailable",
    "OBB_EVIDENCE_UNAVAILABLE": "service_unavailable",
    "PRESET_SCHEMA_INVALID": "protocol_contract_violation",
    "PRESET_HASH_MISMATCH": "protocol_contract_violation",
    "PRESET_PARAMETERS_INVALID": "protocol_contract_violation",
    "SEMANTIC_ROLE_INVALID": "protocol_contract_violation",
    "RGBD_OBSERVATION_UNAVAILABLE": "service_unavailable",
    "EXTERIOR_RGBD_UNAVAILABLE": "service_unavailable",
    "DETECTION_BOX_INVALID": "detector_no_candidate",
    "VLM_CROP_TOURNAMENT_FAILED": "protocol_contract_violation",
    "UNDECLARED_TOOL_DISPATCH": "protocol_contract_violation",
    "PAPER_MANIPULATION_INTERNAL_ERROR": "protocol_contract_violation",
}


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
    paper_outcome: dict[str, Any]
    semantic_evidence: list[dict[str, Any]]
    observation: dict[str, Any]
    mask: Any


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


def _tool(
    ctx: NodeContext,
    name: str,
    code: str,
    *,
    paper_branch: str | None = None,
    paper_operation: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    try:
        if paper_branch is not None and paper_operation is not None:
            kwargs["_paper_evidence"] = {
                "branch": paper_branch,
                "operation": paper_operation,
            }
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
    except PaperManipulationError:
        raise
    except Exception as error:
        raise PaperManipulationError(_SERVICE_FAILURE_CODE[name]) from error


def _selected_index(text: str, candidate_count: int) -> int:
    raw = text.strip()
    try:
        structured = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        structured = None
    if isinstance(structured, dict):
        if set(structured) != {"label"} or isinstance(structured["label"], bool):
            raise PaperManipulationError("VLM_DISAMBIGUATION_FAILED")
        raw = str(structured["label"]).strip()
        if not re.fullmatch(r"[A-Za-z]|[1-9][0-9]*", raw):
            raise PaperManipulationError("VLM_DISAMBIGUATION_FAILED")

    token = raw.upper()
    sole_label = re.fullmatch(r"\**\s*([A-Z])\s*\**", token)
    if sole_label:
        explicit_labels = {sole_label.group(1)}
    elif token.isdigit():
        index = int(token) - 1
        if 0 <= index < candidate_count:
            return index
        raise PaperManipulationError("VLM_DISAMBIGUATION_FAILED")
    else:
        explicit_labels = {
            match.group(1)
            for pattern in (
                r"\b(?:ANSWER|LABEL|MATCH|CHOICE)\b[^.!?\n]*?\bIS\s*\**([A-Z])\**\s*(?=$|[.!?])",
                r"\b(?:ANSWER|LABEL|MATCH|CHOICE)\s*:\s*\**([A-Z])\**\s*(?=$|[.!?])",
                r"\bI\s+(?:CHOOSE|PICK|SELECT)\s+\**([A-Z])\**\s*(?=$|[.!?])",
                r"(?<![A-Z0-9_])\**([A-Z])\**\s+IS\s+(?:THE\s+)?(?:BETTER|CORRECT|CLOSER)\s+(?:MATCH|CHOICE|ANSWER|LABEL)\b",
            )
            for match in re.finditer(pattern, token)
        }
    if len(explicit_labels) != 1:
        raise PaperManipulationError("VLM_DISAMBIGUATION_FAILED")
    index = ord(next(iter(explicit_labels))) - ord("A")
    if not 0 <= index < candidate_count:
        raise PaperManipulationError("VLM_DISAMBIGUATION_FAILED")
    return index


def _affirmative(text: str) -> bool:
    answers = set(re.findall(r"(?<![A-Z0-9_])(YES|NO)(?![A-Z0-9_])", text.strip().upper()))
    if answers == {"YES"}:
        return True
    if answers == {"NO"}:
        return False
    raise PaperManipulationError("VLM_DISAMBIGUATION_FAILED")


def _semantic_query(query: str) -> str:
    normalized = re.sub(r"_\d+$", "", query.strip()).replace("_", " ")
    return " ".join(normalized.split()) or query


def _box_xyxy(box: Any) -> tuple[float, float, float, float]:
    if isinstance(box, dict):
        return tuple(float(box[key]) for key in ("x1", "y1", "x2", "y2"))
    x1, y1, x2, y2 = box
    return float(x1), float(y1), float(x2), float(y2)


def _drop_contained_fragments(
    candidates: list[dict[str, Any]],
    containment_threshold: float = 0.7,
    max_fragment_side: float = 80.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def area(box: Any) -> float:
        x1, y1, x2, y2 = _box_xyxy(box)
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def intersection(first: Any, second: Any) -> float:
        ax1, ay1, ax2, ay2 = _box_xyxy(first)
        bx1, by1, bx2, by2 = _box_xyxy(second)
        return max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
            0.0, min(ay2, by2) - max(ay1, by1)
        )

    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        candidate_area = area(candidate["box"])
        containing_index = None
        if candidate_area > 0:
            for other_index, other in enumerate(candidates):
                if other_index == index or area(other["box"]) <= candidate_area:
                    continue
                if intersection(candidate["box"], other["box"]) / candidate_area >= containment_threshold:
                    containing_index = other_index
                    break
        x1, y1, x2, y2 = _box_xyxy(candidate["box"])
        longest_side = max(x2 - x1, y2 - y1)
        if containing_index is not None and longest_side < max_fragment_side:
            rejected.append(
                {
                    "candidate_index": index,
                    "contained_in_candidate_index": containing_index,
                    "reason": "small_fragment",
                }
            )
        else:
            kept.append(candidate)
    return kept, rejected


def _crop_tournament(image: Any, candidates: list[dict[str, Any]]) -> Any:
    """Build official-style padded, upscaled A/B crops without a full-image fallback."""
    import cv2
    import numpy as np

    source = np.asarray(image, dtype=np.uint8)
    height, width = source.shape[:2]
    tile_size = 384
    pad = 6
    cell_size = tile_size + 2 * pad
    colors = ((255, 0, 0), (0, 255, 0))
    sheet = np.full((cell_size, cell_size * len(candidates), 3), 255, dtype=np.uint8)
    for index, candidate in enumerate(candidates):
        x1, y1, x2, y2 = _box_xyxy(candidate["box"])
        box_width, box_height = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        x_pad, y_pad = int(0.30 * box_width), int(0.30 * box_height)
        left = max(0, int(x1) - x_pad)
        top = max(0, int(y1) - y_pad)
        right = min(width, int(x2) + x_pad)
        bottom = min(height, int(y2) + y_pad)
        if right <= left or bottom <= top:
            raise ValueError("degenerate padded crop")
        crop = source[top:bottom, left:right]
        scale = tile_size / max(crop.shape[:2])
        resized_height = max(1, int(round(crop.shape[0] * scale)))
        resized_width = max(1, int(round(crop.shape[1] * scale)))
        interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        resized = cv2.resize(crop, (resized_width, resized_height), interpolation=interpolation)
        tile = np.full((tile_size, tile_size, 3), 128, dtype=np.uint8)
        y_offset = (tile_size - resized_height) // 2
        x_offset = (tile_size - resized_width) // 2
        tile[y_offset : y_offset + resized_height, x_offset : x_offset + resized_width] = resized
        cv2.rectangle(tile, (0, 0), (tile_size - 1, 34), colors[index], -1)
        cv2.putText(
            tile,
            chr(65 + index),
            (8, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        x_start = index * cell_size + pad
        sheet[pad : pad + tile_size, x_start : x_start + tile_size] = tile
    return sheet


def _disambiguate(
    ctx: NodeContext,
    image: Any,
    candidates: list[dict[str, Any]],
    query: str,
    semantic_role: Literal["target", "destination"],
) -> tuple[int, str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate one candidate or run the canonical pairwise crop tournament."""
    object_name = _semantic_query(query)
    if len(candidates) == 1:
        try:
            crop_sheet = _crop_tournament(image, candidates)
        except Exception as error:
            raise PaperManipulationError("VLM_CROP_TOURNAMENT_FAILED") from error
        response = _tool(
            ctx,
            "vlm.query",
            "VLM_DISAMBIGUATION_FAILED",
            prompt=(
                f"Candidate A is the only detected object crop for {semantic_role} '{object_name}'. "
                "Is candidate A the requested object? Answer YES or NO."
            ),
            image=crop_sheet,
            paper_branch=semantic_role,
            paper_operation="vlm_selection",
        )
        evidence = _evidence(response, "vlm", "VLM_EVIDENCE_UNAVAILABLE")
        if not _affirmative(str(response.get("text", ""))):
            raise PaperManipulationError("VLM_DISAMBIGUATION_FAILED")
        return (
            0,
            "single_candidate_validation",
            [
                {
                    "round_index": 0,
                    "match_index": 0,
                    "candidate_indices": [0],
                    "selected_candidate_index": 0,
                    "raw_response_sha256": _hash(response.get("text", "")),
                }
            ],
            [evidence],
        )

    bracket = list(range(len(candidates)))
    records: list[dict[str, Any]] = []
    evidence_records: list[dict[str, Any]] = []
    round_index = 0
    while len(bracket) > 1:
        next_round: list[int] = []
        for match_index, offset in enumerate(range(0, len(bracket), 2)):
            if offset + 1 >= len(bracket):
                next_round.append(bracket[offset])
                continue
            left, right = bracket[offset], bracket[offset + 1]
            try:
                crop_sheet = _crop_tournament(image, [candidates[left], candidates[right]])
            except Exception as error:
                raise PaperManipulationError("VLM_CROP_TOURNAMENT_FAILED") from error
            response = _tool(
                ctx,
                "vlm.query",
                "VLM_DISAMBIGUATION_FAILED",
                prompt=(
                    f"The image contains two object crops labeled A and B. Exactly one is the "
                    f"better match for {semantic_role} '{object_name}'. Which one is it? "
                    "Answer with just A or B."
                ),
                image=crop_sheet,
                paper_branch=semantic_role,
                paper_operation="vlm_selection",
            )
            evidence_records.append(_evidence(response, "vlm", "VLM_EVIDENCE_UNAVAILABLE"))
            local_index = _selected_index(str(response.get("text", "")), 2)
            winner = (left, right)[local_index]
            next_round.append(winner)
            records.append(
                {
                    "round_index": round_index,
                    "match_index": match_index,
                    "candidate_indices": [left, right],
                    "selected_candidate_index": winner,
                    "raw_response_sha256": _hash(response.get("text", "")),
                }
            )
        bracket = next_round
        round_index += 1
    return bracket[0], "pairwise_crop_tournament", records, evidence_records


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
        ctx,
        "grounding-dino.detect",
        "DETECTION_FAILED",
        paper_branch=semantic_role,
        paper_operation="detection",
        image=camera["rgb"],
        query="object.",
    )
    detector_evidence = _evidence(detected, "learned_model", "DETECTION_EVIDENCE_UNAVAILABLE")
    raw_candidates = list(detected.get("detections", []))
    if not raw_candidates:
        raise PaperManipulationError("DETECTION_CANDIDATES_INSUFFICIENT")
    valid_candidates: list[dict[str, Any]] = []
    valid_source_indices: list[int] = []
    candidate_filtering: list[dict[str, Any]] = []
    for source_index, candidate in enumerate(raw_candidates):
        if _valid_box(candidate.get("box"), camera["rgb"]):
            valid_candidates.append(candidate)
            valid_source_indices.append(source_index)
        else:
            candidate_filtering.append(
                {"candidate_index": source_index, "reason": "invalid_box"}
            )
    if not valid_candidates:
        raise PaperManipulationError("DETECTION_BOX_INVALID")
    candidates, fragment_rejections = _drop_contained_fragments(valid_candidates)
    candidate_source_indices = [
        valid_source_indices[
            next(index for index, valid in enumerate(valid_candidates) if valid is candidate)
        ]
        for candidate in candidates
    ]
    for rejection in fragment_rejections:
        candidate_filtering.append(
            {
                "candidate_index": valid_source_indices[rejection["candidate_index"]],
                "contained_in_candidate_index": valid_source_indices[
                    rejection["contained_in_candidate_index"]
                ],
                "reason": rejection["reason"],
            }
        )
    if len(candidates) > 8:
        candidate_filtering.extend(
            {"candidate_index": source_index, "reason": "tournament_limit"}
            for source_index in candidate_source_indices[8:]
        )
        candidates = candidates[:8]
        candidate_source_indices = candidate_source_indices[:8]
    selected_index, selection_mode, tournament_records, vlm_evidence = _disambiguate(
        ctx, camera["rgb"], candidates, query, semantic_role
    )
    selected = candidates[selected_index]
    segmented = _tool(
        ctx,
        "sam3.segment_box",
        "SEGMENTATION_FAILED",
        paper_branch=semantic_role,
        paper_operation="segmentation",
        image=camera["rgb"],
        box=selected["box"],
    )
    segment_evidence = _evidence(segmented, "learned_model", "SEGMENTATION_EVIDENCE_UNAVAILABLE")
    masks = segmented.get("masks", [])
    if not masks:
        raise PaperManipulationError("SEGMENTATION_EMPTY")
    scores = segmented.get("scores", [])
    try:
        selected_score = float(scores[0])
    except (IndexError, TypeError, ValueError):
        raise PaperManipulationError("SEGMENTATION_LOW_CONFIDENCE") from None
    if not math.isfinite(selected_score) or not 0.3 <= selected_score <= 1.0:
        raise PaperManipulationError("SEGMENTATION_LOW_CONFIDENCE")
    projected = _tool(
        ctx,
        "geometry.mask_to_world_points",
        "DEPTH_PROJECTION_FAILED",
        mask=masks[0],
        depth=camera["depth"],
        intrinsics=camera["intrinsics"],
        camera_pose=camera["pose"],
        paper_branch=semantic_role,
        paper_operation="depth_obb",
    )
    depth_evidence = _evidence(projected, "algorithm_service", "DEPTH_EVIDENCE_UNAVAILABLE")
    if depth_evidence.get("valid_depth_count", 0) <= 0 or not projected.get("points"):
        raise PaperManipulationError("DEPTH_POINTS_EMPTY")
    fitted = _tool(
        ctx,
        "geometry.filter_and_compute_obb",
        "OBB_FIT_FAILED",
        paper_branch=semantic_role,
        paper_operation="depth_obb",
        points=projected["points"],
    )
    obb_evidence = _evidence(fitted, "algorithm_service", "OBB_EVIDENCE_UNAVAILABLE")
    obb = fitted.get("obb")
    if not isinstance(obb, dict):
        raise PaperManipulationError("OBB_FIT_FAILED")
    lineage = {
        "semantic_role": semantic_role,
        "source_camera_index": camera_index,
        "query": query,
        "semantic_query": _semantic_query(query),
        "observation_sha256": _hash(observation),
        "detector_candidates_raw": raw_candidates,
        "detector_candidates": candidates,
        "candidate_filtering": candidate_filtering,
        "candidate_boxes_sha256": _hash([item["box"] for item in candidates]),
        "crop_mapping": [
            {
                "round_index": record["round_index"],
                "match_index": record["match_index"],
                "crop_label": chr(65 + local_index),
                "candidate_index": candidate_index,
                "source_candidate_index": candidate_source_indices[candidate_index],
                "box_sha256": _hash(candidates[candidate_index]["box"]),
            }
            for record in tournament_records
            for local_index, candidate_index in enumerate(record["candidate_indices"])
        ],
        "vlm_decision": {
            "selection_mode": selection_mode,
            "tournament_records": tournament_records,
            "selected_candidate_index": selected_index,
        },
        "selected_candidate_index": selected_index,
        "selected_source_candidate_index": candidate_source_indices[selected_index],
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
            "vlm": vlm_evidence[-1],
            "vlm_rounds": vlm_evidence,
            "segmenter": segment_evidence,
            "depth": depth_evidence,
            "obb": obb_evidence,
        },
        "preset_trace": preset_trace,
        "paper_outcome": {
            "status": "success",
            "failure_code": None,
            "source": "canonical_script",
        },
        "semantic_evidence": [
            {"kind": "vlm_choice", "branch": semantic_role},
        ],
        "fallback_used": False,
        "decision_path": [
            "observe_rgbd",
            "broad_detection",
            selection_mode,
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
        "paper_outcome": lineage["paper_outcome"],
        "semantic_evidence": lineage["semantic_evidence"],
        "observation": observation,
        "mask": masks[0],
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
    failure_code = _PAPER_FAILURE_CODE[code]
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
        "paper_outcome": {
            "status": "failure",
            "failure_code": failure_code,
            "source": "canonical_script",
        },
        "semantic_evidence": [],
        "observation": {},
        "mask": None,
    }
