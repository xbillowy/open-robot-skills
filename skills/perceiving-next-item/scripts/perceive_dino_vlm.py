"""Shared-observation container/item perception with unchanged node outputs.

The first caller performs broad detection, one bounded joint selection, and
segmentation of both selected regions. A complementary caller reuses those
results only for an exactly matching exterior observation in the same episode.
Its query must verify the selected object before returning geometry.

Only a successful explicit whole-scene no-match decision returns found=False.
Observation, selection, verification, segmentation, and geometry errors raise
PerceptionFailed. There are no retries, disk caches, or robot motions.
Safety guard BaseExceptions propagate untouched.
"""
from __future__ import annotations

import builtins
import math
from typing import TypedDict

import numpy as np

from gap import NodeContext
from gap_core.errors import PerceptionFailed
from gap_core.types import CameraFrame, Mask, PointCloud


class Output(TypedDict):
    found: bool
    cloud: PointCloud
    mask: Mask
    score: float


_MAX_BOXES = 32
_REGISTRY_LIMIT = 32
_DIGITS = (
    ('111', '101', '101', '101', '111'),
    ('010', '110', '010', '010', '111'),
    ('111', '001', '111', '100', '111'),
    ('111', '001', '111', '001', '111'),
    ('101', '101', '111', '001', '001'),
    ('111', '100', '111', '001', '111'),
    ('111', '100', '111', '101', '111'),
    ('111', '001', '010', '010', '010'),
    ('111', '101', '111', '101', '111'),
    ('111', '101', '111', '001', '111'),
)
_BROAD_ITEM = (
    'a movable tabletop item still outside the destination container; '
    'exclude the destination container itself, its contents, the robot, '
    'the table, fixtures, shadows, and background'
)


def _empty_cloud() -> PointCloud:
    return {'points': np.zeros((0, 3), dtype=np.float32)}


def _none() -> Output:
    return {
        'found': False,
        'cloud': _empty_cloud(),
        'mask': np.zeros((0, 0), dtype=np.uint8),
        'score': 0.0,
    }


def _role(name: str, description: str) -> str:
    desc = description.lower()
    if any(word in desc for word in ('never', 'exclude', 'except', 'not the')):
        return 'item'
    words = name.lower().replace('-', ' ').split()
    if name.strip().lower() in ('box', 'container', 'bowl', 'tray', 'plate'):
        return 'container'
    if any(word in words for word in ('basket', 'bin', 'tote', 'crate')):
        return 'container'
    if 'container' in words:
        return 'container'
    return 'item'


def _camera(cameras):
    if not isinstance(cameras, list) or not cameras:
        raise PerceptionFailed('A nonempty camera observation is required.')
    exterior = [
        cam for cam in cameras
        if 'eye_in_hand' not in str(cam.get('name') or '').lower()
        and 'wrist' not in str(cam.get('name') or '').lower()
    ]
    if not exterior:
        raise PerceptionFailed('No exterior camera is available for shared perception.')
    cam = sorted(exterior, key=lambda value: str(value.get('name') or ''))[0]
    rgb = np.asarray(cam['rgb'])
    depth = np.asarray(cam['depth'])
    intrinsics = np.asarray(cam['intrinsics'])
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 16:
        raise PerceptionFailed('The exterior RGB observation is malformed.')
    if depth.shape != rgb.shape[:2] or not np.isfinite(depth).any():
        raise PerceptionFailed('The exterior depth observation is malformed.')
    if not np.isfinite(rgb).all() or not np.isfinite(intrinsics).all():
        raise PerceptionFailed('Camera data contains invalid numeric values.')
    if cam.get('pose') is None:
        raise PerceptionFailed('The exterior camera world pose is missing.')
    return cam


def _snapshot(cam):
    snapshot = dict(
        name=cam.get('name'),
        rgb=np.asarray(cam['rgb']).copy(),
        depth=np.asarray(cam['depth']).copy(),
        intrinsics=np.asarray(cam['intrinsics']).copy(),
        pose=repr(cam['pose']),
    )
    return snapshot


def _same(snapshot, cam):
    return (
        snapshot['name'] == cam.get('name')
        and snapshot['pose'] == repr(cam['pose'])
        and all(
            np.array_equal(snapshot[field], np.asarray(cam[field]), equal_nan=True)
            for field in ('rgb', 'depth', 'intrinsics')
        )
    )


def _annotate(rgb, boxes):
    image = np.asarray(rgb).astype(np.uint8).copy()
    height, width = image.shape[:2]
    for index, box in enumerate(boxes, 1):
        x1, y1, x2, y2 = (int(box[key]) for key in ('x1', 'y1', 'x2', 'y2'))
        image[y1:min(y1 + 2, height), x1:x2] = (255, 0, 255)
        image[max(y2 - 2, y1):y2, x1:x2] = (255, 0, 255)
        image[y1:y2, x1:min(x1 + 2, width)] = (255, 0, 255)
        image[y1:y2, max(x2 - 2, x1):x2] = (255, 0, 255)
        digits = str(index)
        left = min(x1, width - len(digits) * 8)
        top = min(y1, height - 12)
        image[top:top + 12, left:left + len(digits) * 8] = 0
        for offset, digit in enumerate(digits):
            for row, pattern in enumerate(_DIGITS[int(digit)]):
                for column, bit in enumerate(pattern):
                    if bit == '1':
                        yy = top + 1 + row * 2
                        xx = left + offset * 8 + column * 2
                        image[yy:yy + 2, xx:xx + 2] = 255
    return image


def _boxes(ctx, cam, box_threshold, text_threshold, dino_prompt):
    response = ctx.tool(
        'grounding-dino.detect', image=cam['rgb'],
        query='object. container. ' + dino_prompt,
        box_threshold=box_threshold, text_threshold=text_threshold,
    )
    detections = response['detections']
    if not isinstance(detections, (list, tuple)):
        raise PerceptionFailed('Broad detection returned a malformed result.')
    height, width = np.asarray(cam['rgb']).shape[:2]
    boxes = []
    for detection in detections:
        values = [float(detection['box'][key]) for key in ('x1', 'y1', 'x2', 'y2')]
        if not all(math.isfinite(value) for value in values):
            raise PerceptionFailed('Broad detection returned a nonfinite box.')
        x1, y1, x2, y2 = values
        x1, x2 = max(0, math.floor(x1)), min(width, math.ceil(x2))
        y1, y2 = max(0, math.floor(y1)), min(height, math.ceil(y2))
        if x2 <= x1 or y2 <= y1:
            raise PerceptionFailed('Broad detection returned a degenerate box.')
        box = dict(zip(('x1', 'y1', 'x2', 'y2'), (x1, y1, x2, y2)))
        if box not in boxes:
            boxes.append(box)
    if len(boxes) > _MAX_BOXES:
        raise PerceptionFailed('Broad detection exceeded the bounded joint-selection capacity.')
    return boxes


def _joint(ctx, cam, boxes, role, query):
    container_query = query if role == 'container' else 'the destination receptacle'
    item_query = query if role == 'item' else _BROAD_ITEM
    prompt = (
        'Use the entire scene and the numbered detection boxes. Select BOTH '
        'the destination container and one remaining eligible item jointly. '
        f'Container requirement: {container_query}. '
        f'Item requirement: {item_query}. '
        'All requirements and exclusions are binding. The item must be a '
        'different physical object from the container, resting outside it, '
        'and its box must cover the whole item rather than a label or fragment. '
        'Objects already inside the container are ineligible. '
        'Use ITEM=NONE only after positively inspecting the entire visible '
        'tabletop and deciding that no eligible item remains. Empty detections '
        'alone do not establish absence. If a potentially eligible object is '
        'unboxed, occluded so absence cannot be decided, or otherwise uncertain, '
        'use ITEM=UNKNOWN. A missing or uncertain container is UNKNOWN. '
        'Return exactly CONTAINER=<number or UNKNOWN>;ITEM=<number or NONE or UNKNOWN>. '
        f'Available boxes in image pixel coordinates: {boxes!r}'
    )
    response = ctx.tool('vlm.query', image=_annotate(cam['rgb'], boxes), prompt=prompt)
    text = response['text'].strip()
    parts = text.split(';')
    if len(parts) != 2 or not parts[0].startswith('CONTAINER=') or not parts[1].startswith('ITEM='):
        raise PerceptionFailed('Joint selection did not return an explicit valid decision.')
    container = parts[0][len('CONTAINER='):].strip()
    item = parts[1][len('ITEM='):].strip()
    if not container.isdecimal() or not 1 <= int(container) <= len(boxes):
        raise PerceptionFailed('The destination container could not be selected confidently.')
    container_index = int(container) - 1
    if item == 'NONE':
        return container_index, None
    if not item.isdecimal() or not 1 <= int(item) <= len(boxes):
        raise PerceptionFailed('The remaining-item decision is unknown or malformed.')
    item_index = int(item) - 1
    if item_index == container_index:
        raise PerceptionFailed('Joint selection confused the item with the container.')
    return container_index, item_index


def _segment(ctx, cam, box, min_score, min_points):
    response = ctx.tool('sam3.segment_box', image=cam['rgb'], box=box)
    masks, scores = response['masks'], response['scores']
    if len(masks) == 0 or len(masks) != len(scores):
        raise PerceptionFailed('A selected region has no usable segmentation.')
    numeric_scores = [float(value) for value in scores]
    if not all(math.isfinite(value) for value in numeric_scores):
        raise PerceptionFailed('Segmentation confidence is malformed.')
    index = max(range(len(numeric_scores)), key=numeric_scores.__getitem__)
    score = numeric_scores[index]
    mask = np.asarray(masks[index])
    if score < min_score or mask.shape != np.asarray(cam['depth']).shape:
        raise PerceptionFailed('A selected region failed segmentation validation.')
    if not np.isfinite(mask).all() or not np.any(mask > 0):
        raise PerceptionFailed('A selected region has an invalid or empty mask.')
    mask = (mask > 0).astype(np.uint8) * 255
    cloud = ctx.tool(
        'geometry.mask_to_world_points', mask=mask, depth=cam['depth'],
        intrinsics=cam['intrinsics'], camera_pose=cam['pose'],
    )['points']
    points = np.asarray(cloud['points'], dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise PerceptionFailed('A selected region has malformed world geometry.')
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < min_points:
        raise PerceptionFailed('A selected region has insufficient valid world points.')
    return {'found': True, 'cloud': {'points': points.copy()}, 'mask': mask.copy(), 'score': score}


def _verify_shared(ctx, cam, boxes, selected, role, query):
    response = ctx.tool(
        'vlm.query_yes_no', image=_annotate(cam['rgb'], boxes),
        prompt=(
            f'In the entire scene, does numbered box {selected + 1} identify '
            f'the requested {role}, satisfying ALL of these binding requirements: '
            f'{query}? Respect every exclusion. '
            + ('The item must be outside and distinct from the destination container. '
               if role == 'item' else 'It must be the destination receptacle. ')
            + 'Answer YES only if this selected whole object clearly satisfies every requirement.'
        ),
    )
    answer = response.get('answer')
    if type(answer) is not bool:
        raise PerceptionFailed('Shared-selection verification returned an unknown decision.')
    if not answer:
        raise PerceptionFailed(
            'The shared selection does not satisfy the requested query; absence has not been established.'
        )


def run(
    ctx: NodeContext,
    cameras: list[CameraFrame],
    object_name: str,
    text_prompts: list[str] | None = None,
    min_points: int = 10,
    min_score: float = 0.3,
    use_multiview: bool = True,
    box_threshold: float = 0.20,
    text_threshold: float = 0.20,
    dino_prompt: str = 'object.',
    object_description: str = '',
    reject_unverified: bool = False,
) -> Output:
    """Return the requested member of a shared container/item localization.

    use_multiview remains accepted; one exterior frame supplies consistent
    masks and clouds. reject_unverified remains accepted; every query now
    requires an explicit selection regardless of its value. text_prompts are
    binding appearance alternatives, not segmentation retries.
    """
    registry = None
    episode_key = None
    try:
        if not isinstance(object_name, str) or not object_name.strip():
            raise PerceptionFailed('A nonempty object name is required.')
        if not isinstance(object_description, str) or not isinstance(dino_prompt, str):
            raise PerceptionFailed('Perception descriptions must be strings.')
        if type(min_points) is not int or min_points < 1:
            raise PerceptionFailed('min_points must be a positive integer.')
        if any(not math.isfinite(float(value)) or not 0 <= float(value) <= 1
               for value in (min_score, box_threshold, text_threshold)):
            raise PerceptionFailed('Perception confidence thresholds must lie within zero and one.')
        if text_prompts is not None and (
            not isinstance(text_prompts, list)
            or any(not isinstance(value, str) or not value.strip() for value in text_prompts)
        ):
            raise PerceptionFailed('Appearance alternatives must be nonempty strings.')
        query = object_name + '. ' + object_description
        if text_prompts:
            query += ' Appearance alternatives: ' + ' OR '.join(text_prompts)
        role = _role(object_name, object_description)
        cam = _camera(cameras)
        owner = getattr(ctx, '_trace', None)
        if owner is None:
            owner = ctx
        episode_key = id(owner)
        registry = builtins.__dict__.setdefault('_ors_next_item_joint_observation_v1', {})
        previous = registry.pop(episode_key, None)
        settings = (box_threshold, text_threshold, dino_prompt)
        reusable = (
            previous is not None
            and previous['owner'] is owner
            and previous['role'] != role
            and previous['settings'] == settings
            and _same(previous['snapshot'], cam)
        )
        if reusable:
            selected = previous['indices'][role]
            out = previous['outputs'][role]
            if selected is not None:
                _verify_shared(ctx, cam, previous['boxes'], selected, role, query)
                if out['score'] < min_score or len(out['cloud']['points']) < min_points:
                    raise PerceptionFailed('The shared localization does not meet the requested quality thresholds.')
            # A container-first NONE explicitly covers the broad movable-item
            # set, including any requested subset. Item-first reuse can only
            # return the positively selected container, never another item.
            return {
                'found': out['found'],
                'cloud': {'points': out['cloud']['points'].copy()},
                'mask': out['mask'].copy(),
                'score': out['score'],
            }
        boxes = _boxes(ctx, cam, box_threshold, text_threshold, dino_prompt)
        container_index, item_index = _joint(ctx, cam, boxes, role, query)
        indices = {'container': container_index, 'item': item_index}
        outputs = {
            'container': _segment(ctx, cam, boxes[container_index], min_score, min_points),
            'item': (_none() if item_index is None else
                     _segment(ctx, cam, boxes[item_index], min_score, min_points)),
        }
        while len(registry) >= _REGISTRY_LIMIT:
            del registry[next(iter(registry))]
        registry[episode_key] = {
            'owner': owner, 'role': role, 'settings': settings,
            'snapshot': _snapshot(cam), 'boxes': boxes,
            'indices': indices, 'outputs': outputs,
        }
        out = outputs[role]
        return {
            'found': out['found'],
            'cloud': {'points': out['cloud']['points'].copy()},
            'mask': out['mask'].copy(),
            'score': out['score'],
        }
    except PerceptionFailed:
        if registry is not None and episode_key is not None:
            registry.pop(episode_key, None)
        raise
    except Exception as exc:
        if registry is not None and episode_key is not None:
            registry.pop(episode_key, None)
        raise PerceptionFailed('Shared container/item perception failed; no absence decision is available.') from exc
