from typing import TypedDict, Optional
import numpy as np
from gap import NodeContext
from gap_core.errors import PerceptionFailed, PlanningFailed
from gap_core.types import OrientedBoundingBox, Se3Pose, Vec3

class Output(TypedDict):
    found: bool
    handle_obb: Optional[OrientedBoundingBox]
    grasp_pose: Optional[Se3Pose]
    pull_axis: Optional[Vec3]

def vector(v):
    if isinstance(v, dict):
        return np.array([v['x'], v['y'], v['z']], dtype=float)
    return np.asarray(v, dtype=float)

def cloud_array(points):
    if isinstance(points, dict):
        points = points['points']
    if len(points) and isinstance(points[0], dict):
        return np.array([vector(p) for p in points], dtype=float)
    return np.asarray(points, dtype=float)

def vec(v):
    return dict(zip(('x', 'y', 'z'), map(float, v)))

def run(ctx: NodeContext, parent_prompt: str = 'cabinet', subpart_prompt: str = 'drawer handle. pull. knob.', height_rank: str = '') -> Output:
    empty = {'found': False, 'handle_obb': None, 'grasp_pose': None, 'pull_axis': None}
    obs = ctx.tool('robot.get_observation')
    hand = vector(ctx.tool('robot.get_ee_pose')['pose']['position'])
    candidates = []
    # Three views bound acquisition work.
    for camera in obs['cameras'][:3]:
        image = camera.get('rgb', camera.get('image'))
        camera_pose = camera.get('pose', camera.get('camera_pose'))
        if image is None or camera_pose is None or camera.get('depth') is None:
            continue
        parents = ctx.tool('grounding-dino.detect', image=image, query=parent_prompt)['detections']
        if not parents:
            continue
        parent_box = max(parents, key=lambda d: d['score'])['box']
        masks = ctx.tool('sam3.segment_box', image=image, box=parent_box)['masks']
        if not masks:
            continue
        points = ctx.tool('geometry.mask_to_world_points', mask=masks[0], depth=camera['depth'], intrinsics=camera['intrinsics'], camera_pose=camera_pose)['points']
        try:
            parent = ctx.tool('geometry.compute_obb', points=points)['obb']
        except PerceptionFailed:
            continue
        center = vector(parent['center'])
        try:
            front = ctx.tool('geometry.front_grasp_from_obb', obb=parent, approach_hint=vec(hand-center))
        except PlanningFailed:
            continue
        normal = vector(front['approach_direction'])
        normal[2] = 0.0
        length = float(np.linalg.norm(normal))
        # Reject numerically degenerate axes.
        if length < 1e-6:
            continue
        normal /= length
        if float(normal @ (hand-center)) < 0:
            normal = -normal
        # Build a parent-front basis without assuming an OBB rotation field.
        tangent = np.array([-normal[1], normal[0], 0.0])
        axes = np.column_stack((tangent, normal, np.cross(tangent, normal)))
        parent_points = cloud_array(points)
        # Four valid points are the minimum for a spatial extent.
        if parent_points.ndim != 2 or parent_points.shape[1] != 3 or len(parent_points) < 4:
            continue
        parent_points = parent_points[np.all(np.isfinite(parent_points), axis=1)]
        if len(parent_points) < 4:
            continue
        projected = (parent_points-center) @ axes
        # Robust percentiles suppress isolated depth outliers.
        low, high = np.percentile(projected, [2, 98], axis=0)
        detections = ctx.tool('grounding-dino.detect', image=image, query=subpart_prompt)['detections']
        # Sixteen candidates bound segmentation work.
        for detection in sorted(detections, key=lambda d: d['score'], reverse=True)[:16]:
            masks = ctx.tool('sam3.segment_box', image=image, box=detection['box'])['masks']
            if not masks:
                continue
            part_points = ctx.tool('geometry.mask_to_world_points', mask=masks[0], depth=camera['depth'], intrinsics=camera['intrinsics'], camera_pose=camera_pose)['points']
            try:
                part = ctx.tool('geometry.compute_obb', points=part_points)['obb']
            except PerceptionFailed:
                continue
            p = vector(part['center'])
            local = (p-center) @ axes
            # Six centimetres allows projecting hardware, not remote fixtures.
            if np.any(local < low - 0.06) or np.any(local > high + 0.06):
                continue
            # Ten centimetres bounds uncertainty behind the observed front.
            if local[1] < high[1] - 0.10:
                continue
            sizes = np.sort(vector(part['extent']))
            # Millimetre floor bounds the elongation denominator.
            elongation = float(sizes[2] / max(sizes[1], 0.001))
            try:
                grasp = ctx.tool('geometry.front_grasp_from_obb', obb=part, approach_hint=vec(normal))['grasp_pose']
            except PlanningFailed:
                continue
            candidates.append((float(p[2]), elongation, float(detection['score']), part, grasp, vec(normal)))
        if candidates:
            break
    if not candidates:
        return empty
    # Aspect ratio threshold prefers bars while retaining round-pull fallback.
    bars = [c for c in candidates if c[1] >= 1.6]
    pool = bars if bars else candidates
    pool.sort(key=lambda c: c[0])
    rank = height_rank.lower().strip()
    if rank in ('top', 'upper', 'highest'):
        chosen = pool[-1]
    elif rank in ('bottom', 'lower', 'lowest'):
        chosen = pool[0]
    elif rank == 'middle':
        chosen = pool[len(pool)//2]
    else:
        # Cap elongation influence before confidence tie-breaking.
        chosen = max(pool, key=lambda c: (min(c[1], 10.0), c[2]))
    return {'found': True, 'handle_obb': chosen[3], 'grasp_pose': chosen[4], 'pull_axis': chosen[5]}
