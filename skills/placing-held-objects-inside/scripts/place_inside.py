from typing import TypedDict
import logging
import math
import numpy as np
from scipy.spatial.transform import Rotation
from gap import NodeContext
from gap_core.types import OrientedBoundingBox, Se3Pose, Vec3, Mask, PointCloud

class Output(TypedDict):
    status: str
    reason: str
    place_position: Vec3

def run(ctx: NodeContext, container_obb: OrientedBoundingBox, ee_pose_at_grasp: Se3Pose, target_obb: OrientedBoundingBox, target_mask: Mask | None = None, container_mask: Mask | None = None, container_cloud: PointCloud | None = None, strategy: str = 'open', region: str = 'center', disposition: str = 'release', object_description: str = '', tcp_length: float = 0.103, clearance: float = 0.16, clearance_step: float = 0.04, rim_offset: float = 0.06, seating_margin: float = 0.012, position_tolerance: float = 0.018, angle_tolerance: float = 0.12, support_tolerance: float = 0.035, footprint_margin: float = 0.005, region_shift: float = 0.52, region_long_scale: float = 0.30, region_short_scale: float = 0.42, insertion_step: float = 0.025, opening_fraction: float = 0.8, pinch_min: float = 0.025, pinch_max: float = 0.94, depth_change: float = 0.012, visible_height: float = 0.06, minimum_points: int = 25, settle_steps: int = 60, slot: float = 0.0) -> Output:
    log = logging.getLogger('placement')
    keys = ('x', 'y', 'z')
    qkeys = ('x', 'y', 'z', 'w')
    place = dict(container_obb['center'])
    verified_inside = False
    opened = False

    def vec(d):
        return np.array([d[k] for k in keys], dtype=float)

    def xyz(a):
        return dict(zip(keys, map(float, a)))

    def rot(q):
        return Rotation.from_quat([q[k] for k in qkeys])

    def tcp(p):
        return vec(p['position']) + rot(p['rotation']).apply(np.array([0, 0, 1]) * tcp_length)

    def read():
        p = ctx.tool('robot.get_ee_pose')['pose']
        return tcp(p), rot(p['rotation'])

    def result(status, reason):
        if status == 'failed':
            log.warning('%s', reason)
        return {'status': status, 'reason': reason, 'place_position': place}

    def move(goal, desired, vertical=False):
        command = np.asarray(goal, dtype=float).copy()
        q = dict(zip(qkeys, map(float, desired.as_quat())))
        for attempt in range(3):
            pose = {'position': xyz(command), 'rotation': q}
            try:
                if attempt == 0 and vertical:
                    current, unused = read()
                    delta = goal - current
                    distance = float(np.linalg.norm(delta))
                    if distance > position_tolerance:
                        js = ctx.tool('robot.get_observation')['arms'][0]['joint_state']
                        plan = ctx.tool('curobo.plan_directed_linear', start_joint_position=js, explicit_direction=xyz(delta / distance), distance=distance, allowed_axes=['Z'], endpoint_mode='DISTANCE', orientation_mode='LOCK')
                        if plan.get('success') and plan.get('trajectory'):
                            ctx.tool('robot.execute_trajectory', trajectory=plan['trajectory'])
                        else:
                            log.info('Directed descent unavailable; trying Cartesian endpoint')
                            ctx.tool('robot.go_to_pose_cartesian', pose=pose)
                elif attempt == 2:
                    ctx.tool('robot.go_to_pose', pose=pose)
                else:
                    ctx.tool('robot.go_to_pose_cartesian', pose=pose)
            except Exception as exc:
                log.info('Motion alternative %s unavailable: %s', attempt, exc)
            try:
                actual, actual_rotation = read()
                error = goal - actual
                if np.linalg.norm(error) <= position_tolerance and (desired.inv() * actual_rotation).magnitude() <= angle_tolerance:
                    return True
                command += np.clip(error, -insertion_step, insertion_step)
            except Exception as exc:
                log.info('Arrival read unavailable: %s', exc)
        log.info('Required arrival was not measured')
        return False

    def cloud(cam, mask=None):
        if mask is None:
            mask = np.ones(np.asarray(cam['depth']).shape, dtype=np.uint8) * 255
        raw = ctx.tool('geometry.mask_to_world_points', mask=mask, depth=cam['depth'], intrinsics=cam['intrinsics'], camera_pose=cam['pose'])['points']
        pts = np.asarray(raw['points'], dtype=float).reshape(-1, 3)
        return pts[np.isfinite(pts).all(axis=1)]

    try:
        if strategy not in ('open', 'shallow', 'sliding', 'divided') or region not in ('center', 'back', 'front', 'left', 'right') or disposition not in ('release', 'stabilize'):
            return result('failed', 'Unsupported placement selection')
        before = ctx.tool('robot.get_observation')
        initial, initial_rotation = read()
        grasp_tip = tcp(ee_pose_at_grasp)
        object_center = vec(target_obb['center']) + initial - grasp_tip
        offset = object_center - initial
        c = vec(container_obb['center'])
        e = vec(container_obb['extent'])
        axes = rot(container_obb['orientation']).as_matrix()
        target_axes = rot(target_obb['orientation']).as_matrix()
        object_extent = vec(target_obb['extent'])
        zone = c.copy()
        zone_extent = e.copy()
        long_axis = int(np.argmax(e[:2]))
        short_axis = 1 - long_axis
        cam = next((v for v in before['cameras'] if 'eye_in_hand' not in v['name']), before['cameras'][0])
        toward_camera = axes[:, short_axis].copy()
        if np.dot(toward_camera, vec(cam['pose']['position']) - c) < 0:
            toward_camera = -toward_camera
        if strategy == 'divided':
            zone_extent[long_axis] *= region_long_scale
            zone_extent[short_axis] *= region_short_scale
        if region in ('back', 'front'):
            sign = -1 if region == 'back' else 1
            zone += toward_camera * e[short_axis] * region_shift * sign
        elif region in ('left', 'right'):
            side = np.cross(np.array([0, 0, 1]), toward_camera)
            zone += side * e[long_axis] * region_shift * (1 if region == 'left' else -1)
        zone += axes[:, long_axis] * e[long_axis] * float(np.clip(slot, -1, 1)) * region_shift
        desired_rotation = initial_rotation
        turn = Rotation.identity()
        if strategy == 'divided':
            object_long = int(np.argmax(object_extent[:2]))
            source_axis = target_axes[:, object_long]
            destination_axis = axes[:, long_axis]
            delta = math.atan2(destination_axis[1], destination_axis[0]) - math.atan2(source_axis[1], source_axis[0])
            delta = (delta + math.pi / 2) % math.pi - math.pi / 2
            turn = Rotation.from_euler('z', delta)
            desired_rotation = turn * initial_rotation
        offset = turn.apply(offset)
        held_axes = turn.as_matrix() @ target_axes
        world_extent = np.abs(held_axes) @ object_extent
        local_extent = np.abs(axes.T @ held_axes) @ object_extent
        rim = c[2] + e[2]
        floor = c[2] - e[2]
        goal = zone - offset
        if strategy == 'open':
            goal[2] = rim + rim_offset
        elif strategy in ('shallow', 'sliding'):
            desired_center_z = (rim if strategy == 'shallow' else floor) + world_extent[2] + seating_margin
            goal[2] = desired_center_z - offset[2]
        else:
            goal[2] = rim + insertion_step
        place = xyz(goal)
        baseline = {}
        for camera in before['cameras'][:2]:
            baseline[camera['name']] = camera

        def inside(center):
            local = (center - zone) @ axes
            return bool(np.all(np.abs(local[:2]) < np.maximum(zone_extent[:2] - footprint_margin, footprint_margin)))

        if not inside(goal + offset):
            return result('failed', 'Computed object center lies outside the selected opening')
        high = max(initial[2], rim + clearance + max(0.0, world_extent[2] - offset[2]), goal[2] + clearance)
        transported = False
        for alternative in range(2):
            height = high + alternative * clearance_step
            lift = initial.copy()
            lift[2] = height
            if not move(lift, initial_rotation, vertical=True):
                log.info('Vertical lift unavailable; trying raised destination approach')
            above = goal.copy()
            above[2] = height
            if move(above, initial_rotation):
                if move(above, desired_rotation):
                    transported = True
                    break
        if not transported:
            return result('failed', 'No measured arrival above the receiving region')
        arrived = False
        for alignment in range(2):
            current, unused = read()
            if strategy == 'divided':
                count = min(16, max(1, int(math.ceil(abs(current[2] - goal[2]) / max(insertion_step, position_tolerance)))))
                start = current.copy()
                for step in range(1, count + 1):
                    intermediate = goal.copy()
                    intermediate[2] = start[2] + (goal[2] - start[2]) * step / count
                    if not move(intermediate, desired_rotation, vertical=True):
                        break
                else:
                    arrived = True
            else:
                arrived = move(goal, desired_rotation, vertical=True)
            actual, actual_rotation = read()
            measured_center = actual + offset
            if arrived and inside(measured_center):
                verified_inside = True
                break
            residual = float(np.linalg.norm(actual - goal))
            bottom = measured_center[2] - world_extent[2]
            support_level = rim if strategy == 'shallow' else floor
            supported = abs(bottom - support_level) <= support_tolerance
            if inside(measured_center) and supported and residual <= support_tolerance and (desired_rotation.inv() * actual_rotation).magnitude() <= angle_tolerance:
                verified_inside = True
                place = xyz(actual)
                break
            log.info('Descent did not establish aligned placement; trying one raised realignment')
            up = actual.copy()
            up[2] = high
            if not move(up, desired_rotation, vertical=True):
                break
            above = goal.copy()
            above[2] = high
            if not move(above, desired_rotation):
                break
        if not verified_inside:
            return result('failed', 'Measured object placement remains outside the release tolerance')
        if disposition == 'stabilize':
            width = float(ctx.tool('robot.get_gripper')['position'])
            if not pinch_min < width < pinch_max:
                return result('failed', 'Inserted placement has no observed occupied pinch')
        else:
            for release_attempt in range(2):
                try:
                    ctx.tool('robot.open_gripper', settle_steps=settle_steps)
                    width = float(ctx.tool('robot.get_gripper')['position'])
                    if width >= opening_fraction:
                        opened = True
                        break
                except Exception as exc:
                    log.info('Opening attempt unavailable: %s', exc)
            if not opened:
                return result('failed', 'Measured fingers remain closed after bounded opening')
            actual, unused = read()
            retreat = actual.copy()
            retreat[2] = max(high, actual[2] + clearance)
            if not move(retreat, desired_rotation, vertical=True):
                log.info('Full withdrawal unavailable after verified inside release')

        evidence = False
        outside = False
        after = ctx.tool('robot.get_observation')
        for camera in after['cameras'][:2]:
            try:
                if object_description:
                    seg = ctx.tool('sam3.segment_text', image=camera['rgb'], query=object_description, max_results=1)
                    if len(seg.get('masks', [])):
                        pts = cloud(camera, seg['masks'][0])
                        if len(pts) >= minimum_points:
                            middle = np.median(pts, axis=0)
                            settled = floor - support_tolerance < middle[2] < rim + visible_height
                            if inside(middle) and settled:
                                evidence = True
                            elif not inside(middle):
                                outside = True
                old = baseline.get(camera['name'])
                if old is not None:
                    old_pose = old['pose']
                    same_view = np.linalg.norm(vec(old_pose['position']) - vec(camera['pose']['position'])) < position_tolerance and (rot(old_pose['rotation']).inv() * rot(camera['pose']['rotation'])).magnitude() < angle_tolerance
                    if same_view:
                        depth = np.asarray(camera['depth'])
                        previous = np.asarray(old['depth'])
                        if depth.shape == previous.shape:
                            mask = ((previous - depth > depth_change) & (depth > 0)).astype(np.uint8) * 255
                            pts = cloud(camera, mask)
                            local = (pts - zone) @ axes
                            keep = np.all(np.abs(local[:, :2]) < zone_extent[:2], axis=1) & (pts[:, 2] > floor + seating_margin) & (pts[:, 2] < rim + visible_height)
                            if int(np.sum(keep)) >= minimum_points:
                                evidence = True
            except Exception as exc:
                log.info('Fresh spatial view inconclusive: %s', exc)
        if evidence:
            return result('done', 'Fresh receiving-region evidence confirms placement' if disposition == 'release' else 'Inserted placement observed with occupied stabilizing pinch')
        if outside:
            return result('failed', 'Fresh target measurement lies outside the selected receiving region')
        return result('done', 'Inside arrival and opening measured; final spatial view could not be confirmed' if disposition == 'release' else 'Inside insertion and occupied pinch measured; final opening view could not be confirmed')
    except Exception as exc:
        log.warning('Placement branch unavailable: %s', exc)
        if verified_inside and opened:
            return result('done', 'Inside arrival and opening measured; subsequent observation or withdrawal unavailable')
        return result('failed', 'Placement did not establish the required measured arrival or gripper state: ' + str(exc))
