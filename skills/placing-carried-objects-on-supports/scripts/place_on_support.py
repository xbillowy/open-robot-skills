from gap import NodeContext
from gap_core.types import OrientedBoundingBox, Se3Pose, Vec3, Mask, PointCloud
from typing import TypedDict
from scipy.spatial.transform import Rotation
import numpy as np
import math
import logging

class Output(TypedDict):
    status: str
    reason: str
    place_position: Vec3

def run(ctx: NodeContext, container_obb: OrientedBoundingBox, ee_pose_at_grasp: Se3Pose, target_obb: OrientedBoundingBox, target_mask: Mask | None = None, container_mask: Mask | None = None, container_cloud: PointCloud | None = None, mode: str = 'surface', relation: str = 'center', object_description: str = '', support_description: str = '', tcp_offset: float = 0.103, clearance: float = 0.012, approach_height: float = 0.13, side_gap: float = 0.04, association_tolerance: float = 0.04) -> Output:
    log = logging.getLogger('placing_carried_objects_on_supports')
    keys = ('x', 'y', 'z')
    up = np.array([0., 0., 1.])
    position = dict(target_obb['center'])
    released = False
    valid_release = False
    def v(d):
        return np.array([float(d[k]) for k in keys])
    def xyz(p):
        return dict(zip(keys, map(float, p)))
    def rot(q):
        return Rotation.from_quat([q[k] for k in ('x', 'y', 'z', 'w')])
    def quat(r):
        return dict(zip(('x', 'y', 'z', 'w'), map(float, r.as_quat())))
    def tip(p):
        return v(p['position']) + rot(p['rotation']).apply(up * tcp_offset)
    def read():
        return ctx.tool('robot.get_ee_pose')['pose']
    def result(status, reason):
        if status == 'failed':
            log.warning('%s', reason)
        return {'status': status, 'reason': reason, 'place_position': position}
    def camera(obs):
        return next(c for c in obs['cameras'] if c['name'] == 'agentview')
    def cloud(cam, mask):
        out = ctx.tool('geometry.mask_to_world_points', mask=mask, depth=cam['depth'], intrinsics=cam['intrinsics'], camera_pose=cam['pose'])
        p = np.asarray(out['points']['points'], dtype=float).reshape(-1, 3)
        return p[np.isfinite(p).all(axis=1)]
    def segment(obs, query):
        if not query:
            return None
        try:
            cam = camera(obs)
            out = ctx.tool('sam3.segment_text', image=cam['rgb'], query=query, max_results=1)
            if not out.get('masks'):
                return None
            p = cloud(cam, out['masks'][0])
            return p if len(p) >= 20 else None
        except Exception as exc:
            log.info('Fresh measurement unavailable: %s', exc)
            return None
    def move(goal, q):
        for attempt in range(3):
            pose = {'position': xyz(goal), 'rotation': q}
            try:
                if attempt == 0:
                    ctx.tool('robot.go_to_pose_cartesian', pose=pose)
                elif attempt == 1:
                    ctx.tool('robot.go_to_pose', pose=pose)
                else:
                    obs = ctx.tool('robot.get_observation')
                    plan = ctx.tool('curobo.plan_to_pose', target_pose=pose, start_joint_position=obs['arms'][0]['joint_state'])
                    if not plan.get('success') or not plan.get('trajectory'):
                        log.info('Planning alternative returned no usable path')
                        continue
                    ctx.tool('robot.execute_trajectory', trajectory=plan['trajectory'])
                actual = read()
                error = float(np.linalg.norm(tip(actual) - goal))
                angle = float((rot(q).inv() * rot(actual['rotation'])).magnitude())
                if error <= clearance * 1.5 and angle <= 0.15:
                    return True
                log.info('Arrival residual: distance=%s angle=%s', error, angle)
            except Exception as exc:
                log.info('Motion alternative %s failed: %s', attempt, exc)
        return False
    try:
        obs = ctx.tool('robot.get_observation')
        live = read()
        if float(ctx.tool('robot.get_gripper')['position']) < clearance:
            return result('failed', 'Measured gripper is closed empty before transport')
        extent = v(target_obb['extent'])
        original = v(target_obb['center'])
        grasp_tip = tip(ee_pose_at_grasp)
        predicted = original + tip(live) - grasp_tip
        body = predicted.copy()
        held = segment(obs, object_description)
        if held is not None:
            lo, hi = np.percentile(held, [2, 98], axis=0)
            midpoint = (lo + hi) / 2.
            size_ok = np.all(hi - lo <= extent * 2. + association_tolerance * 2.)
            associated = np.linalg.norm(midpoint - predicted) <= association_tolerance
            bottom_ok = abs(float(lo[2] - (predicted[2] - extent[2]))) <= association_tolerance
            if size_ok and associated and bottom_ok:
                lower = held[held[:, 2] <= lo[2] + max(clearance * 2., extent[2] * 0.5)]
                if len(lower) >= 15:
                    blo, bhi = np.percentile(lower[:, :2], [2, 98], axis=0)
                    candidate = predicted.copy()
                    candidate[:2] = (blo + bhi) / 2.
                    candidate[2] = lo[2] + extent[2]
                    if np.linalg.norm(candidate - predicted) <= association_tolerance:
                        body = candidate
            else:
                log.info('Held measurement disagrees with recorded grasp reference; retaining recorded offset')
        initial_rotation = rot(live['rotation'])
        local = initial_rotation.inv().apply(body - tip(live))
        center = v(container_obb['center'])
        support_extent = v(container_obb['extent'])
        support_rotation = rot(container_obb['orientation'])
        surface = center.copy()
        surface[2] += support_extent[2]
        normal = up.copy()
        points = segment(obs, support_description)
        if points is None and container_cloud is not None:
            points = np.asarray(container_cloud['points'], dtype=float).reshape(-1, 3)
            points = points[np.isfinite(points).all(axis=1)]
        if points is not None:
            relative = support_rotation.inv().apply(points - center)
            keep = np.all(np.abs(relative[:, :2]) <= support_extent[:2] + clearance, axis=1)
            keep &= np.abs(points[:, 2] - center[2]) <= support_extent[2] + association_tolerance
            points = points[keep]
            if len(points) < 30:
                points = None
        if points is not None and mode != 'inclined':
            slab = points[np.abs(points[:, 2] - surface[2]) <= association_tolerance]
            if len(slab) >= 30:
                z = float(np.percentile(slab[:, 2], 85))
                slab = slab[np.abs(slab[:, 2] - z) <= clearance]
                if len(slab) >= 15:
                    lo, hi = np.percentile(slab, [3, 97], axis=0)
                    candidate = (lo + hi) / 2.
                    if np.linalg.norm(candidate[:2] - center[:2]) <= max(association_tolerance, float(np.min(support_extent[:2]))):
                        surface[:2] = candidate[:2]
                    surface[2] = float(np.median(slab[:, 2]))
        rotation = initial_rotation
        if relation != 'center':
            cam = camera(obs)
            right = rot(cam['pose']['rotation']).as_matrix()[:, 0].copy()
            right[2] = 0.
            right /= max(float(np.linalg.norm(right)), clearance)
            if relation == 'left':
                right = -right
            toward = v(cam['pose']['position']) - center
            toward[2] = 0.
            toward -= right * np.dot(toward, right)
            toward /= max(float(np.linalg.norm(toward)), clearance)
            radius = float(np.sum(np.abs(support_rotation.inv().apply(right)) * support_extent))
            surface = center + right * (radius + side_gap) + toward * clearance
            surface[2] = center[2] - support_extent[2]
            rotation = Rotation.from_euler('z', math.pi / 2.) * Rotation.from_euler('x', math.pi)
        elif mode == 'inclined' and points is not None:
            panel = points.copy()
            for iteration in range(3):
                a = np.column_stack((panel[:, :2], np.ones(len(panel))))
                fit = np.linalg.lstsq(a, panel[:, 2], rcond=None)[0]
                error = np.abs(a @ fit - panel[:, 2])
                keep = error <= max(clearance, float(np.percentile(error, 75)))
                if int(keep.sum()) >= 20:
                    panel = panel[keep]
            normal = np.array([-fit[0], -fit[1], 1.])
            normal /= np.linalg.norm(normal)
            surface = np.median(panel, axis=0)
            surface[2] = fit[0] * surface[0] + fit[1] * surface[1] + fit[2]
            downhill = normal.copy()
            downhill[2] = 0.
            downhill /= max(float(np.linalg.norm(downhill)), clearance)
            if abs(float(initial_rotation.as_matrix()[:, 1] @ downhill)) < 0.5:
                axis = np.cross(normal, up)
                if np.linalg.norm(axis) > clearance:
                    axis /= np.linalg.norm(axis)
                    if axis[0] > 0.:
                        axis = -axis
                    zaxis = -axis
                    rotation = Rotation.from_matrix(np.column_stack((normal, np.cross(zaxis, normal), zaxis)))
        object_rotation = rotation * rot(ee_pose_at_grasp['rotation']).inv() * rot(target_obb['orientation'])
        radius = float(np.sum(np.abs(object_rotation.inv().apply(normal)) * extent))
        goal_center = surface + normal * (radius + clearance)
        if relation != 'center':
            goal_center[2] += clearance * 4.
        goal = goal_center - rotation.apply(local)
        position = xyz(goal)
        q = quat(rotation)
        transported = False
        for approach_attempt in range(3):
            fraction = max(0.3, 1. - 0.4 * approach_attempt)
            current = tip(read())
            high = max(float(current[2]), float(goal[2] + approach_height * fraction))
            lift = current.copy()
            lift[2] = high
            above = goal.copy()
            above[2] = high
            if move(lift, read()['rotation']) and move(above, q):
                transported = True
                break
            log.info('Approach height alternative did not establish arrival; trying lower clearance')
        if not transported:
            return result('failed', 'No measured arrival above destination after motion and height alternatives')
        if float(ctx.tool('robot.get_gripper')['position']) < clearance:
            return result('failed', 'Measured gripper closed empty during transport')
        move(goal, q)
        actual = read()
        actual_center = tip(actual) + rot(actual['rotation']).apply(local)
        delta = actual_center - goal_center
        lateral = delta - normal * np.dot(delta, normal)
        separation = float(np.dot(actual_center - surface, normal) - radius)
        angle = float((rotation.inv() * rot(actual['rotation'])).magnitude())
        height_limit = clearance * (7. if relation != 'center' else 3.)
        valid_release = bool(np.linalg.norm(lateral) <= clearance * 2. and -clearance <= separation <= height_limit and angle <= 0.15)
        if not valid_release:
            return result('failed', 'Measured object release pose remains outside the destination region')
        for attempt in range(2):
            ctx.tool('robot.open_gripper', settle_steps=40)
            if float(ctx.tool('robot.get_gripper')['position']) >= 0.9:
                released = True
                break
        if not released:
            return result('failed', 'Measured gripper remains closed after release retries')
        retreat = tip(read()) + up * approach_height
        withdrew = move(retreat, actual['rotation'])
        final_obs = ctx.tool('robot.get_observation')
        final_points = segment(final_obs, object_description)
        if final_points is not None:
            lo, hi = np.percentile(final_points, [2, 98], axis=0)
            final_center = (lo + hi) / 2.
            associated = np.linalg.norm(final_center - goal_center) <= association_tolerance + radius
            plausible_size = np.all(hi - lo <= extent * 2. + association_tolerance * 2.)
            base = float(np.percentile((final_points - surface) @ normal, 2))
            lateral_final = final_center - surface
            lateral_final -= normal * np.dot(lateral_final, normal)
            if associated and plausible_size and np.linalg.norm(lateral_final) <= association_tolerance + float(np.max(extent[:2])) and -association_tolerance <= base <= height_limit + association_tolerance:
                return result('done', 'Measured valid release and opening; fresh associated object geometry supports the destination relation')
            log.info('Final segmentation does not establish a reliable object-support measurement; inspecting destination depth')
        try:
            cam = camera(final_obs)
            p = cloud(cam, np.ones(np.asarray(cam['depth']).shape, dtype=np.uint8) * 255)
            offset = p - surface
            heights = offset @ normal
            planar = offset - heights[:, None] * normal
            near = (np.linalg.norm(planar, axis=1) < max(float(np.max(extent[:2])), clearance * 3.)) & (heights > clearance) & (heights < radius * 2. + height_limit)
            if int(near.sum()) >= 15:
                return result('done', 'Measured valid object arrival and opening; fresh destination depth supports released-object presence')
        except Exception as exc:
            log.info('Destination depth unavailable: %s', exc)
        reason = 'Measured valid object arrival and open gripper; final support could not be independently resolved'
        if not withdrew:
            reason += '; withdrawal arrival was not confirmed'
        return result('done', reason)
    except Exception as exc:
        if released and valid_release:
            return result('done', 'Measured valid arrival and opening; subsequent inspection was unavailable: ' + str(exc))
        return result('failed', 'Operation stopped before verified release: ' + str(exc))
