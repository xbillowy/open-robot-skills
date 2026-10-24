from typing import TypedDict
import logging
import numpy as np
from scipy.spatial.transform import Rotation
from gap import NodeContext
from gap_core.types import OrientedBoundingBox, Se3Pose, Vec3, Mask, PointCloud


class Output(TypedDict):
    status: str
    reason: str
    place_position: Vec3


def run(ctx: NodeContext, container_obb: OrientedBoundingBox, ee_pose_at_grasp: Se3Pose, target_obb: OrientedBoundingBox, target_mask: Mask | None = None, container_mask: Mask | None = None, container_cloud: PointCloud | None = None, strategy: str = 'surface', relation: str = 'center', clearance: float = 0.012, approach_height: float = 0.13, tcp_offset: float = 0.103, tolerance: float = 0.015, side_gap: float = 0.04, withdrawal: float = 0.12) -> Output:
    log = logging.getLogger('placing_held_objects_on_supports')
    keys = ('x', 'y', 'z')
    vertical = np.array([0., 0., 1.])
    place = dict(target_obb['center'])
    released = False
    arrived = False

    def vector(p):
        return np.array([p[k] for k in keys], dtype=float)

    def xyz(p):
        return {k: float(p[i]) for i, k in enumerate(keys)}

    def rotation(q):
        return Rotation.from_quat([q[k] for k in ('x', 'y', 'z', 'w')])

    def tcp(p):
        return vector(p['position']) + rotation(p['rotation']).apply(vertical * tcp_offset)

    def read():
        return ctx.tool('robot.get_ee_pose')['pose']

    def finish(status, reason):
        if status == 'failed':
            log.warning('Placement branch failed: %s', reason)
        return {'status': status, 'reason': reason, 'place_position': place}

    def move(goal, q):
        pose = {'position': xyz(goal), 'rotation': q}
        desired = rotation(q)
        for method in range(3):
            try:
                if method == 0:
                    ctx.tool('robot.go_to_pose_cartesian', pose=pose)
                elif method == 1:
                    ctx.tool('robot.go_to_pose', pose=pose)
                else:
                    obs = ctx.tool('robot.get_observation')
                    plan = ctx.tool('curobo.plan_to_pose', target_pose=pose, start_joint_position=obs['arms'][0]['joint_state'])
                    if not plan.get('success') or plan.get('trajectory') is None:
                        log.info('Endpoint planner found no path')
                        continue
                    ctx.tool('robot.execute_trajectory', trajectory=plan['trajectory'])
                actual = read()
                distance = float(np.linalg.norm(tcp(actual) - goal))
                angle = float((desired.inv() * rotation(actual['rotation'])).magnitude())
                if distance <= tolerance and angle <= tolerance * 10:
                    return True
                log.info('Waypoint not reached: distance=%s angle=%s', distance, angle)
            except Exception as exc:
                log.info('Motion alternative %s unavailable: %s', method, exc)
        return False

    def camera(obs):
        return next(c for c in obs['cameras'] if c['name'] == 'agentview')

    def cloud(cam, mask):
        result = ctx.tool('geometry.mask_to_world_points', mask=mask, depth=cam['depth'], intrinsics=cam['intrinsics'], camera_pose=cam['pose'])
        p = np.asarray(result['points']['points'], dtype=float).reshape(-1, 3)
        return p[np.isfinite(p).all(axis=1)]

    try:
        c = vector(container_obb['center'])
        e = vector(container_obb['extent'])
        original = vector(target_obb['center'])
        extent = vector(target_obb['extent'])
        grasp_rotation = rotation(ee_pose_at_grasp['rotation'])
        local = grasp_rotation.inv().apply(original - tcp(ee_pose_at_grasp))
        q = dict(ee_pose_at_grasp['rotation'])
        final_rotation = grasp_rotation
        support_rotation = rotation(container_obb['orientation']).as_matrix()
        normal = vertical.copy()
        surface = c.copy()
        surface[2] += e[2]
        obs = ctx.tool('robot.get_observation')
        current = tcp(read())
        cam = camera(obs)

        if relation != 'center':
            right = rotation(cam['pose']['rotation']).as_matrix()[:, 0].copy()
            right[2] = 0.
            right /= max(float(np.linalg.norm(right)), tolerance)
            toward = vector(cam['pose']['position']) - c
            toward[2] = 0.
            toward -= right * float(toward @ right)
            toward /= max(float(np.linalg.norm(toward)), tolerance)
            if relation == 'right':
                direction = right
            elif relation == 'left':
                direction = -right
            elif relation == 'front':
                direction = toward
            elif relation == 'behind':
                direction = -toward
            else:
                return finish('failed', 'Unsupported side relation')
            radius = float(np.sum(np.abs(support_rotation.T @ direction) * e))
            surface = c + direction * (radius + side_gap)
            surface += toward * clearance
            surface[2] = c[2] - e[2]
        elif strategy == 'panel':
            pts = np.empty((0, 3))
            if container_cloud is not None:
                pts = np.asarray(container_cloud['points'], dtype=float).reshape(-1, 3)
                pts = pts[np.isfinite(pts).all(axis=1)]
            if len(pts) < 30:
                fresh = np.ones(np.asarray(cam['depth']).shape, dtype=np.uint8) * 255
                pts = cloud(cam, fresh)
            delta = (pts - c) @ support_rotation
            pts = pts[np.all(np.abs(delta) <= e + tolerance * 2, axis=1)]
            fitted = False
            for iteration in range(4):
                if len(pts) < 20:
                    break
                a = np.column_stack((pts[:, 0], pts[:, 1], np.ones(len(pts))))
                fit = np.linalg.lstsq(a, pts[:, 2], rcond=None)[0]
                residual = np.abs(a @ fit - pts[:, 2])
                keep = residual <= max(tolerance / 2, float(np.percentile(residual, 75)))
                pts = pts[keep]
                fitted = True
            if not fitted or len(pts) < 20:
                return finish('failed', 'No measured panel region available for arrival')
            normal = np.array([-float(fit[0]), -float(fit[1]), 1.])
            normal /= np.linalg.norm(normal)
            surface = np.median(pts, axis=0)
            surface[2] = fit[0] * surface[0] + fit[1] * surface[1] + fit[2]
            slope = normal.copy()
            slope[2] = 0.
            if np.linalg.norm(slope) > tolerance:
                slope /= np.linalg.norm(slope)
                lay = abs(float(grasp_rotation.as_matrix()[:, 1] @ slope)) < 0.5
                if lay:
                    axis = np.cross(normal, vertical)
                    axis /= np.linalg.norm(axis)
                    if axis[0] > 0:
                        axis = -axis
                    zaxis = -axis
                    xaxis = normal
                    yaxis = np.cross(zaxis, xaxis)
                    final_rotation = Rotation.from_matrix(np.column_stack((xaxis, yaxis, zaxis)))
                    q = dict(zip(('x', 'y', 'z', 'w'), map(float, final_rotation.as_quat())))
                    surface[:2] = c[:2]
                    surface[2] = fit[0] * surface[0] + fit[1] * surface[1] + fit[2]
        elif strategy not in ('surface', 'stack'):
            return finish('failed', 'Unsupported support strategy')

        change = final_rotation * grasp_rotation.inv()
        object_axes = change.as_matrix() @ rotation(target_obb['orientation']).as_matrix()
        radius = float(np.sum(np.abs(object_axes.T @ normal) * extent))
        if strategy == 'stack' and relation == 'center':
            radius *= 0.5
        goal_center = surface + normal * (radius + clearance)
        drop = goal_center - final_rotation.apply(local)
        place = xyz(drop)
        transported = False
        for attempt in range(2):
            live = read()
            current = tcp(live)
            high = max(float(current[2]), float(drop[2]) + approach_height * (1 + attempt * 0.5))
            lift = current.copy()
            lift[2] = high
            if not move(lift, live['rotation']):
                continue
            hover = drop.copy()
            hover[2] = high
            if move(hover, q):
                transported = True
                break
        if not transported:
            return finish('failed', 'Raised transport did not reach its measured waypoint')
        if not move(drop, q):
            return finish('failed', 'Descent did not reach the selected release region')
        actual = read()
        actual_center = tcp(actual) + rotation(actual['rotation']).apply(local)
        if np.linalg.norm(actual_center - goal_center) > tolerance * 2:
            return finish('failed', 'Measured held-object center is outside the release region')
        arrived = True
        for attempt in range(2):
            try:
                ctx.tool('robot.open_gripper', settle_steps=40)
                width = float(ctx.tool('robot.get_gripper')['position'])
                if width >= 0.9:
                    released = True
                    break
                log.info('Opening retry: observed open fraction %s', width)
            except Exception as exc:
                log.info('Opening observation unavailable: %s', exc)
        if not released:
            width = float(ctx.tool('robot.get_gripper')['position'])
            if width < 0.9:
                return finish('failed', 'Gripper remained closed after bounded opening attempts')
            released = True
        retreat = tcp(read()) + vertical * withdrawal
        withdrew = move(retreat, q)
        reason = 'Measured arrival and open gripper confirmed; released-body view unconfirmed'
        try:
            fresh_obs = ctx.tool('robot.get_observation')
            fresh_cam = camera(fresh_obs)
            expected = goal_center.copy()
            pc = rotation(fresh_cam['pose']['rotation']).inv().apply(expected - vector(fresh_cam['pose']['position']))
            if pc[2] > 0:
                intr = np.asarray(fresh_cam['intrinsics'])
                u = float(intr[0, 0] * pc[0] / pc[2] + intr[0, 2])
                v = float(intr[1, 1] * pc[1] / pc[2] + intr[1, 2])
                h, w = np.asarray(fresh_cam['depth']).shape
                if 0 <= u < w and 0 <= v < h:
                    seg = ctx.tool('sam3.segment_point', image=fresh_cam['rgb'], pixel_x=u, pixel_y=v)
                    if len(seg.get('masks', [])):
                        pts = cloud(fresh_cam, seg['masks'][0])
                        if len(pts) >= 20:
                            lo, hi = np.percentile(pts, [3, 97], axis=0)
                            measured = (lo + hi) / 2
                            if np.linalg.norm(measured - goal_center) <= float(np.linalg.norm(extent)) + tolerance * 2 and np.max(hi - lo) <= float(np.max(extent)) * 3 + tolerance:
                                reason = 'Measured arrival, open gripper, and released body near the support confirmed'
        except Exception as exc:
            log.info('Released-body view inconclusive: %s', exc)
        if not withdrew:
            reason += '; withdrawal could not be completed'
        return finish('done', reason)
    except Exception as exc:
        log.warning('Placement operation interrupted: %s', exc)
        if arrived and released:
            return finish('done', 'Arrival and opening were measured; subsequent observation or withdrawal was unavailable')
        return finish('failed', 'Required arrival or release was not completed after available alternatives')
