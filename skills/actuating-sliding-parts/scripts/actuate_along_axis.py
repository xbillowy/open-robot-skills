from typing import TypedDict, Optional
import math
import numpy as np
from scipy.spatial.transform import Rotation
from gap import NodeContext
from gap_core.types import Se3Pose, Vec3

class Output(TypedDict):
    moved: bool
    displacement_m: float
    arrival_ok: bool
    retry_needed: bool

def vector(v):
    return np.array([v['x'], v['y'], v['z']], dtype=float)

def vec(v):
    return dict(zip(('x', 'y', 'z'), map(float, v)))

def rot(q):
    return Rotation.from_quat([q['x'], q['y'], q['z'], q['w']])

def shifted(pose, delta):
    return {'position': vec(vector(pose['position']) + delta), 'rotation': dict(pose['rotation'])}

def run(ctx: NodeContext, grasp_pose: Se3Pose, pull_axis: Vec3, direction: str = 'pull', total_distance: float = 0.06, step_length: float = 0.01, stall_limit: int = 3, approach_distance: float = 0.06, goal_fraction: float = 0.8, ee_to_tcp: Optional[Se3Pose] = None) -> Output:
    # None declares matching references; otherwise use the configured rigid
    # transform, never an offset inferred from an unverified command.
    def transform_feedback(raw):
        if ee_to_tcp is None:
            return raw
        r = rot(raw['rotation'])
        p = vector(raw['position']) + r.apply(vector(ee_to_tcp['position']))
        q = (r * rot(ee_to_tcp['rotation'])).as_quat()
        return {'position': vec(p), 'rotation': {'w': float(q[3]), 'x': float(q[0]), 'y': float(q[1]), 'z': float(q[2])}}

    def arrive(target, cartesian):
        # Three attempts bound residual-pose corrections.
        for unused in range(3):
            try:
                if cartesian:
                    ctx.tool('robot.go_to_pose_cartesian', pose=target)
                else:
                    ctx.tool('robot.go_to_pose', pose=target)
                measured = transform_feedback(ctx.tool('robot.get_ee_pose')['pose'])
            except Exception:
                return False
            error = np.linalg.norm(vector(measured['position']) - vector(target['position']))
            angle = (rot(target['rotation']).inv() * rot(measured['rotation'])).magnitude()
            # Arrival limits bound translation and angular mismatch.
            if error <= 0.015 and angle <= 0.15:
                return True
        return False

    result = {'moved': False, 'displacement_m': 0.0, 'arrival_ok': False, 'retry_needed': True}
    axis = vector(pull_axis)
    norm = float(np.linalg.norm(axis))
    # Reject degenerate directions rather than inventing an axis.
    if norm < 1e-6 or direction not in ('pull', 'push') or total_distance <= 0:
        return result
    axis /= norm
    # Bound total travel, individual increments, goal tolerance and retries.
    distance = min(float(total_distance), 0.30)
    step = min(max(float(step_length), 0.003), 0.015)
    fraction = min(max(float(goal_fraction), 0.8), 1.0)
    stalls_allowed = min(max(int(stall_limit), 1), 5)
    # Bound approach clearance without introducing an absolute position.
    approach = min(max(float(approach_distance), 0.02), 0.12)
    pre = shifted(grasp_pose, axis * approach)
    if direction == 'push':
        ctx.tool('robot.close_gripper')
    else:
        ctx.tool('robot.open_gripper')
    if not arrive(pre, False):
        return result
    if not arrive(grasp_pose, True):
        return result
    if direction == 'pull':
        ctx.tool('robot.close_gripper')
    initial = transform_feedback(ctx.tool('robot.get_ee_pose')['pose'])
    origin = vector(initial['position'])
    travel_axis = axis if direction == 'pull' else -axis
    displacement = 0.0
    previous = 0.0
    stalled = 0
    command = grasp_pose
    # Ninety iterations cap work; three passes allow incomplete-step recovery.
    limit = min(90, int(math.ceil(distance / step)) * 3 + stalls_allowed)
    for unused in range(limit):
        if displacement >= distance * fraction:
            break
        remaining = distance - max(displacement, 0.0)
        increment = min(step, remaining)
        target = shifted(command, travel_axis * increment)
        motion_failed = False
        try:
            ctx.tool('robot.go_to_pose_cartesian', pose=target)
        except Exception:
            motion_failed = True
        try:
            raw_pose = ctx.tool('robot.get_ee_pose')['pose']
            measured = transform_feedback(raw_pose)
            displacement = float((vector(measured['position']) - origin) @ travel_axis)
        except Exception:
            return {'moved': False, 'displacement_m': float(displacement), 'arrival_ok': True, 'retry_needed': True}
        if motion_failed:
            return {'moved': False, 'displacement_m': float(displacement), 'arrival_ok': True, 'retry_needed': True}
        # A fifth of the requested step is minimum useful measured progress.
        if displacement - previous < increment * 0.2:
            stalled += 1
        else:
            stalled = 0
        previous = displacement
        achieved = min(max(displacement, 0.0), distance)
        command = shifted(grasp_pose, travel_axis * achieved)
        if stalled >= stalls_allowed:
            break
    moved = displacement >= distance * fraction
    return {'moved': bool(moved), 'displacement_m': float(displacement), 'arrival_ok': True, 'retry_needed': not bool(moved)}
