from typing import TypedDict
import numpy as np
from scipy.spatial.transform import Rotation
from gap import NodeContext
from gap_core.types import Se3Pose
from gap_core.errors import VerificationFailed


class Output(TypedDict):
    retained: bool
    ee_pose_after_lift: Se3Pose


def _rotation(q):
    return Rotation.from_quat([q['x'], q['y'], q['z'], q['w']])


def _tcp(measured, transform):
    r = _rotation(measured['rotation'])
    t = transform['position']
    p = measured['position']
    xyz = np.array([p['x'], p['y'], p['z']], dtype=float) + r.apply([t['x'], t['y'], t['z']])
    q = (r * _rotation(transform['rotation'])).as_quat()
    return {'position': dict(zip(('x', 'y', 'z'), map(float, xyz))),
            'rotation': {'w': float(q[3]), 'x': float(q[0]), 'y': float(q[1]), 'z': float(q[2])}}


def run(ctx: NodeContext, object_description: str,
        feedback_to_tcp: Se3Pose, lift_distance: float = 0.04) -> Output:
    # Bound the test displacement to a small vertical lift, not transport.
    if not object_description or not 0.02 <= lift_distance <= 0.06:
        raise VerificationFailed('A selected object and a bounded test lift are required')
    before = ctx.tool('robot.get_observation')
    if not before['cameras']:
        raise VerificationFailed('No public images for retention verification')
    measured = ctx.tool('robot.get_ee_pose')['pose']
    start = _tcp(measured, feedback_to_tcp)
    p = start['position']
    target = {'position': {'x': p['x'], 'y': p['y'], 'z': p['z'] + lift_distance},
              'rotation': start['rotation']}
    ctx.tool('robot.go_to_pose_cartesian', pose=target)
    after_pose = ctx.tool('robot.get_ee_pose')['pose']
    attained = _tcp(after_pose, feedback_to_tcp)
    a = attained['position']
    b = target['position']
    distance = float(np.linalg.norm([a[k] - b[k] for k in ('x', 'y', 'z')]))
    angle = float((_rotation(target['rotation']).inv() * _rotation(attained['rotation'])).magnitude())
    # Bound acceptable physical endpoint error in metres and radians.
    if distance > 0.015 or angle > 0.12:
        raise VerificationFailed('Test-lift endpoint was not reached')
    after = ctx.tool('robot.get_observation')
    if not after['cameras']:
        raise VerificationFailed('No post-lift images')
    # Observation camera ordering is preserved; compare matching views only.
    n = min(len(before['cameras']), len(after['cameras']))
    images = [c['rgb'] for c in before['cameras'][:n]] + [c['rgb'] for c in after['cameras'][:n]]
    verdict = ctx.tool('vlm.query_yes_no', images=images, prompt=(
        'The first half of these images precedes a short vertical test lift; '
        'the second half contains the corresponding views afterward. '
        'The selected object is: ' + object_description + '. '
        'Is there clear evidence BOTH that the fingers remain partly separated '
        'by grasped material AND that this specific object rose with the hand '
        'off its original support? Do not count the hand itself, a neighboring '
        'object, or finger aperture alone as object rise. Answer NO if occlusion '
        'or ambiguity prevents either check.'))
    if not verdict['answer']:
        raise VerificationFailed('Object-specific retention was not established')
    return {'retained': True, 'ee_pose_after_lift': after_pose}
