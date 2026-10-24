from typing import TypedDict
from gap import NodeContext


class Output(TypedDict):
    verified: bool
    needs_withdrawal: bool
    retry_engagement: bool
    evidence: str


def run(ctx: NodeContext, target_name: str, parent_prompt: str = '', goal: str = 'fully seated closed without a residual opening', hand_withdrawn: bool = False) -> Output:
    if not hand_withdrawn:
        return {'verified': False, 'needs_withdrawal': True, 'retry_engagement': False, 'evidence': 'Withdraw the hand and expose the intended part before inspection.'}
    evidence = ''
    # Two fresh inspections bound observation retries.
    for _ in range(2):
        observation = ctx.tool('robot.get_observation')
        images = [camera['rgb'] for camera in observation['cameras']]
        if not images:
            evidence = 'No camera evidence; refresh perception.'
            continue
        result = ctx.tool(
            'vlm.query_yes_no',
            images=images,
            prompt=(
                'Inspect only the intended part: ' + target_name +
                '. Parent context: ' + parent_prompt +
                '. Is this physical result clearly established now: ' + goal +
                '? Require the hand to be withdrawn and not obscuring the relevant geometry. '
                'For closure require seating against the associated frame without a residual gap; '
                'near alignment alone is insufficient. For turning inspect the control itself, '
                'not the wrist. Answer NO if occluded, uncertain, obstructed, or the part identity is ambiguous.'
            )
        )
        evidence = result['text']
        if result['answer']:
            return {'verified': True, 'needs_withdrawal': False, 'retry_engagement': False, 'evidence': evidence}
    return {'verified': False, 'needs_withdrawal': False, 'retry_engagement': True, 'evidence': evidence}
