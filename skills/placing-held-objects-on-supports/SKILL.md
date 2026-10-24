---
name: placing-held-objects-on-supports
description: Place a held object on a perceived support using the measured grasp displacement, TCP-aware
  feedback, raised transport, checked descent, release, and withdrawal. Separate strategies handle ordinary
  surfaces, nested stacking, and inclined panels. Camera-relative side placement is available without
  reusing stale masks. Motion alternatives are bounded; release requires measured arrival and an observed
  open gripper.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags:
  - experience
  - apply-created
gap:
  exit_conditions:
    completed: The operation's observed completion checks hold; outputs bound.
    failed: The operation could not be completed or verified (reached by raising). Coordinator routes
      on.
  canonical_scripts:
  - place_on_support: scripts/place_on_support.py
  - raise_failure: scripts/raise_failure.py
  streaming: false
---

# placing-held-objects-on-supports

Place a held object on a perceived support using the measured grasp displacement, TCP-aware feedback, raised transport, checked descent, release, and withdrawal. Separate strategies handle ordinary surfaces, nested stacking, and inclined panels. Camera-relative side placement is available without reusing stale masks. Motion alternatives are bounded; release requires measured arrival and an observed open gripper.

This skill was written by the memory system from confirmed experience: earlier tasks failed at this kind of step, were repaired, and the repair was confirmed by a rerun. Honor the coordinator's contract for your sub-graph (its inputs, outputs and exit values) exactly; this skill has no fixed tool whitelist, so compose the registered tools listed below. Its exits are `completed` (success) and `failed` (on_error).

## When to use

- The object is already held, and its pre-grasp OBB, grasp-time hand pose, and destination OBB are available.
- The instruction requests placement on a surface, stacking, inclined-panel support, or placement beside a perceived support.

## When NOT to use

- The object must be inserted through tall walls or into a closed enclosure.
- The supplied destination OBB does not identify the requested support or named sub-region.
- Transport requires navigating a densely obstructed route.

## Procedure

1. Wire one main script node taking the interface inputs. Route status done to success noop completed; route failed to fail running raise_failure.py with reason, then the sub-graph failed exit.
2. Convert grasp-time and live hand poses to TCP poses. Preserve the grasp rotation and object-to-TCP displacement rather than treating the old object OBB as its current location.
3. Use the support top for surface placement. For a side relation, derive horizontal directions from the fixed camera and place beyond the measured support edge at its base level. Never project an old mask onto fresh depth.
4. Select stack for nesting and panel for inclined supports. Panel placement robustly fits supplied or freshly cropped points, then chooses upright or laid placement from the grasp direction and measured slope.
5. Lift vertically, traverse above the goal, and descend. Check TCP position and wrist orientation after each leg. Try Cartesian, direct-pose, and planner alternatives, with a second approach height if necessary.
6. Release only after measured arrival. Retry opening once, inspect the measured open fraction, then withdraw. A fresh point-segmented view can confirm the released body; an inconclusive view does not override measured arrival and opening.

## Completion checks

- Each required waypoint must be reached within the TCP distance tolerance and wrist-angle tolerance. Exhausted motion alternatives return failed without releasing.
- Completion requires a measured object-center arrival at the selected release region and an observed open fraction of at least 0.9. A persistently closed gripper is positive evidence of release failure.
- Fresh verification uses at most one returned segmentation mask from the same image as its depth. Missing or unsuitable geometry is reported as unconfirmed, not as a placement failure.

## Canonical scripts

Use them as `type: script` nodes by their path (their run() inputs and Output fields are listed below); wire their inputs from your sub-graph's inputs or upstream nodes.

- `scripts/place_on_support.py`
- `scripts/raise_failure.py`

## Parameters (main script)

Bind an explicit parameter as a literal input of the `main` node only when the task instruction determines it; otherwise omit it (the default applies).

- `strategy` (default `"surface"`): Use surface for 'on' or 'on top of' a horizontal support; stack for 'stack' or nesting bowls; panel for 'on the rack' when the support is an inclined panel. Allowed values: surface, stack, panel.
- `relation` (default `"center"`): Instruction phrases 'to the right of', 'to the left of', 'in front of', and 'behind' set right, left, front, and behind respectively, relative to the fixed camera. Ordinary on-support placement sets center. Allowed values: center, right, left, front, behind.
- `clearance` (default `0.012`): Release gap in metres above the measured support along its normal. Keep the default unless the instruction explicitly requests a gap; allowed range 0 to 0.05.
- `approach_height` (default `0.13`): Vertical transport clearance above the computed TCP release height, in metres. Keep the default for ordinary placement; an explicit request for extra clearance may increase it. Allowed range 0.05 to 0.25.
- `tcp_offset` (default `0.103`): Calibrated hand-reference-to-fingertip distance in metres along local positive Z. Derived from the robot calibration, not the task wording; must be positive.
- `tolerance` (default `0.015`): Measured TCP arrival tolerance in metres; wrist tolerance is ten times this value in radians. Keep the default unless measured feedback accuracy requires adjustment. Allowed range 0.005 to 0.02.
- `side_gap` (default `0.04`): Distance in metres from the support edge to the held object's center for a side relation. Keep the default for unqualified beside-placement; an explicit distance sets this value. Allowed range 0.02 to 0.2.
- `withdrawal` (default `0.12`): Vertical fingertip withdrawal after opening, in metres. Keep the default unless the instruction explicitly limits overhead clearance. Allowed range 0.03 to 0.2.

## Sub-graph wiring (system)

- `main`: `type: script`, `scripts/place_on_support.py`, inputs bound from the sub-graph inputs (`in.<name>`); optional parameters only when the task states them.
- `conditional_edges.main`: `router_field: status`, mapping `done` -> `completed` and `failed` -> `fail`; you may insert a retry or alternative branch before `fail`.
- `fail`: `type: script`, `scripts/raise_failure.py`, input `reason` = `main.reason`; edge `fail` -> END (it raises, so the sub-graph leaves through its on_error exit `failed`).
- `completed`: `type: noop`, edge `completed` -> END; `exit: {router_field: null, success_values: [completed]}`, `on_error: failed`.
- Sub-graph outputs bind to `main.<field>` of the main script's Output.
