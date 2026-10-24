---
name: placing-carried-objects-on-supports
description: Place a retained object on an observed support while compensating the grasp offset. Lift,
  traverse, descend using measured TCP feedback, release, withdraw, and inspect fresh destination evidence.
  Separate strategies cover ordinary surfaces, stacking, adjacent placement, and inclined panels. Accept
  fresh measurements only when they agree with supplied spatial references.
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

# placing-carried-objects-on-supports

Place a retained object on an observed support while compensating the grasp offset. Lift, traverse, descend using measured TCP feedback, release, withdraw, and inspect fresh destination evidence. Separate strategies cover ordinary surfaces, stacking, adjacent placement, and inclined panels. Accept fresh measurements only when they agree with supplied spatial references.

This skill was written by the memory system from confirmed experience: earlier tasks failed at this kind of step, were repaired, and the repair was confirmed by a rerun. Honor the coordinator's contract for your sub-graph (its inputs, outputs and exit values) exactly; this skill has no fixed tool whitelist, so compose the registered tools listed below. Its exits are `completed` (success) and `failed` (on_error).

## When to use

- The object is already held and upstream object, support, and grasp-time pose references are available.
- The goal requires supported placement, stacking, or placement beside a known support.

## When NOT to use

- Deep insertion, articulated manipulation, or supported pushing.
- Transport through clutter requiring a separately established route.

## Procedure

1. Wire one main script node running place_on_support with the sub-graph inputs. Route status done to success noop completed. Route failed to fail running raise_failure.py with reason, then the failed exit. Bind explicit parameters from the instruction.
2. Reconstruct the carried object using the original object box and measured TCP displacement from grasp. Accept fresh supporting-body and receiving-region measurements only when spatially consistent with these references. Never project an old mask into fresh depth.
3. Select the surface, stack, or inclined strategy. Adjacent placement uses the receiving rim and external-camera directions. Inclined placement fits the supplied or freshly measured panel and chooses upright or laid-down orientation from the grasp direction.
4. Check retention, lift, and traverse. Try successively lower approach clearances if a high waypoint is unreachable. Each waypoint tries Cartesian, endpoint, and planned motion and checks measured TCP position and orientation.
5. Require a measured valid object-relative release pose before opening. Retry opening once, withdraw, and inspect fresh destination evidence. Ambiguous or mismatched segmentation cannot override valid arrival and measured opening.

## Completion checks

- Feedback is converted from the hand reference to TCP using its measured rotation. Object-relative lateral alignment, supporting height, and orientation are checked before release.
- Nearly empty closure prevents loaded motion. A fresh held-body refinement must agree with the predicted carried position, original size, and supporting height; otherwise the recorded grasp relationship remains authoritative.
- After measured opening and withdrawal, fresh destination depth or associated object geometry provides support evidence. An unresolved final view is reported as a limitation, not as evidence of failed placement.
- All motion alternatives are caught and logged; unsuccessful plans are not executed. Lower approach alternatives address unreachable high poses. At most nine guarded perception calls are made.

## Canonical scripts

Use them as `type: script` nodes by their path (their run() inputs and Output fields are listed below); wire their inputs from your sub-graph's inputs or upstream nodes.

- `scripts/place_on_support.py`
- `scripts/raise_failure.py`

## Parameters (main script)

Bind an explicit parameter as a literal input of the `main` node only when the task instruction determines it; otherwise omit it (the default applies).

- `mode` (default `"surface"`): Allowed values surface, stack, inclined. Ordinary 'on' or 'on top' selects surface; 'stack on' selects stack; 'on the rack' referring to an inclined receiving panel selects inclined. Stack retains the carried orientation and seats its base over the receiving rim.
- `relation` (default `"center"`): Allowed values center, right, left. 'On' selects center; 'to the right of' selects right; 'to the left of' selects left. Side placement is relative to the external camera and beyond the measured receiving rim.
- `object_description` (default `""`): Bind the instruction's referring phrase for the carried object, without the action or destination. Empty disables text segmentation and retains the upstream grasp relationship.
- `support_description` (default `""`): Bind the instruction's receiving object or usable supporting region. Empty uses the upstream support box or cloud. Fresh measurements must agree spatially with that box.
- `tcp_offset` (default `0.103`): Positive configured hand-reference-to-TCP distance in metres along local positive z; obtained from the robot configuration.
- `clearance` (default `0.012`): Positive general support clearance and positional tolerance in metres. Retain the default unless measured shape and support resolution justify another value.
- `approach_height` (default `0.13`): Positive initial vertical clearance above the derived release TCP in metres. The script also tries sixty and thirty percent of this clearance when high approaches are unreachable.
- `side_gap` (default `0.04`): Positive distance in metres beyond the observed receiving rim for adjacent placement; adjust only from measured space and carried-object dimensions.
- `association_tolerance` (default `0.04`): Positive spatial agreement tolerance in metres for associating fresh measurements with upstream references. A mismatch preserves the recorded reference rather than substituting an unrelated measurement.

## Sub-graph wiring (system)

- `main`: `type: script`, `scripts/place_on_support.py`, inputs bound from the sub-graph inputs (`in.<name>`); optional parameters only when the task states them.
- `conditional_edges.main`: `router_field: status`, mapping `done` -> `completed` and `failed` -> `fail`; you may insert a retry or alternative branch before `fail`.
- `fail`: `type: script`, `scripts/raise_failure.py`, input `reason` = `main.reason`; edge `fail` -> END (it raises, so the sub-graph leaves through its on_error exit `failed`).
- `completed`: `type: noop`, edge `completed` -> END; `exit: {router_field: null, success_values: [completed]}`, `on_error: failed`.
- Sub-graph outputs bind to `main.<field>` of the main script's Output.
