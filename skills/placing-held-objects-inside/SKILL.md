---
name: placing-held-objects-inside
description: Transport an already held object into a grounded receiving region using grasp-offset compensation,
  measured TCP arrivals, and bounded motion alternatives. Separate strategies handle open containers,
  shallow receivers, exposed sliding cavities, and divided openings. Release and withdraw before checking
  fresh spatial evidence, or retain a stabilizing pinch when explicitly requested.
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
  - place_inside: scripts/place_inside.py
  - raise_failure: scripts/raise_failure.py
  streaming: false
---

# placing-held-objects-inside

Transport an already held object into a grounded receiving region using grasp-offset compensation, measured TCP arrivals, and bounded motion alternatives. Separate strategies handle open containers, shallow receivers, exposed sliding cavities, and divided openings. Release and withdraw before checking fresh spatial evidence, or retain a stabilizing pinch when explicitly requested.

This skill was written by the memory system from confirmed experience: earlier tasks failed at this kind of step, were repaired, and the repair was confirmed by a rerun. Honor the coordinator's contract for your sub-graph (its inputs, outputs and exit values) exactly; this skill has no fixed tool whitelist, so compose the registered tools listed below. Its exits are `completed` (success) and `failed` (on_error).

## When to use

- The object is already held and both object and receiver OBBs are available.
- A receiving cavity is already exposed; any subsequent closing operation belongs to another skill.
- A named compartment can be selected relative to the receiver frame and the external camera.

## When NOT to use

- The object still needs to be grasped, or the receiver must first be opened.
- The transport route requires a separately reconstructed obstacle-aware carrying plan.
- The full object cannot fit the receiving region in an available orientation.

## Procedure

1. Wire one main script node with the interface inputs. Route status done to success noop completed; route failed to fail running raise_failure.py with reason, then the failed exit. Invoke once per held object.
2. Convert the supplied grasp reference and live hand feedback to TCP coordinates. Propagate the pre-grasp object center through the measured displacement. Select the receiver strategy and compensate the complete object-to-TCP offset.
3. For a divided opening, select the requested side relative to the external camera, align the object's long direction with the opening, and insert in bounded stages. Explicit stabilization retains the pinch rather than performing a release.
4. Lift above the measured rim before lateral transport. Check TCP position and orientation after each leg. Try bounded Cartesian corrections, an endpoint fallback, and alternate clearance heights; do not release after unresolved misalignment.
5. Open with settling, verify the measured open fraction, and withdraw vertically. Inspect fresh receiver-local depth changes and, when supplied, a target description from two views. Distinguish measured failure from a hidden final view after verified placement.

## Completion checks

- Required transport arrivals are checked in TCP coordinates with position and rotation tolerances before descent or release.
- The propagated object center, not just the hand, must lie inside the selected footprint before release. A shortened descent is accepted only with a small residual and plausible measured support.
- After withdrawal, check the measured open fraction and fresh target or receiver-local elevated depth evidence. A measured outside target fails; an unavailable view after verified inside arrival is reported as done with the uncertainty stated.
- Stabilized insertion requires measured inside placement, an occupied pinch, and fresh opening-local depth evidence where visible. It does not claim that release occurred.

## Canonical scripts

Use them as `type: script` nodes by their path (their run() inputs and Output fields are listed below); wire their inputs from your sub-graph's inputs or upstream nodes.

- `scripts/place_inside.py`
- `scripts/raise_failure.py`

## Parameters (main script)

Bind an explicit parameter as a literal input of the `main` node only when the task instruction determines it; otherwise omit it (the default applies).

- `strategy` (default `"open"`): Allowed: open for 'in a basket/bin'; shallow for 'in a tray' with support-relative seating; sliding for 'inside an opened drawer'; divided for 'in a compartment'. Selects distinct motion and release-height strategies.
- `region` (default `"center"`): Allowed: center, back, front, left, right. Bind from the named receiving sub-region. Directions are relative to the external camera; center applies when no sub-region is requested.
- `disposition` (default `"release"`): Allowed: release or stabilize. Ordinary 'put/place in' selects release. Select stabilize only when the requested operation explicitly requires insertion while retaining the stabilizing pinch.
- `object_description` (default `""`): The held-object noun phrase from the instruction, used for optional fresh target-specific verification. Empty disables semantic localization and uses receiver-local depth evidence.
- `tcp_length` (default `0.103`): Configured hand-reference-to-TCP distance in metres along local positive z.
- `clearance` (default `0.16`): Minimum vertical carrying and withdrawal clearance in metres, added to runtime receiver and held-object dimensions.
- `clearance_step` (default `0.04`): Additional metres for the second bounded transport-height candidate.
- `rim_offset` (default `0.06`): Open-container TCP release height above the measured rim, in metres.
- `seating_margin` (default `0.012`): Object-bottom margin above the measured receiving support, in metres.
- `position_tolerance` (default `0.018`): Maximum measured TCP arrival error in metres.
- `angle_tolerance` (default `0.12`): Maximum measured wrist rotation error in radians.
- `support_tolerance` (default `0.035`): Maximum residual and object-bottom support discrepancy for accepting a measured shortened seating descent, in metres.
- `footprint_margin` (default `0.005`): Inset from selected receiving-region bounds for object-center checks, in metres.
- `region_shift` (default `0.52`): Fraction of the receiver half-size used to locate a named side or allocated packing slot.
- `region_long_scale` (default `0.3`): Long-axis half-size fraction of the central divided opening.
- `region_short_scale` (default `0.42`): Short-axis half-size fraction of the selected divided opening.
- `insertion_step` (default `0.025`): Preferred staged insertion spacing and maximum Cartesian correction per attempt, in metres.
- `opening_fraction` (default `0.8`): Minimum measured open fraction that confirms opening after release.
- `pinch_min` (default `0.025`): Lower measured occupied-pinch fraction for explicit stabilization.
- `pinch_max` (default `0.94`): Upper measured occupied-pinch fraction for explicit stabilization.
- `depth_change` (default `0.012`): Minimum fresh depth change in metres for new receiving-region material.
- `visible_height` (default `0.06`): Maximum visible target-center or surface height above the receiver rim, in metres.
- `minimum_points` (default `25`): Minimum finite depth samples supporting a fresh spatial check.
- `settle_steps` (default `60`): Bounded settling duration supplied to each opening attempt.
- `slot` (default `0.0`): Normalized long-axis allocation in [-1,1]. For several objects in one receiver, assign distinct signed values when the instruction requires shared packing; zero uses the selected region center.

## Sub-graph wiring (system)

- `main`: `type: script`, `scripts/place_inside.py`, inputs bound from the sub-graph inputs (`in.<name>`); optional parameters only when the task states them.
- `conditional_edges.main`: `router_field: status`, mapping `done` -> `completed` and `failed` -> `fail`; you may insert a retry or alternative branch before `fail`.
- `fail`: `type: script`, `scripts/raise_failure.py`, input `reason` = `main.reason`; edge `fail` -> END (it raises, so the sub-graph leaves through its on_error exit `failed`).
- `completed`: `type: noop`, edge `completed` -> END; `exit: {router_field: null, success_values: [completed]}`, `on_error: failed`.
- Sub-graph outputs bind to `main.<field>` of the main script's Output.
