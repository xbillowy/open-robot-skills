---
name: placing-carried-objects-on-supports
description: Place a retained object using its measured grasp offset and usable receiving surface. Separate
  strategies cover ordinary support, stacking, adjacent placement, inclined panels, and placement followed
  by a supported slide beneath an overhang. Check TCP arrival, retention, opening, and fresh destination
  evidence. Unresolved final visibility does not negate a measured valid release.
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

Place a retained object using its measured grasp offset and usable receiving surface. Separate strategies cover ordinary support, stacking, adjacent placement, inclined panels, and placement followed by a supported slide beneath an overhang. Check TCP arrival, retention, opening, and fresh destination evidence. Unresolved final visibility does not negate a measured valid release.

This skill was written by the memory system from confirmed experience: earlier tasks failed at this kind of step, were repaired, and the repair was confirmed by a rerun. Honor the coordinator's contract for your sub-graph (its inputs, outputs and exit values) exactly; this skill has no fixed tool whitelist, so compose the registered tools listed below. Its exits are `completed` (success) and `failed` (on_error).

## When to use

- The object is already held and object, destination, and grasp-time references are available.
- The instruction requires supported placement, stacking, adjacent placement, or a supported slide beneath an overhang.

## When NOT to use

- Deep insertion or articulated manipulation; switching a control must be completed separately.
- Transport through clutter requiring a separately established route.

## Procedure

1. Wire one main node running place_on_support with the interface inputs. Route status done to success noop completed; route failed to fail running raise_failure.py with reason, then the failed exit. Bind explicit parameters from the instruction.
2. Reconstruct the carried object from grasp-time TCP displacement. Associate fresh body and receiving-region measurements with supplied references. Use the lower supporting body when visible; never project an old mask into fresh depth.
3. For adjacent placement, derive a direction and receiving rim, allow sufficient object-width clearance, preserve the carried orientation, and release above the neighboring support. An inclined receiving reference can select the camera-facing direction.
4. For centered placement, align the supporting body rather than the hand. A precise support_description can localize the usable receiving region within the supplied destination envelope.
5. For mode under, first place at the open side at the original supporting level. Then perform at most two supported pushes using fresh object measurements. Require observed displacement and the final beneath-overhang relation; hand travel alone never establishes success.
6. Check retention, lift, and traverse with bounded feedback corrections. Try Cartesian, endpoint, and planned alternatives plus lower approach clearances. Check object-relative release alignment before opening.
7. Measure opening, withdraw, and inspect fresh object and destination depth. Log unavailable or inconsistent observations. Ordinary placement can finish with a visibility limitation after measured valid arrival and opening; the supported-slide branch requires observed object effects.

## Completion checks

- Convert measured hand feedback to TCP using its rotation. Check release-body lateral error, base separation, and orientation before opening.
- Reject measured empty closure before transport and descent. Accept fresh held-body refinement only when size, position, and base agree with the predicted carried object.
- Use spatially associated receiving depth and lower-body measurements; reject unrelated or oversized measurements rather than replacing authoritative grasp references.
- After measured opening and withdrawal, inspect associated object geometry and destination depth. An inconclusive view alone is not positive evidence that a measured valid placement failed.
- After each supported push, withdraw and measure the object. Require measured forward displacement, an end beneath the receiving footprint, and height below the overhang.
- Catch and log motion failures, never execute unsuccessful plans, and bound every retry. The longest branch uses at most fourteen guarded perception calls.

## Canonical scripts

Use them as `type: script` nodes by their path (their run() inputs and Output fields are listed below); wire their inputs from your sub-graph's inputs or upstream nodes.

- `scripts/place_on_support.py`
- `scripts/raise_failure.py`

## Parameters (main script)

Bind an explicit parameter as a literal input of the `main` node only when the task instruction determines it; otherwise omit it (the default applies).

- `mode` (default `"surface"`): Allowed: surface, stack, inclined, under. Ordinary 'on' selects surface; 'stack on' selects stack; placement on an inclined panel selects inclined; 'under the shelf' selects under, including placement followed by a supported slide.
- `relation` (default `"center"`): Allowed: center, right, left. 'On' selects center; 'to the right of' selects right; 'to the left of' selects left. Sides use external-camera directions; measured inclined receiving geometry selects its camera-facing side. Use center with mode under.
- `object_description` (default `""`): The instruction's referring phrase for the carried object, excluding action and destination. Empty disables text measurements and preserves the recorded grasp reference. For mode under, supply the phrase so observed slide effects can be measured.
- `support_description` (default `""`): The instruction's receiving object or usable supporting region, such as its heating surface rather than its surrounding housing. Empty uses the supplied box or cloud. Fresh points must agree with the destination envelope.
- `tcp_offset` (default `0.103`): Configured positive hand-reference-to-TCP distance in metres along local positive z. Obtain from robot configuration, not the task wording.
- `clearance` (default `0.012`): Positive base clearance and general spatial tolerance in metres. Keep the default unless measured dimensions and depth resolution justify a different value.
- `approach_height` (default `0.13`): Positive initial vertical clearance above the derived release TCP in metres. Lower alternatives use sixty and thirty percent; also sets the withdrawal distance.
- `side_gap` (default `0.04`): Positive adjacent rim clearance in metres, enlarged when the carried body requires more room. Also provides supported-slide approach and withdrawal room. Set from measured available space, not scene coordinates.
- `association_tolerance` (default `0.04`): Positive spatial agreement tolerance in metres for associating fresh measurements with recorded references. A mismatch preserves the original reference.

## Sub-graph wiring (system)

- `main`: `type: script`, `scripts/place_on_support.py`, inputs bound from the sub-graph inputs (`in.<name>`); optional parameters only when the task states them.
- `conditional_edges.main`: `router_field: status`, mapping `done` -> `completed` and `failed` -> `fail`; you may insert a retry or alternative branch before `fail`.
- `fail`: `type: script`, `scripts/raise_failure.py`, input `reason` = `main.reason`; edge `fail` -> END (it raises, so the sub-graph leaves through its on_error exit `failed`).
- `completed`: `type: noop`, edge `completed` -> END; `exit: {router_field: null, success_values: [completed]}`, `on_error: failed`.
- Sub-graph outputs bind to `main.<field>` of the main script's Output.
