---
name: operating-articulated-parts
description: Open or close articulated parts, turn controls, or push supported objects using observed
  geometry and measured effects. Use when the requested result requires physical actuation, not merely
  placement on a fixture. Includes engagement, incremental motion, bounded correction and verification
  after withdrawal.
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
  - verify_exposed_state: scripts/verify_exposed_state.py
  streaming: false
---

# operating-articulated-parts

Open or close articulated parts, turn controls, or push supported objects using observed geometry and measured effects. Use when the requested result requires physical actuation, not merely placement on a fixture. Includes engagement, incremental motion, bounded correction and verification after withdrawal.

This skill was written by the memory system from confirmed experience: earlier tasks failed at this kind of step, were repaired, and the repair was confirmed by a rerun. Honor the coordinator's contract for your sub-graph (its inputs, outputs and exit values) exactly; this skill has no fixed tool whitelist, so compose the registered tools listed below. Its exits are `completed` (success) and `failed` (on_error).

## When to use

- The requested result requires articulation or supported pushing.

## When NOT to use

- The fixture serves only as a placement support.
- Grasp-and-carry placement alone suffices.

## Procedure

1. Before actuation, observe the intended part and parent together. Refresh moved geometry. Infer travel, hinge or spindle from the parent and physical constraint, not a thin handle's narrow axis or a previous approach. Use consistent configured wrist-to-fingertip transforms.
2. Before closing an occupied container, inspect the entire released object and closing swept volume. If anything protrudes or seating is uncertain, refresh interior perception and reposition it before pushing.
3. For pushing, close the hand and align its pressure surface with the observed pushing face and support height. For pulling, align and close around the handle. Derive motion direction from the mechanism, not a generic grasp approach.
4. When closing an occupied compartment, withdraw clear before rotating. Select an outside, raised approach and tilt the hand if nearby hardware obstructs it. Preserve that posture during approach and check clearance from contents.
5. At approach and engagement, compare fresh hand position and orientation with the target in the same configured reference frame. Correct up to three times. If still outside tolerance, refresh geometry and choose another reachable approach before advancing.
6. Actuate in small bounded strokes along observed travel or hinge arcs. Measure hand arrival and part displacement after each stroke. On slipping or stalling, refresh geometry and re-engage; retry up to three strokes before changing approach.
7. For a raised grip on an observed vertical spindle, pinch from above and turn incrementally around its center. Derive the required direction and remaining angle from markings and current orientation, not a fixed turn. Release and withdraw before inspecting the control.
8. Near closure, apply a short seating push toward the intended part's associated closed reference. Follow the hinge arc for hinged parts. Withdraw before inspecting residual gaps; if occluded, obtain another view rather than accepting apparent alignment.
9. After withdrawal, run verify_exposed_state with an explicit physical goal. If unverified, reobserve and perform up to three corrective cycles, then change the pushing, engagement or viewing approach rather than declaring completion.
10. For supported pushing, derive the destination region from observed support geometry. Use clearance-preserving strokes and re-localize the object after each stroke; correct from object displacement rather than commanded hand travel.

## Completion checks

- Require measured position and orientation arrival before engagement; unresolved error requires renewed perception or planning.
- Accept only an exposed observation of seated closure, sufficient opening, required control orientation or arrival in the destination region. Ambiguity requires renewed observation and correction.

## Canonical scripts

Use them as `type: script` nodes by their path (their run() inputs and Output fields are listed below); wire their inputs from your sub-graph's inputs or upstream nodes.

- `scripts/verify_exposed_state.py`
