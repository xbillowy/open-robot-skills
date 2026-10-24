---
name: verifying-object-retention
description: Checks a provisional finger grasp before transport. Performs a short vertical test lift from
  the measured hand pose using an explicitly configured feedback-reference-to-command-TCP transform, verifies
  measured lift arrival, and compares chronological public images for obstructed fingers and rise of the
  selected object. Use after grasping and before lateral transport; failed or ambiguous retention aborts
  the transfer.
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
  - verify_retention: scripts/verify_retention.py
  streaming: false
---

# verifying-object-retention

Checks a provisional finger grasp before transport. Performs a short vertical test lift from the measured hand pose using an explicitly configured feedback-reference-to-command-TCP transform, verifies measured lift arrival, and compares chronological public images for obstructed fingers and rise of the selected object. Use after grasping and before lateral transport; failed or ambiguous retention aborts the transfer.

This skill was written by the memory system from confirmed experience: earlier tasks failed at this kind of step, were repaired, and the repair was confirmed by a rerun. Honor the coordinator's contract for your sub-graph (its inputs, outputs and exit values) exactly; this skill has no fixed tool whitelist, so compose the registered tools listed below. Its exits are `completed` (success) and `failed` (on_error).

## When to use

- When a finger grasp must lift a movable object before transport and the object is visually observable.
- When the configured rigid transform between pose feedback and the Cartesian command TCP is available, including an explicitly supplied identity transform when the references coincide.

## When NOT to use

- When the action is pushing or articulation rather than lifting a movable payload.
- When the feedback-to-command reference transform is unknown or object-specific rise cannot be observed.

## Procedure

1. When entered after provisional closure, capture the measured hand pose and cameras, transform the pose with the configured rigid transform, and derive a short upward target; keep orientation fixed and perform no lateral motion; done when the bounded lift command has been attempted.
2. When the test lift returns, capture fresh pose feedback and images; compare the transformed endpoint with the lift target and inspect the selected object's motion separately from the hand; done when arrival, obstructed fingers and visible object rise are all confirmed.
3. When arrival, obstruction or object rise is absent or unclear, exit failed and block transport without retrying from stale geometry; when all pass, exit retained and leave the hand closed; done when the exit reflects those checks.

## Completion checks

- Fresh feedback transformed into the command TCP reference must be within 15 mm and 0.12 rad of the test-lift endpoint.
- Chronological image comparison must show the specifically selected object's surfaces rising with the hand and fingers remaining obstructed; hand geometry alone or aperture alone cannot pass.
- Missing images, an unknown reference transform, non-arrival or ambiguous visual evidence must fail rather than authorize transport.

## Canonical scripts

Use them as `type: script` nodes by their path (their run() inputs and Output fields are listed below); wire their inputs from your sub-graph's inputs or upstream nodes.

- `scripts/verify_retention.py`
