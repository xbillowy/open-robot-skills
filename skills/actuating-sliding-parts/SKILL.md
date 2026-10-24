---
name: actuating-sliding-parts
description: Locate and actuate a sliding panel through its handle or front pushing surface. Use for requested
  drawer opening or closure, with current parent-relative geometry, guarded engagement, incremental translation
  and measured progress. Verify the panel after releasing and withdrawing; hand travel alone is not completion.
  Supports ranked handles and clearance-aware closure of occupied compartments.
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
    failed: The operation could not be completed or verified. Retryable script failures return
      found false or moved false so the coordinator can re-perceive and retry.
  canonical_scripts:
  - locate_handle: scripts/locate_handle.py
  - actuate_along_axis: scripts/actuate_along_axis.py
  streaming: false
---

# actuating-sliding-parts

Locate and actuate a sliding panel through its handle or front pushing surface. Use for requested drawer opening or closure, with current parent-relative geometry, guarded engagement, incremental translation and measured progress. Verify the panel after releasing and withdrawing; hand travel alone is not completion. Supports ranked handles and clearance-aware closure of occupied compartments.

This skill was written by the memory system from confirmed experience: earlier tasks failed at this kind of step, were repaired, and the repair was confirmed by a rerun. Honor the coordinator's contract for your sub-graph (its inputs, outputs and exit values) exactly; this skill has no fixed tool whitelist, so compose the registered tools listed below. Its exits are `completed` (success) and `failed` (on_error).

## When to use

- The requested interaction translates a sliding panel outward or inward.
- Current observations identify the parent structure and its associated handle or front.

## When NOT to use

- The fixture is only a placement support and no articulation is required.
- The required motion follows a hinge arc or rotates a spindle.
- The requested action pushes a loose object across a surface.

## Procedure

1. Before each sliding interaction, run locate_handle from a fresh observation. Supply parent and part descriptions and any requested height rank. If found is false, obtain another unobstructed view and retry at most twice, then return a re-perception-needed result for recovery.
2. For pulling, use the selected handle grasp. For pushing, choose a closed-hand surface pose from the observed handle or panel front, with inward motion opposite pull_axis. Estimate travel from the selected panel's extension relative to its own fixed frame, not an unrelated background face.
3. When closing an occupied compartment, withdraw clear of the cavity before reorienting. Choose pushing orientation and approach together to clear contents and neighboring hardware; incline the hand only where observed clearance requires it. Preserve that orientation during approach.
4. Run actuate_along_axis with grasp_pose and pull_axis wired by name. Supply the configured feedback-to-command rigid transform when references differ. The script checks approach and engagement arrival, then measures fresh hand feedback after every bounded translation.
5. After a stroke, release and withdraw outward to expose the panel. Reobserve it against its associated fixed frame. If closure remains short, refresh pushing geometry and make at most two short seating strokes; do not demand large travel near the closed frame.
6. If arrival, travel or panel-effect checks fail, block dependent actions. Re-perceive and retry with a corrected reachable approach at most twice; then return recovery-needed with the failed check so the enclosing graph can select another view or approach.

## Completion checks

- Accept a target only when its geometry associates it with the intended parent and requested height rank. Otherwise refresh the view and candidate selection; never substitute an unrelated knob.
- Require measured position and orientation arrival before engagement progression. A false moved result triggers re-perception and bounded retry, not success; commanded distance and normal tool return are not proof of travel.
- Accept closure only when fresh unobstructed observation shows the intended panel seated at its own closed reference and remaining there after withdrawal. Retry a short seating push for a residual gap; request clearance recovery if obstructed.
- Accept opening only when fresh panel or aperture observations establish the requested access. Hand displacement is only a stroke diagnostic; if the panel did not follow, reselect engagement and retry within the bounded recovery sequence.

## Canonical scripts

Use them as `type: script` nodes by their path (their run() inputs and Output fields are listed below); wire their inputs from your sub-graph's inputs or upstream nodes.

- `scripts/locate_handle.py`
- `scripts/actuate_along_axis.py`
