---
name: transporting-objects
description: >
  Paper-replication transport cell that plans and validates held-object motion
  against separate target and destination lineages, executes placement and
  release, then validates retreat. Use when generating the sealed manipulation
  workflow for the paper setting.
compatibility: requires gap>=0.1
metadata: {category: motion, tags: [motion, transport, paper-replication, curobo]}
gap:
  allowed_tools:
    - geometry.compute_drop_position
    - curobo.plan_with_grasped_object
    - curobo.plan_directed_linear
    - curobo.validate_joint_trajectory_robot
    - curobo.validate_joint_trajectory_grasped
    - robot.execute_trajectory
    - robot.get_observation
    - robot.open_gripper
  exit_conditions:
    done: Placement, release, and validated retreat completed.
    rejected: Lineage, planning, collision, placement, release, or retreat evidence was rejected.
    error: The paper transport cell raised an execution error.
  required_inputs:
    held_grasp_json: str
    target_lineage_json: str
    destination_obb: OrientedBoundingBox
    destination_lineage_json: str
    world_config: WorldConfig
    target_name: str
    preset_json: str
  produces_outputs:
    terminal_result_json: str
  hard_rules:
    - "Use only plan_validate_transport.py."
    - "Consume separate target and destination lineages plus the grasp-built world_config."
    - "Publish only terminal_result_json."
    - "Do not add legacy container, mask, target OBB, end-effector-pose, or direct-placement inputs."
  canonical_scripts:
    - transport_descend_linear: scripts/transport_descend_linear.py
    - place_release: scripts/place_release.py
    - compute_drop_pose: scripts/compute_drop_pose.py
    - drop_offset_pose: scripts/drop_offset_pose.py
    - approach_above: scripts/approach_above.py
    - descend_release: scripts/descend_release.py
    - descend_release_linear: scripts/descend_release_linear.py
    - lift_grasped: scripts/lift_grasped.py
    - waypoint_move: scripts/waypoint_move.py
    - waypoint_move_carve: scripts/waypoint_move_carve.py
    - perceive_placement_zone: scripts/perceive_placement_zone.py
    - plan_validate_transport: scripts/plan_validate_transport.py
  prompts:
    vlm_select_zone: prompts/vlm_select_zone.md
  examples:
    - title: Canonical paper validated held-object transport cell
      path: examples/canonical_subgraph.json
  streaming: false
---

# transporting-objects

## Purpose

This dedicated paper cell verifies distinct target and destination lineages,
computes placement from the destination OBB, plans held-object transport,
validates robot and attachment trajectories, executes and revalidates a linear
placement descent, releases, and plans and validates retreat.

## Contract

Inputs are exactly `held_grasp_json`, `target_lineage_json`,
`destination_obb`, `destination_lineage_json`, `world_config`, `target_name`,
and canonical `preset_json`. The sole output is `terminal_result_json`.

## Exact graph

```text
START → plan_validate_transport
          ├─ success=True  → done → END
          └─ success=False → rejected → END
exception → error
```

The only script is `scripts/plan_validate_transport.py`; bind all seven inputs
directly from the subgraph contract and publish its terminal record.

## Prohibitions

Do not add container OBB/mask, target OBB/mask, end-effector pose, direct
Cartesian placement, fixed-height release, placement-zone perception,
re-perception, fallback motion, native state, or task-success evaluation.
