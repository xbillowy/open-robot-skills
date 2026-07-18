---
name: grasping-with-planner
description: >
  Paper-replication grasp cell that reconstructs the collision world, plans
  and validates a sealed top-down candidate set, then executes and visually
  verifies the grasp and lift. Use when generating the sealed manipulation
  workflow for the paper setting.
compatibility: requires gap>=0.1
metadata: {category: grasping, tags: [grasping, paper-replication, collision-aware, curobo]}
gap:
  allowed_tools:
    - geometry.top_down_grasp_candidates
    - geometry.build_world_config
    - geometry.mask_to_world_points
    - curobo.batch_grasp_feasibility
    - curobo.plan_directed_linear
    - curobo.plan_to_grasp_poses
    - curobo.validate_joint_trajectory_robot
    - curobo.validate_joint_trajectory_grasped
    - robot.execute_trajectory
    - robot.get_observation
    - robot.close_gripper
    - vlm.query
  exit_conditions:
    done: Grasp, visual hold verification, and evidenced lift completed.
    rejected: Candidate, planning, validation, or hold evidence was rejected.
    error: The paper grasp cell raised an execution error.
  required_inputs:
    target_obb: OrientedBoundingBox
    target_mask: Mask
    target_observation: Observation
    target_lineage_json: str
    target_name: str
    preset_json: str
  produces_outputs:
    held_grasp_json: str
    world_config: WorldConfig
  hard_rules:
    - "Use exactly build_world.py → plan_validate_grasp.py → execute_verify_grasp.py."
    - "Build world_config inside this cell from target_observation, target_mask, and target_obb."
    - "Never declare world_config as a grasp-cell input."
    - "Publish only held_grasp_json and the internally built world_config."
    - "Do not use grasp_descend_linear, direct Cartesian, best-effort, or privileged-state flows."
  canonical_scripts:
    - build_world: scripts/build_world.py
    - grasp_descend_linear: scripts/grasp_descend_linear.py
    - select_short_axis: scripts/select_short_axis.py
    - plan_validate_grasp: scripts/plan_validate_grasp.py
    - execute_verify_grasp: scripts/execute_verify_grasp.py
  examples:
    - title: Canonical paper world-build → validated-plan → verified-execution cell
      path: examples/canonical_subgraph.json
  streaming: false
---

# grasping-with-planner

## Purpose

This dedicated paper cell builds an observation-grounded collision world,
plans over the sealed candidate count and IK seed count, validates robot and
held-object trajectories, executes one admitted candidate, verifies the hold
with fresh visual evidence, and performs the sealed evidenced lift.

## Contract

Inputs are exactly `target_obb`, `target_mask`, `target_observation`,
`target_lineage_json`, `target_name`, and canonical `preset_json`.
`world_config` is built inside the cell and is not an input. Outputs are
exactly `held_grasp_json` and `world_config`.

## Exact graph

```text
START → build_world → plan_validate_grasp
                         ├─ success=True  → execute_verify_grasp
                         │                    ├─ success=True  → done → END
                         │                    └─ success=False → rejected → END
                         └─ success=False → rejected → END
exception → error
```

Use only `scripts/build_world.py`, `scripts/plan_validate_grasp.py`, and
`scripts/execute_verify_grasp.py`. Bind `plan.validated_grasp_json` into
execute; bind `build_world.config` into both admitted scripts and publish it as
`world_config`.

## Prohibitions

Do not add open/compute/goto/observe/close nodes, `grasp_descend_linear`, a
world-config input, fixed-height Cartesian motion, fallback planners,
best-effort validation, gripper-fraction proof, or privileged/native state.
