---
name: perceiving-objects
description: >
  Paper-replication perception cell for role-bound target or destination
  localization with DINO candidates, VLM disambiguation, SAM3 segmentation,
  depth reconstruction, and evidenced OBB fitting. Use when generating the
  sealed manipulation workflow for the paper setting.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, paper-replication, dino, vlm, sam3]
gap:
  allowed_tools:
    - robot.get_observation
    - grounding-dino.detect
    - vlm.query
    - sam3.segment_box
    - geometry.mask_to_world_points
    - geometry.filter_and_compute_obb
  exit_conditions:
    done: Role-bound perception and lineage evidence produced.
    rejected: Required perception evidence was rejected.
    error: The paper perception cell raised an execution error.
  required_inputs:
    query: str
    semantic_role: str
    preset_json: str
  produces_outputs:
    target_obb: OrientedBoundingBox
    target_mask: Mask
    target_observation: Observation
    target_lineage_json: str
    destination_obb: OrientedBoundingBox
    destination_lineage_json: str
  errors:
    - "NOT_FOUND: Object not detected in the exterior observation."
  hard_rules:
    - "Use only perceive_disambiguate_segment.py."
    - "Run separate target and destination cells; never reuse lineage_json across roles."
    - "Bind the generic script lineage_json to the matching role-specific output name."
    - "Do not add perceive_dino_vlm, wrist fallback, native state, fixed geometry, or fallback nodes."
  canonical_scripts:
    - perceive_dino_vlm: scripts/perceive_dino_vlm.py
    - perceive_disambiguate_segment: scripts/perceive_disambiguate_segment.py
  examples:
    - title: Canonical target-role paper perception cell
      path: examples/canonical_subgraph.json
    - title: Canonical destination-role paper perception cell
      path: examples/canonical_destination_subgraph.json
  streaming: false
---

# perceiving-objects

## Purpose

This dedicated paper cell performs exactly one exterior RGB-D observation,
broad Grounding-DINO candidate detection, VLM crop disambiguation, SAM3 box
segmentation, depth back-projection, and evidenced OBB fitting. Missing or
fallback evidence fails closed.

## Contract

Inputs are exactly `query`, `semantic_role`, and canonical
`recast.paper_manipulation.v1` `preset_json`. Invoke one instance with role
`target` and another with role `destination`.

The target instance publishes `target_obb`, `target_mask`,
`target_observation`, and `target_lineage_json`. The destination instance
publishes `destination_obb` and `destination_lineage_json`. Each role-specific
lineage output aliases that instance's `perceive.lineage_json`.

## Exact graph

```text
START → perceive_disambiguate_segment
          ├─ success=True  → done → END
          └─ success=False → rejected → END
exception → error
```

Use `examples/canonical_subgraph.json` for target and
`examples/canonical_destination_subgraph.json` for destination. Their only
script is `scripts/perceive_disambiguate_segment.py`.

## Prohibitions

Do not emit observation, detector, generic perception, wrist-camera, geometry
shortcut, fallback, or retry nodes. Do not publish generic `lineage_json`
across cells, and never substitute one role's lineage for the other.
