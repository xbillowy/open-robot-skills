## Planning requirements from confirmed experience

When you split this task into sub-graphs and choose their skills, follow each requirement whose condition holds for this task. Each one comes from failures of earlier tasks that were repaired and confirmed by a rerun.

1. When a step matches: The object is already held, and its pre-grasp OBB, grasp-time hand pose, and destination OBB are available.; The instruction requests placement on a surface, stacking, inclined-panel support, or placement beside a perceived support. -- assign that sub-graph the skill `placing-held-objects-on-supports` (its main script `place_on_support` as one `main` node; route on `status`: done -> `completed`, failed -> the `fail` node running `raise_failure`) instead of composing baseline grasp / transport scripts for it.
