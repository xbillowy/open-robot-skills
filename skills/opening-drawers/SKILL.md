---
name: opening-drawers
description: >-
  Open a drawer to expose its interior. For requests to put an object in a
  drawer, coordinate this order: open the drawer with the hand free, clear
  the hand, re-perceive the opened interior, then grasp the object and place
  it into that fresh interior box. Never use a closed-drawer box as the
  placement destination. Verify containment after release and hand clearance;
  close the drawer only if requested and containment is established.
gap:
  exit_conditions:
    opened: Drawer opening is confirmed from current observations, the graspable feature is released, and the hand is clear for fresh interior perception.
    failed: Opening or hand clearance fails or cannot be confirmed. Route through on_error to abort the dependent placement flow.
---

# Opening drawers

Use this skill to open a drawer before accessing its interior. Ground the
opening motion in current perceived drawer and graspable-feature geometry.

## Flow-template example hint: placing an object inside a drawer

The coordinator should arrange the following sequence across the opening,
perception, grasping, and placement skills. This is an ordering hint for graph
generation, not an executable procedure or a new canonical script.

1. With the hand free, perceive the requested drawer and its graspable feature,
   then open the drawer. Complete opening before grasping the object to place.
2. Release the drawer feature and clear the hand from the opening so the
   interior is visible and accessible. Confirm that opening succeeded before
   advancing to the `opened` exit and placement preparation. Route opening
   failures through `on_error: failed`.
3. Acquire a fresh observation and re-perceive the exposed interior of the
   same drawer. Identify the usable interior box, including its floor and
   boundaries; a box enclosing the drawer exterior or handle is insufficient.
   Discard destination geometry obtained while the drawer was closed.
4. Perceive and grasp the object, then transport it to the freshly perceived
   interior box. Wire that post-opening interior box explicitly into placement
   inputs. Derive placement from its current geometry and the held object's
   dimensions, keeping the object within the interior boundaries.
5. Release only after reaching the intended placement pose inside the drawer.
   Clear the hand, then use a fresh observation to verify that the released
   object is contained in the opened interior. Reaching a pose above the
   drawer or issuing a release command alone does not establish containment.
6. If closing was requested, close only after containment is established and
   the hand is clear. Otherwise leave the drawer open.

Opening changes the destination geometry: never place against a stale
closed-drawer box. If the drawer moves again before placement, refresh its
interior geometry before proceeding. If opening, interior perception, or
containment verification fails or is unavailable, route to the relevant
failure exit; do not advance using an assumed interior or close on an
unverified placement.
