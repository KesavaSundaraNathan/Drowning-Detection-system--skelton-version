# Life Guard labelling guide (v1, 2026-10-28)

For: your reviews of automatic labels, the 5 % audit, and the hand-labelled test set. CVAT: http://localhost:8080,
project **Life Guard swimmers v1**. When unsure, **flag, don't guess**: add a `review` tag with a `review_reason`.

## 1. Boxes
- Draw the box around the **full visible body**, including the part you can see under water (refracted legs, a shape
  under the surface). Do not extend the box to where you *think* hidden limbs are.
- **Every visible head gets a box**, even if only the head is above water (that person is a `swimmer`).
- Tight: the box edges touch the outermost visible pixels (hair, fingertips, feet). Splash is not part of the body.
- Partly out of frame: box only the visible part; set **truncated** if CVAT offers it, otherwise add `review` →
  `review_reason: truncated`.
- Minimum: anything you can recognise as a person, however small. Very small (under ~12 px tall) or unsure → box it and
  tag `review: tiny`.

## 2. `swimmer` or `person`
- **`swimmer`: any part of the body is in the water** (inside the `pool_water` mask): swimming, standing in the
  shallow end, sitting on the steps with legs in the water, holding the edge or a ladder from the water side.
- **`person`: completely out of the water**: walking on the deck, lifeguard, sitting on the edge with feet out of the
  water, spectators.
- A person mid-jump / mid-dive above the water: `swimmer` once any part touches the water; `person` before that.
- Sitting on the edge with feet in the water: `swimmer` (safety rule: anyone in contact with the water is watched).
- Reflections, posters, screens, mannequins: **never** a box. If a reflection is very person-like, tag
  `review: reflection` so the detectors learn from it.

## 3. Pose (`pose` skeleton, COCO-17)
Visibility per keypoint:
| Value | Meaning | In CVAT |
|---|---|---|
| **v = 2** | Visible: you can see the joint itself | normal point |
| **v = 1** | Under water or refracted, but you can judge its position confidently | point marked **occluded** |
| **v = 0** | Cannot be placed (hidden, out of frame, too blurred) | point marked **outside** (not placed) |
- **Never guess a joint.** An uncertain joint is v = 0; the training loss ignores it.
- Left / right are the **person's** left and right, not the image's. For a swimmer face-down, check the shoulders and
  hips against the direction of travel.
- No skeleton for people under ~40 px tall: box only.
- Wrists and ankles are the most often wrong: if a hand is lost in splash, its wrist is v = 0.

## 4. `activity` (track attribute on swimmer / person boxes)
| Value | Definition |
|---|---|
| `freestyle` | face down / rotating, alternating overarm strokes, flutter kick |
| `breaststroke` | face forward, simultaneous circular arm pull, frog kick, head rises each stroke |
| `backstroke` | face up, alternating overarm strokes backwards |
| `butterfly` | simultaneous overarm recovery, undulating body, dolphin kick |
| `treading_vertical` | upright in deep water, staying in place, calm sculling / cycling legs (normal) |
| `floating` | horizontal and still on the surface, face up or face down, relaxed (normal — but note face-down floats) |
| `standing_wading` | standing or walking in shallow water, feet on the bottom |
| `entering_diving` | jumping, diving or climbing in or out (ladder, steps, edge) |
| `underwater` | fully below the surface (swimming under water, pushing off the wall) |
| `unknown` | cannot tell, or the person is out of the water (deck `person`) |
- The activity can change within a track: set it at the frame where it changes (CVAT keyframe).
- This dataset is **normal swimming only**. If you ever see something that looks like real distress, tag
  `review: possible distress` and tell the operator/lifeguard team: it must not silently become a "normal" example.

## 5. Edge cases
- **Overlapping people:** one box per person, even if heavily overlapped; if you cannot tell two people apart, box the
  visible one(s) and tag `review: overlap`.
- **Splashing:** box the body under the splash if you can see it; if the person is fully hidden by splash for that
  frame, no box (the track continues once visible) and tag `review: splash`.
- **Lane ropes / floats / kickboards:** not persons. A person holding a float: box the person only.
- **People cut by the frame edge:** box the visible part (see section 1).
- **Glare:** if a person is inside a glare patch and you can still see them, box them; if not, tag `review: glare`.

## 6. Reviewing automatic labels fast (CVAT)
- Each review job is 25–40 frames, hardest first; `review_reason` says why the frame was flagged.
- Keys: **N** draw / repeat last shape · **F / D** next / previous frame · **Del** delete selected shape ·
  **Ctrl+S** save · drag corners to fix a box. The full, current list: CVAT's shortcut panel (**F1**).
- Pre-labels in review jobs: `swimmer` / `person` boxes are where at least 2 of 3 models agreed (class from the pool
  mask); orange **`candidate`** boxes are where only one model was confident. **Every `candidate` must end as a
  `swimmer` / `person` box (right-click → change label, fix the box) or be deleted** — a completed job with a
  `candidate` left is sent back by QA. The `review` tag on the frame says why it was picked.
- Order of work on each frame: (1) delete false boxes, (2) add missed people, (3) fix boxes that are off by more than
  ~10 % of their size, (4) fix the class, (5) fix only clearly wrong joints (set to v = 0 rather than re-placing when
  unsure).
- Mark the job **Completed** when done (that is how I know it is finished; no message needed).

## 7. Test set (hand labelling)
- SAM 2 **click-to-box only**: click the person, check the box, adjust. No pre-labels, no automatic labels.
- 300 frames with boxes + class; ~100 of them also with keypoints; one dense 10-s segment (250 frames) with boxes and
  track ids (keep the same id for the same person).
- The test set is never used for training or tuning, and its metrics are only shown at the final evaluation.
