"""Static-background map: boxes the detectors fire on at the SAME place in many different clips (one fixed camera) are fixtures
(floor T-marking, lane-rope ends, pennants, wall patches), not swimmers. Built from labels/auto/dense_consensus.jsonl (all
detector proposals, 19 clips) - no manual labels. Validated against the owner-audited boxes (persons wrongly caught = the cost).
Output: data/static_background.json (cluster boxes + clip counts), reports/static_background.md. Never used to hide a live person.
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import distractor_verifier as D
TR = D.TR
boxes = []  # (clip, frame, box)
for l in open(TR / "labels/auto/dense_consensus.jsonl", encoding="utf-8"):
    r = json.loads(l); clip = r["key"].split("@")[0]
    for i in r["instances"]:
        if i["box"][3] - i["box"][1] > 12: boxes.append((clip, r["key"], i["box"]))
B = np.array([b for _, _, b in boxes]); clips = np.array([c for c, _, _ in boxes]); frames = np.array([f for _, f, _ in boxes])
print(len(B), "proposals,", len(set(clips)), "clips")
def iou_many(b, A):
    w = np.minimum(b[2], A[:, 2]) - np.maximum(b[0], A[:, 0]); h = np.minimum(b[3], A[:, 3]) - np.maximum(b[1], A[:, 1])
    i = np.clip(w, 0, None) * np.clip(h, 0, None); return i / ((b[2]-b[0])*(b[3]-b[1]) + (A[:, 2]-A[:, 0])*(A[:, 3]-A[:, 1]) - i + 1e-9)
# greedy clustering on a sample of 1 in 3 frames for speed, then count clips / frames over ALL boxes per cluster
order = np.random.default_rng(0).permutation(len(B))
centres = []
for k in order[:: 3]:
    if centres:
        C = np.array(centres)
        if iou_many(B[k], C).max() > 0.6: continue
    centres.append(B[k])
C = np.array(centres); print(len(C), "candidate places")
stats = []
for c in C:
    m = iou_many(c, B) > 0.6
    stats.append((int(len(set(clips[m]))), int(len(set(frames[m]))), c.tolist()))
import os
MIN_CLIPS, MIN_FRAMES = int(os.environ.get('MC',8)), int(os.environ.get('MF',150))
fix = [{"box": c, "clips": a, "frames": f} for a, f, c in stats if a >= MIN_CLIPS and f >= MIN_FRAMES]
fix.sort(key=lambda x: -x["frames"])
(TR / "data/static_background.json").write_text(json.dumps(fix))
# validation against owner-audited truth
t = D.truth(); F = np.array([x["box"] for x in fix]) if fix else np.zeros((0, 4))
hit = {i: (iou_many(np.array(D.bm[i][1]), F).max() > 0.5 if len(F) else False) for i in t if i in D.bm}
dis = [i for i in hit if not t[i]]; per = [i for i in hit if t[i]]
L = ["# Static-background map", "", f"{len(B)} detector proposals over {len(set(clips))} clips -> {len(C)} places -> **{len(fix)} fixtures** (seen in >= {MIN_CLIPS} clips and >= {MIN_FRAMES} frames, IoU > 0.6).", "",
     f"Against the owner's audit: catches **{sum(hit[i] for i in dis)}/{len(dis)} distractor boxes ({sum(hit[i] for i in dis)/max(len(dis),1):.0%})**; wrongly catches **{sum(hit[i] for i in per)}/{len(per)} persons ({sum(hit[i] for i in per)/max(len(per),1):.0%})**.", ""]
(TR / "reports/static_background.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))
