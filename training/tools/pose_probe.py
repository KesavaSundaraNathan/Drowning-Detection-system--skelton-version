"""Is pose usable for a stroke first guess? Runs RTMPose-L (training weights, measurement only) on the owner-verified swimmer
crops and reports how many keypoints are confident (shoulders, wrists, nose) per crop, split by box height.
    training\\.venv\\Scripts\\python.exe training\tools\\pose_probe.py"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import distractor_verifier as D
from mmpose.apis import init_model, inference_topdown
cfg = D.TR / "third_party/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_aic-coco-256x192.py"
m = init_model(str(cfg), str(D.TR / "weights/rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.pth"), device="cuda:0")
t = D.truth(); ids = [i for i in t if t[i] and i in D.bm]
rows = []
for i in ids:
    key, b = D.bm[i]
    img = __import__("cv2").imread(str(D.TR / "data/frames" / key))
    r = inference_topdown(m, img, np.array([b], np.float32), bbox_format="xyxy")[0].pred_instances
    s = r.keypoint_scores[0]; rows.append((b[3] - b[1], s))
S = np.array([s for _, s in rows]); H = np.array([h for h, _ in rows])
names = {"nose": 0, "shoulders": [5, 6], "wrists": [9, 10], "hips": [11, 12]}
L = ["# Pose probe on owner-verified swimmer boxes", "", f"{len(rows)} boxes, RTMPose-L (score > 0.5 = confident).", "", "| group | all | h < 60 px | h >= 60 px |", "|---|---|---|---|"]
for n, k in names.items():
    c = (S[:, k] > 0.5).reshape(len(S), -1).all(1)
    L.append(f"| {n} | {c.mean():.0%} | {c[H < 60].mean():.0%} | {c[H >= 60].mean():.0%} |")
L.append(f"\nBoxes: {int((H < 60).sum())} under 60 px, {int((H >= 60).sum())} at 60 px or more.")
(D.TR / "reports/pose_probe.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))
