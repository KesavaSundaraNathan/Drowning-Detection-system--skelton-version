"""Distractor verifier: person vs reflection / underwater marking / glare / rope / equipment, trained ONLY on the owner's
owner-verified boxes (audit rounds 1+2 and the answer sheet) - Claude's unaudited answers are never used.

Frozen DINOv2-base (Apache-2.0) features of the box crop (+35 % context) -> logistic regression. Evaluated by
leave-clips-out cross-validation (a clip never appears in train and test). Output: training/work_dirs/verifier/
verifier.joblib + reports/distractor_verifier.md. Used to re-score detector boxes (never changes labels).

    training\\.venv\\Scripts\\python.exe training\tools\\distractor_verifier.py
"""
import json, sys
from pathlib import Path
import cv2, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import joblib

TR = Path(__file__).resolve().parents[1]
DL = Path.home() / "Downloads"
fs = json.load(open(TR / "data/audit/frames_state.json"))
bm = {i: (k.split("|", 1)[1], b) for k, L in fs.items() for i, b, v in L}
water = cv2.imread(str(TR / "data/pool_mask/pool_water_mask.png"), 0) > 0


def truth() -> dict:
    t = {}
    for f in ("audit_r1_result.json", "audit_r2_result.json"):
        for r in json.load(open(DL / f)):
            if not r["owner"]:
                continue
            isp = r["verdict"] != "not_a_person"
            t[r["id"]] = isp if r["owner"] == "right" else (not isp)
    for r in json.load(open(DL / "answer_r1_result.json")):  # owner's answer sheet (ground truth)
        if r["owner"]:
            t[r["id"]] = r["owner"] != "not_a_person"
    return t


def crop(key, b, ctx=0.35):
    img = cv2.imread(str(TR / "data/frames" / key))
    h, w = img.shape[:2]
    x1, y1, x2, y2 = b
    bw, bh = max(x2 - x1, 8), max(y2 - y1, 8)
    X1, Y1, X2, Y2 = int(max(0, x1 - ctx * bw)), int(max(0, y1 - ctx * bh)), int(min(w, x2 + ctx * bw)), int(min(h, y2 + ctx * bh))
    c = cv2.cvtColor(img[Y1:Y2, X1:X2], cv2.COLOR_BGR2RGB)
    return cv2.resize(c, (224, 224), interpolation=cv2.INTER_CUBIC)


def features(items):
    from transformers import AutoModel
    m = AutoModel.from_pretrained("facebook/dinov2-base").cuda().eval().half()
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    out = []
    with torch.no_grad():
        for i in range(0, len(items), 32):
            x = np.stack([(crop(*it) / 255.0 - mean) / std for it in items[i:i + 32]]).transpose(0, 3, 1, 2)
            o = m(pixel_values=torch.tensor(x, dtype=torch.float16).cuda()).last_hidden_state
            out.append(torch.cat([o[:, 0], o[:, 1:].mean(1)], 1).float().cpu().numpy())
    return np.concatenate(out)


def main():
    t = truth()
    ids = [i for i in t if i in bm]
    items = [bm[i] for i in ids]
    y = np.array([t[i] for i in ids], int)
    groups = np.array([k.split("/")[0] for k, _ in items])
    X = features(items)
    # cheap geometry features help for the floor T-marking / rope boxes
    geo = np.array([[(b[2] - b[0]) / 100, (b[3] - b[1]) / 100, float(water[int(b[1]):max(int(b[3]), int(b[1]) + 1), int(b[0]):max(int(b[2]), int(b[0]) + 1)].mean())] for _, b in items])
    X = np.hstack([X, geo])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd
    res = {}
    for C in (0.001, 0.003, 0.01, 0.03, 0.1):
        p = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=min(5, len(set(groups)))).split(Xn, y, groups):
            p[te] = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Xn[tr], y[tr]).predict_proba(Xn[te])[:, 1]
        res[C] = (roc_auc_score(y, p), p)
    bestC = max(res, key=lambda c: res[c][0])
    auc, p = res[bestC]
    lines = ["# Distractor verifier (owner-audited boxes only)", "",
             f"{len(y)} boxes: {int(y.sum())} person, {int((1 - y).sum())} distractor, {len(set(groups))} clips; leave-clips-out 5-fold CV.", "",
             f"Best C={bestC}: **AUC {auc:.3f}**.", "", "| threshold | distractors removed | persons kept |", "|---|---|---|"]
    for th in (0.3, 0.4, 0.5, 0.6, 0.7):
        lines.append(f"| {th} | {((p < th) & (y == 0)).sum() / max((y == 0).sum(), 1):.1%} | {((p >= th) & (y == 1)).sum() / max(y.sum(), 1):.1%} |")
    lines += ["", "AUC by C: " + ", ".join(f"{c}: {r[0]:.3f}" for c, r in res.items())]
    out = TR / "work_dirs/verifier"
    out.mkdir(parents=True, exist_ok=True)
    clf = LogisticRegression(C=bestC, max_iter=3000, class_weight="balanced").fit(Xn, y)
    joblib.dump({"clf": clf, "mu": mu, "sd": sd, "model": "facebook/dinov2-base", "ctx": 0.35}, out / "verifier.joblib")
    (TR / "reports/distractor_verifier.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
