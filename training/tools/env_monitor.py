"""Offline environment monitor (replay, live stays off): per clip, how many detector boxes look like pool fixtures /
reflections rather than people, so drift across the evening (lighting 18:14 -> 19:13) shows up before it reaches live.
Inputs: labels/auto/pool_consensus.jsonl (detector proposals on the 4,664 pool frames, score >= 0.35), the distractor verifier
(CPU; the GPU is held by training) and the static-background map. Output: reports/env_monitor.md + data/env_monitor.json.
Boxes the verifier was trained on (owner-audited) are excluded, so the numbers are out-of-sample for the verifier.
    training\\.venv\\Scripts\\python.exe training\tools\\env_monitor.py [--per-clip 120]
"""
import json, sys, random
from collections import defaultdict
from pathlib import Path
import numpy as np, torch, joblib
sys.path.insert(0, str(Path(__file__).parent))
import distractor_verifier as D
TR = D.TR
per_clip = int(sys.argv[sys.argv.index("--per-clip") + 1]) if "--per-clip" in sys.argv else 120
audited = {tuple(D.bm[i][0:1]) + tuple(np.round(D.bm[i][1], 0)) for i in D.truth() if i in D.bm}
rows = defaultdict(list)
for l in open(TR / "labels/auto/pool_consensus.jsonl", encoding="utf-8"):
    r = json.loads(l)
    for i in r["instances"]:
        if max(i["voters"].values(), default=0) >= 0.35 and i["box"][3] - i["box"][1] > 12 and (r["key"], *np.round(i["box"], 0)) not in audited:
            rows[r["key"].split("/")[0]].append((r["key"], i["box"]))
rng = random.Random(0)
fix = json.loads((TR / "data/static_background.json").read_text())
F = np.array([f["box"] for f in fix]) if fix else np.zeros((0, 4))
v = joblib.load(TR / "work_dirs/verifier/verifier.joblib")
from transformers import AutoModel
m = AutoModel.from_pretrained("facebook/dinov2-base").eval()
torch.set_num_threads(6)
def iou(b, A):
    if not len(A): return 0
    w = np.minimum(b[2], A[:, 2]) - np.maximum(b[0], A[:, 0]); h = np.minimum(b[3], A[:, 3]) - np.maximum(b[1], A[:, 1])
    i = np.clip(w, 0, None) * np.clip(h, 0, None); return (i / ((b[2]-b[0])*(b[3]-b[1]) + (A[:, 2]-A[:, 0])*(A[:, 3]-A[:, 1]) - i)).max()
mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
out = {}
for clip, L in sorted(rows.items()):
    S = rng.sample(L, min(per_clip, len(L)))
    X = []
    with torch.no_grad():
        for i in range(0, len(S), 16):
            x = np.stack([(D.crop(*it) / 255.0 - mean) / std for it in S[i:i + 16]]).transpose(0, 3, 1, 2)
            o = m(pixel_values=torch.tensor(x, dtype=torch.float32)).last_hidden_state
            X.append(torch.cat([o[:, 0], o[:, 1:].mean(1)], 1).numpy())
    X = np.concatenate(X)
    geo = np.array([[(b[2]-b[0])/100, (b[3]-b[1])/100, float(D.water[int(b[1]):max(int(b[3]), int(b[1])+1), int(b[0]):max(int(b[2]), int(b[0])+1)].mean())] for _, b in S])
    p = v["clf"].predict_proba((np.hstack([X, geo]) - v["mu"]) / v["sd"])[:, 1]
    out[clip] = {"boxes": len(L), "sampled": len(S), "distractor_rate": float((p < 0.3).mean()), "fixture_hits": float(np.mean([iou(np.array(b), F) > 0.5 for _, b in S])), "mean_p_person": float(p.mean())}
    print(clip, out[clip], flush=True)
(TR / "data/env_monitor.json").write_text(json.dumps(out, indent=1))
L = ["# Environment monitor (offline replay)", "", "Out-of-sample for the verifier (audited boxes excluded). distractor rate = share of sampled detector boxes with p(person) < 0.30.", "",
     "| Clip (start time) | proposals | sampled | distractor rate | fixture hits | mean p(person) |", "|---|---|---|---|---|---|"]
for c, r in out.items():
    L.append(f"| {c} | {r['boxes']} | {r['sampled']} | {r['distractor_rate']:.0%} | {r['fixture_hits']:.0%} | {r['mean_p_person']:.2f} |")
d = np.array([r["distractor_rate"] for r in out.values()])
L += ["", f"Range {d.min():.0%}-{d.max():.0%}, median {np.median(d):.0%}. A clip well above the median is a place where fixtures/glare dominate the detector output."]
(TR / "reports/env_monitor.md").write_text("\n".join(L) + "\n", encoding="utf-8")
