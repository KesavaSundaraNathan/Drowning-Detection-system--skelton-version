"""Use the distractor verifier to find Claude's likely-wrong unverified answers and put them back on "?" in CVAT for the owner.
Only boxes Claude answered (labels/auto/claude_check_r1/r2.jsonl) that the owner has not audited, and whose current CVAT
verdict is still Claude's. Flag rule: Claude said person but p(person) < LOW, or Claude said not_a_person but p >= HIGH.
    --apply writes to CVAT; default is a dry run."""
import json, sys
from pathlib import Path
import numpy as np, joblib
sys.path.insert(0, str(Path(__file__).parent))
import distractor_verifier as D, claude_check_sheets as c
LOW, HIGH = 0.30, 0.60
apply = "--apply" in sys.argv
truth = D.truth()
claude = {}
for f in ("claude_check_r1.jsonl", "claude_check_r2.jsonl"):
    for l in open(D.TR / "labels/auto" / f, encoding="utf-8"):
        r = json.loads(l)
        if str(r.get("src", "")).startswith("owner"):
            continue
        claude[r["id"]] = r  # latest wins
ids = [i for i, r in claude.items() if i not in truth and r["verdict"] != "?" and i in D.bm]
v = joblib.load(D.TR / "work_dirs/verifier/verifier.joblib")
items = [D.bm[i] for i in ids]
X = D.features(items)
geo = np.array([[(b[2]-b[0])/100, (b[3]-b[1])/100, float(D.water[int(b[1]):max(int(b[3]), int(b[1])+1), int(b[0]):max(int(b[2]), int(b[0])+1)].mean())] for _, b in items])
p = v["clf"].predict_proba((np.hstack([X, geo]) - v["mu"]) / v["sd"])[:, 1]
flag = [(i, pp) for i, pp, in zip(ids, p) if (claude[i]["verdict"] != "not_a_person" and pp < LOW) or (claude[i]["verdict"] == "not_a_person" and pp >= HIGH)]
print(len(ids), "unverified Claude answers;", len(flag), "flagged:",
      sum(claude[i]["verdict"] == "not_a_person" for i, _ in flag), "not_a_person->?,", sum(claude[i]["verdict"] != "not_a_person" for i, _ in flag), "person->?")
if apply:
    s = c.session(); pid, vid = c.verdict_spec(s)
    by = {}
    for i, _ in flag:
        by.setdefault(claude[i]["task"], set()).add(i)
    n = 0
    for t, S in by.items():
        sh = s.get(f"{c.BASE}/api/tasks/{t}/annotations").json()["shapes"]
        up = []
        for x in sh:
            if x["id"] in S:
                cur = next((a["value"] for a in x["attributes"] if a["spec_id"] == vid), None)
                if cur == claude[x["id"]]["verdict"]:  # still Claude's answer, owner has not touched it
                    x["attributes"] = [{"spec_id": vid, "value": "?"}]; up.append(x)
        if up:
            s.patch(f"{c.BASE}/api/tasks/{t}/annotations", params={"action": "update"}, json={"version": 0, "shapes": up, "tags": [], "tracks": []}).raise_for_status(); n += len(up)
    with open(D.TR / "labels/auto/verifier_flags.jsonl", "w") as f:
        for i, pp in flag:
            f.write(json.dumps({"id": i, "task": claude[i]["task"], "was": claude[i]["verdict"], "p_person": round(float(pp), 3)}) + "\n")
    print("set back to '?':", n)
