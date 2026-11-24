"""One-page answer sheet for the boxes still on "?" (no CVAT frame-by-frame work).
make : training\\data\audit\answer_r1.html - every "?" proposal as a crop with the verifier's suggestion pre-selected; the owner
       clicks only where the suggestion is wrong (or "Accept all suggestions"), then "Save results" -> answer_r1_result.json.
apply: writes the owner's verdicts to CVAT (only boxes still on "?").
    training\\.venv\\Scripts\\python.exe training\tools\answer_page.py make|apply [--result file]"""
import json, sys
from pathlib import Path
import numpy as np, joblib
sys.path.insert(0, str(Path(__file__).parent))
import distractor_verifier as D, claude_check_sheets as c, claude_audit as A, claude_redo as R


def pending(s, vid, pid):
    seed = json.loads((D.TR / "data/seed_r1.json").read_text()); out = []
    for t in c.check_tasks(s):
        split = "train" if " train " in t["name"] else "val"
        meta = s.get(f"{c.BASE}/api/tasks/{t['id']}/data/meta").json()["frames"]
        for sh in s.get(f"{c.BASE}/api/tasks/{t['id']}/annotations").json()["shapes"]:
            if sh["type"] == "rectangle" and sh["label_id"] == pid and any(a["spec_id"] == vid and a["value"] == "?" for a in sh["attributes"]):
                out.append({"task": t["id"], "id": sh["id"], "key": seed[split][int(meta[sh["frame"]]["name"][:4])]["key"], "box": sh["points"][:4]})
    return out


def make():
    s = c.session(); pid, vid = c.verdict_spec(s)
    P = pending(s, vid, pid)
    v = joblib.load(D.TR / "work_dirs/verifier/verifier.joblib")
    X = D.features([(p["key"], p["box"]) for p in P])
    geo = np.array([[(p["box"][2]-p["box"][0])/100, (p["box"][3]-p["box"][1])/100, float(D.water[int(p["box"][1]):max(int(p["box"][3]), int(p["box"][1])+1), int(p["box"][0]):max(int(p["box"][2]), int(p["box"][0])+1)].mean())] for p in P])
    pr = v["clf"].predict_proba((np.hstack([X, geo]) - v["mu"]) / v["sd"])[:, 1]
    for p, q in zip(P, pr):
        p["sug"] = R.verdict_for_person(D.water, p["box"]) if q >= 0.5 else "not_a_person"; p["p"] = round(float(q), 2)
    cards = []
    for i, p in enumerate(P):
        cards.append(f"""<div class="card" id="c{i}"><img src="data:image/jpeg;base64,{A.crop_b64(p['key'], p['box'])}">
<p>#{i+1} &middot; suggestion: <b>{p['sug'].replace('_',' ')}</b> ({p['p']:.0%} person)</p>
<button onclick="set({i},'swimmer')">Person in water</button><button onclick="set({i},'person')">Person on deck</button><button onclick="set({i},'not_a_person')">NOT a person</button></div>""")
    meta = [{k: p[k] for k in ("task", "id", "key", "sug")} for p in P]
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Answer sheet</title><style>
body{{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:16px}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:12px}}
.card{{background:#222;padding:8px;border-radius:6px;border:3px solid #444}}.card img{{max-width:100%}}.card.swimmer{{border-color:#2a2}}.card.person{{border-color:#29d}}.card.not_a_person{{border-color:#d33}}
button{{font-size:15px;padding:6px 10px;margin-right:4px}}#top button{{font-size:18px;padding:10px 20px;margin:6px}}</style></head><body>
<h2>Answer sheet: {len(P)} boxes</h2><p>The yellow box is the question. Colour = current answer (green person in water, blue person on deck, red not a person). A loose or group box that holds a person counts as a person. Grey = not answered yet (keeps the suggestion if you press Accept).</p>
<div id="top"><button onclick="acc()">Accept all suggestions for unanswered</button><button onclick="save()">Save results</button> <span id="count"></span></div>
<div class="grid">{''.join(cards)}</div><script>const meta={json.dumps(meta)};const res={{}};
function set(i,v){{res[i]=v;document.getElementById('c'+i).className='card '+v;cnt();}}
function acc(){{meta.forEach((m,i)=>{{if(!res[i])set(i,m.sug)}});}}
function cnt(){{document.getElementById('count').textContent=Object.keys(res).length+' / '+meta.length+' answered';}}cnt();
function save(){{const out=meta.map((m,i)=>Object.assign({{owner:res[i]||null}},m));const a=document.createElement('a');
a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{{type:'application/json'}}));a.download='answer_r1_result.json';a.click();}}</script></body></html>"""
    (D.TR / "data/audit/answer_r1.html").write_text(html, encoding="utf-8")
    print(len(P), "boxes ->", D.TR / "data/audit/answer_r1.html", dict(__import__("collections").Counter(p["sug"] for p in P)))


def apply(path):
    s = c.session(); pid, vid = c.verdict_spec(s)
    rows = [r for r in json.loads(Path(path).read_text()) if r["owner"]]
    by = {}
    for r in rows: by.setdefault(r["task"], {})[r["id"]] = r["owner"]
    n = 0
    for t, vd in by.items():
        sh = s.get(f"{c.BASE}/api/tasks/{t}/annotations").json()["shapes"]; up = []
        for x in sh:
            if x["id"] in vd and any(a["spec_id"] == vid and a["value"] == "?" for a in x["attributes"]):
                x["attributes"] = [{"spec_id": vid, "value": vd[x["id"]]}]; up.append(x)
        if up:
            s.patch(f"{c.BASE}/api/tasks/{t}/annotations", params={"action": "update"}, json={"version": 0, "shapes": up, "tags": [], "tracks": []}).raise_for_status(); n += len(up)
    with open(D.TR / "labels/auto/owner_answers_r1.jsonl", "a") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print("wrote", n, "verdicts of", len(rows))


if __name__ == "__main__":
    make() if sys.argv[1] == "make" else apply(sys.argv[sys.argv.index("--result") + 1] if "--result" in sys.argv else str(Path.home() / "Downloads/answer_r1_result.json"))
