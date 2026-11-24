"""Offline HTML tools so the owner can label in a browser instead of CVAT (nothing is uploaded).
    make missing : data/audit/missing_r1.html  - the 250 seed frames with their boxes drawn; drag a box around anybody who has none
    make test    : data/test_upload/label_test.html - the 300 TEST A/B frames, boxes only (hand labels, no model help); keypoints stay in CVAT
    make stroke  : data/audit/stroke_r1.html   - one looping clip per SAM 2 track in STROKE 1-4, owner picks the stroke
    apply missing|stroke [--result file]       - write the downloaded result into CVAT (missing -> new boxes, stroke -> track `activity`)
    export test [--result file]                - data/coco/test_hand.json (COCO) from the downloaded test labels
Keys on the box pages: drag = new box; 1 swimmer, 2 person (class of the next box); Z undo; right-click a new box deletes it;
right arrow / A next frame, left arrow / D previous. Progress autosaves in the browser; Save results downloads the JSON.
"""
import base64
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))
import claude_check_sheets as c
import distractor_verifier as D

TR = D.TR
DL = Path.home() / "Downloads"
COL = {"swimmer": "#2ecc40", "person": "#39a0ff"}
SEEDS = ["2026-10-22-18-42-47", "2026-10-22-18-49-23", "2026-10-22-18-50-32", "2026-10-22-18-31-55"]

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>@TITLE@</title><style>
body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:10px}#bar{position:sticky;top:0;background:#111;padding:6px 0;z-index:5}
button{font-size:15px;padding:6px 12px;margin-right:6px}#cv{border:2px solid #555;cursor:crosshair;max-width:100%}.k{color:#aaa}
</style></head><body><div id="bar"><b>@TITLE@</b> &middot; <span id="pos"></span> &middot; class: <b id="cls"></b>
<span class="k">(1 swimmer, 2 person; drag = new box; Z undo; right-click deletes a NEW box; <b>double-click a yellow box = false detection, marks it red (double-click again to undo)</b>; arrows or A/D change frame)</span><br>
<button onclick="go(-1)">&larr; Prev</button><button onclick="go(1)">Next &rarr;</button><button onclick="undo()">Undo</button><button onclick="save()">Save results</button>
<span id="done"></span></div><canvas id="cv"></canvas><script>
const ITEMS=@ITEMS@, KEY="@KEY@", COL=@COL@; let i=0, cls="swimmer", res={}; try{res=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
const cv=document.getElementById("cv"), g=cv.getContext("2d"); let img=new Image(), drag=null, S=1;
function persist(){try{localStorage.setItem(KEY,JSON.stringify(res))}catch(e){}}
function boxes(){return res[ITEMS[i].id]=res[ITEMS[i].id]||[]}
const DEL=res.__del=res.__del||{};
function draw(){g.drawImage(img,0,0);g.lineWidth=2;for(const b of ITEMS[i].fixed||[]){const d=DEL[b[4]];g.strokeStyle=d?"#ff2d2d":"#ffd400";g.setLineDash([6,4]);g.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);if(d){g.beginPath();g.moveTo(b[0],b[1]);g.lineTo(b[2],b[3]);g.moveTo(b[2],b[1]);g.lineTo(b[0],b[3]);g.stroke()}}
g.setLineDash([]);g.lineWidth=3;for(const b of boxes()){g.strokeStyle=COL[b[4]];g.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1])}
if(drag){g.strokeStyle=COL[cls];g.strokeRect(drag[0],drag[1],drag[2]-drag[0],drag[3]-drag[1])}
document.getElementById("pos").textContent=(i+1)+" / "+ITEMS.length+" (new boxes here: "+boxes().length+")";document.getElementById("cls").textContent=cls;document.getElementById("cls").style.color=COL[cls];
let n=Object.keys(res).filter(k=>res[k].length).length;document.getElementById("done").textContent=" frames with new boxes: "+n}
function load(){img=new Image();img.onload=()=>{cv.width=img.width;cv.height=img.height;draw()};img.src=ITEMS[i].src}
function go(d){i=Math.max(0,Math.min(ITEMS.length-1,i+d));res[ITEMS[i].id]=boxes();persist();load()}
function pt(e){const r=cv.getBoundingClientRect();S=cv.width/r.width;return [(e.clientX-r.left)*S,(e.clientY-r.top)*S]}
cv.onmousedown=e=>{if(e.button!==0)return;const p=pt(e);drag=[p[0],p[1],p[0],p[1]]};
cv.onmousemove=e=>{if(!drag)return;const p=pt(e);drag[2]=p[0];drag[3]=p[1];draw()};
cv.onmouseup=e=>{if(!drag)return;const b=[Math.min(drag[0],drag[2]),Math.min(drag[1],drag[3]),Math.max(drag[0],drag[2]),Math.max(drag[1],drag[3])];drag=null;
if(b[2]-b[0]>=6&&b[3]-b[1]>=6){boxes().push([...b,cls]);persist()}draw()};
cv.oncontextmenu=e=>{e.preventDefault();const p=pt(e),B=boxes();for(let k=B.length-1;k>=0;k--){const b=B[k];if(p[0]>=b[0]&&p[0]<=b[2]&&p[1]>=b[1]&&p[1]<=b[3]){B.splice(k,1);persist();draw();break}}};
cv.ondblclick=e=>{const p=pt(e);let best=null,ba=1e18;for(const b of ITEMS[i].fixed||[]){if(p[0]>=b[0]&&p[0]<=b[2]&&p[1]>=b[1]&&p[1]<=b[3]){const a=(b[2]-b[0])*(b[3]-b[1]);if(a<ba){ba=a;best=b}}}
if(best){if(DEL[best[4]])delete DEL[best[4]];else DEL[best[4]]=1;persist();draw()}};
function undo(){boxes().pop();persist();draw()}
document.onkeydown=e=>{if(e.key=="1")cls="swimmer";else if(e.key=="2")cls="person";else if(e.key=="z"||e.key=="Z")undo();else if(e.key=="ArrowRight"||e.key=="a"||e.key=="A")go(1);else if(e.key=="ArrowLeft"||e.key=="d"||e.key=="D")go(-1);draw()};
function save(){const out=ITEMS.map(it=>({id:it.id,task:it.task,frame:it.frame,key:it.key,deleted:(it.fixed||[]).filter(b=>DEL[b[4]]).map(b=>({id:b[4],kind:b[5]})),boxes:(res[it.id]||[]).map(b=>({x1:b[0],y1:b[1],x2:b[2],y2:b[3],cls:b[4]}))}));
const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{type:"application/json"}));a.download="@OUT@";a.click()}
load();</script></body></html>"""


def page(title, items, key, out):
    return (PAGE.replace("@TITLE@", title).replace("@ITEMS@", json.dumps(items)).replace("@KEY@", key)
            .replace("@COL@", json.dumps(COL)).replace("@OUT@", out))


def final_boxes(s, vid, labels):
    """(task, frame) -> final boxes (swimmer / person / proposal that is not 'not_a_person') for display."""
    seed = json.loads((TR / "data/seed_r1.json").read_text())
    out = {}
    for t in c.check_tasks(s):
        split = "train" if " train " in t["name"] else "val"
        meta = s.get(f"{c.BASE}/api/tasks/{t['id']}/data/meta").json()["frames"]
        for sh in s.get(f"{c.BASE}/api/tasks/{t['id']}/annotations").json()["shapes"]:
            if sh["type"] != "rectangle":
                continue
            name = labels[sh["label_id"]]
            kind = "plain"
            if name == "proposal":
                kind = "proposal"
                name = next((a["value"] for a in sh["attributes"] if a["spec_id"] == vid), "?")
            if name in ("swimmer", "person"):
                key = seed[split][int(meta[sh["frame"]]["name"][:4])]["key"]
                out.setdefault((t["id"], sh["frame"], key), []).append([float(v) for v in sh["points"][:4]] + [sh["id"], kind])
    return out


def make_missing():
    s = c.session()
    pid, vid = c.verdict_spec(s)
    labels = {lb["id"]: lb["name"] for lb in s.get(f"{c.BASE}/api/labels", params={"project_id": 1, "page_size": 100}).json()["results"]}
    fb = final_boxes(s, vid, labels)
    seed = json.loads((TR / "data/seed_r1.json").read_text())
    items = []
    for t in c.check_tasks(s):
        split = "train" if " train " in t["name"] else "val"
        meta = s.get(f"{c.BASE}/api/tasks/{t['id']}/data/meta").json()["frames"]
        for fi, m in enumerate(meta):
            key = seed[split][int(m["name"][:4])]["key"]
            items.append({"id": f"{t['id']}_{fi}", "task": t["id"], "frame": fi, "key": key, "src": "../frames/" + key,
                          "fixed": fb.get((t["id"], fi, key), [])})
    (TR / "data/audit/missing_r1.html").write_text(
        page("Add people who have no box (yellow dashed = existing boxes)", items, "missing_r1", "missing_r1_result.json"), encoding="utf-8")
    print(len(items), "frames ->", TR / "data/audit/missing_r1.html")


def make_test():
    ts = json.loads((TR / "data/test_sets.json").read_text())
    names = ts["A_keypoints"] + ts["B_boxes"]
    items = [{"id": n, "task": 0, "frame": k, "key": n, "src": n, "fixed": []} for k, n in enumerate(names)]
    (TR / "data/test_upload/label_test.html").write_text(
        page("TEST set: draw a box on EVERY person (1 swimmer / 2 person)", items, "test_labels_v1", "test_labels_result.json"), encoding="utf-8")
    print(len(items), "frames ->", TR / "data/test_upload/label_test.html")


STROKES = [("freestyle", "Freestyle"), ("backstroke", "Backstroke"), ("breaststroke", "Breaststroke"), ("butterfly", "Butterfly"),
           ("floating", "Floating / resting"), ("treading_vertical", "Treading water"), ("standing_wading", "Standing / wading"),
           ("underwater", "Underwater"), ("unknown", "Can't tell")]


def make_stroke():
    s = c.session()
    out = []
    for t in sorted(s.get(f"{c.BASE}/api/tasks", params={"search": "STROKE", "page_size": 20}).json()["results"], key=lambda t: t["id"]):
        n = int(t["name"].split()[1])
        clip = SEEDS[n - 1]
        meta = s.get(f"{c.BASE}/api/tasks/{t['id']}/data/meta").json()["frames"]
        for tr in s.get(f"{c.BASE}/api/tasks/{t['id']}/annotations").json()["tracks"]:
            shapes = {sh["frame"]: sh for sh in tr["shapes"] if not sh["outside"]}
            fr = sorted(shapes)[:: max(1, len(shapes) // 14)][:14]
            if len(fr) < 3:
                continue
            imgs = []
            for f in fr:
                x1, y1, x2, y2 = shapes[f]["points"]
                img = cv2.imread(str(TR / "data/stroke" / clip / meta[f]["name"]))
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                half = max(x2 - x1, y2 - y1, 90) * 0.9
                ox, oy = int(max(0, cx - half)), int(max(0, cy - half))
                cr = img[oy:int(min(img.shape[0], cy + half)), ox:int(min(img.shape[1], cx + half))].copy()
                cv2.rectangle(cr, (int(x1 - ox), int(y1 - oy)), (int(x2 - ox), int(y2 - oy)), (0, 255, 255), 1)
                cr = cv2.resize(cr, (300, max(1, int(300 * cr.shape[0] / cr.shape[1]))), interpolation=cv2.INTER_CUBIC)
                imgs.append("data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", cr, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()).decode())
            out.append({"task": t["id"], "track_id": tr.get("id"), "label": tr["label_id"], "clip": clip, "frames": [fr[0], fr[-1]], "imgs": imgs})
    btn = lambda i: "".join(f"<button onclick=\"set({i},'{v}')\">{n}</button>" for v, n in STROKES)
    cards = "".join(f"<div class=\"card\" id=\"c{i}\"><img id=\"im{i}\" src=\"{o['imgs'][0]}\"><p>#{i + 1} &middot; clip {o['clip'][11:]} &middot; frames "
                    f"{o['frames'][0]}-{o['frames'][1]}</p>{btn(i)}</div>" for i, o in enumerate(out))
    meta_js = json.dumps([{k: o[k] for k in ("task", "track_id", "label", "clip", "frames")} for o in out])
    ims_js = json.dumps([o["imgs"] for o in out])
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Stroke tagging</title><style>
body{{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:14px}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:12px}}
.card{{background:#222;padding:8px;border-radius:6px;border:3px solid #444}}.card img{{max-width:100%}}.card.done{{border-color:#2a2}}button{{font-size:13px;padding:5px 8px;margin:2px}}.sel{{background:#2a2;color:#fff}}</style></head><body>
<h2>Stroke tagging: {len(out)} swimmers</h2><p>Each card is a looping clip of one swimmer (yellow box). Pick the stroke they do for most of the clip. Use "Can't tell" instead of guessing. Then <b>Save results</b>.</p>
<button onclick="save()" style="font-size:18px">Save results</button> <span id="n"></span><div class="grid">{cards}</div>
<script>const T={meta_js},IM={ims_js};const res={{}};
IM.forEach((a,i)=>{{let k=0;setInterval(()=>{{k=(k+1)%a.length;document.getElementById('im'+i).src=a[k]}},250)}});
function set(i,v){{res[i]=v;const c=document.getElementById('c'+i);c.className='card done';[...c.querySelectorAll('button')].forEach(b=>b.classList.toggle('sel',b.getAttribute('onclick').includes("'"+v+"'")));document.getElementById('n').textContent=Object.keys(res).length+' / {len(out)}'}}
function save(){{const o=T.map((t,i)=>Object.assign({{stroke:res[i]||null}},t));const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(o,null,1)],{{type:'application/json'}}));a.download='stroke_r1_result.json';a.click()}}</script></body></html>"""
    (TR / "data/audit/stroke_r1.html").write_text(html, encoding="utf-8")
    print(len(out), "tracks ->", TR / "data/audit/stroke_r1.html")


def apply_missing(path):
    s = c.session()
    lab = {lb["name"]: lb["id"] for lb in s.get(f"{c.BASE}/api/labels", params={"project_id": 1, "page_size": 100}).json()["results"]}
    rows = json.load(open(path))
    pid, vid = c.verdict_spec(s)
    dels = {}
    for r in rows:
        for d in r.get("deleted", []):
            dels.setdefault(r["task"], []).append(d)
    for t, L in dels.items():
        shapes = s.get(f"{c.BASE}/api/tasks/{t}/annotations").json()["shapes"]
        prop = {d["id"] for d in L if d["kind"] == "proposal"}
        plain = {d["id"] for d in L if d["kind"] != "proposal"}
        up = []
        for sh in shapes:
            if sh["id"] in prop:
                sh["attributes"] = [{"spec_id": vid, "value": "not_a_person"}]
                up.append(sh)
        if up:
            s.patch(f"{c.BASE}/api/tasks/{t}/annotations", params={"action": "update"}, json={"version": 0, "shapes": up, "tags": [], "tracks": []}).raise_for_status()
        gone = [sh for sh in shapes if sh["id"] in plain]
        if gone:
            s.patch(f"{c.BASE}/api/tasks/{t}/annotations", params={"action": "delete"}, json={"version": 0, "shapes": gone, "tags": [], "tracks": []}).raise_for_status()
        print("task", t, "false detections removed:", len(up) + len(gone))
    by = {}
    for r in rows:
        for b in r["boxes"]:
            by.setdefault(r["task"], []).append({"type": "rectangle", "frame": r["frame"], "label_id": lab[b["cls"]],
                                                 "points": [b["x1"], b["y1"], b["x2"], b["y2"]], "occluded": False, "outside": False,
                                                 "z_order": 0, "group": 0, "source": "manual", "attributes": []})
    for t, sh in by.items():
        s.patch(f"{c.BASE}/api/tasks/{t}/annotations", params={"action": "create"},
                json={"version": 0, "shapes": sh, "tags": [], "tracks": []}).raise_for_status()
        print("task", t, "added", len(sh), "boxes")
    if not by:
        print("no new boxes in the file")


def apply_stroke(path):
    s = c.session()
    act = {lb["id"]: next(a for a in lb["attributes"] if a["name"] == "activity")
           for lb in s.get(f"{c.BASE}/api/labels", params={"project_id": 1, "page_size": 100}).json()["results"] if lb["name"] in ("swimmer", "person")}
    by = {}
    for r in json.load(open(path)):
        if r["stroke"]:
            by.setdefault(r["task"], {})[r["track_id"]] = r["stroke"]
    n = 0
    for t, d in by.items():
        ann = s.get(f"{c.BASE}/api/tasks/{t}/annotations").json()
        up = []
        for tr in ann["tracks"]:
            if tr["id"] in d:
                for sh in tr["shapes"]:  # `activity` is a mutable attribute: it lives on the track's shapes
                    sh["attributes"] = [{"spec_id": act[tr["label_id"]]["id"], "value": d[tr["id"]]}]
                up.append(tr)
        if up:
            s.patch(f"{c.BASE}/api/tasks/{t}/annotations", params={"action": "update"},
                    json={"version": 0, "shapes": [], "tags": [], "tracks": up}).raise_for_status()
            n += len(up)
    print("set activity on", n, "tracks")


def export_test(path):
    images, anns = [], []
    for r in json.load(open(path)):
        iid = len(images)
        images.append({"id": iid, "file_name": r["key"], "width": 960, "height": 1072})
        for b in r["boxes"]:
            w, h = b["x2"] - b["x1"], b["y2"] - b["y1"]
            anns.append({"id": len(anns), "image_id": iid, "category_id": 0 if b["cls"] == "swimmer" else 1,
                         "bbox": [b["x1"], b["y1"], w, h], "area": w * h, "iscrowd": 0})
    (TR / "data/coco/test_hand.json").write_text(json.dumps({
        "info": {"description": "TEST A+B hand labels (boxes)"}, "licenses": [], "images": images, "annotations": anns,
        "categories": [{"id": 0, "name": "swimmer"}, {"id": 1, "name": "person_out_of_water"}]}))
    print(len(images), "frames,", len(anns), "boxes -> data/coco/test_hand.json")


if __name__ == "__main__":
    cmd, what = sys.argv[1], sys.argv[2]
    res = sys.argv[sys.argv.index("--result") + 1] if "--result" in sys.argv else None
    if cmd == "make":
        {"missing": make_missing, "test": make_test, "stroke": make_stroke}[what]()
    elif cmd == "apply":
        {"missing": apply_missing, "stroke": apply_stroke}[what](res or DL / f"{what}_r1_result.json")
    elif cmd == "export":
        export_test(res or DL / "test_labels_result.json")
