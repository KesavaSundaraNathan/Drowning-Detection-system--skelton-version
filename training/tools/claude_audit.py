"""Owner audit of Claude's seed-CHECK answers (owner-approved 2026-10-29: 10 % spot-check before training).

make : every "person" answer plus a random 10 % (seed 0, at least 20) of the "not_a_person" answers among Claude's decided answers in training\\labels\\auto\\claude_check_r1.jsonl
       (swimmer / person / not_a_person; "unsure" ones are left to the owner anyway) -> one offline page
       training\\data\\audit\\audit_r1.html (crops embedded, opens in any browser, nothing is uploaded). The owner
       clicks Right / Wrong per crop and presses "Save results", which downloads audit_r1_result.json.
score: reads that file (from Downloads or --result) -> error rate overall and per answer type, written to
       training\\reports\\claude_check_audit_r1.md.

    training\\.venv\\Scripts\\python.exe training\\tools\\claude_audit.py make
    training\\.venv\\Scripts\\python.exe training\\tools\\claude_audit.py score [--result <file>]
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
TR = ROOT / "training"
OUT = TR / "data" / "audit"
LOG = TR / "labels" / "auto" / "claude_check_r1.jsonl"
WORDS = {"swimmer": "a person (in the water)", "person": "a person (out of the water)", "not_a_person": "NOT a person"}


def crop_b64(key: str, box: list[float]) -> str:
    img = cv2.imread(str(TR / "data/frames" / key))
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    X1, Y1 = int(max(0, x1 - max(bw, 60))), int(max(0, y1 - max(bh, 60)))
    X2, Y2 = int(min(w, x2 + max(bw, 60))), int(min(h, y2 + max(bh, 60)))
    c = img[Y1:Y2, X1:X2].copy()
    cv2.rectangle(c, (int(x1 - X1), int(y1 - Y1)), (int(x2 - X1), int(y2 - Y1)), (0, 255, 255), 2)
    r = 360 / max(c.shape[:2])
    c = cv2.resize(c, None, fx=r, fy=r, interpolation=cv2.INTER_CUBIC)
    return base64.b64encode(cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()).decode()


def make() -> int:
    rows = {}
    for line in LOG.open(encoding="utf-8"):
        r = json.loads(line)
        rows[(r["task"], r["id"])] = r  # the latest answer per box wins
    decided = [r for r in rows.values() if r["verdict"] != "?"]
    rng = random.Random(0)
    pos = [r for r in decided if r["verdict"] != "not_a_person"]  # every "person" answer: a wrong one becomes a false label
    neg = [r for r in decided if r["verdict"] == "not_a_person"]
    sample = pos + rng.sample(neg, min(len(neg), max(20, round(0.1 * len(neg)))))
    rng.shuffle(sample)
    OUT.mkdir(parents=True, exist_ok=True)
    cards = []
    for i, r in enumerate(sample):
        cards.append(f"""<div class="card" data-i="{i}"><img src="data:image/jpeg;base64,{crop_b64(r['key'], r['box'])}">
<p>#{i + 1} &middot; Claude said the yellow box is <b>{WORDS[r['verdict']]}</b></p>
<button onclick="mark({i},'right',this)">Right</button> <button onclick="mark({i},'wrong',this)">Wrong</button></div>""")
    meta = [{"task": r["task"], "id": r["id"], "key": r["key"], "verdict": r["verdict"]} for r in sample]
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Audit of Claude's answers</title><style>
body{{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:16px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:12px}}
.card{{background:#222;padding:8px;border-radius:6px;border:3px solid #222}}.card img{{max-width:100%}}
.card.right{{border-color:#2a2}}.card.wrong{{border-color:#d33}}
button{{font-size:16px;padding:6px 18px;margin-right:6px}}#save{{font-size:18px;padding:10px 24px;margin:16px 0}}
</style></head><body><h2>Audit: {len(sample)} of Claude's {len(decided)} answers (10 %)</h2>
<p>For each crop: is Claude's answer about the <b>yellow box</b> right? "A person" means a real person is inside the
box (a loose box still counts). When done, press <b>Save results</b>: it downloads <code>audit_r1_result.json</code>.
Nothing leaves this PC.</p><p id="count">0 / {len(sample)} answered</p>
<div class="grid">{''.join(cards)}</div><button id="save" onclick="save()">Save results</button>
<script>const meta={json.dumps(meta)};const res={{}};
function mark(i,v,b){{res[i]=v;const c=b.parentElement;c.className='card '+v;
document.getElementById('count').textContent=Object.keys(res).length+' / '+meta.length+' answered';}}
function save(){{const out=meta.map((m,i)=>Object.assign({{owner:res[i]||null}},m));
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{{type:'application/json'}}));
a.download='audit_r1_result.json';a.click();}}</script></body></html>"""
    (OUT / "audit_r1.html").write_text(html, encoding="utf-8")
    print(f"{len(sample)} of {len(decided)} decided answers -> {OUT / 'audit_r1.html'} ({dict(Counter(r['verdict'] for r in sample))})")
    return 0


def score(result: Path | None) -> int:
    if result is None:
        cands = sorted((Path.home() / "Downloads").glob("audit_r1_result*.json"), key=lambda p: p.stat().st_mtime)
        if not cands:
            print("no audit_r1_result.json in Downloads; pass --result")
            return 1
        result = cands[-1]
    rows = [r for r in json.loads(result.read_text()) if r["owner"]]
    by = Counter((r["verdict"], r["owner"]) for r in rows)
    wrong = sum(v for (_, o), v in by.items() if o == "wrong")
    lines = ["# Audit of Claude's seed-CHECK answers (round 1)", "", f"Source: `{result}`; {len(rows)} crops judged.", "",
             f"**Error rate: {wrong}/{len(rows)} = {wrong / max(len(rows), 1):.1%}** (MASTER_PROMPT stopping rule: <= 3 %).", "",
             "| Claude's answer | right | wrong |", "|---|---|---|"]
    for v in ("swimmer", "person", "not_a_person"):
        lines.append(f"| {v} | {by[(v, 'right')]} | {by[(v, 'wrong')]} |")
    (TR / "reports" / "claude_check_audit_r1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "score"])
    ap.add_argument("--result")
    a = ap.parse_args()
    return make() if a.cmd == "make" else score(Path(a.result) if a.result else None)


if __name__ == "__main__":
    sys.exit(main())
