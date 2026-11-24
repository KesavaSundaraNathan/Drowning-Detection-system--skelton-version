"""Evaluate a containment rule against the owner's audit: a Claude 'not_a_person' box that holds or overlaps a person
box in the same frame is a person (loose / duplicate / group boxes count)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import claude_check_sheets as c
s = c.session(); pid, vid = c.verdict_spec(s)
frames = {}  # (task,key)-> list of (id, box, verdict)
for t in c.check_tasks(s):
    shapes = s.get(f"{c.BASE}/api/tasks/{t['id']}/annotations").json()["shapes"]
    meta = s.get(f"{c.BASE}/api/tasks/{t['id']}/data/meta").json()["frames"]
    seed = json.loads((c.TR/"data/seed_r1.json").read_text())
    split = "train" if " train " in t["name"] else "val"
    for sh in shapes:
        if sh["type"] != "rectangle": continue
        v = next((a["value"] for a in sh["attributes"] if a["spec_id"] == vid), None)
        key = seed[split][int(meta[sh["frame"]]["name"][:4])]["key"]
        frames.setdefault((t["id"], key), []).append((sh["id"], sh["points"][:4], v))
json.dump({f"{k[0]}|{k[1]}": v for k, v in frames.items()}, open("training/data/audit/frames_state.json", "w"))
print(len(frames), "frames")
