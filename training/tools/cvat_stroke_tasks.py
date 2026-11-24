"""Stroke-tagging tasks from TRAIN clips (never the test set): 4 tasks "STROKE n - <clip>", each one 10 s segment (251 frames at 25 fps),
same workflow as TEST C: owner draws a box on the first frame, follow_stroke.cmd lets SAM 2 follow, owner sets the track attribute `activity`.
    training\\.venv\\Scripts\\python.exe training\tools\\cvat_stroke_tasks.py"""
import sys
from pathlib import Path
import cv2
sys.path.insert(0, str(Path(__file__).parent))
import cvat_test_tasks as T
CLIPS = ["2026-10-22-18-42-47", "2026-10-22-18-49-23", "2026-10-22-18-50-32", "2026-10-22-18-31-55"]
START_S = 120
s = T.session()
pid = next(p["id"] for p in s.get(f"{T.BASE}/api/projects").json()["results"] if p["name"] == "Life Guard swimmers v1")
for n, clip in enumerate(CLIPS, 1):
    d = T.TR / "data" / "stroke" / clip
    d.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(T.ROOT / "training_dataset" / f"{clip}.mp4"))
    f0 = START_S * 25
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    files = []
    for i in range(f0, f0 + 251):
        ok, fr = cap.read()
        if not ok:
            break
        f = d / f"f{i:06d}.jpg"
        if not f.exists():
            cv2.imwrite(str(f), fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        files.append(f)
    T.create(s, pid, f"STROKE {n} - {clip[11:]} (train clip, strokes per swimmer)", files, 251, 0)
