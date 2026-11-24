"""Runs sam2_follow.py on each STROKE task that has hand-drawn boxes (frame folder = data/stroke/<clip>)."""
import subprocess
from pathlib import Path
TR = Path(__file__).resolve().parents[1]
PY = str(TR / ".venv" / "Scripts" / "python.exe")
CLIPS = ["2026-10-22-18-42-47", "2026-10-22-18-49-23", "2026-10-22-18-50-32", "2026-10-22-18-31-55"]
for n, clip in enumerate(CLIPS, 1):
    print(f"--- STROKE {n} ({clip})")
    subprocess.run([PY, str(TR / "tools/sam2_follow.py"), "--task-prefix", f"STROKE {n} ", "--frames-dir", str(TR / "data/stroke" / clip)])
