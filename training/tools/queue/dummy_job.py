"""Queue self-test job: N 'epochs' of S seconds, a checkpoint after each; --resume continues from the checkpoint."""

import argparse
import json
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--seconds", type=float, default=3)
ap.add_argument("--resume", action="store_true")
a = ap.parse_args()
ck = Path(a.ckpt)
start = json.loads(ck.read_text())["epoch"] if a.resume and ck.exists() else 0
print(f"start at epoch {start} (resume={a.resume})", flush=True)
for e in range(start, a.epochs):
    time.sleep(a.seconds)
    ck.parent.mkdir(parents=True, exist_ok=True)
    ck.write_text(json.dumps({"epoch": e + 1}))
    print(f"epoch {e + 1}/{a.epochs} checkpointed", flush=True)
print("finished", flush=True)
