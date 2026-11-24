"""Life Guard training job queue + watchdog (training/MASTER_PROMPT.md section 4). Standard library only, so it keeps
working when the training venv is broken; run it with the system Python 3.12.

    python jobqueue.py add --name train_v0_rtmdet --resume-arg "--resume" -- <exe> <args...>
    python jobqueue.py add --experiment --name exp_1280 -- <exe> <args...>
    python jobqueue.py list
    python jobqueue.py run          # runs jobs one after another until the queue is empty (run_queue.ps1)
    python jobqueue.py watchdog     # one check: GPU log, temperature pause, restart dead runner / failed jobs
    python jobqueue.py cancel <id>

State: queue_state.json next to this file (ignored by git). Logs: training/logs/<job id>.log (appended on restart).
Jobs must resume themselves: a restart appends --resume-arg (e.g. "--resume") to the command, and the job continues
from its last checkpoint. Main jobs always run before experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAINING = HERE.parent.parent
STATE = HERE / "queue_state.json"
LOCK = HERE / "runner.lock"
PAUSE = HERE / "PAUSED_TEMPERATURE"
LOGS = TRAINING / "logs"
NOTIFY = TRAINING / "tools" / "notify.ps1"
MAX_RESTARTS = 2
TEMP_PAUSE_C = float(os.environ.get("LIFEGUARD_TRAIN_TEMP_PAUSE_C", 83))  # env override: self-test only
TEMP_RESUME_C = float(os.environ.get("LIFEGUARD_TRAIN_TEMP_RESUME_C", 75))


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load() -> dict:
    if not STATE.exists():
        return {"next_id": 1, "jobs": []}
    for _ in range(20):  # the runner and the watchdog may write at the same moment
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, PermissionError):
            time.sleep(0.1)
    raise RuntimeError(f"cannot read {STATE}")


def save(state: dict) -> None:
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    for _ in range(20):
        try:
            tmp.replace(STATE)
            return
        except PermissionError:
            time.sleep(0.1)
    raise RuntimeError(f"cannot write {STATE}")


def notify(title: str, message: str) -> None:
    log("notify", f"{title}: {message}")
    try:
        subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(NOTIFY), "-Title", title,
             "-Message", message],
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError:
        pass


def log(name: str, line: str) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / f"queue_{name}.log", "a", encoding="utf-8") as f:
        f.write(f"{now()}  {line}\n")


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def kill_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    else:
        os.kill(pid, 9)


def runner_pid() -> int | None:
    try:
        pid = int(LOCK.read_text().strip())
    except (OSError, ValueError):
        return None
    return pid if pid_alive(pid) else None


def next_job(state: dict) -> dict | None:
    for kind in ("main", "experiment"):
        for j in state["jobs"]:
            if j["status"] == "queued" and j["kind"] == kind:
                return j
    return None


# ── commands ──────────────────────────────────────────────────────────────────────────────────────────────────
def cmd_add(a: argparse.Namespace) -> None:
    state = load()
    jid = f"{state['next_id']:03d}_{a.name}"
    state["next_id"] += 1
    state["jobs"].append({
        "id": jid, "name": a.name, "kind": "experiment" if a.experiment else "main", "cmd": a.cmd,
        "cwd": a.cwd or str(TRAINING.parent), "resume_arg": a.resume_arg, "status": "queued", "attempts": 0,
        "restarts": 0, "pid": None, "added": now(), "started": None, "finished": None, "exit_code": None,
        "log": str(LOGS / f"{jid}.log"),
    })
    save(state)
    print(jid)


def cmd_list(_: argparse.Namespace) -> None:
    state = load()
    r = runner_pid()
    print(f"runner: {'running (pid ' + str(r) + ')' if r else 'not running'}   paused: {PAUSE.exists()}")
    for j in state["jobs"]:
        print(f"{j['id']:32} {j['kind']:10} {j['status']:9} attempts {j['attempts']} restarts {j['restarts']} "
              f"exit {j['exit_code']}  {j['started'] or ''} -> {j['finished'] or ''}")


def cmd_cancel(a: argparse.Namespace) -> None:
    state = load()
    for j in state["jobs"]:
        if j["id"] == a.id:
            if j["status"] == "running" and j["pid"]:
                kill_tree(j["pid"])
            j["status"] = "cancelled"
            j["finished"] = now()
    save(state)


def cmd_run(_: argparse.Namespace) -> None:
    if runner_pid():
        print("a runner is already active")
        return
    LOCK.write_text(str(os.getpid()))
    log("runner", f"runner started (pid {os.getpid()})")
    try:
        while True:
            if PAUSE.exists():
                time.sleep(30)
                continue
            state = load()
            job = next_job(state)
            if job is None:
                log("runner", "queue empty, runner exits")
                return
            cmd = list(job["cmd"])
            if job["attempts"] > 0 and job["resume_arg"]:
                cmd += job["resume_arg"].split()
            job["attempts"] += 1
            job["status"] = "running"
            job["started"] = now()
            Path(job["log"]).parent.mkdir(parents=True, exist_ok=True)
            with open(job["log"], "a", encoding="utf-8") as lf:
                lf.write(f"\n===== {now()} attempt {job['attempts']}: {' '.join(cmd)}\n")
                lf.flush()
                # UTF-8 output: a job's log is a file, and Windows' default cp1252 crashes on any non-ASCII print.
                # FOR_DISABLE_CONSOLE_CTRL_HANDLER: the Intel Fortran runtime inside NumPy / SciPy aborts the whole job
                # ("forrtl: error (200): program aborting due to window-CLOSE event") when a console closes or the user
                # signs out (measured 2026-10-28); jobs also get their own process group and no console window.
                env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "FOR_DISABLE_CONSOLE_CTRL_HANDLER": "1"}
                flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)
                p = subprocess.Popen(cmd, cwd=job["cwd"], stdout=lf, stderr=subprocess.STDOUT, env=env,
                                     creationflags=flags)
                job["pid"] = p.pid
                save(state)
                log("runner", f"{job['id']} started (pid {p.pid}, attempt {job['attempts']})")
                code = p.wait()
            state = load()  # the watchdog may have changed the job while it ran
            j = next(x for x in state["jobs"] if x["id"] == job["id"])
            j["pid"] = None
            j["exit_code"] = code
            if j["status"] == "running":  # not paused / cancelled by the watchdog meanwhile
                if code == 0:
                    j["status"], j["finished"] = "done", now()
                    log("runner", f"{j['id']} done")
                elif j["restarts"] < MAX_RESTARTS:
                    j["restarts"] += 1
                    j["status"] = "queued"
                    log("runner", f"{j['id']} exit {code}: restart {j['restarts']}/{MAX_RESTARTS} from checkpoint")
                else:
                    j["status"], j["finished"] = "failed", now()
                    log("runner", f"{j['id']} FAILED after {MAX_RESTARTS} restarts (exit {code})")
                    notify("Life Guard training", f"Job {j['id']} failed after {MAX_RESTARTS} restarts")
            save(state)
    finally:
        try:
            if LOCK.read_text().strip() == str(os.getpid()):
                LOCK.unlink()
        except OSError:
            pass


def gpu_reading() -> dict | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout.strip().split(",")
        t, p, mu, mt, u = (float(x) for x in out[:5])
        return {"temp_c": t, "power_w": p, "mem_used_mb": mu, "mem_total_mb": mt, "util_pct": u}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def start_runner_detached() -> None:
    flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "run"], cwd=str(TRAINING.parent),
                     creationflags=flags, close_fds=True)


def cmd_watchdog(_: argparse.Namespace) -> None:
    g = gpu_reading()
    LOGS.mkdir(parents=True, exist_ok=True)
    gcsv = LOGS / "gpu_watch.csv"
    new = not gcsv.exists()
    with open(gcsv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "temp_c", "power_w", "mem_used_mb", "mem_total_mb", "util_pct", "paused"])
        w.writerow([now(), *(g.values() if g else ["", "", "", "", ""]), PAUSE.exists()])
    state = load()
    running = [j for j in state["jobs"] if j["status"] == "running"]
    # temperature pause: stop the running job (it resumes from its checkpoint later; not counted as a restart)
    if g and g["temp_c"] > TEMP_PAUSE_C and not PAUSE.exists():
        PAUSE.write_text(now())
        for j in running:
            if j["pid"]:
                kill_tree(j["pid"])
            j["status"], j["pid"] = "queued", None
            j["attempts"] = max(j["attempts"], 1)  # its restart passes the resume argument
        save(state)
        log("watchdog", f"GPU {g['temp_c']} C > {TEMP_PAUSE_C} C: paused")
        notify("Life Guard training", f"Paused: GPU at {g['temp_c']:.0f} C")
        return
    if PAUSE.exists() and g and g["temp_c"] < TEMP_RESUME_C:
        PAUSE.unlink()
        log("watchdog", f"GPU {g['temp_c']} C < {TEMP_RESUME_C} C: resumed")
    # a job marked running whose runner is gone (crash, kill, reboot): requeue it within the restart limit
    if not runner_pid():
        changed = False
        for j in running:
            if pid_alive(j["pid"]):
                continue  # an orphaned job still running: leave it, a new runner would start a second one
            changed = True
            j["pid"] = None
            if j["restarts"] < MAX_RESTARTS:
                j["restarts"] += 1
                j["status"] = "queued"
                log("watchdog", f"{j['id']} lost its runner: restart {j['restarts']}/{MAX_RESTARTS}")
            else:
                j["status"], j["finished"] = "failed", now()
                notify("Life Guard training", f"Job {j['id']} failed after {MAX_RESTARTS} restarts")
        if changed:
            save(state)
        if next_job(load()) and not PAUSE.exists() and not any(pid_alive(j["pid"]) for j in running):
            log("watchdog", "queued work and no runner: starting the runner")
            start_runner_detached()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd_name", required=True)
    a = sub.add_parser("add")
    a.add_argument("--name", required=True)
    a.add_argument("--experiment", action="store_true")
    a.add_argument("--cwd")
    a.add_argument("--resume-arg", default="")
    a.add_argument("cmd", nargs=argparse.REMAINDER)
    a.set_defaults(fn=cmd_add)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    sub.add_parser("run").set_defaults(fn=cmd_run)
    sub.add_parser("watchdog").set_defaults(fn=cmd_watchdog)
    c = sub.add_parser("cancel")
    c.add_argument("id")
    c.set_defaults(fn=cmd_cancel)
    args = ap.parse_args()
    if getattr(args, "cmd", None) and args.cmd and args.cmd[0] == "--":
        args.cmd = args.cmd[1:]
    args.fn(args)


if __name__ == "__main__":
    main()
