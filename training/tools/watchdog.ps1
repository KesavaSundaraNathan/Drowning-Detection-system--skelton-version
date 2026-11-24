<#
.SYNOPSIS
  Training watchdog, one check per call (Task Scheduler: at log-on and every 5 minutes): logs GPU temperature, power
  and VRAM to training\logs\gpu_watch.csv; pauses training above 83 C and resumes below 75 C; requeues a job whose
  runner died (at most 2 restarts, then failed + alert); starts the queue runner after a reboot; queues v0 training when the seed review reaches 80 % (seed_trigger.py).
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File training\tools\watchdog.ps1
#>
$ErrorActionPreference = "Stop"
$py = "C:\Program Files\Python312\python.exe"
& $py (Join-Path $PSScriptRoot "queue\jobqueue.py") watchdog
$rc = $LASTEXITCODE
# Phase 6 trigger: queue the v0 export + training once 80 % of the seed review is Completed (queues once)
try { & (Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe") (Join-Path $PSScriptRoot "seed_trigger.py") | Out-Null } catch {}
exit $rc
