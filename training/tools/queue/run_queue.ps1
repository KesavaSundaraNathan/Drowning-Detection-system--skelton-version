<#
.SYNOPSIS
  Run the training job queue: jobs one after another (main before experiments), each logged to training\logs.
  Runs without Claude Code. A second copy exits at once (single runner).
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File training\tools\queue\run_queue.ps1
#>
$ErrorActionPreference = "Stop"
$py = "C:\Program Files\Python312\python.exe"
& $py (Join-Path $PSScriptRoot "jobqueue.py") run
exit $LASTEXITCODE
