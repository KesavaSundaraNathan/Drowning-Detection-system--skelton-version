<#
.SYNOPSIS
  Phase 4 exit criterion: chaos tests - kill the API, kill a camera stream,
  stop the supervisor; each must produce a FAULT notification within STALL_S + 10 s.
.DESCRIPTION
  The camera-stream scenario needs MediaMTX + FFmpeg (tools\rtsp_sim\README.md); without them it is
  skipped and reported as such. Report: ci-artifacts\chaos_report.json.
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools\chaos.ps1
#>
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$env:PYTHONUTF8 = "1"
& (Join-Path $Root ".venv\Scripts\python.exe") -m pytest -m chaos tests\chaos -s -p no:cacheprovider --junitxml ci-artifacts\pytest-chaos.xml
exit $LASTEXITCODE
