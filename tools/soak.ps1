<#
.SYNOPSIS
  Phase 4 exit criterion: 24 h soak with N simulated RTSP cameras at the ADMITTED camera count.
.DESCRIPTION
  Needs MediaMTX (MIT) and FFmpeg - see tools\rtsp_sim\README.md (not downloaded by the repo).
  Runs on the target GPU box. Report: ci-artifacts\soak_report.json. Pass: every camera
  live_analysed_fraction >= 0.99, no FAULT delivered, bounded API memory growth.
.EXAMPLE
  $env:LIFEGUARD_MEDIAMTX = "C:\tools\mediamtx\mediamtx.exe"; $env:LIFEGUARD_FFMPEG = "C:\tools\ffmpeg\bin\ffmpeg.exe"
  powershell -ExecutionPolicy Bypass -File tools\soak.ps1 -Hours 24
#>
param([double]$Hours = 24, [double]$Fps = 25, [string]$Clips = "", [int]$Cameras = 0)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
if (-not $env:LIFEGUARD_MEDIAMTX -or -not $env:LIFEGUARD_FFMPEG) {
  Write-Host "Set LIFEGUARD_MEDIAMTX and LIFEGUARD_FFMPEG first (tools\rtsp_sim\README.md)" -ForegroundColor Red
  exit 2
}
$env:LIFEGUARD_SOAK_HOURS = "$Hours"
$env:LIFEGUARD_SOAK_FPS = "$Fps"
if ($Clips) { $env:LIFEGUARD_SOAK_CLIPS = $Clips }
if ($Cameras -gt 0) { $env:LIFEGUARD_SOAK_CAMERAS = "$Cameras" }
$env:PYTHONUTF8 = "1"
& (Join-Path $Root ".venv\Scripts\python.exe") -m pytest -m soak tests\soak -s -p no:cacheprovider --junitxml ci-artifacts\pytest-soak.xml
exit $LASTEXITCODE
