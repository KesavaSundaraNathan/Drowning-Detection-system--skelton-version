<#
.SYNOPSIS
  Site acceptance test for an INSTALLED Life Guard (plan Phase 8 / Verification 5).

.DESCRIPTION
  1. starts simulated cameras: MediaMTX (MIT) on 127.0.0.1:8554 only (RTMP/HLS/WebRTC/SRT/API off) and one
     FFmpeg (LGPL build) publisher per camera looping a clip with -c copy (no re-encode); both binaries are
     sha256-checked against deploy\windows\TOOLS.json;
  2. runs tools\acceptance.py against the public HTTPS entry point (TLS, headers, health, 401s, WebSocket refusal,
     CUDA provider, capacity, N cameras for -Minutes with >= 0.99 analysed and nothing dropped, release status,
     and LAST the 30-bad-logins lockout probe - it locks THIS machine's IP out of logins for 15 minutes);
  3. stops everything it started, also on failure.
  The JSON report (default <DataDir>\acceptance\acceptance-<time>.json) carries a harness-format "live" block
  for `python -m lifeguard.eval.release_gate sign --live`.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools\acceptance.ps1 -Base https://lifeguard.pool.local
  powershell -ExecutionPolicy Bypass -File tools\acceptance.ps1 -Base https://localhost:8443 -DataDir build\stage\data -Minutes 5   # staged
#>
param(
  [Parameter(Mandatory = $true)][string]$Base,
  [string]$InstallDir = "",
  [string]$DataDir = "C:\ProgramData\LifeGuard",
  [string]$ToolsDir = "",
  [string]$Python = "",
  [string]$Clip = "",
  [int]$Cameras = 0,
  [double]$Minutes = 60,
  [string]$Out = "",
  [switch]$NoLockoutProbe
)
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $InstallDir) { $InstallDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path) }
if (-not $ToolsDir) { $ToolsDir = Join-Path $InstallDir "build\tools" }
if (-not $Python) {
  foreach ($cand in @((Join-Path $InstallDir ".venv\Scripts\python.exe"), (Join-Path $InstallDir "build\stage\venv\Scripts\python.exe"))) {
    if (Test-Path $cand) { $Python = $cand; break }
  }
}
if (-not $Clip) { $Clip = Join-Path $InstallDir "build\v2_baseline\backend\storage\uploads\5696203ffd4b445d9508dcb788477f2f.mp4" }
if (-not (Test-Path $Clip)) { throw "clip not found: $Clip (pass -Clip <an H.264 .mp4>)" }
if (-not $Out) { $Out = Join-Path $DataDir ("acceptance\acceptance-" + (Get-Date -Format "yyyyMMdd-HHmmss") + ".json") }
$Ca = Join-Path $DataDir "caddy\caddy\pki\authorities\local\root.crt"
if (-not (Test-Path $Ca)) { throw "site root certificate not found at $Ca (tls internal); for an ACME site pass the public CA bundle path via -DataDir layout" }

function Get-Sha256([string]$Path) {
  # .NET directly: Get-FileHash in Windows PowerShell 5.1 silently returns nothing under -WhatIf
  $sha = [System.Security.Cryptography.SHA256]::Create()
  $fs = [System.IO.File]::OpenRead($Path)
  try { return ([BitConverter]::ToString($sha.ComputeHash($fs)) -replace "-", "").ToLowerInvariant() } finally { $fs.Dispose(); $sha.Dispose() }
}

function Get-Tool([string]$Key) {
  $manifest = Get-Content (Join-Path $InstallDir "deploy\windows\TOOLS.json") -Raw | ConvertFrom-Json
  $t = $manifest.tools.$Key
  $path = Join-Path $ToolsDir ($t.exe -replace "/", "\")
  if (-not (Test-Path $path)) { throw "$Key not found at $path" }
  if ((Get-Sha256 $path) -ne $t.exe_sha256) { throw "$Key at $path fails its sha256 pin" }
  return $path
}

$work = Join-Path $env:TEMP ("lifeguard-acceptance-" + [guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Force -Path $work | Out-Null
$procs = New-Object System.Collections.Generic.List[System.Diagnostics.Process]
try {
  $mtx = Get-Tool "mediamtx"
  $ff = Get-Tool "ffmpeg"
  $cfg = Join-Path $work "mediamtx.yml"
  Set-Content -Path $cfg -Encoding ASCII -Value @(
    "logLevel: warn", "rtsp: true", "rtspAddress: 127.0.0.1:8554", "rtspTransports: [tcp]",
    "rtmp: false", "hls: false", "webrtc: false", "srt: false", "api: false", "metrics: false", "pprof: false", "playback: false",
    "moq: false",  # [Phase 11 WP6] MediaMTX v1.21: MoQ is ON by default on ALL interfaces
    "paths:", "  all_others:"
  )
  # MediaMTX writes auto.crt / auto.key into its working directory: keep them in the temp work dir
  $procs.Add((Start-Process -FilePath $mtx -ArgumentList "`"$cfg`"" -WorkingDirectory $work -PassThru -WindowStyle Hidden -RedirectStandardOutput (Join-Path $work "mediamtx.out") -RedirectStandardError (Join-Path $work "mediamtx.err")))
  Start-Sleep -Seconds 2
  $n = $Cameras
  if ($n -le 0) { $n = 8 }  # publish enough streams; acceptance.py uses the admitted count unless -Cameras is given
  for ($i = 0; $i -lt $n; $i++) {
    $a = "-hide_banner -loglevel error -re -stream_loop -1 -i `"$Clip`" -an -c:v copy -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/acc-$i"
    $procs.Add((Start-Process -FilePath $ff -ArgumentList $a -PassThru -WindowStyle Hidden -RedirectStandardError (Join-Path $work "ffmpeg-$i.err")))
  }
  Start-Sleep -Seconds 3
  foreach ($p in $procs) { if ($p.HasExited) { throw "a simulated-camera process exited early (see $work)" } }
  $pyArgs = @((Join-Path $InstallDir "tools\acceptance.py"), "--base", $Base, "--ca", $Ca, "--minutes", "$Minutes", "--out", $Out, "--rtsp", "rtsp://127.0.0.1:8554/acc")
  if ($Cameras -gt 0) { $pyArgs += @("--cameras", "$Cameras") }
  if ($NoLockoutProbe) { $pyArgs += "--no-lockout-probe" }
  & $Python @pyArgs
  $code = $LASTEXITCODE
} finally {
  foreach ($p in $procs) { if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } }
}
exit $code
